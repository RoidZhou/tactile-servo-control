"""Single-threaded capture/TCP pairing for active contact probing.

Only SDK capture times and mapped RTDE state times are compared.  The robot
clock mapping still has an unknown minimum transport delay; interpolation
does not eliminate that calibration uncertainty.
"""
from bisect import bisect_left
from collections import deque
import time

import numpy as np
from scipy.spatial.transform import Rotation


class SynchronizedProbeSource:
    """Poll existing clients on their owning thread; never share ZMQ sockets.

    A successful poll consumes both capture cursors.  Unsuccessful polls leave
    them unchanged.  ``clock_fault`` is latched until explicit ``reset()`` or
    reconstruction, so a clock restart cannot silently mix reference epochs.
    ``max_tcp_skew_s`` bounds each endpoint's distance from the pair midpoint
    (and therefore bounds the interpolation span by twice that value).
    """

    def __init__(self, robot, tactile, max_lr_skew_s=0.020,
                 max_tcp_skew_s=0.020, max_age_s=0.15, history_s=2.0,
                 clock=None):
        for name, value in (("max_lr_skew_s", max_lr_skew_s),
                            ("max_tcp_skew_s", max_tcp_skew_s),
                            ("max_age_s", max_age_s), ("history_s", history_s)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if history_s < max_age_s:
            raise ValueError("history_s must cover max_age_s")
        self.robot = robot
        self.tactile = tactile
        self.max_lr_skew_s = float(max_lr_skew_s)
        self.max_tcp_skew_s = float(max_tcp_skew_s)
        self.max_age_s = float(max_age_s)
        self.history_s = float(history_s)
        self._clock = time.time if clock is None else clock
        self.reset()

    def reset(self):
        self.states = deque(maxlen=4096)
        self.cursors = [None, None]
        self.last_seen_captures = [None, None]
        self.last_native = None
        self.clock_epoch = None
        self._epoch_seen = False
        self.last_host_time = None
        self.clock_fault = False
        self.last_reason = "waiting for synchronized captures and TCP history"
        self.last_diagnostics = {}
        self.accepted_count = 0

    def _unavailable(self, reason, clock_fault=False):
        self.last_reason = str(reason)
        self.clock_fault = self.clock_fault or clock_fault
        self.last_diagnostics = {"reason": self.last_reason,
                                 "clock_fault": self.clock_fault,
                                 "tcp_history_samples": len(self.states)}
        return None

    @staticmethod
    def _timestamp(value):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("non-finite or non-positive timestamp")
        return value

    def _now(self):
        now = self._timestamp(self._clock())
        if self.last_host_time is not None and now < self.last_host_time:
            raise RuntimeError("host clock moved backwards; reset required")
        self.last_host_time = now
        return now

    def _add_state(self, observation, now):
        if not isinstance(observation, dict) or not observation.get(
                "rtde_timestamp_consistent", False):
            return "waiting for a consistent mapped RTDE snapshot"
        stamp = self._timestamp(observation.get("observation_timestamp"))
        tcp = np.asarray(observation.get("tcp_pose"), dtype=np.float64)
        if tcp.shape != (6,) or not np.isfinite(tcp).all():
            raise ValueError("invalid TCP pose")
        if stamp > now:
            raise ValueError("RTDE state timestamp is in the future")
        epoch = observation.get("rtde_clock_epoch")
        if self._epoch_seen and epoch != self.clock_epoch:
            self.states.clear()
            self._unavailable("RTDE clock epoch changed; reset required", True)
            return self.last_reason
        self.clock_epoch, self._epoch_seen = epoch, True
        native = observation.get("rtde_timestamp")
        if native is not None:
            native = float(native)
            if not np.isfinite(native):
                raise ValueError("invalid native RTDE timestamp")
            if self.last_native is not None and native < self.last_native:
                self.states.clear()
                self._unavailable("native RTDE clock moved backwards; reset required", True)
                return self.last_reason
        if self.states and stamp < self.states[-1][0]:
            self.states.clear()
            self._unavailable("mapped RTDE clock moved backwards; reset required", True)
            return self.last_reason
        if self.states and (stamp == self.states[-1][0]
                            or (native is not None and native == self.last_native)):
            return "duplicate RTDE snapshot"
        self.states.append((stamp, tcp.copy()))
        self.last_native = native
        while self.states and self.states[0][0] < now - self.history_s:
            self.states.popleft()
        return None

    def _tcp_at(self, stamp):
        if not self.states:
            return None
        times = [item[0] for item in self.states]
        index = bisect_left(times, stamp)
        if index < len(times) and times[index] == stamp:
            return self.states[index][1].copy(), 0.0, 0.0
        if index == 0 or index == len(times):
            return None  # Never extrapolate beyond measured robot feedback.
        t0, p0 = self.states[index - 1]
        t1, p1 = self.states[index]
        skew = max(stamp - t0, t1 - stamp)
        if skew > self.max_tcp_skew_s:
            return None
        alpha = (stamp - t0) / (t1 - t0)
        first, second = Rotation.from_rotvec(p0[3:]), Rotation.from_rotvec(p1[3:])
        # Geodesic interpolation on SO(3), including rotations across +/-pi.
        relative = (first.inv() * second).as_rotvec()
        rotation = first * Rotation.from_rotvec(alpha * relative)
        tcp = np.r_[p0[:3] + alpha * (p1[:3] - p0[:3]), rotation.as_rotvec()]
        return tcp, skew, t1 - t0

    def _captures(self, client, side, now):
        values = client.recording_timestamps(self.cursors[side])
        now = self._now()  # The RPC may return a capture newer than its request.
        times = sorted(set(self._timestamp(value) for value in values))
        if times and times[-1] > now:
            raise ValueError("tactile capture timestamp is in the future")
        previous = self.last_seen_captures[side]
        if times and previous is not None and times[-1] < previous:
            self._unavailable("tactile capture clock moved backwards; reset required", True)
            return []
        if times:
            self.last_seen_captures[side] = times[-1]
        cursor = self.cursors[side]
        return [value for value in times if now - value <= self.max_age_s
                and (cursor is None or value > cursor)]

    def _valid_frame(self, frame, expected_stamp, cursor, now):
        if not isinstance(frame, dict) or not frame.get("valid", False):
            return False
        stamp = self._timestamp(frame.get("capture_timestamp"))
        if cursor is not None and stamp < cursor:
            raise RuntimeError("returned tactile capture clock moved backwards; reset required")
        if (abs(stamp - expected_stamp) > 1e-6 or stamp > now
                or now - stamp > self.max_age_s
                or (cursor is not None and stamp <= cursor)):
            return False
        force = np.asarray(frame.get("force"), dtype=np.float64)
        wrench = np.asarray(frame.get("force6d"), dtype=np.float64)
        return (force.ndim == 3 and force.shape[-1] == 3 and force.size > 0
                and wrench.shape == (6,) and np.isfinite(force).all()
                and np.isfinite(wrench).all())

    def poll(self):
        """Return a fresh, bounded-age capture pair and interpolated actual TCP.

        ``sample_age_s`` conservatively measures the older of the two captures.
        Diagnostics never substitute a publication/receipt time for capture.
        """
        if self.clock_fault:
            return None
        try:
            observation = self.robot.get_observations()
            now = self._now()
            state_reason = self._add_state(observation, now)
            if self.clock_fault:
                return None
            clients = (self.tactile.left, self.tactile.right)
            available = [self._captures(client, side, now)
                         for side, client in enumerate(clients)]
            if self.clock_fault:
                return None
            if not all(available):
                return self._unavailable("waiting for fresh bilateral captures within age limit")
            pairs = [(0.5 * (left + right), left, right)
                     for left in available[0] for right in available[1]
                     if abs(left - right) <= self.max_lr_skew_s]
            if not pairs:
                return self._unavailable("no left/right capture pair within skew limit")
            for midpoint, left_stamp, right_stamp in sorted(pairs, reverse=True):
                interpolated = self._tcp_at(midpoint)
                if interpolated is None:
                    continue
                stamps = (left_stamp, right_stamp)
                frames = [client.read_at(stamp, 1e-6, cursor)
                          for client, stamp, cursor in zip(clients, stamps, self.cursors)]
                returned_at = self._now()
                if not all(self._valid_frame(frame, stamp, cursor, returned_at)
                           for frame, stamp, cursor in zip(frames, stamps, self.cursors)):
                    return self._unavailable("selected capture changed, expired, or is invalid")
                tcp, tcp_skew, tcp_span = interpolated
                self.cursors = list(stamps)
                self.accepted_count += 1
                self.last_reason = "ok"
                self.last_diagnostics = {"reason": "ok", "clock_fault": False,
                                         "tcp_history_samples": len(self.states)}
                return {"left": frames[0], "right": frames[1], "tcp": tcp,
                        "timestamp": midpoint, "left_stamp": left_stamp,
                        "right_stamp": right_stamp,
                        "left_right_delta_ms": abs(left_stamp - right_stamp) * 1000,
                        "tcp_skew_ms": tcp_skew * 1000,
                        "tcp_span_ms": tcp_span * 1000,
                        "sample_age_s": returned_at - min(stamps)}
            return self._unavailable(state_reason or "no bounded TCP bracket for capture midpoint")
        except (ValueError, TypeError, OverflowError, RuntimeError, KeyError) as exc:
            message = str(exc)
            return self._unavailable(message, "backwards" in message.lower())
