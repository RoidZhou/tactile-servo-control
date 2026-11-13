"""Bounded tactile-deformation probing controller; no hardware or force claim.

All positions are metres, rotations are rotation vectors in radians, and times
are monotonic seconds. The runner must establish contact and latch a stable
tactile reference before constructing this controller. A small relative normal
displacement is held; this is not force control or proof of external contact.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import numpy as np
from scipy.spatial.transform import Rotation


class ProbeControlError(RuntimeError):
    """Invalid controller input; the runner must stop issuing motion commands."""


def _matrix(rotation):
    # SciPy before 1.4 used as_dcm/from_dcm.
    return rotation.as_matrix() if hasattr(rotation, 'as_matrix') else rotation.as_dcm()


def _rotation(matrix):
    return Rotation.from_matrix(matrix) if hasattr(Rotation, 'from_matrix') else Rotation.from_dcm(matrix)


def _transform(pose):
    value = np.asarray(pose, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ProbeControlError('non-finite pose or deformation')
    if value.shape == (6,):
        result = np.eye(4)
        result[:3, :3] = _matrix(Rotation.from_rotvec(value[3:]))
        result[:3, 3] = value[:3]
        return result
    if value.shape == (4, 4):
        if (not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8)
                or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-7)):
            raise ProbeControlError('invalid rigid transform')
        return value.copy()
    raise ProbeControlError('pose must have shape (6,) or (4,4)')


def _pose(transform):
    return np.r_[transform[:3, 3], _rotation(transform[:3, :3]).as_rotvec()]


def _inverse(transform):
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def _clip_norm(value, maximum):
    norm = np.linalg.norm(value)
    return value * min(1., maximum / max(norm, 1e-15))


def _smooth(value):
    value = np.clip(value, 0., 1.)
    return value ** 3 * (10 - 15 * value + 6 * value ** 2)


@dataclass
class ProbeConfig:
    settle_s: float = 1.
    # Original-project timing at 100 Hz: pushdown <=1200 frames,
    # pre-rock tilt 500 frames, rocking 1.25*2000 frames, release 500 frames.
    # Durations, rather than frame counts, keep the profile meaningful with a
    # slower XENSE inference loop.
    ramp_s: float = 12.
    rock_tilt_s: float = 5.
    rock_period_s: float = 20.
    rock_cycles: float = 1.25
    follow_s: float = 20.
    release_s: float = 5.
    release_timeout_s: float = 8.
    angle_x_rad: float = np.deg2rad(1.)
    angle_y_rad: float = np.deg2rad(.6)
    normal_target_m: float = .0003
    translation_gain_s: float = 1.
    rotation_gain_s: float = 1.
    max_linear_speed_m_s: float = .001
    max_angular_speed_rad_s: float = np.deg2rad(.5)
    max_linear_accel_m_s2: float = .002
    max_angular_accel_rad_s2: float = np.deg2rad(1.)
    max_step_m: float = .0005
    max_step_rad: float = np.deg2rad(.5)
    max_total_translation_m: float = .012
    max_normal_translation_m: float = .001
    max_total_rotation_rad: float = np.deg2rad(4.)
    max_dt_s: float = .25
    tracking_position_m: float = .002
    tracking_angle_rad: float = np.deg2rad(1.)
    tracking_timeout_s: float = 2.
    line_stable_s: float = 2.
    line_max_age_s: float = .5
    line_direction_change_rad: float = np.deg2rad(5.)
    line_position_change_m: float = .002
    min_line_observability: float = 5.
    max_line_rms_m: float = .0015
    max_line_angle_rms_rad: float = np.deg2rad(3.)
    min_line_samples: int = 8
    line_min_excitation: float = 1e-4
    release_position_tolerance_m: float = .0001
    release_angle_tolerance_rad: float = np.deg2rad(.05)
    follow_enabled: bool = True

    def __post_init__(self):
        nonnegative = {
            'settle_s', 'angle_x_rad', 'angle_y_rad',
        }
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == 'follow_enabled':
                if not isinstance(value, bool):
                    raise ValueError('follow_enabled must be boolean')
                continue
            if not np.isfinite(value):
                raise ValueError(item.name + ' must be finite')
            if item.name == 'normal_target_m':
                continue
            if value < 0 or (value == 0 and item.name not in nonnegative):
                raise ValueError(item.name + ' must be positive')
        if self.rock_cycles <= 0:
            raise ValueError('rock_cycles must be positive')
        for name in ('min_line_samples',):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(name + ' must be an integer')
        if abs(self.normal_target_m) > self.max_normal_translation_m:
            raise ValueError('normal target exceeds normal travel limit')


class ProbeController:
    """Feedback controller with the original-project rocking profile.

    The active sequence is push_down -> rock_tilt -> rock -> rock_release.
    All rotations are applied about ``H``: the calibrated nut centre in the
    stable reference TCP frame.  Optional estimated-line following starts
    only after that four-stage measurement sequence is complete.

    ``prediction`` is D_t expressed at the original reference TCP. H is the
    fixed TCP->feedback-point transform. E=inv(D_s)D_t is conjugated by H,
    then expressed in the fixed initial feedback axes before velocity control.
    The tactile reference is never replaced during follow.
    """
    def __init__(self, config, initial_tcp, tcp_to_feedback_pose,
                 initial_reference_prediction=None, reference_stable=True):
        self.config = config
        self.initial = _transform(initial_tcp)
        self.H = _transform(tcp_to_feedback_pose)
        self.H_inv = _inverse(self.H)
        self.reference = _transform(np.zeros(6) if initial_reference_prediction is None
                                    else initial_reference_prediction)
        self.initial_feedback = self.initial @ self.H
        self.stage = 'push_down' if reference_stable else 'settle'
        self.elapsed = 0.
        self.last_time = None
        self.velocity = np.zeros(6)
        self.tcp_velocity = np.zeros(6)
        self.last_command = self.initial.copy()
        self.tracking_elapsed = 0.
        self.angle_path_rad = 0.
        self.translation_path_m = 0.
        self.finished = False
        self.status = 'running'
        self.reason = ''
        self.release_reason = ''
        self.desired_feedback = np.zeros(6)
        self.release_from = np.zeros(6)
        self.follow_target = np.zeros(6)
        self.line = None
        self.line_anchor = None
        self.line_stamp = None
        self.line_stable_since = None
        self.line_stable_duration = 0.
        self.line_qualified = False
        self.line_reset_reason = ''

    def pause(self, t_monotonic):
        """Discard velocity and elapsed-gap integration when samples are absent."""
        if not np.isfinite(t_monotonic):
            raise ProbeControlError('non-finite monotonic time')
        if self.last_time is not None and t_monotonic < self.last_time:
            raise ProbeControlError('monotonic time moved backwards')
        self.last_time = float(t_monotonic)
        self.velocity[:] = 0
        self.tcp_velocity[:] = 0
        self._reset_line('sampling_paused')

    def _reset_line(self, reason):
        self.line_qualified = False
        self.line_stable_since = None
        self.line_stable_duration = 0.
        self.line_anchor = None
        self.line_reset_reason = reason

    def _check_line(self, now, estimate):
        c = self.config
        if estimate is None:
            self._reset_line('missing_line')
            return False
        try:
            stamp = float(estimate.get('timestamp', estimate.get('source_timestamp', np.nan)))
            point = np.asarray(estimate['center_base'], dtype=float).reshape(3)
            direction = np.asarray(estimate['direction_base'], dtype=float).reshape(3)
            observability = float(estimate.get('observability', 0.))
            rms = float(estimate.get('point_rms_m', np.inf))
            angle_rms = float(estimate.get('angle_rms_rad', np.inf))
            samples = int(estimate.get('samples', 0))
            singular = np.asarray(estimate.get('singular_values', [1., 1., 0.]), dtype=float)
            valid = (np.isfinite(np.r_[stamp, point, direction, observability, rms, angle_rms]).all()
                     and np.linalg.norm(direction) > 1e-10
                     and estimate.get('valid', True) and not estimate.get('degenerate', False)
                     and estimate.get('optimizer_success', True)
                     and samples >= c.min_line_samples and observability >= c.min_line_observability
                     and 0 <= rms <= c.max_line_rms_m and 0 <= angle_rms <= c.max_line_angle_rms_rad
                     and singular.size >= 2 and np.isfinite(singular).all()
                     and singular[1] > c.line_min_excitation)
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            self._reset_line('invalid_or_degenerate_line')
            return False
        if stamp > now + 1e-6 or now - stamp > c.line_max_age_s:
            self._reset_line('stale_or_future_line')
            return False
        if self.line_stamp is not None and stamp < self.line_stamp:
            self._reset_line('line_time_moved_backwards')
            return False
        if self.line_stamp is not None and stamp == self.line_stamp:
            # Repeated cached estimates never acquire extra stable duration.
            return self.line_anchor is not None
        direction /= np.linalg.norm(direction)
        if self.line_stamp is not None and stamp - self.line_stamp > c.line_max_age_s:
            self._reset_line('line_observation_gap')
        self.line_stamp = stamp
        changed = False
        if self.line_anchor is not None:
            p0, d0 = self.line_anchor
            if direction @ d0 < 0:
                direction = -direction
            angle = np.arccos(np.clip(direction @ d0, -1., 1.))
            distance = np.linalg.norm(np.cross(point - p0, d0))
            changed = angle > c.line_direction_change_rad or distance > c.line_position_change_m
        if self.line_anchor is None or changed:
            self.line_anchor = (point.copy(), direction.copy())
            self.line_stable_since = stamp
            self.line_stable_duration = 0.
            self.line_qualified = False
            self.line_reset_reason = 'line_axis_changed' if changed else 'new_line_candidate'
        else:
            self.line_stable_duration = stamp - self.line_stable_since
            self.line_qualified = self.line_stable_duration >= c.line_stable_s
        self.line = (point.copy(), direction.copy())
        return True

    def _transition(self, stage):
        self.stage = stage
        self.elapsed = 0.

    def _release(self, reason):
        self.release_from = self.desired_feedback.copy()
        self.release_reason = reason
        self._transition('release')

    def _target(self):
        c = self.config
        desired = np.zeros(6)
        if self.stage == 'push_down':
            desired[2] = c.normal_target_m * _smooth(self.elapsed / c.ramp_s)
        elif self.stage == 'rock_tilt':
            # Original rocking() first ramps desired roll from zero to one
            # degree while desired pitch remains zero (500 frames at 100 Hz).
            desired[2] = c.normal_target_m
            desired[3] = c.angle_x_rad * _smooth(self.elapsed / c.rock_tilt_s)
        elif self.stage == 'rock':
            phase = 2 * np.pi * self.elapsed / c.rock_period_s
            desired[2] = c.normal_target_m
            desired[3:5] = [c.angle_x_rad * np.cos(phase), c.angle_y_rad * np.sin(phase)]
        elif self.stage == 'rock_release':
            # At 1.25 cycles the ellipse ends at (roll, pitch)=(0,+Ay).
            # Match the original 500-frame release by reducing pitch to zero.
            desired[2] = c.normal_target_m
            desired[4] = c.angle_y_rad * (1 - _smooth(self.elapsed / c.release_s))
        elif self.stage == 'follow':
            desired = self.follow_target.copy()
        elif self.stage == 'release':
            desired = self.release_from * (1 - _smooth(self.elapsed / c.release_s))
        self.desired_feedback = desired
        return desired

    def _result(self, actual, desired, measured, error, dt, increment=None,
                candidate_command=None, travel_diagnostics=None):
        return dict(command=_pose(actual), stage=self.stage, status=self.status,
                    reason=self.reason, finished=self.finished, dt=float(dt),
                    release_reason=self.release_reason,
                    desired_deformation=_pose(self.reference @ self.H @ _transform(desired) @ self.H_inv),
                    measured_deformation=_pose(self.reference @ self.H @ _transform(measured) @ self.H_inv),
                    measured_feedback_deformation=measured.copy(), error=error.copy(),
                    desired_feedback_deformation=desired.copy(),
                    line_qualified=bool(self.line_qualified), line_stable_s=self.line_stable_duration,
                    line_reset_reason=self.line_reset_reason, velocity=self.velocity.copy(),
                    tcp_velocity_world=self.tcp_velocity.copy(),
                    command_increment=np.zeros(6) if increment is None else increment.copy(),
                    candidate_command=_pose(actual if candidate_command is None else candidate_command),
                    travel_diagnostics=travel_diagnostics,
                    angular_path_rad=self.angle_path_rad, translation_path_m=self.translation_path_m)

    def _fail(self, actual, reason, desired, measured, error, dt,
              candidate_command=None, travel_diagnostics=None):
        self.finished = True
        self.status = 'failed'
        self.reason = reason
        self.velocity[:] = 0
        self.tcp_velocity[:] = 0
        return self._result(actual, desired, measured, error, dt,
                            candidate_command=candidate_command,
                            travel_diagnostics=travel_diagnostics)

    def step(self, t_monotonic, tcp, prediction, line_estimate=None):
        c = self.config
        actual, D = _transform(tcp), _transform(prediction)
        if not np.isfinite(t_monotonic):
            raise ProbeControlError('non-finite monotonic time')
        now = float(t_monotonic)
        if self.last_time is not None and now < self.last_time:
            raise ProbeControlError('monotonic time moved backwards')
        dt = 0. if self.last_time is None else now - self.last_time
        self.last_time = now
        measured = _pose(self.H_inv @ _inverse(self.reference) @ D @ self.H)
        desired = self._target()
        Ds = self.reference @ self.H @ _transform(desired) @ self.H_inv
        E = self.H_inv @ _inverse(Ds) @ D @ self.H
        # E is in desired feedback axes. Rotate it into the initial feedback
        # axes used by velocities; do not subtract Euler angles.
        error = np.r_[_transform(desired)[:3, :3] @ E[:3, 3],
                      _transform(desired)[:3, :3] @ _rotation(E[:3, :3]).as_rotvec()]
        if self.finished:
            return self._result(actual, desired, measured, error, 0.)
        line_valid = self._check_line(now, line_estimate)
        if self.stage == 'follow' and (not line_valid or not self.line_qualified):
            return self._fail(actual, 'follow_line_invalid_or_unstable', desired, measured, error, dt)
        if dt == 0:
            return self._result(actual, desired, measured, error, dt)
        if dt > c.max_dt_s:
            self.velocity[:] = 0
            self.tcp_velocity[:] = 0
            self._reset_line('sample_gap_paused')
            self.last_command = actual.copy()
            self.reason = 'sample_gap_paused'
            return self._result(actual, desired, measured, error, 0.)
        tracking = _pose(_inverse(actual) @ self.last_command)
        if (np.linalg.norm(tracking[:3]) > c.tracking_position_m
                or np.linalg.norm(tracking[3:]) > c.tracking_angle_rad):
            self.velocity[:] = 0
            self.tcp_velocity[:] = 0
            self.tracking_elapsed += dt
            if self.tracking_elapsed >= c.tracking_timeout_s:
                return self._fail(actual, 'tracking_timeout', desired, measured, error, dt)
            self.reason = 'tracking_paused'
            return self._result(actual, desired, measured, error, 0.)
        self.tracking_elapsed = 0.
        self.reason = ''
        self.elapsed += dt
        if self.stage == 'settle' and self.elapsed >= c.settle_s:
            self._transition('push_down')
        elif self.stage == 'push_down' and self.elapsed >= c.ramp_s:
            self._transition('rock_tilt')
        elif self.stage == 'rock_tilt' and self.elapsed >= c.rock_tilt_s:
            self._transition('rock')
        elif self.stage == 'rock':
            if self.elapsed >= c.rock_period_s * c.rock_cycles:
                self._transition('rock_release')
        elif self.stage == 'rock_release' and self.elapsed >= c.release_s:
            # Completion is handled after recomputing the final release target
            # and its feedback error below.
            pass
        elif self.stage == 'follow' and self.elapsed >= c.follow_s:
            self._release('completed')
        desired = self._target()
        Ds = self.reference @ self.H @ _transform(desired) @ self.H_inv
        E = self.H_inv @ _inverse(Ds) @ D @ self.H
        desired_R = _transform(desired)[:3, :3]
        error = np.r_[desired_R @ E[:3, 3], desired_R @ _rotation(E[:3, :3]).as_rotvec()]
        if self.stage == 'rock_release' and self.elapsed >= c.release_s:
            # The original implementation advances by a fixed 500 frames and
            # stops; it does not gate phase completion on tactile convergence.
            if c.follow_enabled and self.line_qualified:
                self.follow_target = desired.copy()
                self._transition('follow')
                desired = self._target()
            else:
                self.stage, self.finished = 'done', True
                self.status = 'no_valid_line' if c.follow_enabled else 'completed'
                self.reason = 'no_valid_line' if c.follow_enabled else 'profile_completed'
                self.velocity[:] = 0
                self.tcp_velocity[:] = 0
                return self._result(actual, desired, measured, error, dt)
        if self.stage == 'release' and self.elapsed >= c.release_s:
            if (np.linalg.norm(error[:3]) <= c.release_position_tolerance_m
                    and np.linalg.norm(error[3:]) <= c.release_angle_tolerance_rad):
                self.stage, self.finished = 'done', True
                self.status = 'completed' if self.release_reason in ('completed', 'follow_disabled') else 'no_valid_line'
                self.reason = self.release_reason
                self.velocity[:] = 0
                self.tcp_velocity[:] = 0
                return self._result(actual, desired, measured, error, dt)
            if self.elapsed >= c.release_s + c.release_timeout_s:
                return self._fail(actual, 'release_timeout', desired, measured, error, dt)
        if self.stage == 'settle':
            self.velocity[:] = 0
            self.tcp_velocity[:] = 0
            self.last_command = actual.copy()
            return self._result(actual, desired, measured, error, dt)
        request = np.r_[-c.translation_gain_s * error[:3], -c.rotation_gain_s * error[3:]]
        world_R = self.initial_feedback[:3, :3]
        if self.stage == 'follow':
            direction_local = world_R.T @ self.line[1]
            # Along-line motion is an explicit small sinusoidal probe. Only
            # perpendicular angle error is corrected, avoiding opposing the probe.
            request[3:] -= direction_local * (request[3:] @ direction_local)
            request[3:] += direction_local * (c.angle_x_rad * 2 * np.pi / c.rock_period_s
                                              * np.cos(2 * np.pi * self.elapsed / c.rock_period_s))
        request[:3] = _clip_norm(request[:3], c.max_linear_speed_m_s)
        request[3:] = _clip_norm(request[3:], c.max_angular_speed_rad_s)
        self.velocity[:3] += _clip_norm(request[:3] - self.velocity[:3], c.max_linear_accel_m_s2 * dt)
        self.velocity[3:] += _clip_norm(request[3:] - self.velocity[3:], c.max_angular_accel_rad_s2 * dt)
        increment = np.r_[_clip_norm(self.velocity[:3] * dt, c.max_step_m),
                          _clip_norm(self.velocity[3:] * dt, c.max_step_rad)]
        feedback = actual @ self.H
        world_omega = world_R @ increment[3:]
        position = feedback[:3, 3].copy()
        orientation = feedback[:3, :3].copy()
        if self.stage == 'follow':
            point, direction = self.line
            parallel = direction * (world_omega @ direction)
            R_parallel = _matrix(Rotation.from_rotvec(parallel))
            position = point + R_parallel @ (position - point)
            orientation = _matrix(Rotation.from_rotvec(world_omega - parallel)) @ R_parallel @ orientation
        else:
            orientation = _matrix(Rotation.from_rotvec(world_omega)) @ orientation
        feedback[:3, :3] = orientation
        feedback[:3, 3] = position + world_R @ increment[:3]
        command = feedback @ self.H_inv
        # Limit the complete TCP motion as well: pivot rotation creates TCP
        # translation even when feedback-point translation is zero.
        world_step = np.r_[command[:3, 3] - actual[:3, 3],
                           _rotation(command[:3, :3] @ actual[:3, :3].T).as_rotvec()]
        tcp_request = world_step / dt
        tcp_request[:3] = _clip_norm(tcp_request[:3], min(c.max_linear_speed_m_s, c.max_step_m / dt))
        tcp_request[3:] = _clip_norm(tcp_request[3:], min(c.max_angular_speed_rad_s, c.max_step_rad / dt))
        self.tcp_velocity[:3] += _clip_norm(tcp_request[:3] - self.tcp_velocity[:3],
                                           c.max_linear_accel_m_s2 * dt)
        self.tcp_velocity[3:] += _clip_norm(tcp_request[3:] - self.tcp_velocity[3:],
                                           c.max_angular_accel_rad_s2 * dt)
        command[:3, 3] = actual[:3, 3] + self.tcp_velocity[:3] * dt
        command[:3, :3] = _matrix(Rotation.from_rotvec(self.tcp_velocity[3:] * dt)) @ actual[:3, :3]
        travel = _pose(_inverse(self.initial) @ command)
        normal_travel = (world_R.T @ ((command @ self.H)[:3, 3] - self.initial_feedback[:3, 3]))[2]
        translation_m = np.linalg.norm(travel[:3])
        normal_m = abs(normal_travel)
        rotation_rad = np.linalg.norm(travel[3:])
        travel_diagnostics = {
            'translation_mm': float(translation_m * 1000.),
            'translation_limit_mm': float(c.max_total_translation_m * 1000.),
            'translation_margin_mm': float((c.max_total_translation_m - translation_m) * 1000.),
            'normal_signed_mm': float(normal_travel * 1000.),
            'normal_abs_mm': float(normal_m * 1000.),
            'normal_limit_mm': float(c.max_normal_translation_m * 1000.),
            'normal_margin_mm': float((c.max_normal_translation_m - normal_m) * 1000.),
            'rotation_deg': float(np.rad2deg(rotation_rad)),
            'rotation_limit_deg': float(np.rad2deg(c.max_total_rotation_rad)),
            'rotation_margin_deg': float(np.rad2deg(c.max_total_rotation_rad - rotation_rad)),
        }
        if (translation_m > c.max_total_translation_m
                or normal_m > c.max_normal_translation_m
                or rotation_rad > c.max_total_rotation_rad):
            return self._fail(actual, 'motion_travel_limit', desired, measured, error, dt,
                              candidate_command=command, travel_diagnostics=travel_diagnostics)
        step = _pose(_inverse(actual) @ command)
        self.angle_path_rad += np.linalg.norm(step[3:])
        self.translation_path_m += np.linalg.norm(step[:3])
        self.last_command = command.copy()
        return self._result(command, desired, measured, error, dt, step,
                            candidate_command=command, travel_diagnostics=travel_diagnostics)
