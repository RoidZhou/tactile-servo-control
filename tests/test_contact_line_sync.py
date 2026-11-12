"""Capture/TCP synchronization tests using simulated I/O only."""
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from tactile_policy.contact_line_sync import SynchronizedProbeSource


def frame(stamp):
    return {"capture_timestamp": stamp, "valid": True,
            "force": np.ones((35, 20, 3)), "force6d": np.ones(6)}


def state(stamp, pose=None, epoch=0):
    return {"observation_timestamp": stamp,
            "tcp_pose": np.zeros(6) if pose is None else np.asarray(pose),
            "rtde_timestamp": stamp - 900,
            "rtde_timestamp_consistent": True, "rtde_clock_epoch": epoch}


class CaptureClient:
    def __init__(self, *stamps):
        self.frames = [frame(stamp) for stamp in stamps]
        self.returned = None

    def recording_timestamps(self, after=None):
        return [f["capture_timestamp"] for f in self.frames
                if after is None or f["capture_timestamp"] > after]

    def read_at(self, stamp, skew, after=None):
        if self.returned is not None:
            return self.returned
        candidates = [f for f in self.frames if after is None
                      or f["capture_timestamp"] > after]
        if not candidates:
            return None
        result = min(candidates, key=lambda f: abs(f["capture_timestamp"] - stamp))
        return result if abs(result["capture_timestamp"] - stamp) <= skew else None


class ProbeSyncTests(unittest.TestCase):
    def setUp(self):
        self.clock = SimpleNamespace(now=1000.03)
        self.robot = SimpleNamespace(observation=state(1000.0))
        self.robot.get_observations = lambda: self.robot.observation
        self.tactile = SimpleNamespace(left=CaptureClient(1000.009),
                                       right=CaptureClient(1000.011))
        self.source = SynchronizedProbeSource(
            self.robot, self.tactile, clock=lambda: self.clock.now)

    def prime(self):
        self.assertIsNone(self.source.poll())
        self.robot.observation = state(1000.02)

    def test_interpolates_translation_and_consumes_once(self):
        self.robot.observation = state(1000.0, [0, 0, 0, 0, 0, 0])
        self.prime()
        self.robot.observation = state(1000.02, [2, 4, 6, 0, 0, 0])
        result = self.source.poll()
        np.testing.assert_allclose(result["tcp"], [1, 2, 3, 0, 0, 0])
        self.assertAlmostEqual(result["left_right_delta_ms"], 2, places=6)
        self.assertAlmostEqual(result["tcp_skew_ms"], 10, places=6)
        self.assertAlmostEqual(result["sample_age_s"], .021, places=6)
        self.assertEqual(self.source.cursors, [1000.009, 1000.011])
        self.assertIsNone(self.source.poll())

    def test_selects_newest_match_from_history(self):
        self.tactile.left.frames.append(frame(1000.017))
        self.tactile.right.frames.append(frame(1000.019))
        self.prime()
        result = self.source.poll()
        self.assertAlmostEqual(result["timestamp"], 1000.018)
        self.assertEqual(self.source.cursors, [1000.017, 1000.019])

    def test_rotation_uses_shortest_so3_path_across_pi(self):
        self.robot.observation = state(1000.0, [0, 0, 0, 0, 0, np.deg2rad(170)])
        self.assertIsNone(self.source.poll())
        self.robot.observation = state(1000.02, [0, 0, 0, 0, 0, np.deg2rad(-170)])
        result = self.source.poll()
        rv = result["tcp"][3:]
        np.testing.assert_allclose(Rotation.from_rotvec(rv).as_rotvec(),
                                   [0, 0, np.pi], atol=1e-8)

    def test_no_extrapolation_and_too_wide_tcp_gap(self):
        self.prime()
        self.robot.observation = state(1000.06)
        self.clock.now = 1000.07
        self.assertIsNone(self.source.poll())
        self.assertEqual(self.source.cursors, [None, None])
        self.assertIn("TCP bracket", self.source.last_reason)

    def test_rejected_lr_pair_does_not_consume_captures(self):
        self.tactile.right.frames = [frame(1000.039)]
        self.clock.now = 1000.05
        self.prime()
        self.assertIsNone(self.source.poll())
        self.assertEqual(self.source.cursors, [None, None])
        # Keep the rejected right capture; a newer left capture can pair
        # with it because rejection did not consume either cursor.
        self.tactile.left.frames.append(frame(1000.035))
        self.robot.observation = state(1000.04)
        self.assertIsNotNone(self.source.poll())

    def test_stale_or_missing_captures_are_rejected(self):
        self.prime()
        self.clock.now = 1000.3
        self.assertIsNone(self.source.poll())
        self.assertEqual(self.source.cursors, [None, None])
        self.tactile.left.frames = []
        self.assertIsNone(self.source.poll())

    def test_invalid_and_changed_returned_frame_are_not_consumed(self):
        self.prime()
        for changed in (dict(frame(1000.009), valid=False), frame(1000.010),
                        dict(frame(1000.009), force=np.full((35, 20, 3), np.nan)),
                        {"timestamp": 1000.009, "valid": True}):
            self.tactile.left.returned = changed
            self.assertIsNone(self.source.poll())
            self.assertEqual(self.source.cursors, [None, None])
        self.tactile.left.returned = None
        self.assertIsNotNone(self.source.poll())

    def test_future_and_nonfinite_timestamps_are_rejected(self):
        for stamp in (1001, float("nan"), float("inf"), 0, -1):
            self.source.reset()
            self.tactile.left.frames = [frame(stamp)]
            self.assertIsNone(self.source.poll())
            self.assertEqual(self.source.cursors, [None, None])

    def test_robot_clock_rollback_or_epoch_change_latches_fault(self):
        for observation in (state(999.99), state(1000.02, epoch=1)):
            self.source.reset()
            self.robot.observation = state(1000.0)
            self.assertIsNone(self.source.poll())
            self.robot.observation = observation
            self.assertIsNone(self.source.poll())
            self.assertTrue(self.source.clock_fault)
            self.robot.observation = state(1000.02)
            self.assertIsNone(self.source.poll())

    def test_duplicate_native_with_recomputed_wall_time_cannot_trigger_rollback(self):
        self.robot.observation = state(1000.0)
        self.source.poll()
        original = self.source.states[-1][1].copy()
        for jitter in (-.000001, .000001, -.000000238):
            duplicate = state(1000.0+jitter, np.ones(6))
            duplicate['rtde_timestamp'] = 100.0
            self.robot.observation = duplicate
            self.assertIsNone(self.source.poll())
            self.assertFalse(self.source.clock_fault)
            self.assertEqual(len(self.source.states), 1)
            self.assertEqual(self.source.states[-1][0], 1000.0)
            np.testing.assert_array_equal(self.source.states[-1][1], original)
            self.assertEqual(self.source.cursors, [None, None])
        self.robot.observation = state(1000.02)
        sample = self.source.poll()
        self.assertIsNotNone(sample)
        self.assertFalse(self.source.clock_fault)

    def test_native_progress_with_reversed_mapped_time_is_still_a_clock_fault(self):
        self.robot.observation = state(1000.0)
        self.source.poll()
        observation = state(999.999999)
        observation['rtde_timestamp'] = 100.008
        self.robot.observation = observation
        self.assertIsNone(self.source.poll())
        self.assertTrue(self.source.clock_fault)
        self.assertIn('mapped RTDE clock moved backwards', self.source.last_reason)
        self.assertIn('native 100.0 -> 100.008', self.source.last_reason)
        self.assertEqual(len(self.source.states), 0)
        self.assertEqual(self.source.cursors, [None, None])

    def test_tactile_clock_rollback_and_host_clock_rollback(self):
        self.prime()
        self.tactile.left.frames = [frame(1000.005)]
        self.assertIsNone(self.source.poll())
        self.assertTrue(self.source.clock_fault)
        self.source.reset()
        self.assertIsNone(self.source.poll())
        self.clock.now = 1000.029
        self.assertIsNone(self.source.poll())
        self.assertTrue(self.source.clock_fault)

    def test_inconsistent_tcp_snapshot_does_not_enter_history(self):
        self.robot.observation["rtde_timestamp_consistent"] = False
        self.assertIsNone(self.source.poll())
        self.assertEqual(len(self.source.states), 0)

    def test_future_nonfinite_or_invalid_tcp_observation_is_rejected(self):
        for observation in (state(1001), state(float("nan")),
                            state(1000.0, [0, 0, np.nan, 0, 0, 0])):
            self.source.reset()
            self.robot.observation = observation
            self.assertIsNone(self.source.poll())
            self.assertEqual(self.source.cursors, [None, None])
            self.assertEqual(len(self.source.states), 0)

    def test_rpc_expiration_does_not_consume_capture_cursors(self):
        self.prime()
        original = self.tactile.right.read_at
        def slow_read(*args):
            result = original(*args)
            self.clock.now += .2
            return result
        self.tactile.right.read_at = slow_read
        self.assertIsNone(self.source.poll())
        self.assertEqual(self.source.cursors, [None, None])

    def test_invalid_configuration(self):
        for config in ({"max_age_s": 0}, {"max_tcp_skew_s": float("nan")},
                       {"history_s": .1, "max_age_s": .15}):
            with self.assertRaises(ValueError):
                SynchronizedProbeSource(self.robot, self.tactile, **config)


if __name__ == "__main__":
    unittest.main()
