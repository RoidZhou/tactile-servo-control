"""Surface geometry and failure gates, using synthetic poses only."""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tactile_policy.contact_line_surface import SurfaceContactLineEstimator


def rx(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def ry(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


class SurfaceContactTests(unittest.TestCase):
    @staticmethod
    def estimator(**kwargs):
        settings = dict(min_samples=12, max_solve_seconds=2.)
        settings.update(kwargs)
        return SurfaceContactLineEstimator([0., 0., 0.], [0., 0., 1.], **settings)

    @staticmethod
    def append_plane_sequence(estimator, center=None, slides=True, offset_noise=None):
        center = np.array([.02, -.01, .03]) if center is None else np.asarray(center)
        angles = np.deg2rad(np.linspace(-4, 4, 25))
        for i, angle in enumerate(angles):
            rotation = rx(angle)
            slide = .005 * np.sin(i / 3.) if slides else 0.
            position = center + rotation @ np.array([0., slide, 0.])
            if offset_noise is not None:
                position += rotation[:, 2] * offset_noise[i]
            estimator.append(1000. + i * .05, position, rotation,
                             np.zeros(6), np.r_[position, angle, 0., 0.])

    def test_known_line_survives_material_point_sliding(self):
        estimator = self.estimator()
        center = np.array([.02, -.01, .03])
        self.append_plane_sequence(estimator, center=center)
        result = estimator.estimate()
        self.assertTrue(result['valid'], result['reason'])
        self.assertGreater(result['normal_excitation_deg'], 7.9)
        self.assertGreater(result['observability'], 1e6)
        self.assertLess(np.linalg.norm(np.cross(result['center_base'] - center, [1., 0., 0.])), 1e-6)
        self.assertGreater(abs(result['direction_base'][0]), .999999)
        self.assertLess(result['raw_plane_rms_mm'], 1e-6)
        self.assertLess(result['raw_angle_rms_deg'], 1e-6)
        self.assertEqual(result['measurement_position_rms_m'], 0.)
        self.assertEqual(result['measurement_rotation_rms_rad'], 0.)
        self.assertTrue(result['external_contact_evidence_required'])
        self.assertLess(result['tcp_only_direction_difference_deg'], 1e-5)
        # No single material q can occupy the fixed world line through these
        # alternating translations: the old Rq+p fixed-point residual is large.
        samples = estimator.snapshot()
        a = np.concatenate([s['R'] - samples[0]['R'] for s in samples[1:]])
        b = np.concatenate([-(s['p'] - samples[0]['p']) for s in samples[1:]])
        q = np.linalg.lstsq(a, b, rcond=None)[0]
        self.assertGreater(np.sqrt(np.mean((a @ q - b)**2)), .001)

    def test_small_measurement_noise_has_finite_metrics(self):
        estimator = self.estimator()
        noise = np.random.RandomState(42).normal(0., 1e-6, 25)
        self.append_plane_sequence(estimator, offset_noise=noise)
        result = estimator.estimate()
        self.assertTrue(result['valid'], result['reason'])
        self.assertLess(result['raw_plane_rms_mm'], .01)
        self.assertTrue(np.isfinite(result['center_base']).all())
        self.assertTrue(np.isfinite(result['direction_base']).all())
        self.assertGreaterEqual(result['estimate_timestamp'], result['source_timestamp'])

    def test_calibrated_plane_offset_and_translated_world_origin(self):
        local_point = np.array([.003, -.004, .012])
        center = np.array([1000.02, -800.01, 200.03])
        estimator = SurfaceContactLineEstimator(local_point, [0., 0., -2.],
                                                max_solve_seconds=2.)
        for i, angle in enumerate(np.deg2rad(np.linspace(-4, 4, 25))):
            rotation = rx(angle)
            point = center + rotation @ np.array([0., .005 * np.sin(i / 3.), 0.])
            position = point - rotation @ local_point
            estimator.append(i + 1., position, rotation, np.zeros(6), np.zeros(6))
        result = estimator.estimate()
        self.assertTrue(result['valid'], result['reason'])
        self.assertLess(np.linalg.norm(np.cross(result['center_base'] - center,
                                               result['direction_base'])), 1e-6)

    def test_excessive_line_state_corrections_are_rejected(self):
        estimator = self.estimator(max_position_correction_mm=1e-5)
        noise = np.random.RandomState(42).normal(0., 1e-6, 25)
        self.append_plane_sequence(estimator, offset_noise=noise)
        result = estimator.estimate()
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'line_position_correction_exceeded')
        self.assertNotIn('center_base', result)
        estimator = self.estimator(max_direction_correction_deg=1e-5)
        for i, angle in enumerate(np.deg2rad(np.linspace(-4, 4, 25))):
            rotation = rx(angle) @ ry(.0001 * np.sin(i))
            estimator.append(i + 1., np.zeros(3), rotation, np.zeros(6), np.zeros(6))
        result = estimator.estimate()
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'line_direction_correction_exceeded')

    def test_stationary_and_tiny_excitation_are_invalid(self):
        for amplitude in (0., 1e-7):
            estimator = self.estimator()
            for i, angle in enumerate(np.linspace(-amplitude, amplitude, 20)):
                estimator.append(i + 1., np.zeros(3), rx(angle), np.zeros(6), np.zeros(6))
            result = estimator.estimate()
            self.assertFalse(result['valid'])
            self.assertEqual(result['reason'], 'insufficient_normal_excitation')
            self.assertNotIn('center_base', result)

    def test_two_axis_common_pivot_has_no_common_line(self):
        estimator = self.estimator()
        rotations = [rx(a) for a in np.deg2rad(np.linspace(-4, 4, 20))]
        rotations += [ry(a) for a in np.deg2rad(np.linspace(-4, 4, 20))]
        pivot = np.array([.02, -.01, .03])
        for i, rotation in enumerate(rotations):
            position = pivot - rotation @ np.array([0., 0., .1])
            estimator.append(i + 1., position, rotation, np.zeros(6), np.zeros(6))
        result = estimator.estimate()
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'no_common_line_direction')
        self.assertLess(result['observability'], 1.01)

    def test_distant_intersection_is_not_accepted(self):
        estimator = self.estimator()
        for i, angle in enumerate(np.deg2rad(np.linspace(-4, 4, 25))):
            rotation = rx(angle)
            estimator.append(i + 1., rotation @ np.array([0., .1, 0.]), rotation,
                             np.zeros(6), np.zeros(6))
        result = estimator.estimate()
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'line_outside_physical_region')
        self.assertGreater(result['max_line_distance_mm'], 99.)

    def test_inconsistent_planes_are_rejected_before_optimization(self):
        estimator = self.estimator()
        self.append_plane_sequence(estimator, offset_noise=np.tile([-.005, .005], 13)[:25])
        result = estimator.estimate()
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'inconsistent_plane_offsets')
        self.assertFalse(result['optimizer_updated'])

    def test_expired_window_does_not_reuse_valid_result(self):
        estimator = self.estimator(capacity=25)
        self.append_plane_sequence(estimator)
        self.assertTrue(estimator.estimate()['valid'])
        for i in range(25):
            estimator.append(2000. + i, np.zeros(3), np.eye(3), np.zeros(6), np.zeros(6))
        result = estimator.estimate()
        self.assertFalse(result['valid'])
        self.assertEqual(result['source_timestamp'], 2024.)
        self.assertNotIn('center_base', result)

    def test_snapshot_is_independent_and_timing_cannot_regress(self):
        estimator = self.estimator()
        estimator.append(1., np.zeros(3), np.eye(3), np.zeros(6), np.zeros(6))
        copied = estimator.snapshot()
        copied[0]['p'][0] = 10.
        self.assertEqual(estimator.snapshot()[0]['p'][0], 0.)
        self.assertFalse(estimator.estimate()['valid'])
        with self.assertRaises(ValueError):
            estimator.append(1., np.zeros(3), np.eye(3), np.zeros(6), np.zeros(6))

    def test_solver_budget_returns_explicit_invalid_result(self):
        estimator = self.estimator(max_solve_seconds=1e-12)
        self.append_plane_sequence(estimator)
        result = estimator.estimate()
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'optimizer_time_budget')

    def test_final_solver_operation_cannot_exceed_time_budget(self):
        estimator = self.estimator(max_solve_seconds=.1)
        self.append_plane_sequence(estimator)
        def delayed_result(_residual, initial, **_kwargs):
            return SimpleNamespace(success=True, x=initial, cost=0., nfev=1)
        with mock.patch('tactile_policy.contact_line_surface.least_squares',
                        side_effect=delayed_result), mock.patch(
                            'tactile_policy.contact_line_surface.time.monotonic',
                            side_effect=[10., 10.2]):
            result = estimator.estimate()
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'optimizer_time_budget')
        self.assertTrue(result['optimizer_updated'])

    def test_invalid_configuration_and_nonrigid_sample(self):
        for settings in ({'min_samples': 2}, {'capacity': 5},
                         {'min_observability': 1.}, {'max_solve_seconds': -1.},
                         {'temporal_direction_sigma_deg': 90.},
                         {'plane_direction_sigma_deg': 180.},
                         {'min_normal_span_deg': 91.},
                         {'max_direction_correction_deg': 91.},
                         {'max_raw_angle_rms_deg': 91.}):
            with self.assertRaises(ValueError):
                self.estimator(**settings)
        estimator = self.estimator()
        with self.assertRaises(ValueError):
            estimator.append(1., np.zeros(3), np.zeros((3, 3)), np.zeros(6), np.zeros(6))


if __name__ == '__main__':
    unittest.main()
