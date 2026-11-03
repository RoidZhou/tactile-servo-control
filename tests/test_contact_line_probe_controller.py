"""Pure simulated feedback checks; no robot or tactile services are imported."""
import unittest

import numpy as np

from tactile_policy.contact_line_probe_controller import (
    ProbeConfig, ProbeController, ProbeControlError, _transform, _pose, _inverse,
)


def config(**kwargs):
    defaults = dict(ramp_s=.4, rock_period_s=1., rock_cycles=2, follow_s=.5,
                    release_s=.2, release_timeout_s=.5, line_stable_s=.2,
                    line_max_age_s=.1, max_dt_s=.15)
    defaults.update(kwargs)
    return ProbeConfig(**defaults)


def line(now, **kwargs):
    value = dict(timestamp=now, center_base=np.zeros(3), direction_base=np.array([1., 0., 0.]),
                 observability=100., point_rms_m=.0001, angle_rms_rad=.001,
                 singular_values=np.array([.03, .01, .0001]), samples=20)
    value.update(kwargs)
    return value


class ProbeControllerTests(unittest.TestCase):
    def test_positive_measured_deformation_commands_negative_feedback(self):
        c = ProbeController(config(angle_x_rad=0, angle_y_rad=0), np.zeros(6), np.zeros(6))
        prediction = np.array([.0002, -.0002, .0001, .001, -.001, .0005])
        c.step(0., np.zeros(6), prediction)
        out = c.step(.05, np.zeros(6), prediction)
        self.assertTrue(np.all(out['command'] * prediction < 0))
        np.testing.assert_allclose(out['error'], prediction, atol=1e-12)

    def test_H_conjugation_cancels_lever_arm_translation_for_fingertip_rotation(self):
        H = _transform(np.array([0., 0., .1, 0., 0., 0.]))
        deformation_F = _transform(np.array([0., 0., 0., .01, 0., 0.]))
        D = H @ deformation_F @ _inverse(H)
        self.assertGreater(np.linalg.norm(D[:3, 3]), .0009)
        c = ProbeController(config(), np.zeros(6), H)
        out = c.step(0., np.zeros(6), _pose(D))
        np.testing.assert_allclose(out['measured_feedback_deformation'][:3], 0., atol=1e-12)
        np.testing.assert_allclose(out['error'][:3], 0., atol=1e-12)
        np.testing.assert_allclose(out['measured_deformation'], _pose(D), atol=1e-12)

    def test_nonzero_initial_reference_is_not_relatched(self):
        reference = np.array([.0001, -.0002, 0., .01, 0., 0.])
        c = ProbeController(config(), np.zeros(6), np.zeros(6), reference)
        out = c.step(0., np.zeros(6), reference)
        np.testing.assert_allclose(out['error'], 0., atol=1e-12)
        np.testing.assert_allclose(out['measured_deformation'], reference, atol=1e-12)

    def test_actual_dt_has_consistent_integrated_feedback(self):
        def simulate(dt):
            c = ProbeController(config(ramp_s=100, angle_x_rad=0, angle_y_rad=0,
                                       max_linear_accel_m_s2=100.), np.zeros(6), np.zeros(6))
            actual = np.zeros(6)
            prediction = np.array([.0002, 0., 0., 0., 0., 0.])
            c.step(0., actual, prediction)
            for now in np.arange(dt, 1. + dt / 2, dt):
                actual = c.step(now, actual, prediction)['command']
            return actual
        np.testing.assert_allclose(simulate(.1), simulate(.05), atol=1e-10)

    def test_complete_tcp_speed_acceleration_and_step_bounds_with_pivot(self):
        cfg = config(ramp_s=.1, angle_x_rad=.3, angle_y_rad=.2, rock_period_s=10.,
                     max_total_rotation_rad=.5)
        c = ProbeController(cfg, np.zeros(6), np.array([0., 0., .1, 0., 0., 0.]))
        actual = np.zeros(6)
        previous_velocity = np.zeros(6)
        c.step(0., actual, np.zeros(6))
        for index in range(1, 31):
            out = c.step(index * .05, actual, np.zeros(6))
            current = _transform(out['command']); old = _transform(actual)
            step = np.r_[current[:3, 3] - old[:3, 3],
                         _pose(current @ _inverse(old))[3:]]
            velocity = step / .05
            self.assertLessEqual(np.linalg.norm(velocity[:3]), cfg.max_linear_speed_m_s + 1e-10)
            self.assertLessEqual(np.linalg.norm(velocity[3:]), cfg.max_angular_speed_rad_s + 1e-10)
            self.assertLessEqual(np.linalg.norm(velocity[:3] - previous_velocity[:3]),
                                 cfg.max_linear_accel_m_s2 * .05 + 1e-10)
            self.assertLessEqual(np.linalg.norm(velocity[3:] - previous_velocity[3:]),
                                 cfg.max_angular_accel_rad_s2 * .05 + 1e-10)
            self.assertLessEqual(np.linalg.norm(step[:3]), cfg.max_step_m + 1e-10)
            self.assertLessEqual(np.linalg.norm(step[3:]), cfg.max_step_rad + 1e-10)
            previous_velocity = velocity
            actual = out['command']

    def test_no_line_releases_and_finishes_explicitly_without_returning_to_start(self):
        c = ProbeController(config(), np.zeros(6), np.zeros(6))
        actual = np.zeros(6)
        seen = set()
        for now in np.arange(0., 4., .05):
            out = c.step(now, actual, np.zeros(6))
            actual = out['command']; seen.add(out['stage'])
            if out['finished']:
                break
        self.assertEqual(out['stage'], 'done')
        self.assertEqual(out['status'], 'no_valid_line')
        self.assertEqual(out['reason'], 'no_valid_line')
        self.assertIn('release', seen)
        self.assertNotIn('follow', seen)
        self.assertGreater(np.linalg.norm(actual[3:]), 0.)

    def enter_follow(self, **changes):
        c = ProbeController(config(**changes), np.zeros(6), np.zeros(6))
        actual = np.zeros(6)
        for now in np.arange(0., 2., .05):
            out = c.step(now, actual, np.zeros(6), line(now))
            actual = out['command']
            if out['stage'] == 'follow':
                self.assertGreaterEqual(now, 1.4 - 1e-9)
                return c, now, actual, out
        self.fail('stable valid line did not enter follow')

    def test_follow_requires_one_full_rock_cycle_and_stale_line_stops(self):
        c, now, actual, out = self.enter_follow()
        stale = line(now - .2)
        stopped = c.step(now + .05, actual, np.zeros(6), stale)
        self.assertTrue(stopped['finished'])
        self.assertEqual(stopped['status'], 'failed')
        np.testing.assert_allclose(stopped['command'], actual, atol=1e-12)
        again = c.step(now + .1, actual, np.zeros(6), line(now + .1))
        self.assertTrue(again['finished'])
        np.testing.assert_allclose(again['command'], actual, atol=1e-12)

    def test_follow_invalid_or_changed_axis_stops_before_motion(self):
        for changed in (None, dict(degenerate=True), dict(direction_base=np.array([0.,1.,0.]))):
            c, now, actual, out = self.enter_follow()
            candidate = None if changed is None else line(now + .05, **changed)
            stopped = c.step(now + .05, actual, np.zeros(6), candidate)
            self.assertTrue(stopped['finished'])
            np.testing.assert_allclose(stopped['command'], actual, atol=1e-12)

    def test_repeated_timestamp_cannot_accumulate_stability_and_degenerate_never_follows(self):
        for bad in ('repeat', 'degenerate'):
            c = ProbeController(config(line_max_age_s=10), np.zeros(6), np.zeros(6))
            actual = np.zeros(6)
            for now in np.arange(0., 3.5, .05):
                candidate = line(0.) if bad == 'repeat' else line(now, singular_values=[1.,0.,0.])
                out = c.step(now, actual, np.zeros(6), candidate)
                actual = out['command']
                self.assertFalse(out['line_qualified'])
                self.assertNotEqual(out['stage'], 'follow')

    def test_line_axis_change_resets_stable_time_and_sign_flip_does_not(self):
        c = ProbeController(config(), np.zeros(6), np.zeros(6))
        for now in (0., .05, .1, .15, .2, .25):
            candidate = line(now, direction_base=np.array([-1.,0.,0.]) if now > .1 else np.array([1.,0.,0.]))
            out = c.step(now, np.zeros(6), np.zeros(6), candidate)
        self.assertTrue(out['line_qualified'])
        out = c.step(.3, np.zeros(6), np.zeros(6), line(.3, direction_base=np.array([0.,1.,0.])))
        self.assertFalse(out['line_qualified'])
        self.assertEqual(out['line_reset_reason'], 'line_axis_changed')

    def test_pause_and_sample_gap_do_not_integrate_missing_time(self):
        c = ProbeController(config(), np.zeros(6), np.zeros(6))
        c.step(0., np.zeros(6), np.zeros(6))
        c.pause(10.)
        out = c.step(10., np.zeros(6), np.zeros(6))
        self.assertEqual(out['dt'], 0.)
        self.assertEqual(c.elapsed, 0.)
        out = c.step(12., np.zeros(6), np.zeros(6))
        self.assertEqual(out['reason'], 'sample_gap_paused')
        np.testing.assert_allclose(out['command'], 0.)
        self.assertEqual(c.elapsed, 0.)

    def test_tracking_timeout_and_travel_limit_stop_without_accumulating_command(self):
        c = ProbeController(config(tracking_timeout_s=.1), np.zeros(6), np.zeros(6))
        c.step(0., np.zeros(6), np.zeros(6))
        c.last_command[0,3] = .003
        c.step(.05, np.zeros(6), np.zeros(6))
        out = c.step(.1, np.zeros(6), np.zeros(6))
        self.assertEqual(out['reason'], 'tracking_timeout')
        self.assertTrue(out['finished'])
        np.testing.assert_allclose(out['command'], 0.)
        c = ProbeController(config(max_total_rotation_rad=.00001), np.zeros(6), np.zeros(6))
        c.step(0., np.zeros(6), np.zeros(6))
        for now in np.arange(.05, 1., .05):
            out = c.step(now, np.zeros(6), np.zeros(6))
            if out['finished']: break
        self.assertTrue(out['finished'])
        self.assertEqual(out['reason'], 'motion_travel_limit')

    def test_release_timeout_does_not_chase_forever(self):
        c = ProbeController(config(angle_x_rad=0, angle_y_rad=0), np.zeros(6), np.zeros(6))
        actual = np.zeros(6)
        for now in np.arange(0., 5., .05):
            prediction = np.array([.0002,0.,0.,0.,0.,0.])
            out = c.step(now, actual, prediction)
            actual = out['command']
            if out['finished']: break
        self.assertTrue(out['finished'])
        self.assertEqual(out['reason'], 'release_timeout')

    def test_nonfinite_inputs_and_bad_transforms_are_rejected(self):
        c = ProbeController(config(), np.zeros(6), np.zeros(6))
        for time, tcp, pred in ((np.nan,np.zeros(6),np.zeros(6)),
                                (0.,np.full(6,np.inf),np.zeros(6)),
                                (0.,np.zeros(6),np.full(6,np.nan))):
            with self.assertRaises(ProbeControlError): c.step(time,tcp,pred)
        with self.assertRaises(ProbeControlError):
            ProbeController(config(), np.zeros(6), np.zeros((4,4)))
        with self.assertRaises(ValueError): ProbeConfig(max_dt_s=0)
        with self.assertRaises(ValueError): ProbeConfig(rock_cycles=1.5)


if __name__ == '__main__':
    unittest.main()
