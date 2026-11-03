"""CLI and complete runtime checks with simulated robot/model and clocks only."""
import argparse
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tactile_policy import contact_line_runtime as runtime
from tactile_policy.contact_line_probe_controller import _transform, _inverse, _pose


class BooleanOptionalAction(argparse.Action):
    # Test adapter for local Python 3.7; production requires Python >=3.9.
    def __init__(self, option_strings, dest, default=None, **kwargs):
        options = [item for option in option_strings for item in (option, '--no-'+option[2:])]
        super().__init__(options, dest, nargs=0, default=default, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, not option_string.startswith('--no-'))


def parse_entry_args(argv):
    """Run actual entry-point parser without loading robot, torch or ZMQ."""
    source = Path(__file__).resolve().parents[1] / 'tactile_policy/infer_tactile_servo_nutbolt_contact_line.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    nodes = [node for node in tree.body if (isinstance(node, ast.Assign)
             and all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets))
             or (isinstance(node, ast.FunctionDef) and node.name in ('six_floats', 'three_floats', 'parse_args'))]
    tree.body = nodes
    environment = dict(np=np, argparse=argparse, Path=Path, __file__=str(source), __doc__='Parser test',
                       add_probe_arguments=runtime.add_probe_arguments, validate_probe_arguments=runtime.validate_probe_arguments)
    with mock.patch.object(argparse, 'BooleanOptionalAction', getattr(argparse, 'BooleanOptionalAction', BooleanOptionalAction), create=True):
        exec(compile(tree, str(source), 'exec'), environment)
        with mock.patch.object(sys, 'argv', ['probe'] + list(argv)):
            return environment['parse_args']()


MANUAL = ['--mode', 'contact-line-probe', '--no-contact-line-auto-prepare', '--contact-line-contact-established']


class RuntimeTests(unittest.TestCase):
    def test_missing_plane_uses_current_standard_geometry_in_mm(self):
        args = parse_entry_args(MANUAL)
        self.assertEqual(args.contact_line_plane_point_mm, [0., 0., 100.])
        self.assertEqual(args.contact_line_plane_point_source, 'standard_tcp_to_nut_center_approximation')
        args = parse_entry_args(MANUAL + ['--standard-tcp-to-nut-center', '0.001,-0.002,0.12'])
        np.testing.assert_allclose(args.contact_line_plane_point_mm, [1., -2., 120.])

    def test_explicit_plane_wins_and_legacy_does_not_require_one(self):
        args = parse_entry_args(MANUAL + ['--contact-line-plane-point-mm', '1', '2', '108'])
        self.assertEqual(args.contact_line_plane_point_mm, [1., 2., 108.])
        self.assertEqual(args.contact_line_plane_point_source, 'explicit_plane_point_mm')
        legacy = parse_entry_args(['--mode', 'contact-line-probe', '--contact-line-strategy', 'waypoints'])
        self.assertEqual(legacy.contact_line_estimator, 'svd')
        self.assertIsNone(legacy.contact_line_plane_point_mm)

    def test_invalid_geometry_and_incompatible_estimator_rejected(self):
        for extra in (['--contact-line-plane-normal', '0', '0', '0'],
                      ['--contact-line-plane-point-mm', 'nan', '0', '100'],
                      ['--contact-line-estimator', 'svd'],
                      ['--contact-line-deformation-roll-deg', '0', '--contact-line-deformation-pitch-deg', '0']):
            with self.subTest(extra=extra), redirect_stdout(io.StringIO()), mock.patch('sys.stderr', io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_entry_args(MANUAL + extra)

    def test_stability_uses_real_pose_and_force_history_and_rejects_gaps(self):
        reference = runtime.StableReference(.2, .0001, .001, .2)
        tcp = np.zeros(6)
        for stamp in np.arange(0., .21, .04):
            ready = reference.update(stamp, tcp, [3., 3.])
        self.assertTrue(ready)
        self.assertFalse(reference.update(.24, tcp, [4., 3.]))
        self.assertFalse(reference.update(.6, tcp, [3., 3.]))
        self.assertEqual(len(reference.samples), 1)

    def test_force_norms_only_mean_contact_when_operator_or_free_baseline_supplied(self):
        self.assertFalse(runtime.force_evidence([3., 4.], None, .5)[0])
        self.assertFalse(runtime.force_evidence([3., 4.], [3., 4.], .5)[0])
        self.assertTrue(runtime.force_evidence([3.6, 4.], [3., 4.], .5)[0])
        evidence = runtime.force_evidence([3., 4.], None, .5, manual=True)
        self.assertTrue(evidence[0])
        self.assertEqual(evidence[1], 'operator_established_contact')

    def test_feedback_axes_are_rigid_and_follow_normal(self):
        H = runtime.feedback_transform([.001, .002, .1], [1., 0., 0.])
        np.testing.assert_allclose(H[:3, :3].T @ H[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(H[:3, 2], [1., 0., 0.])
        self.assertAlmostEqual(np.linalg.det(H[:3, :3]), 1.)

    def test_contact_hysteresis_retains_near_threshold_but_pauses_weak_evidence(self):
        gate = runtime.ContactEvidenceGate([5., 4.])
        self.assertFalse(gate.update(1., [5.4, 4.])['valid'])
        self.assertTrue(gate.update(1.1, [5.6, 4.])['valid'])
        self.assertTrue(gate.update(1.2, [5.35, 4.])['valid'])
        low = gate.update(1.3, [5.1, 4.1])
        self.assertFalse(low['valid'])
        self.assertFalse(low['lost'])
        self.assertTrue(gate.update(1.4, [5.3, 4.])['valid'])
        self.assertFalse(gate.update(1.5, [5.1, 4.])['lost'])

    def test_contact_loss_requires_consecutive_fresh_samples_not_unsynchronized_gaps(self):
        gate = runtime.ContactEvidenceGate([5., 4.])
        gate.update(1., [5.6, 4.])
        gate.update(1.1, [5.1, 4.])
        gap = gate.update(1.6, [5.1, 4.])
        self.assertFalse(gap['lost'])
        self.assertEqual(gap['low_samples'], 1)
        for stamp in [1.7, 1.8, 1.91]:
            low = gate.update(stamp, [5.1, 4.])
        self.assertTrue(low['lost'])
        self.assertGreaterEqual(low['low_samples'], 3)
        with self.assertRaises(ValueError):
            gate.update(1.91, [5.1, 4.])

    def test_simulated_real_contact_loss_is_logged_with_rejected_frames_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            outcome, commands, metadata = self.simulate(directory, lose_contact=True)
            self.assertIn('external-contact force-change evidence lost:', outcome)
            self.assertIn('delta_N=', outcome)
            self.assertGreater(len(commands), 0)
            self.assertEqual(metadata['last_contact_evidence']['state'], 'evidence_lost')
            import csv
            with (Path(directory)/'probe.csv').open(encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            rejected = [row for row in rows if row['event'] == 'contact_evidence']
            self.assertGreaterEqual(len(rejected), 3)
            self.assertEqual(rows[-1]['event'], 'contact_evidence_lost')
            self.assertEqual(rows[-1]['motion_command_sent'], 'False')

    def test_worker_invalid_results_replace_previous_valid_line(self):
        completed = threading.Event()
        estimator = SimpleNamespace(estimate=mock.Mock(side_effect=[
            {'valid': True, 'source_timestamp': 10.},
            {'valid': False, 'source_timestamp': 11., 'reason': 'degenerate'}]))
        original = estimator.estimate
        def estimate():
            result = original()
            completed.set()
            return result
        estimator.estimate = estimate
        worker = runtime.LatestEstimateWorker(estimator)
        try:
            worker.submit()
            self.assertTrue(completed.wait(1.))
            # Wait under the result lock, not by assuming optimizer timing.
            for _ in range(100):
                if worker.latest().get('source_timestamp') == 10.:
                    break
                threading.Event().wait(.001)
            self.assertTrue(worker.latest()['valid'])
            completed.clear()
            worker.submit()
            self.assertTrue(completed.wait(1.))
            for _ in range(100):
                if worker.latest().get('source_timestamp') == 11.:
                    break
                threading.Event().wait(.001)
            self.assertFalse(worker.latest()['valid'])
        finally:
            worker.close()
        self.assertFalse(worker.thread.is_alive())

    def simulate(self, directory, inference_delay=0., missing=False, lose_contact=False):
        args = parse_entry_args(MANUAL + ['--contact-line-no-follow', '--contact-line-reference-stable-s', '.1',
            '--contact-line-reference-timeout-s', '2', '--contact-line-ramp-s', '.1',
            '--contact-line-rock-period-s', '.4', '--contact-line-release-s', '.1',
            '--contact-line-release-timeout-s', '2', '--contact-line-output', str(Path(directory)/'probe.csv')])
        clock = SimpleNamespace(now=0.)
        if lose_contact:
            args.contact_line_contact_established = False
            args.contact_line_free_grasp_forces_n = [2., 2.]
        robot = SimpleNamespace(pose=np.zeros(6))
        robot.get_tcp_pose = lambda: robot.pose.copy()
        class Source:
            accepted_count = 0
            clock_fault = False
            last_reason = 'no fresh pair'
            def poll(self):
                clock.now += .02
                if missing:
                    return None
                self.accepted_count += 1
                stamp = 1000.+clock.now
                force = 2.1 if lose_contact and clock.now > .5 else 3.
                frame = dict(force6d=np.array([force, 0., 0., 0., 0., 0.]), prediction=robot.pose.copy())
                return dict(left=frame, right=frame, tcp=robot.pose.copy(), timestamp=stamp,
                            left_stamp=stamp, right_stamp=stamp, left_right_delta_ms=0.,
                            tcp_skew_ms=0., tcp_span_ms=0., sample_age_s=0.)
        class Worker:
            def __init__(self, estimator): pass
            def submit(self): pass
            def latest(self): return {'valid': False, 'reason': 'synthetic_no_line', 'source_timestamp': None}
            def close(self): pass
        class SafetyStop(RuntimeError): pass
        commands = []
        def command(robot, pose, execute):
            commands.append(np.asarray(pose).copy())
            robot.pose = np.asarray(pose).copy()
        def predict(*values):
            clock.now += inference_delay
            return values[3].copy()
        api = SimpleNamespace(format_tactile_frame=lambda frame, *rest: frame, check_safety=lambda *values: None,
                              paired_force_input=lambda left, right: left['prediction'].copy(),
                              predict_pair_motion=predict,
                              inferred_object_frame=lambda tcp, pred: ((_transform(tcp) @ _inverse(_transform(pred)))[:3, 3],
                                                                        (_transform(tcp) @ _inverse(_transform(pred)))[:3, :3]),
                              command_pose=command, SafetyStop=SafetyStop)
        model = SimpleNamespace(tactile_input_kind='paired_force')
        norm = SimpleNamespace(target_limit=np.array([.01, .01, .01, .1, .1, .1]))
        with mock.patch.object(runtime, 'SynchronizedProbeSource', return_value=Source()), \
             mock.patch.object(runtime, 'LatestEstimateWorker', Worker), \
             mock.patch.object(runtime.time, 'monotonic', side_effect=lambda: clock.now), \
             mock.patch.object(runtime.time, 'time', side_effect=lambda: 1000.+clock.now), \
             mock.patch.object(runtime.time, 'sleep', side_effect=lambda delay: setattr(clock, 'now', clock.now+delay)), \
             redirect_stdout(io.StringIO()):
            try:
                runtime.run_feedback_probe(args, robot, None, None, model, norm, None, api)
            except SafetyStop as exc:
                outcome = str(exc)
            else:
                outcome = None
        return outcome, commands, json.loads((Path(directory)/'probe.metadata.json').read_text(encoding='utf-8'))

    def test_complete_feedback_rock_and_release_with_fake_robot(self):
        with tempfile.TemporaryDirectory() as directory:
            outcome, commands, metadata = self.simulate(directory)
            self.assertIsNone(outcome)
            self.assertGreater(len(commands), 5)
            self.assertEqual(metadata['outcome'], 'completed')
            self.assertEqual(metadata['reason'], 'follow_disabled')
            self.assertEqual(metadata['args']['contact_line_plane_point_mm'], [0., 0., 100.])
            self.assertEqual(metadata['external_contact_proof'], 'operator assumption')
            self.assertLess(np.linalg.norm(commands[-1][3:]), .003)

    def test_missing_capture_or_slow_inference_stops_without_motion(self):
        for kwargs in ({'missing': True}, {'inference_delay': .2}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                outcome, commands, metadata = self.simulate(directory, **kwargs)
                self.assertIsNotNone(outcome)
                self.assertEqual(commands, [])
                self.assertEqual(metadata['outcome'], 'stopped')


if __name__ == '__main__':
    unittest.main()
