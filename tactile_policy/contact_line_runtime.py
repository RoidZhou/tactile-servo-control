"""Runtime for bounded, synchronized tactile-deformation contact probing.

Hardware/model functions are injected by the existing inference entry point.
The estimator thread only sees copied numeric samples, never ZMQ clients.
"""
import csv
import json
from collections import deque
from dataclasses import asdict
from pathlib import Path
import threading
import time

import numpy as np

from tactile_policy.contact_line_probe_controller import (
    ProbeConfig, ProbeController, ProbeControlError, _transform, _inverse, _pose,
)
from tactile_policy.contact_line_surface import SurfaceContactLineEstimator
from tactile_policy.contact_line_sync import SynchronizedProbeSource


def add_probe_arguments(parser):
    group = parser.add_argument_group('Feedback contact-line probe (SI internally)')
    group.add_argument('--contact-line-strategy', choices=('feedback', 'waypoints'), default='feedback')
    group.add_argument('--contact-line-plane-point-mm', nargs=3, type=float, default=None,
                       help='Calibrated point on nut bottom in initial-reference TCP axes, mm.')
    group.add_argument('--contact-line-plane-normal', nargs=3, type=float, default=[0., 0., 1.],
                       help='Nut-bottom normal in initial-reference TCP axes; sign defines feedback +z.')
    group.add_argument('--contact-line-feedback-point-mm', nargs=3, type=float, default=None,
                       help='Control point in reference TCP; default standard-tcp-to-nut-center.')
    evidence = group.add_mutually_exclusive_group()
    evidence.add_argument('--contact-line-free-grasp-forces-n', nargs=2, type=float, default=None,
                          help='Left/right force norms before external contact, at unchanged grip.')
    evidence.add_argument('--contact-line-contact-established', action='store_true',
                          help='Operator-established bottom/bolt contact; no automatic external-contact proof.')
    group.add_argument('--contact-line-no-follow', action='store_true',
                       help='Collect feedback rocking only; do not pivot about estimated line.')
    defaults = {
        'poll-hz': 100., 'tcp-skew-ms': 40., 'max-age-ms': 150., 'missing-timeout-s': 1.,
        'reference-stable-s': 1., 'reference-timeout-s': 15.,
        'reference-position-mm': .1, 'reference-angle-deg': .1, 'reference-force-n': .2,
        'contact-change-n': .5, 'contact-loss-s': .3,
        'ramp-s': 3., 'rock-period-s': 20., 'deformation-roll-deg': .3,
        'deformation-pitch-deg': .2, 'normal-deformation-mm': 0.,
        'follow-s': 20., 'release-s': 3., 'release-timeout-s': 8.,
        'translation-gain': 1., 'rotation-gain': 1.,
        'max-speed-mm-s': 1., 'max-speed-deg-s': .5,
        'max-accel-mm-s2': 2., 'max-accel-deg-s2': 1.,
        'max-step-mm': .5, 'max-step-deg': .5, 'max-travel-mm': 12.,
        'max-normal-travel-mm': 1., 'max-rotation-deg': 4.,
        'line-stable-s': 2., 'line-max-age-s': .5,
        'min-normal-span-deg': 1., 'min-observability': 8.,
        'max-plane-rms-mm': .5, 'max-angle-rms-deg': .5,
        'max-line-distance-mm': 30., 'solve-budget-s': .2,
    }
    for name, value in defaults.items():
        group.add_argument('--contact-line-' + name, type=float, default=value)


def validate_probe_arguments(parser, args):
    if args.contact_line_estimator is None:
        args.contact_line_estimator = 'surface' if args.contact_line_strategy == 'feedback' else 'svd'
    if args.mode != 'contact-line-probe':
        return
    if args.contact_line_strategy == 'waypoints':
        if args.contact_line_estimator == 'surface':
            parser.error('surface estimator requires --contact-line-strategy feedback')
        return
    if args.contact_line_estimator != 'surface':
        parser.error('feedback probing requires --contact-line-estimator surface')
    if args.contact_line_plane_point_mm is None:
        parser.error('feedback probing requires --contact-line-plane-point-mm X Y Z (calibrated nut bottom)')
    for name in ('plane_point_mm', 'plane_normal', 'feedback_point_mm', 'free_grasp_forces_n'):
        value = getattr(args, 'contact_line_' + name)
        if value is not None and not np.isfinite(value).all():
            parser.error('contact-line-' + name.replace('_', '-') + ' must be finite')
    if np.linalg.norm(args.contact_line_plane_normal) < 1e-10 or not np.isfinite(np.linalg.norm(args.contact_line_plane_normal)):
        parser.error('contact-line-plane-normal must be nonzero and finite')
    if args.contact_line_free_grasp_forces_n is not None and min(args.contact_line_free_grasp_forces_n) < 0:
        parser.error('free-grasp force norms must be nonnegative')
    if (not args.contact_line_auto_prepare and args.contact_line_free_grasp_forces_n is None
            and not args.contact_line_contact_established):
        parser.error('manual preparation requires free-grasp force baseline or --contact-line-contact-established')
    for name, value in vars(args).items():
        if name.startswith('contact_line_') and isinstance(value, float):
            if not np.isfinite(value) or (value <= 0 and name != 'contact_line_normal_deformation_mm'
                                         and name != 'contact_line_hold_s'):
                parser.error(name.replace('_', '-') + ' must be positive and finite')
    if not np.isfinite(args.control_hz) or not 1 <= args.control_hz <= 100:
        parser.error('feedback control-hz must be in [1,100]')
    if args.contact_line_poll_hz < args.control_hz or args.contact_line_poll_hz > 250:
        parser.error('contact-line-poll-hz must be >= control-hz and <=250')
    if args.contact_line_reference_timeout_s <= args.contact_line_reference_stable_s:
        parser.error('reference timeout must exceed stable duration')
    if args.contact_line_max_age_ms >= args.contact_line_missing_timeout_s * 1000:
        parser.error('sample max age must be shorter than missing-sample timeout')
    if args.contact_line_max_rotation_deg > 4 or args.contact_line_min_observability <= 1:
        parser.error('max-rotation-deg must be <=4 and min-observability must exceed 1')
    try:
        make_controller_config(args)
        make_estimator(args)
    except ValueError as exc:
        parser.error(str(exc))


def make_controller_config(a):
    return ProbeConfig(
        settle_s=a.contact_line_reference_stable_s, ramp_s=a.contact_line_ramp_s,
        rock_period_s=a.contact_line_rock_period_s, rock_cycles=a.contact_line_probe_cycles,
        follow_s=a.contact_line_follow_s, release_s=a.contact_line_release_s,
        release_timeout_s=a.contact_line_release_timeout_s,
        angle_x_rad=np.deg2rad(a.contact_line_deformation_roll_deg),
        angle_y_rad=np.deg2rad(a.contact_line_deformation_pitch_deg),
        normal_target_m=a.contact_line_normal_deformation_mm / 1000,
        translation_gain_s=a.contact_line_translation_gain, rotation_gain_s=a.contact_line_rotation_gain,
        max_linear_speed_m_s=a.contact_line_max_speed_mm_s / 1000,
        max_angular_speed_rad_s=np.deg2rad(a.contact_line_max_speed_deg_s),
        max_linear_accel_m_s2=a.contact_line_max_accel_mm_s2 / 1000,
        max_angular_accel_rad_s2=np.deg2rad(a.contact_line_max_accel_deg_s2),
        max_step_m=a.contact_line_max_step_mm / 1000, max_step_rad=np.deg2rad(a.contact_line_max_step_deg),
        max_total_translation_m=a.contact_line_max_travel_mm / 1000,
        max_normal_translation_m=a.contact_line_max_normal_travel_mm / 1000,
        max_total_rotation_rad=np.deg2rad(a.contact_line_max_rotation_deg),
        line_stable_s=a.contact_line_line_stable_s, line_max_age_s=a.contact_line_line_max_age_s,
        min_line_observability=a.contact_line_min_observability, min_line_samples=a.contact_line_min_samples,
        max_line_rms_m=a.contact_line_max_plane_rms_mm / 1000,
        max_line_angle_rms_rad=np.deg2rad(a.contact_line_max_angle_rms_deg),
        follow_enabled=not a.contact_line_no_follow,
    )


def make_estimator(a):
    return SurfaceContactLineEstimator(
        np.asarray(a.contact_line_plane_point_mm) / 1000, a.contact_line_plane_normal,
        capacity=a.contact_line_window, min_samples=a.contact_line_min_samples,
        min_normal_span_deg=a.contact_line_min_normal_span_deg, min_observability=a.contact_line_min_observability,
        max_raw_plane_rms_mm=a.contact_line_max_plane_rms_mm,
        max_raw_angle_rms_deg=a.contact_line_max_angle_rms_deg,
        max_line_distance_mm=a.contact_line_max_line_distance_mm,
        temporal_position_sigma_mm=a.contact_line_position_sigma_mm,
        temporal_direction_sigma_deg=a.contact_line_rotation_sigma_deg,
        plane_sigma_mm=a.contact_line_contact_sigma_mm,
        plane_direction_sigma_deg=a.contact_line_direction_sigma_deg,
        max_nfev=a.contact_line_factor_graph_max_nfev, max_solve_seconds=a.contact_line_solve_budget_s,
    )


def feedback_transform(point, normal):
    """Feedback z follows bottom normal; x is projected reference TCP x."""
    z = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    seed = np.array([1., 0., 0.]) if abs(z[0]) < .9 else np.array([0., 1., 0.])
    x = seed - z * (seed @ z)
    x /= np.linalg.norm(x)
    H = np.eye(4)
    H[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    H[:3, 3] = point
    return H


class LatestEstimateWorker:
    """Bounded wake-up queue; invalid results replace previous valid results."""
    def __init__(self, estimator):
        self.estimator = estimator
        self._wake, self._stop = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        self._latest = {'valid': False, 'reason': 'waiting_for_samples', 'source_timestamp': None}
        self.thread = threading.Thread(target=self._run, name='contact-line-estimator', daemon=True)
        self.thread.start()

    def submit(self):
        self._wake.set()

    def _run(self):
        while not self._stop.is_set():
            if not self._wake.wait(.05):
                continue
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                result = self.estimator.estimate()
            except Exception as exc:
                result = {'valid': False, 'reason': 'estimator_error: ' + str(exc), 'source_timestamp': None}
            with self._lock:
                self._latest = result

    def latest(self):
        with self._lock:
            return dict(self._latest)

    def close(self):
        self._stop.set()
        self._wake.set()
        self.thread.join(timeout=1.)


class StableReference:
    def __init__(self, duration, position_m, angle_rad, force_n, max_gap_s=.2):
        self.duration, self.position_m, self.angle_rad, self.force_n = duration, position_m, angle_rad, force_n
        self.max_gap_s = max_gap_s
        self.samples = deque()

    def update(self, stamp, tcp, forces):
        if self.samples and (stamp <= self.samples[-1][0] or stamp-self.samples[-1][0] > self.max_gap_s):
            self.samples.clear()
        self.samples.append((stamp, _transform(tcp), np.asarray(forces)))
        while len(self.samples) > 1 and stamp - self.samples[1][0] >= self.duration:
            self.samples.popleft()
        if len(self.samples) < 5 or stamp-self.samples[0][0] < self.duration:
            return False
        first = self.samples[0][1]
        differences = np.stack([_pose(_inverse(first) @ item[1]) for item in self.samples])
        forces = np.stack([item[2] for item in self.samples])
        return (np.linalg.norm(differences[:, :3], axis=1).max() <= self.position_m
                and np.linalg.norm(differences[:, 3:], axis=1).max() <= self.angle_rad
                and np.ptp(forces, axis=0).max() <= self.force_n)


def force_evidence(forces, baseline, threshold, manual=False):
    if manual:
        return True, 'operator_established_contact', np.full(2, np.nan)
    if baseline is None:
        return False, 'missing_free_grasp_baseline', np.full(2, np.nan)
    delta = np.asarray(forces) - np.asarray(baseline)
    return bool(np.any(np.abs(delta) >= threshold)), 'force_change_from_free_grasp', delta


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def run_feedback_probe(args, robot, tactile, force_reader, model, norm, device, api):
    """Execute only through the entry point's existing --execute gate."""
    if getattr(model, 'tactile_input_kind', '') != 'paired_force':
        raise ValueError('feedback probing requires a paired relative SE3 checkpoint')
    baseline = args.contact_line_free_grasp_forces_n
    preparation = {}
    if args.contact_line_auto_prepare:
        if args.preparation_checkpoint is None:
            raise ValueError('automatic preparation requires --preparation-checkpoint')
        prep_model, prep_norm = api.load_model(args.preparation_checkpoint, device)
        if getattr(prep_model, 'tactile_input_kind', '') == 'paired_force':
            raise ValueError('preparation checkpoint must predict absolute offset')
        preparation = api.prepare_contact_line_probe(args, robot, tactile, force_reader, prep_model, prep_norm, device)
        baseline = preparation['free_grasp_forces_n']
    config = make_controller_config(args)
    estimator = make_estimator(args)
    point = (np.asarray(args.standard_tcp_to_nut_center) if args.contact_line_feedback_point_mm is None
             else np.asarray(args.contact_line_feedback_point_mm) / 1000)
    H = feedback_transform(point, args.contact_line_plane_normal)
    source = SynchronizedProbeSource(robot, tactile, args.tactile_sync_max_skew_ms/1000,
                                     args.contact_line_tcp_skew_ms/1000, args.contact_line_max_age_ms/1000)
    stable = StableReference(args.contact_line_reference_stable_s, args.contact_line_reference_position_mm/1000,
                             np.deg2rad(args.contact_line_reference_angle_deg), args.contact_line_reference_force_n)
    target_limit = np.asarray(norm.target_limit, dtype=float)
    if target_limit.shape != (6,) or not np.isfinite(target_limit).all() or np.any(target_limit <= 0):
        raise ValueError('paired checkpoint requires six finite positive target limits')
    output = Path(args.contact_line_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output.with_suffix('.metadata.json')
    metadata = dict(schema_version=2, strategy='feedback', contact_geometry='nut_bottom_plane',
                    args=vars(args), controller=asdict(config), feedback_transform=H,
                    checkpoint=str(Path(args.checkpoint).resolve()), target_limit=target_limit,
                    preparation=preparation, free_grasp_forces_n=baseline,
                    started=time.time(), outcome='running', units='m, rad, s, N; *_mm and *_deg explicit',
                    object_pose_convention='T_WOstar = T_WG @ inverse(inverse(D_self) @ D_raw)',
                    external_contact_proof='operator assumption' if args.contact_line_contact_established else 'force-change heuristic')
    def save_metadata():
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_value), encoding='utf-8')
    save_metadata()
    fields = ['time_s', 'timestamp', 'left_stamp', 'right_stamp', 'left_right_delta_ms', 'tcp_skew_ms', 'tcp_span_ms',
              'sample_age_s', 'phase', 'status', 'reason', 'dt', 'reference_id', 'external_contact_evidence',
              'contact_evidence_source', 'left_force_n', 'right_force_n', 'left_force_change_n', 'right_force_change_n',
              'line_valid', 'line_reason', 'line_source_timestamp', 'line_age_s', 'line_stable_s', 'line_qualified',
              'normal_excitation_deg', 'observability', 'raw_plane_rms_mm', 'raw_angle_rms_deg', 'optimizer_seconds']
    for prefix in ('tcp', 'control_tcp', 'pred', 'raw_pred', 'desired', 'measured_feedback', 'desired_feedback', 'error', 'command'):
        fields += [prefix + '_' + axis for axis in ('x', 'y', 'z', 'rx', 'ry', 'rz')]
    fields += ['line_' + name + '_' + axis for name in ('point', 'direction') for axis in ('x', 'y', 'z')]
    started = time.monotonic()
    last_valid, last_step, last_print = started, -np.inf, -np.inf
    controller = reference = reference_self = None
    loss_since = None
    count = 0
    worker = LatestEstimateWorker(estimator)
    print('[contact-line] feedback: stable reference -> ramp -> rocking -> qualified line follow -> release', flush=True)
    print('[contact-line] bottom plane (mm), normal:', args.contact_line_plane_point_mm, args.contact_line_plane_normal, flush=True)
    print('[contact-line] CSV:', output, '; metadata:', meta_path, flush=True)
    try:
        with output.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            while True:
                tick = time.monotonic()
                if controller is None and tick-started > args.contact_line_reference_timeout_s:
                    raise api.SafetyStop('stable synchronized contact reference timeout: ' + source.last_reason)
                # Poll actual robot history faster than inference. No shared ZMQ sockets.
                sample = source.poll()
                if source.clock_fault:
                    raise api.SafetyStop('capture/RTDE clock fault: ' + source.last_reason)
                if sample is None:
                    if tick-last_valid > args.contact_line_missing_timeout_s:
                        raise api.SafetyStop('synchronized sample timeout: ' + source.last_reason)
                    if controller is not None and tick-last_valid > args.contact_line_max_age_ms/1000:
                        controller.pause(tick)
                    if tick-last_print >= 1.:
                        print('[contact-line] waiting:', source.last_reason, flush=True)
                        last_print = tick
                    time.sleep(max(0., 1/args.contact_line_poll_hz - (time.monotonic()-tick)))
                    continue
                left, right = [api.format_tactile_frame(sample[side], args.xense_force_height, args.xense_force_width)
                               for side in ('left', 'right')]
                api.check_safety(args, force_reader, left, right)
                forces = np.array([np.linalg.norm(frame['force6d'][:3]) for frame in (left, right)])
                if min(forces) < args.min_tactile_peak:
                    raise api.SafetyStop('feedback probe lost bilateral grasp contact')
                evidence, evidence_kind, delta = force_evidence(forces, baseline, args.contact_line_contact_change_n,
                                                               args.contact_line_contact_established)
                last_valid = time.monotonic()
                if not evidence:
                    stable.samples.clear()
                    if controller is not None:
                        controller.pause(last_valid)
                        loss_since = last_valid if loss_since is None else loss_since
                        if last_valid-loss_since >= args.contact_line_contact_loss_s:
                            raise api.SafetyStop('external-contact force-change evidence lost')
                    continue
                loss_since = None
                tcp = sample['tcp']
                if controller is None:
                    if not stable.update(sample['timestamp'], tcp, forces):
                        continue
                    reference = api.paired_force_input(left, right)
                    reference_self = api.predict_pair_motion(model, norm, reference, reference, device)
                    if not np.isfinite(reference_self).all() or np.any(np.abs(reference_self) >= .25*target_limit):
                        raise api.SafetyStop('reference self-prediction too large; check model/preprocessing')
                    controller = ProbeController(config, tcp, H, reference_self, reference_stable=True)
                    metadata.update(reference_timestamp=sample['timestamp'], reference_tcp=tcp, reference_self_prediction=reference_self)
                    save_metadata()
                    print('[contact-line] stable reference latched; model self-prediction:', reference_self, flush=True)
                if last_valid-last_step < 1/args.control_hz:
                    time.sleep(max(0., 1/args.contact_line_poll_hz-(time.monotonic()-tick)))
                    continue
                raw_prediction = api.predict_pair_motion(model, norm, reference, api.paired_force_input(left, right), device)
                now, wall = time.monotonic(), time.time()
                if wall-min(sample['left_stamp'], sample['right_stamp']) > args.contact_line_max_age_ms/1000:
                    controller.pause(now)
                    if now-last_step > args.contact_line_missing_timeout_s and np.isfinite(last_step):
                        raise api.SafetyStop('inference latency exceeds sample age budget')
                    continue
                if not np.isfinite(raw_prediction).all() or np.any(np.abs(raw_prediction) >= .9*target_limit):
                    raise api.SafetyStop('prediction nonfinite or near checkpoint output limit')
                prediction = _pose(_inverse(_transform(reference_self)) @ _transform(raw_prediction))
                object_p, object_R = api.inferred_object_frame(tcp, prediction)
                estimator.append(sample['timestamp'], object_p, object_R, prediction, tcp)
                count += 1
                if count % args.contact_line_factor_graph_every == 0:
                    worker.submit()
                estimate = worker.latest()
                source_stamp = estimate.get('source_timestamp')
                if source_stamp is not None:
                    estimate['timestamp'] = now - (wall-source_stamp)
                # Use current measured TCP for command generation, synchronized TCP only for geometry.
                actual = np.asarray(robot.get_tcp_pose(), dtype=float)
                result = controller.step(now, actual, raw_prediction, estimate)
                if np.any(np.abs(result['desired_deformation']) >= .9*target_limit):
                    raise api.SafetyStop('desired deformation exceeds checkpoint output range')
                last_step = now
                row = dict(time_s=now-started, timestamp=sample['timestamp'],
                           phase=result['stage'], status=result['status'], reason=result['reason'], dt=result['dt'],
                           reference_id=1, external_contact_evidence=evidence, contact_evidence_source=evidence_kind,
                           left_force_n=forces[0], right_force_n=forces[1], left_force_change_n=delta[0], right_force_change_n=delta[1],
                           line_valid=estimate.get('valid', False), line_reason=estimate.get('reason'),
                           line_source_timestamp=source_stamp, line_age_s=wall-source_stamp if source_stamp is not None else None,
                           line_stable_s=result['line_stable_s'], line_qualified=result['line_qualified'])
                for name in ('left_stamp', 'right_stamp', 'left_right_delta_ms', 'tcp_skew_ms', 'tcp_span_ms', 'sample_age_s'):
                    row[name] = sample[name] if name != 'sample_age_s' else wall-min(sample['left_stamp'], sample['right_stamp'])
                for name in ('normal_excitation_deg', 'observability', 'raw_plane_rms_mm', 'raw_angle_rms_deg', 'optimizer_seconds'):
                    row[name] = estimate.get(name)
                for prefix, values in (('tcp', tcp), ('control_tcp', actual), ('pred', prediction), ('raw_pred', raw_prediction),
                                       ('desired', result['desired_deformation']), ('error', result['error']),
                                       ('measured_feedback', result['measured_feedback_deformation']),
                                       ('desired_feedback', result['desired_feedback_deformation']), ('command', result['command'])):
                    row.update({prefix+'_'+axis: value for axis, value in zip(('x', 'y', 'z', 'rx', 'ry', 'rz'), values)})
                if estimate.get('valid'):
                    for name, key in (('point', 'center_base'), ('direction', 'direction_base')):
                        row.update({'line_'+name+'_'+axis: value for axis, value in zip(('x', 'y', 'z'), estimate[key])})
                writer.writerow(row)
                handle.flush()
                if result['status'] == 'failed':
                    raise api.SafetyStop('feedback controller: ' + result['reason'])
                api.command_pose(robot, result['command'], args.execute)
                if now-last_print >= 1.:
                    print('[contact-line] phase={} line={} ({}) age={} stable={:.2f}s'.format(
                        result['stage'], estimate.get('valid', False), estimate.get('reason'), row['line_age_s'],
                        result['line_stable_s']), flush=True)
                    last_print = now
                if result['finished']:
                    metadata['outcome'] = result['status']
                    metadata['reason'] = result['reason']
                    print('[contact-line] finished:', result['status'], result['reason'], flush=True)
                    break
                time.sleep(max(0., 1/args.contact_line_poll_hz-(time.monotonic()-tick)))
    except BaseException as exc:
        metadata.update(outcome='stopped', reason=str(exc))
        raise
    finally:
        worker.close()
        metadata.update(finished=time.time(), synchronized_samples=source.accepted_count, inference_samples=count)
        save_metadata()
