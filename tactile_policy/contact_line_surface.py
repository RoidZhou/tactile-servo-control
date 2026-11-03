"""Estimate a contact line from a calibrated object surface, allowing in-plane slip.

Each measured object pose transforms a calibrated local plane into the base frame.
Normals and offsets must support one approximately common geometric line before a
soft temporal fit is attempted. The fit has one line state per sample; it never
fixes an object material point. Measurement poses are not optimized, so optimizer
success cannot manufacture normal excitation or erase inconsistent measurements.

Geometry alone cannot establish external contact: a free object executing a
single-axis programmed motion can pass these checks. The controller must supply
independent contact/preload evidence. Shared tactile-reference errors are not
modeled as independent covariance measurements, and no posterior covariance is
claimed. All positions use metres and angles use radians internally.
"""
from collections import deque
import threading
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix


class _SolveDeadline(RuntimeError):
    pass


class SurfaceContactLineEstimator:
    """Fit time-varying lines constrained by measured object-surface planes.

    ``plane_point_local`` and ``plane_normal_local`` belong to O*, the object-attached
    frame initially coincident with the reference TCP. They must be re-expressed if
    the tactile reference changes. ``valid`` means geometric consistency only.
"""

    def __init__(self, plane_point_local, plane_normal_local, capacity=60, min_samples=12,
                 min_normal_span_deg=1.0, min_observability=8.0,
                 max_raw_plane_rms_mm=0.5, max_raw_angle_rms_deg=0.5,
                 max_line_distance_mm=30.0, temporal_position_sigma_mm=0.5,
                 temporal_direction_sigma_deg=0.5, max_position_correction_mm=2.0,
                 max_direction_correction_deg=2.0, max_nfev=20,
                 max_solve_seconds=0.2, plane_sigma_mm=0.2,
                 plane_direction_sigma_deg=0.2):
        self.plane_point_local = np.asarray(plane_point_local, dtype=float).reshape(3).copy()
        self.plane_normal_local = np.asarray(plane_normal_local, dtype=float).reshape(3).copy()
        normal_norm = np.linalg.norm(self.plane_normal_local)
        if (not np.isfinite(self.plane_point_local).all()
                or not np.isfinite(self.plane_normal_local).all()
                or not np.isfinite(normal_norm) or normal_norm < 1e-12):
            raise ValueError("Plane point/normal must be finite and normal nonzero")
        self.plane_normal_local /= normal_norm
        if (isinstance(capacity, bool) or isinstance(min_samples, bool)
                or int(capacity) != capacity or int(min_samples) != min_samples
                or not 3 <= min_samples <= capacity):
            raise ValueError("Require integer capacity >= min_samples >= 3")
        if isinstance(max_nfev, bool) or int(max_nfev) != max_nfev or max_nfev < 1:
            raise ValueError("max_nfev must be a positive integer")
        self.capacity, self.min_samples = int(capacity), int(min_samples)
        self.max_nfev = int(max_nfev)
        settings = dict(
            min_normal_span_deg=min_normal_span_deg, min_observability=min_observability,
            max_raw_plane_rms_mm=max_raw_plane_rms_mm, max_raw_angle_rms_deg=max_raw_angle_rms_deg,
            max_line_distance_mm=max_line_distance_mm,
            temporal_position_sigma_mm=temporal_position_sigma_mm,
            temporal_direction_sigma_deg=temporal_direction_sigma_deg,
            max_position_correction_mm=max_position_correction_mm,
            max_direction_correction_deg=max_direction_correction_deg,
            max_solve_seconds=max_solve_seconds, plane_sigma_mm=plane_sigma_mm,
            plane_direction_sigma_deg=plane_direction_sigma_deg,
        )
        for name, value in settings.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be positive and finite")
            setattr(self, name, float(value))
        if min_observability <= 1:
            raise ValueError("min_observability must exceed one")
        for name in ('temporal_direction_sigma_deg', 'plane_direction_sigma_deg'):
            if getattr(self, name) >= 90:
                raise ValueError(name + " must be less than 90 degrees")
        for name in ('min_normal_span_deg', 'max_raw_angle_rms_deg',
                     'max_direction_correction_deg'):
            if getattr(self, name) > 90:
                raise ValueError(name + " must not exceed 90 degrees")
        self.samples = deque(maxlen=self.capacity)
        self._lock = threading.Lock()

    def append(self, timestamp, position, rotation, prediction, tcp):
        sample = dict(timestamp=float(timestamp),
                      p=np.asarray(position, dtype=float).reshape(3).copy(),
                      R=np.asarray(rotation, dtype=float).reshape(3, 3).copy(),
                      prediction=np.asarray(prediction, dtype=float).reshape(6).copy(),
                      tcp=np.asarray(tcp, dtype=float).reshape(6).copy())
        if (not np.isfinite(sample['timestamp'])
                or not all(np.isfinite(sample[key]).all() for key in ('p', 'R', 'prediction', 'tcp'))
                or not np.allclose(sample['R'].T @ sample['R'], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(sample['R']), 1.0, atol=1e-5)):
            raise ValueError("Sample must contain a finite rigid pose and finite timing/prediction")
        with self._lock:
            if self.samples and sample['timestamp'] <= self.samples[-1]['timestamp']:
                raise ValueError("Sample capture timestamps must increase")
            self.samples.append(sample)

    def snapshot(self):
        """Return independent samples for a worker-owned estimator or diagnostics."""
        with self._lock:
            return [{key: value.copy() if isinstance(value, np.ndarray) else value
                     for key, value in sample.items()} for sample in self.samples]

    @staticmethod
    def _rotvec_matrix(vector):
        theta = np.linalg.norm(vector)
        if theta < 1e-12:
            return np.eye(3)
        x, y, z = vector / theta
        cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        return np.eye(3) + np.sin(theta) * cross + (1 - np.cos(theta)) * (cross @ cross)

    def estimate(self):
        samples = self.snapshot()
        result = dict(valid=False, reason='insufficient_samples', samples=len(samples),
                      source_timestamp=samples[-1]['timestamp'] if samples else None,
                      estimate_timestamp=time.time(), normal_excitation_deg=0.0,
                      observability=0.0, raw_plane_rms_mm=float('nan'),
                      raw_angle_rms_deg=float('nan'), optimizer_success=False,
                      optimizer_updated=False, optimizer_seconds=0.0,
                      measurement_position_rms_m=0.0, measurement_rotation_rms_rad=0.0,
                      measurement_pose_corrected=False, external_contact_evidence_required=True,
                      tcp_only_direction_difference_deg=float('nan'))

        def invalid(reason):
            result.update(valid=False, reason=reason, estimate_timestamp=time.time())
            return result

        if len(samples) < self.min_samples:
            return result
        rotations = np.stack([s['R'] for s in samples])
        positions = np.stack([s['p'] for s in samples])
        points = positions + (rotations @ self.plane_point_local[:, None])[..., 0]
        normals = (rotations @ self.plane_normal_local[:, None])[..., 0]
        origin = points.mean(axis=0)
        points_relative = points - origin
        dots = np.clip(np.abs(normals @ normals.T), 0.0, 1.0)
        result['normal_excitation_deg'] = float(np.rad2deg(np.arccos(dots.min())))
        if result['normal_excitation_deg'] < self.min_normal_span_deg:
            return invalid('insufficient_normal_excitation')
        u, singular, vt = np.linalg.svd(normals, full_matrices=False)
        result['normal_singular_values'] = singular.copy()
        result['observability'] = float(singular[1] / max(singular[2], 1e-12))
        if singular[1] < 1e-8 or result['observability'] < self.min_observability:
            return invalid('no_common_line_direction')
        direction = vt[-1].copy()
        if direction[np.argmax(np.abs(direction))] < 0:
            direction *= -1
        rhs = np.einsum('ij,ij->i', normals, points_relative)
        center = vt[:2].T @ ((u[:, :2].T @ rhs) / singular[:2])
        plane_errors = normals @ center - rhs
        angles = np.arcsin(np.clip(np.abs(normals @ direction), 0., 1.))
        result.update(raw_plane_rms_mm=float(np.sqrt(np.mean(plane_errors**2)) * 1000),
                      raw_angle_rms_deg=float(np.rad2deg(np.sqrt(np.mean(angles**2)))),
                      raw_center_base=origin + center, raw_direction_base=direction.copy())
        if result['raw_plane_rms_mm'] > self.max_raw_plane_rms_mm:
            return invalid('inconsistent_plane_offsets')
        if result['raw_angle_rms_deg'] > self.max_raw_angle_rms_deg:
            return invalid('inconsistent_plane_directions')
        distances = np.linalg.norm(np.cross(points_relative - center, direction), axis=1)
        result['max_line_distance_mm'] = float(distances.max() * 1000)
        if result['max_line_distance_mm'] > self.max_line_distance_mm:
            return invalid('line_outside_physical_region')

        # This diagnostic compares geometric directions only; it is not a contact test.
        tcp_normals = np.stack([self._rotvec_matrix(s['tcp'][3:]) @ self.plane_normal_local
                                for s in samples])
        _, tcp_singular, tcp_vt = np.linalg.svd(tcp_normals, full_matrices=False)
        result['tcp_only_observability'] = float(tcp_singular[1] / max(tcp_singular[2], 1e-12))
        if tcp_singular[1] > 1e-8:
            result['tcp_only_direction_difference_deg'] = float(np.rad2deg(np.arccos(
                np.clip(abs(tcp_vt[-1] @ direction), 0., 1.))))

        n = len(samples)
        seed = np.tile(np.r_[center, direction], (n, 1))
        sp = self.plane_sigma_mm / 1000
        sd = np.sin(np.deg2rad(self.plane_direction_sigma_deg))
        tp = self.temporal_position_sigma_mm / 1000
        td = np.sin(np.deg2rad(self.temporal_direction_sigma_deg))
        started = time.perf_counter()

        def unpack(x):
            state = x.reshape(n, 6)
            lengths = np.linalg.norm(state[:, 3:], axis=1)
            dirs = state[:, 3:] / np.maximum(lengths[:, None], 1e-12)
            return state[:, :3], dirs, lengths

        def residual(x):
            if time.perf_counter() - started > self.max_solve_seconds:
                raise _SolveDeadline()
            centers, dirs, lengths = unpack(x)
            geometry = np.c_[np.einsum('ij,ij->i', normals, centers - points_relative) / sp,
                             np.einsum('ij,ij->i', normals, dirs) / sd,
                             (lengths - 1) / .01,
                             np.einsum('ij,ij->i', centers, dirs) / sp]
            delta = centers[1:] - centers[:-1]
            transverse = delta - np.einsum('ij,ij->i', delta, dirs[:-1])[:, None] * dirs[:-1]
            temporal = np.c_[transverse / tp, np.cross(dirs[:-1], dirs[1:]) / td]
            return np.r_[geometry.ravel(), temporal.ravel()]

        # Analytic block Jacobian avoids graph-coloured finite differences on
        # every solve. This retains the same residuals, robust loss and gates.
        geometry_rows = np.broadcast_to(4*np.arange(n)[:, None, None] + np.arange(4)[None, :, None], (n, 4, 6))
        geometry_cols = np.broadcast_to(6*np.arange(n)[:, None, None] + np.arange(6)[None, None, :], (n, 4, 6))
        temporal_rows = np.broadcast_to(4*n + 6*np.arange(n-1)[:, None, None] + np.arange(6)[None, :, None], (n-1, 6, 12))
        temporal_cols = np.broadcast_to(6*np.arange(n-1)[:, None, None] + np.arange(12)[None, None, :], (n-1, 6, 12))
        rows = np.r_[geometry_rows.ravel(), temporal_rows.ravel()]
        cols = np.r_[geometry_cols.ravel(), temporal_cols.ravel()]

        def cross_matrices(vectors):
            matrices = np.zeros((len(vectors), 3, 3))
            matrices[:, 0, 1], matrices[:, 0, 2] = -vectors[:, 2], vectors[:, 1]
            matrices[:, 1, 0], matrices[:, 1, 2] = vectors[:, 2], -vectors[:, 0]
            matrices[:, 2, 0], matrices[:, 2, 1] = -vectors[:, 1], vectors[:, 0]
            return matrices

        def jacobian(x):
            if time.perf_counter() - started > self.max_solve_seconds:
                raise _SolveDeadline()
            centers, dirs, lengths = unpack(x)
            derivative = (np.eye(3)[None] - dirs[:, :, None]*dirs[:, None, :]) / np.maximum(lengths[:, None, None], 1e-12)
            geometry = np.zeros((n, 4, 6))
            geometry[:, 0, :3] = normals / sp
            geometry[:, 1, 3:] = np.einsum('ni,nij->nj', normals, derivative) / sd
            geometry[:, 2, 3:] = dirs / .01
            geometry[:, 3, :3] = dirs / sp
            geometry[:, 3, 3:] = np.einsum('ni,nij->nj', centers, derivative) / sp
            delta = centers[1:] - centers[:-1]
            d0, d1 = dirs[:-1], dirs[1:]
            projection = np.eye(3)[None] - d0[:, :, None]*d0[:, None, :]
            temporal = np.zeros((n-1, 6, 12))
            temporal[:, :3, :3] = -projection / tp
            temporal[:, :3, 6:9] = projection / tp
            along = np.einsum('ni,ni->n', delta, d0)
            temporal[:, :3, 3:6] = -((d0[:, :, None]*delta[:, None, :]
                + along[:, None, None]*np.eye(3)[None]) @ derivative[:-1]) / tp
            temporal[:, 3:, 3:6] = -cross_matrices(d1) @ derivative[:-1] / td
            temporal[:, 3:, 9:12] = cross_matrices(d0) @ derivative[1:] / td
            values = np.r_[geometry.ravel(), temporal.ravel()]
            return coo_matrix((values, (rows, cols)), shape=(4*n + 6*(n-1), 6*n)).tocsr()
        try:
            fit = least_squares(residual, seed.ravel(), jac=jacobian,
                                loss='soft_l1', x_scale='jac', max_nfev=self.max_nfev,
                                tr_options={'maxiter': 40, 'atol': 1e-4, 'btol': 1e-4},
                                ftol=1e-5, xtol=1e-5, gtol=1e-5)
        except _SolveDeadline:
            result['optimizer_seconds'] = time.perf_counter() - started
            return invalid('optimizer_time_budget')
        except (ValueError, np.linalg.LinAlgError):
            result['optimizer_seconds'] = time.perf_counter() - started
            return invalid('optimizer_numerical_failure')
        result.update(optimizer_success=bool(fit.success), optimizer_updated=True,
                      optimizer_seconds=time.perf_counter() - started,
                      optimizer_cost=float(fit.cost), optimizer_nfev=int(fit.nfev))
        if result['optimizer_seconds'] > self.max_solve_seconds:
            return invalid('optimizer_time_budget')
        if not fit.success or not np.isfinite(fit.x).all():
            return invalid('optimizer_not_converged')
        centers, dirs, _ = unpack(fit.x)
        corrections = centers - center
        corrections -= (corrections @ direction)[:, None] * direction
        correction_mm = np.linalg.norm(corrections, axis=1) * 1000
        correction_deg = np.rad2deg(np.arccos(np.clip(np.abs(dirs @ direction), 0., 1.)))
        result.update(max_line_position_correction_mm=float(correction_mm.max()),
                      max_line_direction_correction_deg=float(correction_deg.max()))
        if correction_mm.max() > self.max_position_correction_mm:
            return invalid('line_position_correction_exceeded')
        if correction_deg.max() > self.max_direction_correction_deg:
            return invalid('line_direction_correction_exceeded')
        distances = np.linalg.norm(np.cross(points_relative - centers, dirs), axis=1)
        result['max_line_distance_mm'] = float(distances.max() * 1000)
        if result['max_line_distance_mm'] > self.max_line_distance_mm:
            return invalid('line_outside_physical_region')
        plane_errors = np.einsum('ij,ij->i', normals, centers - points_relative)
        angle_errors = np.arcsin(np.clip(np.abs(np.einsum('ij,ij->i', normals, dirs)), 0., 1.))
        result.update(valid=True, reason='geometry_consistent', estimate_timestamp=time.time(),
                      center_base=origin + centers[-1], direction_base=dirs[-1].copy(),
                      plane_point_base=points[-1].copy(), plane_normal_base=normals[-1].copy(),
                      line_states_center_base=origin + centers, line_states_direction_base=dirs,
                      point_rms_m=float(np.sqrt(np.mean(plane_errors**2))),
                      angle_rms_rad=float(np.sqrt(np.mean(angle_errors**2))),
                      raw_line_distance_mm=float(np.linalg.norm(np.cross(points_relative[-1]-center, direction))*1000))
        return result
