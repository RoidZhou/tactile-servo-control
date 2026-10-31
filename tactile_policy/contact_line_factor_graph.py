"""Sliding-window factor graph solved with SciPy sparse nonlinear least squares.

This is a batch factor-graph backend, not GTSAM/iSAM2. Measured object poses
already combine robot feedback and paired tactile motion. Their uncertainty
is approximated as independent; shared-reference correlation is not modeled.
"""
from collections import deque
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix


class ContactLineFactorGraph:
    def __init__(self, capacity, min_samples, initializer_class,
                 position_sigma_m=0.0005, rotation_sigma_rad=np.deg2rad(0.5),
                 contact_sigma_m=0.0002, direction_sigma_rad=np.deg2rad(0.2),
                 max_nfev=30, update_every=5):
        self.samples = deque(maxlen=int(capacity))
        self.min_samples = int(min_samples)
        self.initializer_class = initializer_class
        self.scales = np.array([position_sigma_m, rotation_sigma_rad,
                                contact_sigma_m, direction_sigma_rad])
        if not np.isfinite(self.scales).all() or np.any(self.scales <= 0):
            raise ValueError("Factor sigmas must be finite and positive")
        self.max_nfev = int(max_nfev)
        self.update_every = int(update_every)
        if self.max_nfev <= 0 or self.update_every <= 0:
            raise ValueError("max_nfev and update_every must be positive")
        self.append_count = 0
        self.cached_estimate = None
        self.last_diagnostics = {}

    def append(self, timestamp, position, rotation, prediction, tcp):
        self.samples.append(dict(timestamp=float(timestamp),
                                 p=np.asarray(position).copy(), R=np.asarray(rotation).copy(),
                                 prediction=np.asarray(prediction).copy(), tcp=np.asarray(tcp).copy()))
        self.append_count += 1

    def estimate(self):
        if len(self.samples) < self.min_samples:
            return None
        if (self.cached_estimate is not None
                and self.append_count % self.update_every != 0):
            cached = dict(self.cached_estimate)
            cached["optimizer_updated"] = False
            cached["samples"] = len(self.samples)
            return cached
        initial = self.initializer_class(len(self.samples), self.min_samples)
        initial.samples.extend(self.samples)
        seed = initial.estimate()
        if seed is None:
            return None
        start = time.monotonic()
        samples = list(self.samples)
        p = np.stack([s['p'] for s in samples])
        r = np.stack([s['R'] for s in samples])
        n = len(samples) - 1
        # First pose fixed to measured pose: anchors the window. Remaining
        # poses have local rotation corrections and base-frame translations.
        x0 = np.r_[np.zeros(6*n), seed['center_local'], seed['direction_local']]
        sp, sr, sc, sd = self.scales

        def unpack(x):
            delta = x[:6*n].reshape(n, 6)
            pp = p.copy(); rr = r.copy()
            pp[1:] += delta[:, :3]
            rr[1:] = r[1:] @ Rotation.from_rotvec(delta[:, 3:]).as_matrix()
            q = x[-6:-3]; d = x[-3:]
            d = d / max(np.linalg.norm(d), 1e-12)
            return pp, rr, q, d, delta

        def residual(x):
            pp, rr, q, d, delta = unpack(x)
            # Unary tactile/robot measurements; sticking point and line
            # direction factors couple each pose to shared q,d variables.
            measurement = np.c_[delta[:, :3]/sp, delta[:, 3:]/sr]
            centers = (rr @ q[..., None])[..., 0] + pp
            directions = (rr @ d[..., None])[..., 0]
            blocks = np.c_[measurement, (centers[1:]-centers[0])/sc,
                           (directions[1:]-directions[0])/sd]
            # Unit direction plus representative-point gauge q dot d = 0.
            return np.r_[blocks.ravel(), (np.linalg.norm(x[-3:])-1)*100,
                         np.dot(q, d)/sc]

        sparsity = lil_matrix((12*n+2, 6*n+6), dtype=int)
        for i in range(n):
            sparsity[12*i:12*i+12, 6*i:6*i+6] = 1
            sparsity[12*i+6:12*i+12, -6:] = 1
        sparsity[-2:, -6:] = 1
        fit = least_squares(residual, x0, jac_sparsity=sparsity.tocsr(),
                            loss='soft_l1', x_scale='jac', max_nfev=self.max_nfev,
                            ftol=1e-5, xtol=1e-5, gtol=1e-5)
        self.last_diagnostics = dict(optimizer_success=bool(fit.success),
                                    optimizer_cost=float(fit.cost),
                                    optimizer_seconds=time.monotonic()-start,
                                    optimizer_updated=True)
        if not fit.success or not np.isfinite(fit.x).all():
            return None
        pp, rr, q, d, delta = unpack(fit.x)
        centers = (rr @ q[..., None])[..., 0] + pp
        directions = (rr @ d[..., None])[..., 0]
        direction = directions.mean(axis=0); direction /= np.linalg.norm(direction)
        if direction[np.argmax(abs(direction))] < 0:
            direction *= -1; d *= -1
        angles = np.arccos(np.clip(abs(directions @ direction), 0, 1))
        result = dict(seed)
        result.update(center_base=centers.mean(axis=0), direction_base=direction,
                      center_local=q, direction_local=d,
                      point_rms_m=float(np.sqrt(np.mean(np.sum((centers-centers[0])**2, axis=1)))),
                      angle_rms_rad=float(np.sqrt(np.mean(angles**2))),
                      optimized_position=pp[-1], optimized_rotation=rr[-1],
                      measurement_position_rms_m=float(np.sqrt(np.mean(np.sum(delta[:, :3]**2, axis=1)))),
                      measurement_rotation_rms_rad=float(np.sqrt(np.mean(np.sum(delta[:, 3:]**2, axis=1)))),
                      **self.last_diagnostics)
        # Observability remains from RAW measured motion, not optimizer-enforced
        # line consistency. No posterior confidence is claimed.
        self.cached_estimate = dict(result)
        return result
