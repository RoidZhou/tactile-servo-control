#!/usr/bin/env python3
"""Focused real-UR5 active contact-line inference.

This entry point intentionally implements one task only::

    stable bilateral reference
      -> push-down -> bounded roll/pitch rocking -> back-and-forth translation
      -> paired tactile relative-SE(3) inference
      -> capture-time synchronized UR5 pose fusion
      -> sliding-window contact-line optimization

The organization follows the active sensing idea used by
``active_extrinsic_contact/codes_real_experiments/main_wall.py``: establish a
contact reference, actively excite the contact by rocking, and continuously
feed tactile motion plus robot motion into a contact estimator.  It does not
contain grasp, transport, descent, screw, fixed-cycle, or pose-validation
modes.  Before starting, the operator must already have the nut firmly grasped
and lightly contacting the bolt.

No robot command is sent without ``--execute``.  The default only checks the
checkpoint, robot connection, and bilateral tactile frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.tactile_node import ZMQClientTactile
from tactile_policy.contact_line_runtime import (
    run_feedback_probe,
    validate_probe_arguments,
)
from tactile_policy.data_relative_se3 import PairNormalization, signed_log1p
from tactile_policy.model import load_pairwise_model


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "tactile_policy/outputs/relative_se3_600_200_native_depth_cnn/best_model.pt"
)
PAIRWISE_INPUT_KINDS = frozenset(("paired_force", "paired_depth"))
TACTILE_FORCE_LIMIT_N = np.asarray([30.0, 25.0, 25.0])
TACTILE_TORQUE_LIMIT_NMM = np.asarray([100.0, 100.0, 100.0])


class SafetyStop(RuntimeError):
    """Stop active probing without issuing another motion command."""


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def is_pairwise_model(model) -> bool:
    return getattr(model, "tactile_input_kind", "") in PAIRWISE_INPUT_KINDS


def load_model(path: Path, device: torch.device):
    """Load only paired relative-SE(3) checkpoints used by this program."""
    path = path.expanduser().resolve()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type")
    if model_type not in (
        "pairwise_tactile_se3", "pairwise_backbone_se3",
        "pairwise_native_depth_se3",
    ):
        raise ValueError(f"Unsupported contact-line model_type: {model_type!r}")
    model = load_pairwise_model(checkpoint, device)
    normalization = PairNormalization(**{
        key: np.asarray(checkpoint["normalization"][key], dtype=np.float32)
        for key in ("input_mean", "input_std", "target_limit")
    })
    channels = 2 if model_type == "pairwise_native_depth_se3" else 6
    if (normalization.input_mean.shape != (channels,)
            or normalization.input_std.shape != (channels,)
            or normalization.target_limit.shape != (6,)
            or not all(np.isfinite(value).all() for value in (
                normalization.input_mean, normalization.input_std,
                normalization.target_limit,
            ))
            or np.any(normalization.input_std <= 0)
            or np.any(normalization.target_limit <= 0)):
        raise ValueError("Invalid paired-model normalization")
    if model_type == "pairwise_native_depth_se3":
        preprocessing = checkpoint.get("input_preprocessing", {})
        if (checkpoint.get("input_modality") != "depth"
                or checkpoint.get("input_keys") != [
                    "observation.tactile_left.depth",
                    "observation.tactile_right.depth",
                ]
                or checkpoint.get("input_shape_per_frame") != [2, *model.input_hw]
                or preprocessing.get("channel_order_per_time") != ["left", "right"]
                or preprocessing.get("time_order") != ["reference", "current"]
                or preprocessing.get("png_channel") != 0
                or preprocessing.get("resize") is not None
                or preprocessing.get("signed_log1p") is not False
                or preprocessing.get("per_image_normalization") is not False):
            raise ValueError("Depth checkpoint preprocessing is incompatible")
        divisor = float(preprocessing.get("divisor", np.nan))
        if (preprocessing.get("type") not in (
                "depth_png_uint8_div255", "depth_png_uint8_train_max")
                or not np.isfinite(divisor) or divisor <= 0
                or not np.all(normalization.input_mean == 0)
                or not np.allclose(normalization.input_std, divisor, rtol=1e-6)):
            raise ValueError("Depth normalization disagrees with checkpoint metadata")
    print(
        f"Loaded {model_type} checkpoint={path} epoch={checkpoint.get('epoch')} "
        f"target_limit={normalization.target_limit.tolist()}", flush=True,
    )
    return model, normalization


def _fit_force(value, height: int, width: int) -> tuple[np.ndarray, bool]:
    output = np.zeros((height, width, 3), dtype=np.float32)
    if value is None:
        return output, False
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0 or not np.isfinite(array).all():
        return output, False
    if array.ndim == 3 and array.shape[-1] == 3:
        h, w = min(height, array.shape[0]), min(width, array.shape[1])
        output[:h, :w] = array[:h, :w]
        return output, True
    try:
        flattened = array.reshape(-1, 3)
    except ValueError:
        return output, False
    count = min(len(flattened), height * width)
    output.reshape(-1, 3)[:count] = flattened[:count]
    return output, count > 0


def format_tactile_frame(frame: dict, height: int, width: int) -> dict:
    force, force_ok = _fit_force(frame.get("force"), height, width)
    wrench = np.asarray(frame.get("force6d", np.zeros(6)), dtype=np.float32).reshape(-1)
    force6d = np.zeros(6, dtype=np.float32)
    force6d[:min(6, len(wrench))] = wrench[:6]
    valid = bool(frame.get("valid", force_ok)) and force_ok
    if not np.isfinite(force6d).all():
        valid = False
    return {
        "force": force, "force6d": force6d, "depth": frame.get("depth"),
        "capture_timestamp": frame.get("capture_timestamp", 0),
        "timestamp": frame.get("timestamp", 0), "valid": valid,
    }


class TactileSource:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.left = ZMQClientTactile(
            port=args.left_tactile_port, host=args.host,
            timeout_ms=args.tactile_timeout_ms,
        )
        self.right = ZMQClientTactile(
            port=args.right_tactile_port, host=args.host,
            timeout_ms=args.tactile_timeout_ms,
        )

    def read(self) -> tuple[dict, dict]:
        frames = [
            format_tactile_frame(client.read(), self.args.xense_force_height,
                                 self.args.xense_force_width)
            for client in (self.left, self.right)
        ]
        if not all(frame["valid"] for frame in frames):
            raise SafetyStop("one or both XENSE tactile frames are invalid")
        return frames[0], frames[1]

    def close(self) -> None:
        self.left.close()
        self.right.close()


def _depth_uint8(value, input_hw: tuple[int, int]) -> np.ndarray:
    if value is None:
        raise SafetyStop("paired_depth model requires SDK depth on both sensors")
    array = np.squeeze(np.asarray(value))
    if array.ndim == 3:
        if array.shape[-1] in (1, 3, 4):
            array = array[..., 0]
        elif array.shape[0] in (1, 3, 4):
            array = array[0]
    if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
        raise SafetyStop(f"invalid XENSE depth shape: {array.shape}")
    if array.shape != input_hw:
        from PIL import Image
        array = np.asarray(
            Image.fromarray(array.astype(np.float32)).resize(
                (input_hw[1], input_hw[0])
            ), dtype=np.float32,
        )
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    array = np.asarray(array, dtype=np.float32)
    if float(array.max()) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def paired_tactile_input(model, left: dict, right: dict) -> np.ndarray:
    kind = model.tactile_input_kind
    if kind == "paired_force":
        values = [np.asarray(frame["force"], dtype=np.float32)
                  for frame in (left, right)]
        if any(value.ndim != 3 or value.shape[-1] != 3
               or not np.isfinite(value).all() for value in values):
            raise SafetyStop("invalid native XENSE force maps")
        return signed_log1p(
            np.concatenate(values, axis=-1).transpose(2, 0, 1)
        ).astype(np.float32)
    if kind == "paired_depth":
        return np.stack([
            _depth_uint8(frame.get("depth"), model.input_hw)
            for frame in (left, right)
        ]).astype(np.float32)
    raise ValueError(f"unsupported tactile input kind: {kind!r}")


def predict_pair_motion(model, normalization, reference, current, device):
    if reference.shape != current.shape:
        raise SafetyStop("reference/current tactile shapes differ")
    inputs = np.stack((reference, current)).astype(np.float32)
    inputs = (
        inputs - normalization.input_mean[None, :, None, None]
    ) / normalization.input_std[None, :, None, None]
    with torch.no_grad():
        output = model(torch.from_numpy(inputs)[None].to(device))
    return output.cpu().numpy()[0].astype(np.float64) * normalization.target_limit


def rotvec_to_matrix(vector: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(vector, dtype=np.float64)).as_matrix()


def inferred_object_frame(actual: np.ndarray, relative_motion: np.ndarray):
    """Fuse synchronized robot pose and tactile D=inv(T_NG0)@T_NGt."""
    actual_rotation = rotvec_to_matrix(actual[3:])
    relative_rotation = rotvec_to_matrix(relative_motion[3:])
    inferred_rotation = actual_rotation @ relative_rotation.T
    inferred_position = actual[:3] - inferred_rotation @ relative_motion[:3]
    return inferred_position, inferred_rotation


def tactile_peaks(left: dict, right: dict) -> tuple[float, float]:
    return tuple(float(np.linalg.norm(frame["force6d"][:3]))
                 for frame in (left, right))


def check_safety(args, force_reader, left: dict, right: dict) -> None:
    del args, force_reader
    for name, frame in (("left", left), ("right", right)):
        wrench = np.asarray(frame["force6d"], dtype=np.float64)
        if np.any(np.abs(wrench[:3]) >= TACTILE_FORCE_LIMIT_N):
            raise SafetyStop(f"{name} tactile force limit exceeded: {wrench[:3]}")
        if np.any(np.abs(wrench[3:]) >= TACTILE_TORQUE_LIMIT_NMM):
            raise SafetyStop(f"{name} tactile torque limit exceeded: {wrench[3:]}")


def command_pose(robot, target: np.ndarray, execute: bool) -> None:
    if execute:
        robot.command_tcp_pose(np.asarray(target, dtype=np.float64))


def smoothstep5(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def interpolate_pose(start: np.ndarray, target: np.ndarray, blend: float) -> np.ndarray:
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    first = Rotation.from_rotvec(start[3:])
    relative = (first.inv() * Rotation.from_rotvec(target[3:])).as_rotvec()
    rotation = first * Rotation.from_rotvec(float(blend) * relative)
    return np.r_[start[:3] + blend * (target[:3] - start[:3]), rotation.as_rotvec()]


def monitored_wait(args, tactile: TactileSource, seconds: float) -> tuple[dict, dict]:
    import time
    deadline = time.monotonic() + seconds
    latest = None
    while time.monotonic() < deadline:
        latest = tactile.read()
        check_safety(args, None, *latest)
        time.sleep(min(1.0 / args.control_hz, max(0.0, deadline - time.monotonic())))
    return latest if latest is not None else tactile.read()


def move_pose(args, robot, tactile, target: np.ndarray, duration_s: float) -> None:
    import time
    start = np.asarray(robot.get_tcp_pose(), dtype=np.float64)
    steps = max(1, int(round(duration_s * args.control_hz)))
    period = 1.0 / args.control_hz
    for index in range(1, steps + 1):
        tick = time.monotonic()
        left, right = tactile.read()
        check_safety(args, None, left, right)
        command_pose(
            robot,
            interpolate_pose(start, target, smoothstep5(index / steps)),
            args.execute,
        )
        time.sleep(max(0.0, period - (time.monotonic() - tick)))


def measure_grasp_baseline(args, tactile: TactileSource) -> np.ndarray:
    samples = []
    for _ in range(args.baseline_samples):
        left, right = tactile.read()
        check_safety(args, None, left, right)
        samples.append(tactile_peaks(left, right))
    baseline = np.median(np.asarray(samples, dtype=np.float64), axis=0)
    print("[prepare] free-grasp force baseline (N):", np.round(baseline, 4))
    return baseline


def descend_until_contact(
    args, robot, tactile: TactileSource, above: np.ndarray,
    contact: np.ndarray, baseline: np.ndarray,
) -> np.ndarray:
    """Slow geometric descent; stop after sustained bilateral force changes."""
    import time
    distance = float(np.linalg.norm(contact[:3] - above[:3]))
    # Quintic smoothstep has a peak derivative of 1.875. Scale duration so
    # --descent-speed-mm-s is an actual upper bound, not merely average speed.
    duration = 1.875 * distance / (args.descent_speed_mm_s / 1000.0)
    steps = max(1, int(np.ceil(duration * args.control_hz)))
    period = 1.0 / args.control_hz
    axis = args.transport_axis_base
    consecutive = 0
    last_command = above.copy()
    for index in range(1, steps + 1):
        tick = time.monotonic()
        progress = index / steps
        command = interpolate_pose(above, contact, smoothstep5(progress))
        command_pose(robot, command, args.execute)
        time.sleep(max(0.0, 0.5 * period - (time.monotonic() - tick)))
        left, right = tactile.read()
        check_safety(args, None, left, right)
        forces = np.asarray(tactile_peaks(left, right))
        changes = np.abs(forces - baseline)
        height_to_contact_mm = max(
            0.0, float(np.dot(command[:3] - contact[:3], axis)) * 1000.0
        )
        inside_window = height_to_contact_mm <= args.descent_contact_window_mm
        changed = bool(np.all(changes >= args.descent_contact_change_n))
        consecutive = consecutive + 1 if inside_window and changed else 0
        last_command = command
        if index % max(1, int(args.control_hz)) == 0 or consecutive:
            print(
                f"[prepare] descent remaining={height_to_contact_mm:.2f}mm "
                f"forces={np.round(forces, 3)}N delta={np.round(changes, 3)}N "
                f"confirm={consecutive}/{args.descent_contact_frames}",
                flush=True,
            )
        if consecutive >= args.descent_contact_frames:
            print("[prepare] sustained bilateral contact change detected; descent stopped.")
            return last_command
        time.sleep(max(0.0, period - (time.monotonic() - tick)))
    # Contact may occur exactly at the calibrated endpoint. Hold without any
    # further descent and allow several fresh tactile frames to confirm it.
    for _ in range(args.descent_contact_frames + 5):
        left, right = tactile.read()
        check_safety(args, None, left, right)
        forces = np.asarray(tactile_peaks(left, right))
        changes = np.abs(forces - baseline)
        consecutive = (consecutive + 1 if np.all(
            changes >= args.descent_contact_change_n
        ) else 0)
        if consecutive >= args.descent_contact_frames:
            print("[prepare] endpoint contact confirmed; descent stopped.")
            return last_command
    raise SafetyStop("contact pose reached without sustained bilateral tactile change")


def prepare_probe(args, robot, tactile: TactileSource) -> np.ndarray:
    """Manual load followed by collision-aware lift, transfer, and descent."""
    print("[prepare] Opening RG2. Keep the object supported by hand.", flush=True)
    robot.command_gripper(0.0, force=args.gripper_force)
    monitored_wait(args, tactile, args.gripper_settle_s)
    answer = input(
        "Place the object between the parallel fingertips, keep hands clear, "
        "then type READY: "
    ).strip()
    if answer != "READY":
        raise SystemExit("Preparation cancelled before closing RG2")

    print("[prepare] Closing RG2.", flush=True)
    robot.command_gripper(1.0, force=args.gripper_force)
    left, right = monitored_wait(args, tactile, args.gripper_settle_s)
    peaks = tactile_peaks(left, right)
    if min(peaks) < args.min_tactile_peak:
        raise SafetyStop(f"object was not grasped bilaterally; peaks={peaks}")

    axis = args.transport_axis_base
    current = np.asarray(robot.get_tcp_pose(), dtype=np.float64)
    lift = current.copy()
    lift[:3] += axis * (args.transport_height_mm / 1000.0)
    contact = np.asarray(args.contact_tcp_pose, dtype=np.float64)
    above = contact.copy()
    above[:3] += axis * (args.approach_height_mm / 1000.0)
    transfer = above.copy()
    transfer_axis = max(float(np.dot(lift[:3], axis)), float(np.dot(above[:3], axis)))
    transfer[:3] += axis * (transfer_axis - float(np.dot(transfer[:3], axis)))

    print("[prepare] lift:", np.round(lift, 6), flush=True)
    move_pose(args, robot, tactile, lift, args.move_duration_s)
    print("[prepare] high transfer:", np.round(transfer, 6), flush=True)
    move_pose(args, robot, tactile, transfer, args.move_duration_s)
    print("[prepare] above target:", np.round(above, 6), flush=True)
    move_pose(args, robot, tactile, above, args.move_duration_s)
    monitored_wait(args, tactile, min(1.0, args.gripper_settle_s))
    baseline = measure_grasp_baseline(args, tactile)
    descend_until_contact(args, robot, tactile, above, contact, baseline)
    monitored_wait(args, tactile, min(1.0, args.gripper_settle_s))
    return baseline


def three_floats(values: str) -> np.ndarray:
    result = np.asarray([float(value) for value in values.split(",")], dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise argparse.ArgumentTypeError("expected three finite comma-separated values")
    return result


def six_floats(values: str) -> np.ndarray:
    result = np.asarray([float(value) for value in values.split(",")], dtype=np.float64)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise argparse.ArgumentTypeError("expected six finite comma-separated values")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=6001)
    parser.add_argument("--robot-timeout-ms", type=int, default=10000)
    parser.add_argument("--left-tactile-port", type=int, default=5100)
    parser.add_argument("--right-tactile-port", type=int, default=5101)
    parser.add_argument("--tactile-timeout-ms", type=int, default=15000)
    parser.add_argument("--xense-force-height", type=int, default=35)
    parser.add_argument("--xense-force-width", type=int, default=20)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--min-tactile-peak", type=float, default=1e-4)
    parser.add_argument(
        "--tactile-sync-max-skew-ms", type=float, default=60.0,
        help="Maximum left/right SDK capture-time difference.",
    )
    parser.add_argument(
        "--standard-tcp-to-nut-center", type=three_floats,
        default=three_floats("0,0,0.100"),
        help="TCP-to-nut-centre vector in the stable reference TCP frame (m).",
    )

    preparation = parser.add_argument_group("Grasp, transport, and contact preparation")
    preparation.add_argument(
        "--prepare-probe", action=argparse.BooleanOptionalAction, default=True,
        help=("Open RG2, wait for manual object placement, close, lift, transfer "
              "above the target, and descend until tactile contact change."),
    )
    preparation.add_argument(
        "--contact-tcp-pose", type=six_floats, default=six_floats("-0.43727224,-0.16484881, 0.20620829, -2.18654589, -2.12657183, 0.05866673"),
        help=("TCP [x,y,z,rx,ry,rz] for nominal first nut/bolt contact. "
              "Required when --prepare-probe is enabled."),
    )
    preparation.add_argument(
        "--transport-axis-base", type=three_floats, default=three_floats("0,0,1"),
        help="World direction used for lift and approach; normally 0,0,1.",
    )
    preparation.add_argument("--transport-height-mm", type=float, default=80.0)
    preparation.add_argument("--approach-height-mm", type=float, default=30.0)
    preparation.add_argument("--move-duration-s", type=float, default=5.0)
    preparation.add_argument("--descent-speed-mm-s", type=float, default=1.0)
    preparation.add_argument("--descent-contact-window-mm", type=float, default=10.0)
    preparation.add_argument(
        "--descent-contact-change-n", type=float, default=None,
        help=("Bilateral force-change threshold during descent. Defaults to "
              "--contact-line-contact-change-n so preparation and probing "
              "use the same acquisition threshold."),
    )
    preparation.add_argument("--descent-contact-frames", type=int, default=3)
    preparation.add_argument("--baseline-samples", type=int, default=10)
    preparation.add_argument("--gripper-force", type=float, default=4.0)
    preparation.add_argument("--gripper-settle-s", type=float, default=2.5)

    geometry = parser.add_argument_group("Known contact geometry")
    geometry.add_argument(
        "--contact-line-plane-point-mm", nargs=3, type=float, default=[0.0, 0.0, 100.0],
        help="A calibrated point on the nut bottom plane in reference-TCP axes.",
    )
    geometry.add_argument(
        "--contact-line-plane-normal", nargs=3, type=float,
        default=[0.0, 0.0, 1.0],
        help="Nut-bottom plane normal in reference-TCP axes.",
    )
    geometry.add_argument(
        "--contact-line-feedback-point-mm", nargs=3, type=float, default=None,
        help="Rocking control point in reference-TCP axes; default is nut centre.",
    )

    evidence = parser.add_argument_group("External-contact evidence")
    evidence_choice = evidence.add_mutually_exclusive_group()
    evidence_choice.add_argument(
        "--contact-line-contact-established", action="store_true",
        help="Operator confirms nut/bolt contact before starting.",
    )
    evidence_choice.add_argument(
        "--contact-line-free-grasp-forces-n", nargs=2, type=float,
        help="Left/right force norms measured before external nut/bolt contact.",
    )
    evidence.add_argument(
        "--contact-line-contact-change-n", type=float, default=0.1,
        help="Force change required to acquire external contact, in N.",
    )
    evidence.add_argument(
        "--contact-line-contact-release-n", type=float, default=0.05,
        help=("Force-change hysteresis threshold used to retain contact; "
              "must be lower than the acquisition threshold."),
    )
    evidence.add_argument("--contact-line-contact-loss-s", type=float, default=0.3)

    reference = parser.add_argument_group("Synchronized stable reference")
    reference.add_argument("--contact-line-poll-hz", type=float, default=20.0)
    reference.add_argument("--contact-line-tcp-skew-ms", type=float, default=100.0)
    reference.add_argument("--contact-line-max-age-ms", type=float, default=200.0)
    reference.add_argument("--contact-line-missing-timeout-s", type=float, default=1.0)
    reference.add_argument("--contact-line-reference-stable-s", type=float, default=1.0)
    reference.add_argument("--contact-line-reference-timeout-s", type=float, default=15.0)
    reference.add_argument("--contact-line-reference-position-mm", type=float, default=0.1)
    reference.add_argument("--contact-line-reference-angle-deg", type=float, default=0.1)
    reference.add_argument("--contact-line-reference-force-n", type=float, default=0.2)

    motion = parser.add_argument_group("Paper-style active probing")
    motion.add_argument("--contact-line-probe-cycles", type=float, default=1.25)
    motion.add_argument(
        "--contact-line-ramp-s", type=float, default=12.0,
        help="Maximum-duration passive push-down ramp (1200 frames / 100 Hz in the original project).",
    )
    motion.add_argument(
        "--contact-line-rock-tilt-s", type=float, default=5.0,
        help="Pre-rock roll ramp (500 frames / 100 Hz in the original project).",
    )
    motion.add_argument("--contact-line-rock-period-s", type=float, default=20.0)
    motion.add_argument("--contact-line-deformation-roll-deg", type=float, default=1.0)
    motion.add_argument("--contact-line-deformation-pitch-deg", type=float, default=0.6)
    motion.add_argument(
        "--contact-line-normal-deformation-mm", type=float, default=0.3,
        help="Positive deformation follows --contact-line-plane-normal (TCP local +z by default).",
    )
    motion.add_argument("--contact-line-translation-gain", type=float, default=1.0)
    motion.add_argument("--contact-line-rotation-gain", type=float, default=1.0)
    motion.add_argument("--contact-line-max-speed-mm-s", type=float, default=1.0)
    motion.add_argument("--contact-line-max-speed-deg-s", type=float, default=0.5)
    motion.add_argument("--contact-line-max-accel-mm-s2", type=float, default=2.0)
    motion.add_argument("--contact-line-max-accel-deg-s2", type=float, default=1.0)
    motion.add_argument("--contact-line-max-step-mm", type=float, default=0.5)
    motion.add_argument("--contact-line-max-step-deg", type=float, default=0.5)
    motion.add_argument("--contact-line-max-travel-mm", type=float, default=12.0)
    motion.add_argument("--contact-line-max-normal-travel-mm", type=float, default=1.0)
    motion.add_argument("--contact-line-max-rotation-deg", type=float, default=4.0)

    estimator = parser.add_argument_group("Sliding-window contact-line estimator")
    estimator.add_argument("--contact-line-window", type=int, default=80)
    estimator.add_argument("--contact-line-min-samples", type=int, default=12)
    estimator.add_argument("--contact-line-factor-graph-every", type=int, default=5)
    estimator.add_argument("--contact-line-factor-graph-max-nfev", type=int, default=30)
    estimator.add_argument("--contact-line-position-sigma-mm", type=float, default=0.5)
    estimator.add_argument("--contact-line-rotation-sigma-deg", type=float, default=0.5)
    estimator.add_argument("--contact-line-contact-sigma-mm", type=float, default=0.2)
    estimator.add_argument("--contact-line-direction-sigma-deg", type=float, default=0.2)
    estimator.add_argument("--contact-line-solve-budget-s", type=float, default=0.2)
    estimator.add_argument("--contact-line-min-normal-span-deg", type=float, default=1.0)
    estimator.add_argument("--contact-line-min-observability", type=float, default=8.0)
    estimator.add_argument("--contact-line-max-plane-rms-mm", type=float, default=0.5)
    estimator.add_argument("--contact-line-max-angle-rms-deg", type=float, default=0.5)
    estimator.add_argument("--contact-line-max-line-distance-mm", type=float, default=30.0)
    estimator.add_argument("--contact-line-line-stable-s", type=float, default=2.0)
    estimator.add_argument("--contact-line-line-max-age-s", type=float, default=0.5)
    estimator.add_argument("--contact-line-follow-s", type=float, default=20.0)
    estimator.add_argument(
        "--contact-line-release-s", type=float, default=5.0,
        help="Rock-release duration (500 frames / 100 Hz in the original project).",
    )
    estimator.add_argument("--contact-line-release-timeout-s", type=float, default=8.0)
    estimator.add_argument(
        "--follow-estimated-line", action=argparse.BooleanOptionalAction,
        default=False,
        help="After estimation, pivot about the estimated line. Off by default for pure inference.",
    )
    estimator.add_argument(
        "--contact-line-output", type=Path,
        default=Path("tactile_policy/logs/contact_line_probe_focused.csv"),
    )

    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the final RUN confirmation.")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()

    # Fixed internal configuration: this file has no task mode or probe
    # strategy selector and never performs automatic grasp/transport/descent.
    args.mode = "contact-line-probe"
    args.contact_line_strategy = "feedback"
    args.contact_line_estimator = "surface"
    args.contact_line_auto_prepare = False
    args.preparation_checkpoint = None
    args.contact_line_no_follow = not args.follow_estimated_line
    args.use_force_safety = False
    args.force_bias = np.zeros(6, dtype=np.float64)

    # Keep one contact-acquisition definition across the preparation descent
    # and the synchronized active probe.  An explicit descent value remains
    # available for reproducing an older calibrated run.
    if args.descent_contact_change_n is None:
        args.descent_contact_change_n = args.contact_line_contact_change_n

    axis_norm = float(np.linalg.norm(args.transport_axis_base))
    if not np.isfinite(axis_norm) or axis_norm < 1e-9:
        parser.error("--transport-axis-base must be finite and nonzero")
    args.transport_axis_base /= axis_norm
    if args.prepare_probe:
        if args.contact_tcp_pose is None:
            parser.error("--contact-tcp-pose is required with --prepare-probe")
        # A real baseline is measured after high-position settling and replaces
        # this placeholder before run_feedback_probe is entered.
        if (args.contact_line_free_grasp_forces_n is None
                and not args.contact_line_contact_established):
            args.contact_line_free_grasp_forces_n = np.zeros(2)
    elif (args.contact_line_free_grasp_forces_n is None
          and not args.contact_line_contact_established):
        parser.error(
            "--no-prepare-probe requires --contact-line-contact-established "
            "or --contact-line-free-grasp-forces-n LEFT RIGHT"
        )
    for name in (
        "transport_height_mm", "approach_height_mm", "move_duration_s",
        "descent_speed_mm_s", "descent_contact_window_mm",
        "descent_contact_change_n", "gripper_force", "gripper_settle_s",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.descent_contact_window_mm > args.approach_height_mm:
        parser.error("contact window cannot exceed approach height")
    if args.descent_contact_frames < 1 or args.baseline_samples < 1:
        parser.error("descent contact frames and baseline samples must be positive")

    if not 0 < args.tactile_sync_max_skew_ms <= 150:
        parser.error("--tactile-sync-max-skew-ms must be in (0,150]")
    if args.xense_force_height <= 0 or args.xense_force_width <= 0:
        parser.error("XENSE force-map dimensions must be positive")
    if args.min_tactile_peak < 0:
        parser.error("--min-tactile-peak must be nonnegative")
    if args.contact_line_window < 6:
        parser.error("--contact-line-window must be at least 6")
    if not 6 <= args.contact_line_min_samples <= args.contact_line_window:
        parser.error("--contact-line-min-samples must be within [6, window]")
    if args.contact_line_probe_cycles < 0:
        parser.error("--contact-line-probe-cycles must be positive")
    validate_probe_arguments(parser, args)
    return args


def make_runtime_api() -> SimpleNamespace:
    """Expose only the hardware/model operations required by the probe loop."""
    return SimpleNamespace(
        load_model=load_model,
        prepare_contact_line_probe=None,
        paired_tactile_input=paired_tactile_input,
        predict_pair_motion=predict_pair_motion,
        inferred_object_frame=inferred_object_frame,
        format_tactile_frame=format_tactile_frame,
        check_safety=check_safety,
        command_pose=command_pose,
        SafetyStop=SafetyStop,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, normalization = load_model(args.checkpoint, device)
    if not is_pairwise_model(model):
        raise ValueError(
            "Contact-line inference requires a paired relative-SE(3) checkpoint "
            "(paired_force or paired_depth)"
        )

    robot = ZMQClientRobot(
        port=args.robot_port, host=args.host, timeout_ms=args.robot_timeout_ms
    )
    tactile = TactileSource(args)
    try:
        if robot.num_dofs() != 7:
            raise RuntimeError("expected a 7-DoF UR5 + RG2 robot server")
        tcp = np.asarray(robot.get_tcp_pose(), dtype=np.float64)
        left, right = tactile.read()
        check_safety(args, None, left, right)
        peaks = tactile_peaks(left, right)
        if not args.prepare_probe and min(peaks) < args.min_tactile_peak:
            raise SafetyStop(
                f"bilateral grasp contact is required; tactile peaks={peaks}"
            )
        print("Focused contact-line probe")
        print("  model:", model.tactile_input_kind, args.checkpoint.resolve())
        print("  current TCP [m,rotvec]:", np.round(tcp, 6))
        print("  tactile peaks:", np.round(peaks, 4))
        print(
            "  preparation:",
            ("open -> manual load -> close -> lift -> transfer -> tactile descent"
             if args.prepare_probe else
             "disabled; nut must already be grasped and lightly touching bolt"),
        )
        print(
            "  sequence: stable reference -> push-down -> rock-tilt -> "
            "1.25-cycle rocking -> rock-release -> line optimization"
        )
        feedback_point = (
            np.asarray(args.standard_tcp_to_nut_center, dtype=np.float64) * 1000.0
            if args.contact_line_feedback_point_mm is None
            else np.asarray(args.contact_line_feedback_point_mm, dtype=np.float64)
        )
        print("  calibrated nut-centre pivot in reference TCP (mm):", np.round(feedback_point, 3))
        print(
            "  contact thresholds: descent={:.3f}N, acquire={:.3f}N, "
            "release={:.3f}N".format(
                args.descent_contact_change_n,
                args.contact_line_contact_change_n,
                args.contact_line_contact_release_n,
            )
        )

        if not args.execute:
            print("Observation-only check complete. Add --execute to start active rocking.")
            return
        if not args.yes:
            answer = input(
                "Real UR5 active rocking will start. Keep E-stop ready. Type RUN: "
            ).strip()
            if answer != "RUN":
                raise SystemExit("Cancelled without sending a motion command")

        if args.prepare_probe:
            args.contact_line_free_grasp_forces_n = prepare_probe(
                args, robot, tactile
            )
            # Use the measured free-grasp baseline so external contact remains
            # independently checked during active rocking.
            args.contact_line_contact_established = False

        run_feedback_probe(
            args, robot, tactile, None, model, normalization, device,
            make_runtime_api(),
        )
    except KeyboardInterrupt:
        print("Interrupted; no further motion commands will be sent.")
    except SafetyStop as exc:
        print(f"SAFETY STOP: {exc}. No further motion commands will be sent.")
        raise
    finally:
        tactile.close()
        robot.close()


if __name__ == "__main__":
    main()
