"""Run HEAL's float BEVFusion model on one six-camera + LiDAR frame."""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

TASK = "bevfusion_pointpillar_henet_multisensor_multitask"
ROOT = Path(__file__).resolve().parents[1]
# Sample numbering is front, front-right, front-left, back, back-left, back-right.
# HEAL reads views left/front/right, then back-left/back/back-right.
DEFAULT_ORDER = ("camera_3", "camera_1", "camera_2", "camera_5", "camera_4", "camera_6")
BOX_COLUMNS = ("cx", "cy", "cz", "w", "l", "h", "yaw", "vx", "vy")


def demo_path(value):
    """Accept repo-relative paths from WSL or drive-letter paths from Windows."""
    value = str(value).replace("\\", "/")
    if len(value) >= 3 and value[1:3] == ":/" and sys.platform.startswith("linux"):
        value = f"/mnt/{value[0].lower()}/{value[3:]}"
    return Path(value).expanduser().resolve()


def prepare_frame(samples, order, point_dim, fifth_column, image_size):
    import numpy as np
    import torch
    from horizon_plugin_pytorch.nn.functional import bgr_to_yuv444
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    from torchvision.transforms._functional_tensor import resize

    calibration = json.loads((samples / "calibration.json").read_text(encoding="utf-8"))
    names = calibration["camera_names"]
    intrinsics = np.asarray(calibration["camera_intrinsics"], dtype=np.float64)
    camera2lidar = np.asarray(calibration["camera2lidar"], dtype=np.float64)
    if (
        len(names) != 6
        or len(set(names)) != 6
        or set(order) != set(names)
        or intrinsics.shape != (6, 3, 3)
        or camera2lidar.shape != (6, 4, 4)
        or not np.isfinite(intrinsics).all()
        or not np.isfinite(camera2lidar).all()
    ):
        raise ValueError(
            "calibration.json must contain six matched cameras, finite (6,3,3) K "
            "and (6,4,4) camera2lidar matrices"
        )

    height, width = image_size
    images, projections = [], []
    for name in order:
        index = names.index(name)
        with Image.open(samples / f"{name}.png") as source:
            original_width, original_height = source.size
            rgb = pil_to_tensor(source.convert("RGB"))
        # Match HEAL's local inference: resize uint8 CHW before YUV and normalization.
        images.append(resize(rgb, [height, width]))
        projection = np.eye(4)
        projection[:3, :3] = intrinsics[index]
        scale = np.diag([width / original_width, height / original_height, 1, 1])
        projections.append(scale @ projection @ np.linalg.inv(camera2lidar[index]))

    rgb = torch.stack(images)
    img = (bgr_to_yuv444(rgb, True).float() - 128.0) / 128.0
    lidar2img = np.asarray(projections, dtype=np.float32)
    path = samples / "points.bin"
    size = path.stat().st_size
    if not size or size % (4 * point_dim):
        raise ValueError(f"{path}: invalid size for little-endian float32 points with {point_dim} columns")
    raw = np.fromfile(path, dtype="<f4").reshape(-1, point_dim)
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: point cloud contains NaN or Inf")
    points = np.zeros((len(raw), 5), dtype=np.float32)
    points[:, :4] = raw[:, :4]
    # nuScenes' raw fifth field is ring index. HEAL expects time lag, zero for a single sweep.
    if point_dim == 5 and fifth_column == "time-lag":
        points[:, 4] = raw[:, 4]

    return {
        "img": img.contiguous(),
        "points": [torch.from_numpy(points)],
        # This config projects from LiDAR coordinates, so no synthetic ego pose is needed.
        "seq_meta": [{"lidar2img": lidar2img[None], "meta": [{"scene": str(samples)}]}],
    }, rgb, lidar2img


def load_float_weights(model, checkpoint):
    import numpy as np
    import torch
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

    # The official checkpoint includes NumPy scalar training metadata.
    # Keep weights_only=True rather than enabling general pickle loading.
    scalar = np._core.multiarray.scalar if hasattr(np, "_core") else np.core.multiarray.scalar
    globals_ = [(scalar, "numpy.core.multiarray.scalar"), np.dtype]
    globals_ += [type(np.dtype(t)) for t in (np.float32, np.float64, np.int32, np.int64)]
    with torch.serialization.safe_globals(globals_):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload.get("model", payload))
    consume_prefix_in_state_dict_if_present(state, "module.")
    model.load_state_dict(state, strict=True)


def run(args):
    import numpy as np
    import torch
    from horizon_plugin_pytorch.march import set_march

    sys.path[:0] = [str(args.heal_root), str(args.heal_root / "configs" / TASK)]
    os.environ["HAT_TRAINING_STEP"] = "float"
    if args.visualize:
        os.environ.setdefault("MPLBACKEND", "Agg")

    import heal
    from common import bev_size, class_names, input_size, march, occ3d_seg_class
    from hat_engine.registry import build_from_registry
    from model import model as model_config

    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("This demo needs a CUDA GPU visible to PyTorch in WSL.")
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    set_march(march)

    print(f"HEAL {heal.__version__} | PyTorch {torch.__version__} | {torch.cuda.get_device_name(device)}", flush=True)
    model = build_from_registry(model_config)
    load_float_weights(model, args.checkpoint)
    model = model.eval().to(device=device, dtype=torch.float32)

    data, rgb, lidar2img = prepare_frame(
        args.samples, args.camera_order, args.point_dim, args.fifth_column, input_size
    )
    points_cpu = data["points"][0]
    data["img"] = data["img"].to(device)
    data["points"] = [points_cpu.to(device)]
    print(f"Input: {tuple(data['img'].shape)}, {tuple(points_cpu.shape)}", flush=True)
    print(f"FP32 forward: {args.warmup} warm-up, {args.runs} timed run(s)", flush=True)
    latencies_ms = []
    with torch.inference_mode():
        torch.cuda.synchronize(device)
        for _ in range(args.warmup):
            model(data)
        if args.warmup:
            torch.cuda.synchronize(device)
        for _ in range(args.runs):
            start = perf_counter()
            det, occ = model(data)
            torch.cuda.synchronize(device)
            latencies_ms.append((perf_counter() - start) * 1000)

    avg_ms = float(np.mean(latencies_ms))
    p50_ms, p95_ms = np.percentile(latencies_ms, [50, 95])
    timing = {
        "runs": args.runs,
        "warmup_runs": args.warmup,
        "per_run_ms": [round(ms, 3) for ms in latencies_ms],
        "mean_ms": round(avg_ms, 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
        "p50_ms": round(float(p50_ms), 3),
        "p95_ms": round(float(p95_ms), 3),
    }
    for index, ms in enumerate(latencies_ms, 1):
        print(f"  Run {index}/{args.runs}: {ms:.2f} ms")
    print(
        f"FP32 forward: mean {avg_ms:.2f} ms | P50 {p50_ms:.2f} ms | "
        f"P95 {p95_ms:.2f} ms | min/max {min(latencies_ms):.2f}/{max(latencies_ms):.2f} ms",
        flush=True,
    )

    # Only save the last timed run's predictions for this repeated frame.
    det = det[0].cpu().numpy()
    occupancy = occ[0][0].cpu().numpy()
    if det.ndim != 2 or det.shape[1] != 11 or not np.isfinite(det).all():
        raise RuntimeError(f"Invalid HEAL detection output: {det.shape}")
    if occupancy.shape != (200, 200, 16) or occupancy.min() < 0 or occupancy.max() > len(occ3d_seg_class):
        raise RuntimeError(f"Invalid HEAL occupancy output: {occupancy.shape}")
    occupancy = occupancy.astype(np.uint8)
    kept = det[det[:, 9] >= args.score_threshold]
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "predictions.npz",
        boxes_3d=kept[:, :9], scores=kept[:, 9], labels=kept[:, 10].astype(np.int64),
        occupancy=occupancy,
    )
    if args.visualize:
        from heal.visualize.nuscenes import NuscenesMultitaskViz
        from heal.visualize.occ import OccViz

        viz = NuscenesMultitaskViz(
            occ_viz=OccViz(vcs_range=(-40, -40, -1, 40, 40, 5.4), vis_bev_2d=True),
            bev_size=bev_size, score_thresh=args.score_threshold,
        )
        viz(
            [{"name": name, "img": img} for name, img in zip(args.camera_order, rgb)],
            [points_cpu], {"bev_det": [det], "occ_det": occupancy},
            {"lidar2img": lidar2img}, save_path=str(args.output), stage="float",
        )
    summary = {
        "checkpoint": str(args.checkpoint), "samples": str(args.samples),
        "device": str(device), "precision": "float32",
        "forward_seconds": round(avg_ms / 1000, 4),  # Mean of timed runs; legacy field.
        "forward_timing": timing,
        "camera_order": args.camera_order, "point_count": len(points_cpu),
        "point_features": ["x", "y", "z", "intensity", "time_lag"],
        "score_threshold": args.score_threshold, "detections": len(kept),
        "detections_before_threshold": len(det), "box_columns": BOX_COLUMNS,
        "box_coordinate_frame": "lidar", "detection_class_names": class_names,
        "occupancy_shape": list(occupancy.shape), "occupancy_axes": ["x", "y", "z"],
        "occupancy_class_names": [*occ3d_seg_class, "free"],
        "occupancy_classes_present": np.unique(occupancy).tolist(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{len(kept)} detections at score >= {args.score_threshold:g}; occupancy {occupancy.shape}")
    print(f"Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=demo_path, default=ROOT / "data" / "samples")
    parser.add_argument("--checkpoint", type=demo_path, default=ROOT / TASK / "float-checkpoint-last.pth.tar")
    parser.add_argument("--heal-root", type=demo_path, default=ROOT / "Heal")
    parser.add_argument("--output", type=demo_path, default=ROOT / "outputs" / "bevfusion_float")
    parser.add_argument("--device", default="cuda:0", help="PyTorch CUDA device, e.g. cuda:0")
    parser.add_argument("--point-dim", type=int, choices=(4, 5), default=5)
    parser.add_argument("--fifth-column", choices=("ring", "time-lag"), default="ring")
    parser.add_argument("--camera-order", nargs=6, default=DEFAULT_ORDER)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--runs", type=int, default=1, help="Number of timed forwards on the same frame (default: 1)")
    parser.add_argument("--warmup", type=int, default=0, help="Number of untimed warm-up forwards (default: 0)")
    parser.add_argument("--visualize", action="store_true", help="Save HEAL's camera and BEV preview PNG")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")
    if args.device != "cuda" and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        parser.error("--device must be cuda or cuda:<index>")
    if len(set(args.camera_order)) != 6:
        parser.error("--camera-order requires six unique camera IDs")
    if not (args.heal_root / "configs" / TASK / "model.py").is_file():
        parser.error(f"HEAL model config not found under {args.heal_root}")
    if not args.checkpoint.is_file():
        parser.error(f"Float checkpoint not found: {args.checkpoint}")
    with args.checkpoint.open("rb") as file:
        if file.read(42).startswith(b"version https://git-lfs.github.com/spec/v1"):
            parser.error(f"{args.checkpoint} is a Git LFS pointer, not model weights")
    try:
        run(args)
    except ModuleNotFoundError as exc:
        if exc.name is None:
            raise
        parser.exit(
            1,
            f"Missing Python module {exc.name!r} in {sys.executable}. "
            "Use a Conda environment with this HEAL checkout's PyTorch/CUDA "
            "and Horizon dependencies.\n",
        )


if __name__ == "__main__":
    main()
