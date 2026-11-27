"""Prepare S600 test inputs, decode board outputs, or visualize predictions."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

from bevfusion_float_demo import DEFAULT_ORDER, ROOT, TASK, demo_path, prepare_frame

DEFAULT_HBM = ROOT / "outputs" / "bevfusion_s600_100" / "model.hbm"
DEFAULT_BUNDLE = ROOT / "outputs" / "bevfusion_board_demo"
SUPPORTED = {"int8", "uint8", "int16", "int32", "float32"}
OUTPUT_NAMES = ("bev_embed", "outputs_classes", "reference_out", "outputs_coords", "occ_preds")
PLANE_NAMES = {f"img_{i}_{plane}" for i in range(6) for plane in ("y", "uv")}


def spec(variable):
    t = variable.type
    name, shape, dtype = variable.name, list(t.shape), t.np_dtype.name
    if not re.fullmatch(r"[A-Za-z0-9_]+", name) or dtype not in SUPPORTED or not shape or min(shape) < 1:
        raise ValueError(f"Unsupported tensor: {name}, {shape}, {dtype}")
    q = t.quant_info
    return dict(name=name, shape=shape, dtype=dtype, quant=None if q is None else {
        "axis": int(q.channel_axis), "scales": list(map(float, q.scales)),
        "zero_points": list(map(int, q.zero_points)),
    })


def qparams(s):
    import numpy as np

    q = s["quant"]
    scales = np.asarray(q["scales"], dtype=np.float64)
    zero = np.asarray(q["zero_points"], dtype=np.float64)
    if not scales.size or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError(f"Invalid quantization scale for {s['name']}")
    if zero.size not in (1, scales.size):
        raise ValueError(f"Invalid zero points for {s['name']}")
    if scales.size == 1:
        return scales[0], zero[0]
    shape, axis = s["shape"], q["axis"]
    if not -len(shape) <= axis < len(shape) or shape[axis] != scales.size:
        raise ValueError(f"Invalid channel axis for {s['name']}")
    broadcast = [1] * len(shape)
    broadcast[axis] = scales.size
    return scales.reshape(broadcast), zero.reshape(broadcast) if zero.size > 1 else zero[0]


def input_array(values, s):
    import numpy as np

    dtype = np.dtype(s["dtype"]).newbyteorder("<")
    if list(values.shape) != s["shape"] or not np.isfinite(values).all():
        raise ValueError(f"{s['name']}: invalid shape or non-finite values: {values.shape}")
    if s["quant"] is not None:
        scale, zero = qparams(s)
        limits = np.iinfo(dtype)
        values = np.clip(np.rint(values.astype(np.float64) / scale) + zero, limits.min, limits.max)
    elif values.dtype.name != dtype.name:
        raise ValueError(f"{s['name']}: no quantization metadata for {values.dtype} -> {dtype.name}")
    return np.ascontiguousarray(values, dtype=dtype)


def read_output(directory, s):
    import numpy as np

    dtype = np.dtype(s["dtype"]).newbyteorder("<")
    path = directory / (s["name"] + ".bin")
    array = np.fromfile(path, dtype=dtype)
    if path.stat().st_size != array.size * dtype.itemsize or array.size != np.prod(s["shape"]):
        raise ValueError(f"Wrong output size: {path}")
    array = array.reshape(s["shape"])
    if s["quant"] is not None:
        scale, zero = qparams(s)
        array = (array.astype(np.float64) - zero) * scale
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite board output: {path}")
    return array


def load_heal(root):
    config = root / "configs" / TASK
    if not (config / "model.py").is_file():
        raise FileNotFoundError(f"HEAL BEVFusion config not found under {root}")
    sys.path[:0] = [str(root), str(config)]
    import heal  # noqa: F401 - register HEAL components


def prepare(args):
    import torch
    from hbdk4.compiler.hbm import Hbm
    from horizon_plugin_pytorch.march import set_march

    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    hbm = Hbm(str(args.model))
    if hbm.march_name != "nash-p" or len(hbm.graphs) != 1:
        raise ValueError("Expected a single-graph nash-p HBM for RDK S600")
    graph = hbm.graphs[0]
    inputs = [spec(v) for v in graph.flatten_inputs]
    outputs = [spec(v) for v in graph.flatten_outputs]
    if [s["name"] for s in outputs] != list(OUTPUT_NAMES):
        raise ValueError("Expected five named HEAL BEVFusion outputs")

    load_heal(args.heal_root)
    from common import input_size
    from hat_engine.registry import build_from_registry
    from heal.models.structures.bevfusion import BevFusionHbirInfer
    from model import model as model_cfg

    torch.set_num_threads(4)
    set_march("nash-p")
    data, rgb, _ = prepare_frame(args.samples, DEFAULT_ORDER, args.point_dim, args.fifth_column, input_size)
    names = {s["name"] for s in inputs}
    lidar_and_grids = {"features", "coors", "queries_rebatch_grid", "restore_bev_grid",
                       "reference_points_rebatch", "bev_pillar_counts"}
    nv12 = names == lidar_and_grids | PLANE_NAMES
    if not nv12 and names != lidar_and_grids | {"img"}:
        raise ValueError(f"Unexpected HBM inputs: {sorted(names)}")
    if len(names) != len(inputs):
        raise ValueError("HBM has duplicate input names")
    if nv12:
        data["img"] = rgb
    # Only the HEAL LiDAR and geometry preprocessors run here, not the network.
    with torch.inference_mode():
        prep = BevFusionHbirInfer(None, build_from_registry(model_cfg).eval(), use_nv12=nv12)
        values = prep.process_input(data)
    arrays = {s["name"]: input_array(values[s["name"]].numpy(), s) for s in inputs}

    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / "inputs"
    raw.mkdir()
    for name, array in arrays.items():
        array.tofile(raw / f"{name}.bin")
    rows = [f"{kind}\t{s['name']}\t{s['dtype']}\t{','.join(map(str, s['shape']))}"
            for kind, specs in (("input", inputs), ("output", outputs)) for s in specs]
    (raw / "tensors.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    packaged_hbm = args.output / "model.hbm"
    shutil.copyfile(args.model, packaged_hbm)
    for name in ("CMakeLists.txt", "board_demo.cpp"):
        shutil.copyfile(ROOT / "board" / name, args.output / name)
    with packaged_hbm.open("rb") as file:
        digest = hashlib.sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = dict(march=hbm.march_name, graph=graph.name, model_sha256=digest.hexdigest(),
                    camera_input="nv12" if nv12 else "ddr", camera_order=DEFAULT_ORDER,
                    samples=str(args.samples), point_dim=args.point_dim, fifth_column=args.fifth_column,
                    point_count=len(data["points"][0]), inputs=inputs, outputs=outputs)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(inputs)} inputs for {graph.name} ({manifest['camera_input']}) in {args.output}")


def decode(args):
    import numpy as np
    import torch
    from horizon_plugin_pytorch.march import set_march

    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["march"] != "nash-p" or [s["name"] for s in manifest["outputs"]] != list(OUTPUT_NAMES):
        raise ValueError("Not a BEVFusion S600 demo manifest")
    bev, cls, ref, coords, occ = (read_output(args.board_output, s) for s in manifest["outputs"])
    load_heal(args.heal_root)
    from hat_engine.registry import build_from_registry
    from model import bev_head

    torch.set_num_threads(4)
    set_march("nash-p")
    with torch.inference_mode():
        decoder = build_from_registry(bev_head).eval()
        preds = decoder.get_outputs([torch.from_numpy(cls)], [torch.from_numpy(ref)],
                                    [torch.from_numpy(coords)], torch.from_numpy(bev))
        boxes = decoder.post_process(preds)[0].numpy()
    if not np.isfinite(boxes).all():
        raise RuntimeError("Decoded boxes contain non-finite values")
    boxes = boxes[boxes[:, 9] >= args.score_threshold]
    occupancy = occ.argmax(-1)[0].astype(np.uint8)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "predictions.npz", boxes_3d=boxes[:, :9], scores=boxes[:, 9],
                        labels=boxes[:, 10].astype(np.int64), occupancy=occupancy)
    print(f"Decoded {len(boxes)} detections and occupancy {occupancy.shape} to {args.output}")


def visualize(args):
    import numpy as np
    import torch
    import matplotlib

    matplotlib.use("Agg")
    with np.load(args.predictions, allow_pickle=False) as predictions:
        boxes = predictions["boxes_3d"]
        scores = predictions["scores"]
        labels = predictions["labels"]
        occupancy = predictions["occupancy"]
    if (boxes.ndim != 2 or boxes.shape[1] != 9 or scores.shape != (len(boxes),)
            or labels.shape != (len(boxes),)):
        raise ValueError("Expected boxes_3d (N, 9), scores (N,), and labels (N,)")
    detections = np.column_stack((boxes, scores, labels))
    if not np.isfinite(detections).all():
        raise ValueError("Predictions contain non-finite detections")
    if (occupancy.shape != (200, 200, 16) or not np.issubdtype(occupancy.dtype, np.integer)
            or occupancy.min() < 0 or occupancy.max() > 17):
        raise ValueError("Expected a (200, 200, 16) occupancy grid with integer labels 0–17")

    manifest = {}
    manifest_path = args.bundle / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = args.samples or demo_path(manifest.get("samples", ROOT / "data" / "samples"))
    order = manifest.get("camera_order", DEFAULT_ORDER)
    point_dim = args.point_dim or manifest.get("point_dim", 5)
    load_heal(args.heal_root)
    from common import bev_size, input_size
    from heal.visualize.nuscenes import NuscenesMultitaskViz
    from heal.visualize.occ import OccViz

    torch.set_num_threads(4)
    data, rgb, lidar2img = prepare_frame(samples, order, point_dim, "ring", input_size)
    viz = NuscenesMultitaskViz(occ_viz=OccViz(vis_bev_2d=True), bev_size=bev_size,
                              score_thresh=args.score_threshold)
    output = args.output or args.predictions.parent
    viz([{"name": name, "img": image} for name, image in zip(order, rgb)],
        data["points"], {"bev_det": [detections], "occ_det": occupancy.astype(np.uint8)},
        {"lidar2img": lidar2img}, save_path=str(output), stage="board")
    print(f"Saved preview to {output / viz.result_filename(0, 'board')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare", help="Stage a one-frame bundle for the board")
    prep.add_argument("--model", type=demo_path, default=DEFAULT_HBM)
    prep.add_argument("--samples", type=demo_path, default=ROOT / "data" / "samples")
    prep.add_argument("--output", type=demo_path, default=DEFAULT_BUNDLE)
    prep.add_argument("--point-dim", type=int, choices=(4, 5), default=5)
    prep.add_argument("--fifth-column", choices=("ring", "time-lag"), default="ring")
    prep.set_defaults(action=prepare)
    dec = modes.add_parser("decode", help="Decode the returned board outputs on the host")
    dec.add_argument("--bundle", type=demo_path, default=DEFAULT_BUNDLE)
    dec.add_argument("--board-output", type=demo_path, required=True)
    dec.add_argument("--output", type=demo_path, default=ROOT / "outputs" / "bevfusion_board_predictions")
    dec.add_argument("--score-threshold", type=float, default=0.3)
    dec.set_defaults(action=decode)
    vis = modes.add_parser("visualize", help="Render a decoded NPZ without rerunning inference")
    vis.add_argument("--predictions", type=demo_path,
                     default=ROOT / "outputs" / "bevfusion_board_predictions" / "predictions.npz")
    vis.add_argument("--bundle", type=demo_path, default=DEFAULT_BUNDLE,
                     help="Bundle providing the original sample path and camera order")
    vis.add_argument("--samples", type=demo_path, help="Override the location of the original demo frame")
    vis.add_argument("--point-dim", type=int, choices=(4, 5), help="Override the original point-file column count")
    vis.add_argument("--output", type=demo_path, help="Output directory (default: next to the NPZ)")
    vis.add_argument("--score-threshold", type=float, default=0.3)
    vis.set_defaults(action=visualize)
    for mode in (prep, dec, vis):
        mode.add_argument("--heal-root", type=demo_path, default=ROOT / "Heal")
    args = parser.parse_args()
    if args.mode in ("decode", "visualize") and not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be in [0, 1]")
    args.action(args)


if __name__ == "__main__":
    main()
