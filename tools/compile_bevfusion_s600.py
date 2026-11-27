"""Calibrate HEAL BEVFusion and compile for RDK S600 (nash-p)."""

import argparse
from collections import Counter
import gc
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import monotonic

from bevfusion_float_demo import ROOT, TASK, DEFAULT_ORDER, demo_path, load_float_weights

MARCH = "nash-p"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=demo_path, default=ROOT / "data" / "nuscenes")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes metadata directory name")
    parser.add_argument("--calibration-frames", type=int, default=100, help="Number of distinct keyframes (default: 100)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=demo_path, default=ROOT / TASK / "float-checkpoint-last.pth.tar")
    parser.add_argument("--heal-root", type=demo_path, default=ROOT / "Heal")
    parser.add_argument("--output", type=demo_path, default=ROOT / "outputs" / "bevfusion_s600")
    parser.add_argument("--format", choices=("hbm", "both", "hbir"), default="hbm",
                        help="HBM only (default), HBM and HBIR, or HBIR only")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device for calibration (default: cuda:0)")
    parser.add_argument("--opt", type=int, choices=(0, 1, 2), default=1, help="HBM optimization level")
    parser.add_argument("--jobs", type=int, default=4, help="HBDK compilation workers")
    parser.add_argument("--camera-input", choices=("nv12", "ddr"), default="nv12",
                        help="HBM camera interface: HEAL NV12 pyramid (default) or normalized YUV444 tensor")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-data", action="store_true", help="List the selected keyframes without loading the model")
    mode.add_argument("--from-hbir", type=demo_path, metavar="QAT_BC",
                      help="Compile a previously exported qat.bc without recalibration")
    mode.add_argument("--samples", type=demo_path, metavar="FRAME_DIR",
                      help="Calibrate on the single demo-format frame for a smoke test only")
    args = parser.parse_args()
    if args.calibration_frames < 1 or args.jobs < 1:
        parser.error("--calibration-frames and --jobs must be positive")
    if args.from_hbir and args.format != "hbm":
        parser.error("--from-hbir is for --format hbm; the input HBIR is not modified")
    if not args.check_data:
        if not (args.heal_root / "configs" / TASK / "model.py").is_file():
            parser.error(f"HEAL model config not found under {args.heal_root}")
        if args.from_hbir:
            if not args.from_hbir.is_file():
                parser.error(f"HBIR not found: {args.from_hbir}")
        elif not args.checkpoint.is_file():
            parser.error(f"Float checkpoint not found: {args.checkpoint}")
        else:
            with args.checkpoint.open("rb") as file:
                if file.read(42).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    parser.error(f"{args.checkpoint} is a Git LFS pointer, not model weights")
    return args


def select_frames(args):
    from nuscenes_calibration import index_calibration_frames

    frames = index_calibration_frames(
        args.dataset, version=args.version, limit=args.calibration_frames, seed=args.seed
    )
    if len(frames) < args.calibration_frames:
        raise ValueError(f"Requested {args.calibration_frames} frames, but found {len(frames)} complete keyframes. "
                         "Lower --calibration-frames explicitly to use fewer.")
    counts = Counter(frame.scene for frame in frames)
    print(f"Selected {len(frames)} keyframes across {len(counts)} scenes", flush=True)
    return frames


def calibrate_and_export(args, source, report, stage):
    import torch
    import horizon_plugin_pytorch as horizon
    from horizon_plugin_pytorch.quantization import FakeQuantState, set_fake_quantize
    from horizon_plugin_pytorch.quantization.hbdk4 import export
    from horizon_plugin_pytorch.utils.global_quant_round_mode import QuantRoundMode

    import heal  # register official HEAL model types
    from common import convert_mode, input_size, qat_mode
    from hat_engine.models.model_convert.converters import FixWeightQScale, Float2Calibration, RepModel2Deploy
    from hat_engine.registry import build_from_registry
    from heal.models.structures.bevfusion import BevFusion
    from model import deploy_inputs, model as model_cfg, output_names
    from qconfig import get_custom_qconfig

    class FlatBevFusion(BevFusion):
        def forward(self, data):
            detection, occupancy = super().forward(data)
            return (*detection, *occupancy)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")
    torch.set_num_threads(min(os.cpu_count() or 1, 4))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    horizon.march.set_march(MARCH)
    horizon.qat_mode.set_qat_mode(qat_mode)
    QuantRoundMode.set(QuantRoundMode.HALF_TO_EVEN)

    stage("select_data")
    if args.samples:
        from bevfusion_float_demo import prepare_frame

        print("Warning: calibrating on one demo frame; not an accuracy benchmark.", flush=True)
        frames = [None]
        tokens = [str(args.samples)]

        def load_frame(_):
            return prepare_frame(args.samples, DEFAULT_ORDER, 5, "ring", input_size)[0]
    else:
        from nuscenes_calibration import load_calibration_frame

        frames = select_frames(args)
        tokens = [frame.token for frame in frames]

        def load_frame(frame):
            return load_calibration_frame(frame, image_size=input_size)

    report.update(calibration_tokens=tokens, calibration_frames=len(frames),
                  calibration_scenes=None if args.samples else dict(Counter(frame.scene for frame in frames)))
    stage("load_float")
    original = build_from_registry(model_cfg).eval()
    load_float_weights(original, args.checkpoint)
    original.to(device=device, dtype=torch.float32)
    # Keep HEAL's voxelization and reference-grid calculation outside the exported graph.
    voxelizer = original.lidar_net.pre_process
    view_transform = original.camera_net.view_transformer
    original.lidar_net.pre_process = None
    view_transform.is_compile = True
    det_head, occ_head = original.bev_decoders
    det_head.is_compile = True
    det_head.criterion = None
    det_head.post_process = None
    occ_head.is_compile = True
    occ_head.loss_occ = None
    model = FlatBevFusion(
        lidar_network=original.lidar_net, camera_network=original.camera_net,
        bev_decoders=original.bev_decoders, fuse_module=original.fuse_module,
        bev_h=original.bev_h, bev_w=original.bev_w,
    ).eval()
    del original

    def make_inputs(frame):
        data = load_frame(frame)
        data["img"] = data["img"].to(device)
        data["points"] = [points.to(device) for points in data["points"]]
        with torch.no_grad():
            features, coors = voxelizer(data["points"], True)
            grids = view_transform.get_reference_points_cam(data)
        values = (features, coors.to(torch.int32), data["img"], *grids)
        inputs = dict(zip(deploy_inputs, values))
        for name, tensor in inputs.items():
            expected = deploy_inputs[name]
            if tensor.shape != expected.shape or tensor.dtype != expected.dtype or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid calibration input {name}: {tensor.shape} {tensor.dtype}")
        return inputs

    example = make_inputs(frames[0])
    stage("prepare_calibration")
    qconfig_setter = get_custom_qconfig(True)
    qconfig_dir = args.output / "qconfig"
    qconfig_dir.mkdir()
    qconfig_setter.save_dir = str(qconfig_dir)
    model = RepModel2Deploy()(model)
    prepared = Float2Calibration(
        convert_mode=convert_mode, example_inputs=(example,), qconfig_setter=qconfig_setter,
    )(model)
    del model
    prepared = FixWeightQScale()(prepared)
    stage("calibrate")
    with torch.no_grad():
        for index, (frame, token) in enumerate(zip(frames, tokens), 1):
            tensors = example if index == 1 else make_inputs(frame)
            outputs = prepared(tensors)
            if len(outputs) != len(output_names) or any(not torch.isfinite(tensor).all() for tensor in outputs):
                raise RuntimeError(f"Non-finite or invalid calibration output for {token}")
            if index in (1, len(frames)) or index % 10 == 0:
                print(f"Calibrated {index}/{len(frames)}: {token}", flush=True)
    del outputs, tensors, voxelizer, view_transform, det_head, occ_head
    prepared.eval()
    set_fake_quantize(prepared, FakeQuantState.VALIDATION)

    stage("export_qat_hbir")
    with torch.no_grad():
        module = export(prepared, (example,), name=TASK, input_names=list(deploy_inputs),
                        output_names=output_names, native_pytree=False)
    from hbdk4 import compiler

    func = module.functions[0]
    inputs = [item.name for item in func.flatten_inputs]
    outputs = [item.name for item in func.flatten_outputs]
    if inputs != list(deploy_inputs) or outputs != output_names:
        raise RuntimeError(f"Unexpected HBIR signature: inputs={inputs}, outputs={outputs}")
    compiler.save(module, str(source))
    del module, prepared, example
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report.update(heal_version=heal.__version__, input_names=inputs, output_names=outputs)


def write_infer_hbir(source, output):
    from hbdk4 import compiler

    module = compiler.load(str(source))
    module._integer_conv = True
    quantized = compiler.convert(module, MARCH, enable_vpu=False)
    compiler.save(quantized, str(output))


def compile_hbm(args, source, work, stage):
    from hbdk4 import compiler
    from hbdk4.compiler.hbm import Hbm
    from hat_engine.utils.hbdk4.hbir_compiler import HbirCompiler
    from model import deploy_inputs

    # Reuse HEAL's NV12 graph rewrite, but skip its automatic performance test.
    stage("convert_deploy_hbir")
    hbm_path = work / "model.hbm"
    nv12 = args.camera_input == "nv12"
    helper = HbirCompiler(
        model_path=str(source), out_path=str(hbm_path), march=MARCH,
        opt=args.opt, jobs=args.jobs, max_time_per_fc=0, enable_vpu=nv12,
        input_source=["pyramid" if nv12 and key == "img" else "ddr" for key in deploy_inputs],
    )
    func = helper.model.functions[0]
    if [item.name for item in func.flatten_inputs] != list(deploy_inputs):
        raise ValueError("Expected the seven named HEAL BEVFusion inputs in qat.bc")
    helper.set_input_nodes(helper.model, helper.input_source)
    expected_inputs = 18 if nv12 else 7
    if len(helper.model.functions[0].flatten_inputs) != expected_inputs:
        raise RuntimeError(f"Expected {expected_inputs} board inputs after the {args.camera_input} rewrite")
    helper.model._integer_conv = True
    deployed = compiler.convert(helper.model, MARCH, enable_vpu=nv12)
    if nv12:
        deployed.functions[0].remove_io_op(op_types=["Quantize", "Dequantize"])
    if args.format == "both":
        compiler.save(deployed, str(args.output / "quantized_deploy.bc"))
    del helper
    stage("compile_hbm")
    compiler.compile(deployed, str(hbm_path), march=MARCH, opt=args.opt, jobs=args.jobs, progress_bar=True)
    metadata = Hbm(str(hbm_path))
    if metadata.march_name != MARCH:
        raise RuntimeError(f"Expected {MARCH}, got {metadata.march_name}")
    del metadata
    path = args.output / "model.hbm"
    hbm_path.replace(path)
    return {"path": path.name, "bytes": path.stat().st_size, "march": MARCH,
            "camera_input": args.camera_input, "input_count": expected_inputs, "board_verified": False}


def run(args):
    if args.check_data:
        for frame in select_frames(args):
            print(frame.token, frame.scene)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output}; select a new --output")
    sys.path[:0] = [str(args.heal_root), str(args.heal_root / "configs" / TASK)]
    os.environ["HAT_TRAINING_STEP"] = "calibration"
    # Keep version-dependent compiler side files out of the source tree.
    os.chdir(args.output)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    start = monotonic()
    report = dict(status="running", march=MARCH, output_format=args.format, dataset=str(args.dataset),
                  dataset_version=args.version, seed=args.seed, checkpoint=None if args.from_hbir else str(args.checkpoint),
                  source_hbir=str(args.from_hbir) if args.from_hbir else None, lidar_history_sweeps=0,
                  opt=args.opt, jobs=args.jobs)

    def stage(name):
        report.update(stage=name, elapsed_seconds=round(monotonic() - start, 2))
        (args.output / "compile_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[{name}]", flush=True)

    stage("environment")
    try:
        report["versions"] = {p: version(p) for p in ("torch", "horizon-plugin-pytorch", "hbdk4-compiler")}
        with TemporaryDirectory(prefix=".work-", dir=args.output) as temp:
            work = Path(temp)
            source = args.from_hbir or (args.output if args.format != "hbm" else work) / "qat.bc"
            if not args.from_hbir:
                calibrate_and_export(args, source, report, stage)
            if args.format != "hbm":
                stage("convert_infer_hbir")
                write_infer_hbir(source, args.output / "quantized_infer.bc")
            if args.format != "hbir":
                report["hbm"] = compile_hbm(args, source, work, stage)
        report["artifacts"] = sorted(p.name for p in args.output.iterdir() if p.suffix in (".hbm", ".bc"))
        report["status"] = "completed"
        stage("completed")
        print(f"Outputs: {args.output}")
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        stage(report["stage"])
        raise


if __name__ == "__main__":
    run(parse_args())
