import argparse
from pathlib import Path

from .system import BagGuardSystem


def main():
    root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Bag Guard System")
    parser.add_argument("--source", default="0", help="0 for webcam or path to a video file")
    parser.add_argument("--model", default=str(root / "models" / "yolo26x.pt"), help="Path to model weights")
    parser.add_argument("--out", "--output", dest="output", default=str(root / "outputs" / "detection_output.mp4"), help="Output video path")
    parser.add_argument("--show", action="store_true", help="Show live window")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    parser.add_argument("--max_fps", type=float, default=12, help="Max processing FPS (0 to disable)")
    parser.add_argument("--skip", type=int, default=0, help="Process every (skip+1)th frame")
    parser.add_argument("--half", action="store_true", help="Use FP16 if CUDA is available")
    parser.add_argument("--tracker-backend", choices=["deepsort", "bytetrack"], default="deepsort", help="Tracking backend: deepsort (default) or bytetrack")
    parser.add_argument("--tracker-profile", choices=["main", "stable"], default="main", help="ByteTrack preset: main (default) or stable (longer persistence)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        candidate = root / model_path
        if candidate.exists():
            model_path = candidate
        elif model_path.parent == Path("."):
            model_path = root / "models" / model_path
        else:
            model_path = root / model_path

    source_arg = str(args.source)
    if source_arg.isdigit():
        video_path = int(source_arg)
    else:
        video_path = Path(source_arg)
        if not video_path.is_absolute():
            if video_path.parent == Path("."):
                video_path = root / "data" / video_path
            else:
                video_path = root / video_path
        video_path = str(video_path)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        if output_path.parent == Path("."):
            output_path = root / "outputs" / output_path
        else:
            output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nResolved paths:")
    print(f"Resolved model path: {model_path}")
    print(f"Resolved source: {video_path}")
    print(f"Resolved output path: {output_path}")

    system = BagGuardSystem(
        str(model_path),
        video_path,
        str(output_path),
        imgsz=args.imgsz,
        max_fps=args.max_fps,
        skip=args.skip,
        half=args.half,
        tracker_backend=args.tracker_backend,
        tracker_profile=args.tracker_profile,
        show=args.show,
    )
    success = system.run()

    if success:
        print("🎉 SUCCESS! Full BGS Specification Implemented!")
        print("\n📋 IMPLEMENTED FEATURES:")
        print("  ✓ Monocular distance estimation (meters)")
        print("  ✓ Ownership persistence (trend-based)")
        print("  ✓ 3-state bag status (OK/POTENTIAL/UNATTENDED)")
        print("  ✓ Ownership locking & confirmation")
        print("  ✓ Distance lines & labels")
        print("  ✓ Frozen parameters (reproducible)")
        print("  ✓ Professional debug overlay")