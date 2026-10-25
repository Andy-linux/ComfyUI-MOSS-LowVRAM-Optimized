"""Offline FP8 export using this plugin's pinned vendor; never downloads weights."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Local dense BF16 single file or HF transformer folder")
    parser.add_argument("--output", required=True, help="New safetensors file; existing files refused")
    parser.add_argument("--variant", required=True, choices=("base", "sft"))
    parser.add_argument("--precision", default="fp8_compute", choices=("fp8_compute", "mxfp8", "fp8_weight_only"))
    parser.add_argument("--recipe", default="row", choices=("row", "tensor", "block"))
    parser.add_argument("--device", default="cpu", help="Quantization device only; no training")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime.fp8_support import export_fp8
    print(json.dumps(export_fp8(args.source, args.output, variant=args.variant,
                               precision=args.precision, recipe=args.recipe, device=args.device),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
