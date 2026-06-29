import argparse
import sys
import torch
import yaml
from   pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.yolo import Model
from utils.torch_utils import model_info


def shape_of(x):
    if torch.is_tensor(x):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return [shape_of(v) for v in x]
    return str(type(x).__name__)


def fmt_shape(x):
    return str(x).replace("torch.Size", "")


def print_table(rows):
    headers = ["idx", "from", "module", "params", "input_shape", "output_shape"]
    table = [
        [
            row["idx"],
            str(row["from"]),
            row["module"],
            f"{row['params']:,}",
            fmt_shape(row["input_shape"]),
            fmt_shape(row["output_shape"]),
        ]
        for row in rows
    ]

    widths = [max(len(str(r[i])) for r in [headers] + table) for i in range(len(headers))]
    print(" | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * w for w in widths))
    for row in table:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def inspect_model(cfg_path, img_size=640, batch_size=1, in_channels=3):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = Model(cfg=str(cfg_path), ch=in_channels, nc=cfg.get("nc"))
    model.eval()

    h = w = img_size
    x = torch.zeros(batch_size, in_channels, h, w)

    rows = []
    layer_defs = cfg.get("backbone", []) + cfg.get("head", [])

    def hook(module, inputs, output):
        rows.append(
            {
                "idx": len(rows),
                "from": layer_defs[len(rows)][0] if len(rows) < len(layer_defs) else None,
                "module": module.__class__.__name__,
                "params": sum(p.numel() for p in module.parameters()),
                "input_shape": shape_of(inputs),
                "output_shape": shape_of(output),
            }
        )

    handles = [module.register_forward_hook(hook) for module in model.model]

    with torch.no_grad():
        output = model(x)

    for handle in handles:
        handle.remove()

    return model, output, rows


def main():
    parser = argparse.ArgumentParser(description="Inspect a YOLOv7-style YAML architecture using the repository's own model builder")
    parser.add_argument("--cfg",      type=str, default=str(ROOT / "cfg/training/udp_dpf_1h_8a.yaml"))
    parser.add_argument("--img",      type=int, nargs="+", default=[320], help="Input image size; use '--img 640' or '--img 640 640'")
    parser.add_argument("--batch",    type=int, default=1)
    parser.add_argument("--channels", type=int, default=3)
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(f"YAML file not found: {cfg_path}")

    if len(args.img) == 1:
        img_size = args.img[0]
    elif len(args.img) == 2:
        img_size = args.img[0]
    else:
        raise ValueError("Use --img 640 or --img 640 640")

    model, output, rows = inspect_model(
        cfg_path=cfg_path,
        img_size=img_size,
        batch_size=args.batch,
        in_channels=args.channels,
    )

    print()
    print("YAML file:")
    print(cfg_path)
    print()
    print("PyTorch module tree:")
    print(model)
    print()
    print("Layer summary:")
    print_table(rows)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print(f"Total parameters:   {total_params:,}")
    print(f"Trainable params:   {trainable_params:,}")
    print(f"Final output shape: {fmt_shape(shape_of(output))}")

    print()
    model_info(model, verbose=False, img_size=img_size)


if __name__ == "__main__":
    main()