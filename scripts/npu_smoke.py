from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal Ascend NPU PyTorch smoke test.")
    parser.add_argument("--device", default="npu:0", help="Logical NPU device, for example npu:0.")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    import torch
    import torch_npu

    device = torch.device(args.device)
    torch_npu.npu.set_device(device)
    x = torch.randn(args.size, args.size, device=device)
    y = torch.randn(args.size, args.size, device=device)
    z = x @ y
    torch_npu.npu.synchronize()
    print(
        {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": str(device),
            "shape": list(z.shape),
            "mean": float(z.float().mean().cpu()),
        }
    )


if __name__ == "__main__":
    main()
