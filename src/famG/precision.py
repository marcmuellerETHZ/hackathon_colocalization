"""Precision guards (DESIGN §5 A engineering, §7.1 item 10): TF32 off and asserted; no fp16/bf16 for LD."""
import os
import torch


def disable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:  # pragma: no cover
        pass


def assert_no_tf32() -> None:
    disable_tf32()
    assert torch.backends.cuda.matmul.allow_tf32 is False, "TF32 matmul must be disabled"
    assert torch.backends.cudnn.allow_tf32 is False, "TF32 cudnn must be disabled"
    assert torch.get_float32_matmul_precision() == "highest", "float32 matmul precision must be 'highest'"


def check_ld_dtype(dtype: torch.dtype) -> None:
    assert dtype in (torch.float32, torch.float64), f"LD dtype {dtype} not allowed (no fp16/bf16 for LD)"


def default_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def env_active() -> bool:
    return os.environ.get("COLOC_ENV_ACTIVE") == "1"
