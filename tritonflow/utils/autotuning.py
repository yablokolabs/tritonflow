"""Autotuning configuration helpers for Triton kernels."""

from __future__ import annotations

from typing import Any

try:
    import triton

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

_DEFAULT_BLOCK_SIZES = [64, 128, 256, 512, 1024]
_DEFAULT_NUM_WARPS = [2, 4, 8]
_DEFAULT_NUM_STAGES = [2, 3, 4]

_MAX_THREADS_PER_SM = 2048
_MAX_WARPS_PER_SM = 64
_MAX_BLOCKS_PER_SM = 32
_MAX_SHARED_MEM_PER_SM = 163840  # 160 KB (Ampere default)


def standard_block_sizes() -> list[int]:
    """Return the standard block sizes used for autotuning."""
    return list(_DEFAULT_BLOCK_SIZES)


def get_autotune_configs(
    block_sizes: list[int] | None = None,
    num_warps_options: list[int] | None = None,
    num_stages_options: list[int] | None = None,
) -> list[Any]:
    """Generate a list of autotuning configurations.

    Returns triton.Config objects if triton is installed, otherwise plain dicts.
    """
    blocks = block_sizes or _DEFAULT_BLOCK_SIZES
    warps = num_warps_options or _DEFAULT_NUM_WARPS
    stages = num_stages_options or _DEFAULT_NUM_STAGES

    configs: list[Any] = []
    for bs in blocks:
        for nw in warps:
            for ns in stages:
                if _TRITON_AVAILABLE:
                    configs.append(
                        triton.Config(
                            {"BLOCK_SIZE": bs},
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
                else:
                    configs.append(
                        {
                            "BLOCK_SIZE": bs,
                            "num_warps": nw,
                            "num_stages": ns,
                        }
                    )

    return configs


def estimate_occupancy(
    block_size: int,
    shared_mem_bytes: int,
    registers_per_thread: int = 32,
) -> float:
    """Estimate theoretical occupancy for a kernel launch configuration.

    Returns a value between 0.0 and 1.0 representing the fraction of
    maximum possible active warps per SM.
    """
    threads_per_block = block_size
    warp_size = 32
    warps_per_block = (threads_per_block + warp_size - 1) // warp_size

    max_blocks_by_warps = _MAX_WARPS_PER_SM // warps_per_block if warps_per_block > 0 else 0

    if shared_mem_bytes > 0:
        max_blocks_by_smem = _MAX_SHARED_MEM_PER_SM // shared_mem_bytes
    else:
        max_blocks_by_smem = _MAX_BLOCKS_PER_SM

    regs_per_block = registers_per_thread * threads_per_block
    max_regs_per_sm = 65536
    max_blocks_by_regs = max_regs_per_sm // regs_per_block if regs_per_block > 0 else 0

    active_blocks = min(max_blocks_by_warps, max_blocks_by_smem, max_blocks_by_regs, _MAX_BLOCKS_PER_SM)
    active_blocks = max(active_blocks, 0)

    active_warps = active_blocks * warps_per_block
    occupancy = active_warps / _MAX_WARPS_PER_SM

    return min(occupancy, 1.0)
