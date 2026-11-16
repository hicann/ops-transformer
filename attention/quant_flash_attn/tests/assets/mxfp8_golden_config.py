# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""QFA MXFP8 CPU golden block profiles (local --plat aliases).

Example: ttk e2e ... --backend cpu --plat arch92 --no-prof --dump in,golden
Use separate --manual-data-dirs for arch35 and arch92. These aliases do not
represent TTK SoC model names for compilation or simulation. Omitted --plat
keeps the arch35 profile; arch92 currently supports Q scale TND and N2TGD.
No device or TTK dependency is loaded at import time.
Profiles do not select the golden path: available NPU metadata enables sectioned
execution as before. In that path only K uses the profile; M/S2 use metadata.
"""

from typing import NamedTuple


class BlockSizes(NamedTuple):
    q: int
    k: int
    v: int


def resolve_target_arch(platform=None):
    """Interpret QFA's offline --plat aliases; omitted platform keeps arch35.

    The aliases are deliberately local to the MXFP8 test asset.
    """
    value = str(platform or "AUTO").strip().lower()
    if value in ("auto", "cpu", ""):
        return "arch35"
    if value in ("arch35", "arch92"):
        return value
    raise ValueError(
        f"QFA MXFP8 CPU golden: unsupported --plat {platform!r}; "
        "use --plat arch35 or --plat arch92"
    )


def get_block_sizes(arch, q_scale_layout, head_dim):
    """Global/fixed-reference Q/K/V blocks; metadata execution uses only K."""
    if arch == "arch35":
        return (
            BlockSizes(128, 128, 256) if head_dim == 256 else BlockSizes(128, 256, 512)
        )
    if arch != "arch92":
        raise ValueError(f"Unsupported MXFP8 golden architecture: {arch!r}")
    layout = str(q_scale_layout).upper()
    if layout == "TND":
        return BlockSizes(64, 128, 256) if head_dim == 256 else BlockSizes(64, 256, 512)
    if layout == "N2TGD":
        return BlockSizes(64, 256, 512)
    raise ValueError(
        f"arch92 MXFP8 CPU profile supports Q scale TND/N2TGD, got {q_scale_layout!r}"
    )


def get_ttk_cpu_profile(q_scale_layout, head_dim, backend=None):
    """Read existing TTK switches only at the adapter boundary.

    None means retain the original block sizes. This function selects only
    blocks, never whether metadata is requested or sectioned execution is used.
    """
    try:
        from ttk.switches import get_global_storage
    except ImportError:
        try:
            from ttk.utilities.container_utils import get_global_storage
        except ImportError:
            get_global_storage = None
    switches = get_global_storage() if get_global_storage is not None else None
    platform = getattr(switches, "dev_plat", None)
    value = str(platform or "AUTO").strip().lower()
    force_cpu = bool(getattr(switches, "force_cpu", False)) or backend == "cpu"
    if value not in ("arch35", "arch92") and not force_cpu:
        return None
    arch = resolve_target_arch(platform)
    return arch, get_block_sizes(arch, q_scale_layout, head_dim)
