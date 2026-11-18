# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""TTK registration for FlashGDN; implementation scripts are lazy-loaded.

Use this file as the TTK TestSpec source and assets/cases/*.csv as the cases.
The CSV API names remain gdn_ttk_ops.flash_gdn[_fwd]. For direct imports outside
the TestSpec loader, expose assets/impl on PYTHONPATH. No NPU is initialized here.
"""

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")


@lru_cache(None)
def _load_adapter():
    """Register the stable CSV API module even when only spec.py was loaded."""
    name = "gdn_ttk_ops"
    path = ASSET_IMPL_DIR / (name + ".py")
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path.resolve():
            raise ImportError(
                "FlashGDN test adapter was loaded from a different asset tree"
            )
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_impl_module(name):
    return _load_adapter().load_asset(name)


def _golden_dispatch(*args, **kwargs):
    return load_impl_module("golden").golden(*args, **kwargs)


def _inputs_dispatch(*args, **kwargs):
    return load_impl_module("inputs").customize_inputs(*args, **kwargs)


class CpuChunkSimulation:
    """Independent FP32 CPU chunk peer, not a GPU or DUT-derived baseline."""

    def __call__(
        self,
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
        **kwargs,
    ):
        inputs = [
            q,
            k,
            v,
            g,
            beta,
            metadata,
            initial_state,
            A_log,
            dt_bias,
            cu_seqlens,
            seqused,
        ]
        return load_impl_module("compare").pack(
            load_impl_module("peer").reference(inputs, kwargs)
        )


class FlashGdnFwdTestSpec:
    golden = staticmethod(_golden_dispatch)
    customize_inputs = staticmethod(_inputs_dispatch)
    # Same native acceptance preset as FlashKDA. No custom comparison hook,
    # per-case relaxation, or synthetic noise in the third-party denominator.
    third_party = {"torch": CpuChunkSimulation}
    tolerance = {
        "bfloat16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
    }


__spec__ = {
    "gdn_ttk_ops.flash_gdn_fwd": "FlashGdnFwdTestSpec",
    "gdn_ttk_ops.flash_gdn": "FlashGdnFwdTestSpec",
}

# Make the unchanged CSV API names importable before TTK resolves them.
_load_adapter()
