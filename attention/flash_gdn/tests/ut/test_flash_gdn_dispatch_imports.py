# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Real Torch dispatcher calls in fresh processes for both adapter layouts."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("layout", ["source", "wheel"])
def test_real_dispatcher_import_layout(layout):
    repo = Path(__file__).resolve().parents[4]
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    wheel_site = env.get("GDN_REVIEW_WHEEL_SITE")
    if layout == "wheel" and not wheel_site:
        pytest.skip(
            "set GDN_REVIEW_WHEEL_SITE to the freshly built GDN wheel installation"
        )
    paths = (
        [str(repo / "torch_extension"), str(repo)]
        if layout == "source"
        else [wheel_site]
    )
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    code = r"""
import importlib
import logging
import sys
import torch
logging.getLogger("cann_ops_transformer.ops").setLevel(logging.ERROR)
if sys.argv[1] == "source":
    from attention.flash_gdn.torch_extension import flash_gdn, flash_gdn_fwd, flash_gdn_metadata
    adapter = importlib.import_module("attention.flash_gdn.torch_extension.flash_gdn")
else:
    from cann_ops_transformer import flash_gdn, flash_gdn_fwd, flash_gdn_metadata
    adapter = importlib.import_module("cann_ops_transformer.ops.attention.flash_gdn.flash_gdn")
implementation = adapter._dsl_module("flash_gdn")
assert implementation.__package__.endswith("flash_gdn.op_dsl"), implementation.__file__
lengths = torch.empty(2, dtype=torch.int32, device="meta")
md = flash_gdn_metadata(lengths, batch_size=1, nheads_qk=1, nheads_v=1, seqlens=2)
assert md.device.type == "meta" and md.shape == (13,)
q = torch.empty((2, 1, 128), dtype=torch.bfloat16, device="meta")
g = torch.empty((2, 1), dtype=torch.float32, device="meta")
for function, arity in ((flash_gdn, 3), (flash_gdn_fwd, 16)):
    result = function(q, q, q, g, g, md, output_final_state=True, return_intermediate_states=True)
    assert len(result) == arity and result[0].shape == q.shape
    assert result[1].shape == (1, 1, 128, 128)
    assert result[2].shape == (2, 1, 128, 128)
try:
    flash_gdn_metadata(batch_size=1, nheads_qk=1, nheads_v=1, seqlens=2, enable_grad=True)
except NotImplementedError:
    pass
else:
    raise AssertionError("real CompositeExplicitAutograd metadata dispatch was not exercised")
print("REAL_DISPATCHER_OK", sys.argv[1], implementation.__file__)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, layout],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REAL_DISPATCHER_OK" in result.stdout
