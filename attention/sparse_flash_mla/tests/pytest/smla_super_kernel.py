# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Opt-in ACLGraph SuperKernel execution for SMLA pytest."""

import os
import shlex
import subprocess
from contextlib import contextmanager
from unittest.mock import patch

import torch


def super_kernel_enabled():
    value = os.environ.get("SMLA_SUPER_KERNEL", "0")
    if value not in ("0", "1"):
        raise ValueError("SMLA_SUPER_KERNEL must be 0 or 1")
    return value == "1"


class _SuperKernelNetwork(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.scope_name = "torch.ops.cann_ops_transformer.sparse_flash_mla"

    def forward(self, *args, **kwargs):
        torch.npu.super_kernel_scope_begin(self.scope_name)
        result = self.model(*args, **kwargs)
        torch.npu.super_kernel_scope_end(self.scope_name)
        return result


@contextmanager
def _require_kernel_compile():
    """Reject compiler failures even when the backend falls back to binaries."""
    original_run = subprocess.run
    failures = []

    def checked_run(*args, **kwargs):
        command = args[0] if args else kwargs.get("args", ())
        parts = shlex.split(command) if isinstance(command, str) else command
        if not parts or os.path.basename(os.fsdecode(parts[0])) != "op_compiler":
            return original_run(*args, **kwargs)
        try:
            result = original_run(*args, **kwargs)
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            OSError,
        ) as error:
            failures.append(str(error))
            raise
        if result.returncode:
            failures.append(f"op_compiler exited with {result.returncode}")
        return result

    with patch.object(subprocess, "run", checked_run):
        yield
    if failures:
        raise RuntimeError("KERNEL_COMPILE_FAILURE: " + "; ".join(failures))


def compile_super_kernel(model):
    """Compile the main operator only; metadata must be prepared outside."""
    import torch_npu  # noqa: F401 registers the npugraph_ex backend

    if not all(
        callable(getattr(torch.npu, name, None))
        for name in ("super_kernel_scope_begin", "super_kernel_scope_end")
    ):
        raise RuntimeError("SMLA SuperKernel requires torch_npu scope APIs")
    options = {
        "static_kernel_compile": True,
        "super_kernel_optimize": True,
        "super_kernel_debug_options": {"debug_per_op_max_core_num": 1},
        "clone_input": False,
    }
    print("SMLA SuperKernel: npugraph_ex, fullgraph=True, online compile")
    compiled = torch.compile(
        _SuperKernelNetwork(model),
        backend="npugraph_ex",
        fullgraph=True,
        dynamic=False,
        options=options,
    )

    def run(*args, **kwargs):
        jit_compile = not torch.npu.is_jit_compile_false()
        try:
            torch.npu.set_compile_mode(jit_compile=True)
            with (
                torch._dynamo.config.patch(suppress_errors=False),
                _require_kernel_compile(),
            ):
                result = compiled(*args, **kwargs)
                torch.npu.synchronize()
                return result
        finally:
            torch.npu.set_compile_mode(jit_compile=jit_compile)

    return run
