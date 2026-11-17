# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""gen -> cpu -> npu -> compare, with the same cache stages as the FP8 suite."""

import importlib
import os
from pathlib import Path
from unittest.mock import patch

import torch

from .quant_flash_attn_int8_golden import golden


def load_qfa():
    package = os.environ.get("QFA_TEST_OP_PACKAGE", "cann_ops_transformer")
    with patch("importlib.metadata.entry_points", return_value=[]):
        return importlib.import_module(f"{package}.ops").quant_flash_attn


def prepare_npu(params, batches, vs):
    """Prepare tensors and metadata before the profiled attention call."""
    import torch_npu  # noqa: F401

    torch.npu.set_device(int(os.environ.get("QFA_TEST_NPU_DEVICE", "0")))
    qfa = load_qfa()
    module = importlib.import_module(qfa.__module__)
    layout, kv_layout = params["layout"], params["layout_kv"]
    args = [
        golden._pack(batches, l, i).npu()
        for i, l in enumerate((layout, kv_layout, kv_layout, layout, kv_layout))
    ]
    args += [vs.npu(), 12]
    options = dict(
        layout_q=layout,
        layout_q_descale=layout,
        layout_kv=kv_layout,
        layout_out=params["layout_out"],
        mask_mode=params["mask_mode"],
        win_left=params["win_left"],
        win_right=params["win_right"],
        max_seqlen_q=params.get("max_seqlen_q", max(params["actual_seq_q"])),
        max_seqlen_kv=params.get("max_seqlen_kv", max(params["actual_seq_kv"])),
    )
    for name, lengths, tensor_layout in (
        ("q", params["actual_seq_q"], layout),
        ("kv", params["actual_seq_kv"], kv_layout),
    ):
        if tensor_layout == "TND":
            options[f"cu_seqlens_{name}"] = torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0)], dtype=torch.int32, device="npu"
            )
    metadata = (
        module.quant_flash_attn_metadata(
            params["N_q"],
            params["N_kv"],
            params["D"],
            12,
            batch_size=None if layout == "TND" else len(batches),
            **options,
        )
        if params["metadata"]
        else None
    )
    options.update(
        metadata=metadata,
        softmax_scale=params["softmax_scale"],
        p_scale=torch.tensor([params["p_scale"]], device="npu"),
        return_softmax_lse=params["enable_lse"],
    )
    return qfa, args, options


def reference(params, batches, vs):
    outputs = []
    for batch, length in zip(batches, params["actual_seq_q"]):
        rows = golden.sample_rows(length, params["sample_rows"])
        options = dict(
            rows=rows,
            causal=params["mask_mode"] == 3,
            win_left=params["win_left"],
            win_right=params["win_right"],
            softmax_scale=params["softmax_scale"],
        )
        outputs.append(
            dict(
                rows=rows,
                golden=golden.attention_reference(
                    batch, vs, high_precision=True, p_scale=params["p_scale"], **options
                ),
                reference=golden.attention_reference(
                    batch, vs, p_scale=params["p_scale"], **options
                ),
                lse=golden.lse_reference(batch, **options)
                if params["enable_lse"]
                else None,
            )
        )
    return outputs


def compare(params, expected, actual):
    output, lse = actual
    assert torch.isfinite(output).all(), "nonfinite attention output"
    if params["enable_lse"]:
        lengths, heads = params["actual_seq_q"], params["N_q"]
        shape = (
            (heads, sum(lengths))
            if params["layout_out"] == "TND"
            else (len(lengths), heads, lengths[0])
        )
        assert (
            lse.dtype == torch.float32
            and tuple(lse.shape) == shape
            and lse.is_contiguous()
        )
        assert not torch.isnan(lse).any() and not torch.isposinf(lse).any()
    unpacked = golden.unpack(output, params["layout_out"], params["actual_seq_q"])
    offset = 0
    for index, (batch, refs, length) in enumerate(
        zip(unpacked, expected, params["actual_seq_q"])
    ):
        rows = refs["rows"]
        if rows:
            golden.assert_atk_accuracy(
                refs["golden"], batch[:, :, rows], refs["reference"]
            )
            if params["enable_lse"]:
                observed = (
                    lse[:, offset : offset + length].unsqueeze(0)
                    if params["layout_out"] == "TND"
                    else lse[index : index + 1]
                )
                torch.testing.assert_close(
                    observed[:, :, rows].double(), refs["lse"], rtol=5e-4, atol=3e-3
                )
        offset += length
    if not params["enable_lse"]:
        assert lse is None or lse.numel() == 0
    return {"pass": True}, {"pass": True} if params["enable_lse"] else None


def execute_test(params, mode, cdir=None):
    directory = Path(cdir) if cdir else Path(__file__).with_name("golden_cache")
    directory.mkdir(parents=True, exist_ok=True)

    def save(stage, value):
        torch.save(
            {"params": params, "value": value},
            directory / f"{params['name']}_{stage}.pt",
        )

    def load(stage):
        payload = torch.load(
            directory / f"{params['name']}_{stage}.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert payload["params"] == params, (
            "cache parameters differ; regenerate this case"
        )
        return payload["value"]

    if "gen" in mode:
        batches, vs = golden._inputs(
            params["actual_seq_q"],
            params["actual_seq_kv"],
            params["N_q"],
            params["N_kv"],
            params["D"],
            params["seed"],
            params.get("input_range", (-8, 9)),
        )
        for batch in batches:
            batch[3].mul_(params.get("q_scale_factor", 1.0))
            if params.get("zero_q_scale", False):
                batch[3][:, :, ::2].zero_()
        save("input", (batches, vs))
    elif mode & {"cpu", "npu"}:
        batches, vs = load("input")
    if "cpu" in mode:
        save("cpu_output", reference(params, batches, vs))
    if "npu" in mode:
        qfa, args, options = prepare_npu(params, batches, vs)
        for _ in range(params["warmup"]):
            qfa(*args, **options)
        torch.npu.synchronize()
        with torch.profiler.record_function(f"qfa_gqa::{params['name']}"):
            for _ in range(params["repeat"]):
                output, lse = qfa(*args, **options)
        torch.npu.synchronize()
        if params["enable_lse"]:
            without, empty = qfa(*args, **{**options, "return_softmax_lse": False})
            torch.testing.assert_close(output, without, rtol=0, atol=0)
            assert empty is None or empty.numel() == 0
        save("npu_output", (output.cpu(), lse.cpu() if lse is not None else None))
    if "compare" in mode:
        return compare(params, load("cpu_output"), load("npu_output"))
    return None, None


def check_results(atten_result, lse_result):
    for result in (atten_result, lse_result):
        if result is not None:
            assert result["pass"], str(result)
