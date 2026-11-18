# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""使用 CUDA FLA 对 FlashKDA 的输出、末态和 chunk 状态进行三方精度比较。"""

import importlib
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

import torch
from arch35.kda_ttk_ops import load_asset


def enabled():
    value = os.environ.get("KDA_CROSS_CHECK_BACKEND", "cpu").lower()
    if value not in ("cpu", "fla"):
        raise ValueError("KDA_CROSS_CHECK_BACKEND must be cpu or fla")
    return value == "fla"


def check_case(c):
    if c["suite"] != "accuracy":
        raise ValueError("FLA cross_check supports accuracy cases only")


def chunk_states_enabled(c):
    """按照当前公开接口的训推条件决定是否比较 chunk 状态。"""
    f = load_asset("inputs").effective(c)
    return (c["inference"] and f["return_intermediate_states"]) or (
        not c["inference"] and f["disable_recompute"]
    )


@lru_cache(None)
def chunk_entry():
    root = os.environ.get("FLA_ROOT")
    if root:
        folder = Path(root).expanduser().resolve()
        if not (folder / "fla/ops/kda/chunk.py").is_file():
            raise ValueError("FLA_ROOT must point to flash-linear-attention")
        sys.path.insert(0, str(folder))
    os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    backends = importlib.import_module("fla.backends")
    if not backends._DISPATCH_DISABLED:
        raise RuntimeError("Start the GPU server with FLA_DISABLE_BACKEND_DISPATCH=1")
    module = importlib.import_module("fla.ops.kda.chunk")
    if root and not Path(module.__file__).resolve().is_relative_to(folder):
        raise RuntimeError("FLA was already imported from a different source tree")
    # 训练和推理统一使用指定仓库的 Triton 实现，关闭 CUTLASS 和 TileLang 分发。
    logging.info("FlashKDA third-party source=%s", module.__file__)
    return module.chunk_kda


class FLAThirdParty:
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
        c = load_asset("inputs").configuration(inputs, kwargs)
        check_case(c)
        if q.device.type != "cuda":
            raise RuntimeError("FLA third party requires a CUDA GPU XPU endpoint")
        for value in inputs:
            if torch.is_tensor(value) and value.device != q.device:
                raise ValueError("FLA inputs must reside on the same CUDA device")

        chunk_kda = chunk_entry()
        f = load_asset("inputs").effective(c)
        output_chunks = chunk_states_enabled(c)
        # FLA 仅允许在关闭重算缓存的无梯度路径显式返回第三项；不修改 NPU 用例参数。
        fla_disable_recompute = False if output_chunks else f["disable_recompute"]
        # Q/K/V 遵循输入布局；定长 g/beta 固定为 BSND/BSN，紧凑输入为 TND/TN。
        gate_c = dict(c, layout="TND" if c["layout"] == "TND" else "BSND")
        sequences = [
            load_asset("inputs").sequences(value, c) for value in inputs[:3]
        ] + [load_asset("inputs").sequences(value, gate_c) for value in inputs[3:5]]
        outputs, states, chunk_states = [], [], []
        with torch.inference_mode(c["inference"]), torch.no_grad():
            for batch, values in enumerate(zip(*sequences)):
                qi, ki, vi, gi, bi = [
                    value.transpose(1, 2).contiguous() for value in values
                ]
                state = (
                    initial_state[batch : batch + 1].contiguous()
                    if c["initial_state_mode"] not in ("none", "omit")
                    else None
                )
                # FLA 接收一维偏置，当前接口允许的一维和二维形式统一按头、通道展开。
                bias = (
                    dt_bias.contiguous().flatten()
                    if c["dt_bias_mode"] != "none" and dt_bias is not None
                    else None
                )
                result = chunk_kda(
                    qi,
                    ki,
                    vi,
                    gi,
                    bi,
                    scale=f["scale"],
                    initial_state=state,
                    output_final_state=f["output_final_state"],
                    use_qk_l2norm_in_kernel=f["use_qk_l2norm_in_kernel"],
                    use_gate_in_kernel=f["use_gate_in_kernel"],
                    use_beta_sigmoid_in_kernel=f["use_beta_sigmoid_in_kernel"],
                    allow_neg_eigval=f["allow_neg_eigval"],
                    disable_recompute=fla_disable_recompute,
                    safe_gate=f["use_gate_in_kernel"] and f["lower_bound"] is not None,
                    lower_bound=f["lower_bound"],
                    return_intermediate_states=output_chunks,
                    state_v_first=True,
                    chunk_size=64,
                    A_log=A_log,
                    dt_bias=bias,
                )
                expected_arity = 3 if output_chunks else 2
                if (
                    not isinstance(result, (tuple, list))
                    or len(result) != expected_arity
                ):
                    raise RuntimeError(f"FLA must return {expected_arity} outputs")
                out, final = result[:2]
                if output_chunks:
                    # 逐序列调用时 FLA 为 [1, J, Nv, Dv, Dk]，J 仅含有效 chunk。
                    # 与 NPU 的 compare.pack 一样按序列、chunk、head 顺序展开，忽略预留槽。
                    chunks = result[2]
                    count = (qi.shape[1] + 63) // 64
                    expected_shape = (1, count, vi.shape[2], vi.shape[-1], qi.shape[-1])
                    if (
                        not isinstance(chunks, torch.Tensor)
                        or tuple(chunks.shape) != expected_shape
                        or chunks.dtype != torch.bfloat16
                        or chunks.device != q.device
                    ):
                        raise RuntimeError(
                            f"FLA chunk_states must have shape {expected_shape}, "
                            f"dtype torch.bfloat16 and device {q.device}; got "
                            f"{getattr(chunks, 'shape', None)}, "
                            f"{getattr(chunks, 'dtype', None)}, "
                            f"{getattr(chunks, 'device', None)}"
                        )
                    chunk_states.append(chunks[0].contiguous().flatten())
                # 按当前 compare.pack 的有效区顺序展开，统一为每条序列头维度在前。
                outputs.append(out.transpose(1, 2).contiguous().flatten())
                if f["output_final_state"]:
                    if final is None:
                        raise RuntimeError("FLA did not return requested final_state")
                    states.append(final.flatten())
            torch.cuda.synchronize(q.device)
        logging.info(
            "FlashKDA peer=FLA CUDA case=%s inference=%s outputs=out,final_state,chunk_states",
            c["id"],
            c["inference"],
        )
        return [
            torch.cat(outputs),
            torch.cat(states)
            if states
            else torch.empty(0, dtype=torch.float32, device=q.device),
            # 第三项未启用时使用空槽，与 NPU 打包结果和 CPU golden 保持一致。
            torch.cat(chunk_states)
            if chunk_states
            else torch.empty(0, dtype=torch.float32, device=q.device),
        ]
