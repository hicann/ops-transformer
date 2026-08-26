#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import os
import sys
from typing import List, Optional

import torch
import torch_npu

# 复用 quant_flash_attn_golden / quant_flash_attn_fp8_golden 的 layout 转换 / e8m0 打包 / prepare_npu_inputs / 全局变量
_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.join(_ASSETS_DIR, "..")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
import quant_flash_attn_golden as mxfp8_golden_mod
import quant_flash_attn_fp8_golden as fp8_golden_mod

logger = logging.getLogger(__name__)


def _apply_golden_globals(params, quant_mode=1):
    """把 case 参数注入 golden 模块全局变量 (按 quant_mode 选择目标模块, 与 qfa_wrapper 一致).

    prepare_npu_inputs 读目标 golden_mod 的全局变量 (B/N_q/N_kv/D/ENABLE_PA/...),
    必须在调 prepare_npu_inputs 前把 csv attributes 全部注入.

    quant_mode=6 → fp8_golden_mod (GQA FP8 全量化路径)
    其他 → mxfp8_golden_mod (MXFP8 路径)
    """
    target = fp8_golden_mod if quant_mode == 6 else mxfp8_golden_mod
    for k, v in params.items():
        setattr(target, k, v)


class QuantFlashAttnAclGraph(torch.nn.Module):
    """aclgraph 编译目标: forward 只调 quant_flash_attn 主算子.

    __init__ (capture 之外) 完成:
      1. 从函数参数取 final-layout fp8 q/k/v + e8m0 descale + fp32 p_scale + int32 block_table
      2. 注入 golden 全局变量 (B/N_q/.../ENABLE_PA/INPUT_LAYOUT/...)
      3. prepare_npu_inputs
      4. cu_seqlens/seqused 直接用入参 _t (slot 8-11, inputs.py 已写入真实值), 空→None
      5. quant_flash_attn_metadata 构建 (capture 之前, metadata 是 int32 tensor)
      6. 所有运行时入参存为 self. 属性, forward 直接读

    入参对齐 15 位置 tensor (与 CSV tensor_view_shapes / qfa_wrapper 一致):
      0 q .. 7 block_table, 8 cu_seqlens_q_t, 9 cu_seqlens_kv_t, 10 seqused_q_t,
      11 seqused_kv_t, 12 sinks_t, 13 attn_mask_t, 14 metadata_t。

    forward() (capture 之内) 只调:
      torch.ops.cann_ops_transformer.quant_flash_attn(self.q, ..., metadata=self.metadata, ...)
    """

    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dequant_scale_q: torch.Tensor,
        dequant_scale_k: torch.Tensor,
        dequant_scale_v: torch.Tensor,
        p_scale: torch.Tensor,
        block_table: torch.Tensor,
        cu_seqlens_q_t: torch.Tensor,
        cu_seqlens_kv_t: torch.Tensor,
        seqused_q_t: torch.Tensor,
        seqused_kv_t: torch.Tensor,
        sinks_t: torch.Tensor,
        attn_mask_t: torch.Tensor,
        metadata_t: torch.Tensor,
        *,
        batch_size: int,
        N_q: int,
        N_kv: int,
        D: int,
        cu_seqlens_q: Optional[List[int]] = None,
        cu_seqlens_kv: Optional[List[int]] = None,
        seqused_q: Optional[List[int]] = None,
        seqused_kv: Optional[List[int]] = None,
        max_seqlen_q: int = -1,
        max_seqlen_kv: int = -1,
        enable_pa: bool = False,
        kv_cache_layout: str = "BnNBsD",
        block_size: int = 0,
        mask_mode: int = 0,
        q_scale_layout: str = "TND",
        quant_mode: int = 1,
        enable_lse: bool = False,
        graph_path: int = 0,
        input_layout: str = "TND",
        is_contiguous: bool = True,
        device_id: int = 0,
        softmax_scale: Optional[float] = None,
        data_range_q: float = 1.0,
        data_range_k: float = 1.0,
        data_range_v: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        torch_npu.npu.set_device(int(device_id))

        # ---- 1. 从函数参数取 final-layout fp8 + e8m0  ----
        # q, k, v 是 torch.float8_e4m3fn, final layout (TND / PA paged)
        # dequant_scale_q/k/v 是 torch.float8_e8m0fnu, final layout (TND / PA paged)
        # p_scale 是 torch.float32, block_table 是 torch.int32
        p_scale_val = (
            float(p_scale.item())
            if isinstance(p_scale, torch.Tensor)
            else float(p_scale)
        )

        # cu_seqlens/seqused 真实值由 inputs.py 写入 _t tensor slot (8-11)；
        # 这里从 tensor 读回 list（与 qfa_wrapper/qfa_main_wrapper 一致）。
        # _t 端可能是 NPU tensor，.cpu().tolist() 读回；空 tensor (numel==0) → None。
        def _tolist_t(t, default=None):
            if t is None:
                return default
            return t.detach().cpu().tolist() if t.numel() > 0 else default

        cu_seqlens_q_list = _tolist_t(cu_seqlens_q_t, [0])
        cu_seqlens_kv_list = _tolist_t(cu_seqlens_kv_t, [0])
        seqused_q_list = _tolist_t(seqused_q_t)
        seqused_kv_list = _tolist_t(seqused_kv_t)

        # batch_size: CSV 原始值 (可为 -1), 透传给 metadata 的 batch_size;
        # B: 从 cu_seqlens_q 推导的正整数, 供 prepare_npu_inputs 生成 BNSD 张量。
        B = (
            max(1, len(cu_seqlens_q_list) - 1)
            if cu_seqlens_q_list and len(cu_seqlens_q_list) >= 2
            else 1
        )

        attn_mask_shape = tuple(attn_mask_t.shape)

        # ---- 2. 注入 golden 全局变量 (prepare_npu_inputs 依赖) ----
        _apply_golden_globals(
            {
                "B": B,
                "BATCH_SIZE": batch_size,
                "N_q": N_q,
                "N_kv": N_kv,
                "D": D,
                "CU_SEQLENS_Q": cu_seqlens_q_list,
                "CU_SEQLENS_KV": cu_seqlens_kv_list,
                "SEQUSED_Q": seqused_q_list,
                "SEQUSED_KV": seqused_kv_list,
                "MAX_SEQLEN_Q": max_seqlen_q,
                "MAX_SEQLEN_KV": max_seqlen_kv,
                "ENABLE_PA": enable_pa,
                "KV_CACHE_LAYOUT": kv_cache_layout,
                "BLOCK_SIZE": block_size,
                "SPARSE_MODE": mask_mode,
                "ATTN_MASK_SHAPE": attn_mask_shape or (2048, 2048),
                "Q_SCALE_LAYOUT": q_scale_layout,
                "P_SCALE": p_scale_val,
                "ENABLE_LSE": enable_lse,
                "FP8_DTYPE": torch.float8_e4m3fn,
                "QUANT_MODE": quant_mode,
                "QUANT_GROUP_SIZE": 32,
                "INPUT_LAYOUT": input_layout,
                "IS_CONTIGUOUS": is_contiguous,
                "DEVICE_ID": device_id,
                "GRAPH_PATH": graph_path,
                "SOFTMAX_SCALE": softmax_scale,
                "SEED_Q": kwargs.get("seed_q"),
                "SEED_K": kwargs.get("seed_k"),
                "SEED_V": kwargs.get("seed_v"),
                "DATA_RANGE_Q": kwargs.get("data_range_q", data_range_q),
                "DATA_RANGE_K": kwargs.get("data_range_k", data_range_k),
                "DATA_RANGE_V": kwargs.get("data_range_v", data_range_v),
            },
            quant_mode=quant_mode,
        )

        # ---- 3. 透传 ----

        def _to_cpu(t):
            return (
                t.detach().cpu()
                if isinstance(t, torch.Tensor) and t.device.type == "npu"
                else t
            )

        q_fp8 = _to_cpu(q)
        k_fp8 = _to_cpu(k)
        v_fp8 = _to_cpu(v)
        dequant_scale_q_cpu = _to_cpu(dequant_scale_q)
        dequant_scale_k_cpu = _to_cpu(dequant_scale_k)
        dequant_scale_v_cpu = _to_cpu(dequant_scale_v)
        p_scale_cpu = _to_cpu(p_scale)
        block_table_cpu = _to_cpu(block_table)

        logger.info(
            "[GRAPH] 透传 fp8+e8m0 (q=%s, k=%s, v=%s, dq=%s, dk=%s, dv=%s, enable_pa=%s)",
            tuple(q_fp8.shape),
            tuple(k_fp8.shape),
            tuple(v_fp8.shape),
            tuple(dequant_scale_q_cpu.shape),
            tuple(dequant_scale_k_cpu.shape),
            tuple(dequant_scale_v_cpu.shape),
            enable_pa,
        )

        # ---- 4. prepare_npu_inputs (按 quant_mode 派发) ----
        if quant_mode == 6:
            inputs = fp8_golden_mod.prepare_npu_inputs_gqa_fp8(
                q_fp8,
                k_fp8,
                v_fp8,
                dequant_scale_q_cpu,
                dequant_scale_k_cpu,
                dequant_scale_v_cpu,
                p_scale_cpu,
                cu_seqlens_q_list,
                cu_seqlens_kv_list,
                seqused_q_list,
                seqused_kv_list,
                max_seqlen_q,
                max_seqlen_kv,
                block_table_cpu
                if isinstance(block_table_cpu, torch.Tensor) and enable_pa
                else None,
            )
        else:
            inputs = mxfp8_golden_mod.prepare_npu_inputs(
                q_fp8,
                k_fp8,
                v_fp8,
                dequant_scale_q_cpu,
                dequant_scale_k_cpu,
                dequant_scale_v_cpu,
                p_scale_cpu,
                cu_seqlens_q_list,
                cu_seqlens_kv_list,
                seqused_q_list,
                seqused_kv_list,
                max_seqlen_q,
                max_seqlen_kv,
                block_table_cpu
                if isinstance(block_table_cpu, torch.Tensor) and enable_pa
                else None,
            )

        # ---- 4. cu_seqlens/seqused 直接用入参 _t (slot 8-11) ----
        layout_q = inputs["layout_q"]
        layout_kv = inputs["layout_kv"]
        is_tnd_q = layout_q == "TND"
        is_tnd_kv = layout_kv == "TND"

        # cu_seqlens/seqused 直接用入参 _t（NPU tensor，inputs.py 已 in-place 写入真实值，
        # 保留 CSV tensor_dtypes：空/异常 dtype 原样传给算子被拦截）。
        # 兼容旧调用：_t 为 None 或空 (numel==0) 时从 list 重建 int32 tensor。
        def _as_npu(t, lst):
            if t is not None:
                return t if t.numel() > 0 else None
            if lst is None:
                return None
            return torch.tensor(list(lst), dtype=torch.int32).npu()

        cu_seqlens_q_t = _as_npu(cu_seqlens_q_t, cu_seqlens_q_list)
        cu_seqlens_kv_t = _as_npu(cu_seqlens_kv_t, cu_seqlens_kv_list)
        seqused_q_t = _as_npu(seqused_q_t, seqused_q_list)
        seqused_kv_t = _as_npu(seqused_kv_t, seqused_kv_list)

        torch.npu.synchronize()

        # ---- 5. metadata 构建 (capture 之前) ----
        # metadata 是 int32 tensor (4096,), 不含可微参数, 在 __init__ 构建.
        # 参数名与传参方式严格对齐 golden_mod._call_npu_fa_op (line 1222-1239):
        #   - batch_size 透传 CSV 原始值 (可为 -1)
        #   - 不传 win_left/win_right (用 schema 默认, 与 golden 一致)
        logger.info("[GRAPH] 构建 metadata (quant_flash_attn_metadata)")
        self.metadata = torch.ops.cann_ops_transformer.quant_flash_attn_metadata(
            int(inputs["q_n"]),
            int(inputs["kv_n"]),
            int(D),
            int(quant_mode),
            cu_seqlens_q=cu_seqlens_q_t if is_tnd_q else None,
            cu_seqlens_kv=cu_seqlens_kv_t if is_tnd_kv else None,
            seqused_q=seqused_q_t,
            seqused_kv=seqused_kv_t,
            v_descale=inputs["dequant_scale_v"],
            batch_size=batch_size,
            mask_mode=int(inputs["sparse_mode"]),
            layout_q=layout_q,
            layout_q_descale=inputs["layout_q_descale"],
            layout_kv=layout_kv,
            layout_out=inputs["layout_out"],
            max_seqlen_q=int(inputs["max_seqlen_q"]),
            max_seqlen_kv=int(inputs["max_seqlen_kv"]),
        )
        # metadata 可能建在错误的 device 上, 对齐 q.device
        if self.metadata.device != inputs["q"].device:
            self.metadata = self.metadata.to(inputs["q"].device)

        # ---- 6. 存所有 forward 需要的入参为 self. 属性 ----
        self.q = inputs["q"]
        self.k = inputs["k"]
        self.v = inputs["v"]
        self.q_descale = inputs["dequant_scale_q"]
        self.k_descale = inputs["dequant_scale_k"]
        self.v_descale = inputs["dequant_scale_v"]
        self.quant_mode = int(quant_mode)
        self.block_table = inputs["block_table"]
        self.p_scale = inputs["p_scale"]
        self.cu_seqlens_q = cu_seqlens_q_t if is_tnd_q else None
        self.cu_seqlens_kv = cu_seqlens_kv_t if is_tnd_kv else None
        self.seqused_q = seqused_q_t
        self.seqused_kv = seqused_kv_t
        self.attn_mask = inputs["mask"]
        self.softmax_scale = inputs["softmax_scale"]
        self.mask_mode = int(inputs["sparse_mode"])
        self.layout_q = layout_q
        self.layout_q_descale = inputs["layout_q_descale"]
        self.layout_kv = layout_kv
        self.layout_out = inputs["layout_out"]
        self.max_seqlen_q = int(inputs["max_seqlen_q"])
        self.max_seqlen_kv = int(inputs["max_seqlen_kv"])
        self.return_softmax_lse = bool(enable_lse)

        logger.info(
            "[GRAPH] __init__ done: q=%s, k=%s, v=%s, layout_q=%s, layout_kv=%s, "
            "metadata=%s, enable_pa=%s",
            self.q.shape,
            self.k.shape,
            self.v.shape,
            self.layout_q,
            self.layout_kv,
            self.metadata.shape,
            enable_pa,
        )

    def forward(self):
        """只调 quant_flash_attn 主算子, 可被 npugraph_ex 捕获.

        所有入参来自 self. 属性 (__init__ 预处理结果), forward 无参数.
        与 SparseFlashMlaAclGraph.forward() 模式一致.
        传参方式严格对齐 golden_mod._call_npu_fa_op 的 main_kwargs (line 1241-1265):
          - q_descale/k_descale/v_descale (不是 dequant_scale_*)
          - 不传 sinks/win_left/win_right (用 schema 默认, 与 golden 一致)
          - return_softmax_lse 由 _get_npu_fa_kwargs 机制对应 (此处直接传 enable_lse)
        """
        atten_out, lse_out = torch.ops.cann_ops_transformer.quant_flash_attn(
            self.q,
            self.k,
            self.v,
            self.q_descale,
            self.k_descale,
            self.v_descale,
            self.quant_mode,
            block_table=self.block_table,
            p_scale=self.p_scale,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_kv=self.cu_seqlens_kv,
            seqused_q=self.seqused_q,
            seqused_kv=self.seqused_kv,
            attn_mask=self.attn_mask,
            metadata=self.metadata,
            softmax_scale=self.softmax_scale,
            mask_mode=self.mask_mode,
            layout_q=self.layout_q,
            layout_q_descale=self.layout_q_descale,
            layout_kv=self.layout_kv,
            layout_out=self.layout_out,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_kv=self.max_seqlen_kv,
            return_softmax_lse=self.return_softmax_lse,
        )

        if not self.return_softmax_lse:
            lse_out = None
        # NPU lse_out 已是 N-major (N, T), 无需 reshape
        return atten_out, lse_out
