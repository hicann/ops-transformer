#!/usr/bin/env python3
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
"""ATK executor for quant_flash_mla INT8 full-quant (arch22 / Ascend 910B).

用例输入只描述逻辑张量(q / q_rope / k / k_rope, bf16)与标量属性(batch / kv_len / layout),
int8 量化、PA_NZ 排布、block_table、metadata 以及 bf16 golden 全部在本执行器内构造,
保证 cpu(标杆) 与 npu(被测) 两个 backend 使用完全相同的量化网格。
"""

import glob
import math
import os

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import AccuracyConfig, TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.single_benchmark_compare import (
    SingleBenchmarkAccuracyCompare,
)

D_NOPE = 512
D_ROPE = 64
BLOCK = 128
QUANT_MODE_INT8 = 2
SCALE = 1.0 / math.sqrt(D_NOPE + D_ROPE)


@ACCURACY_REGISTRY.register("qfmla_benchmark")
class QfmlaBenchmarkCompare(SingleBenchmarkAccuracyCompare):
    """INT8 全量化 MLA 双标杆判据。

    标杆构成:
      - cpu 节点(被比对对象 remote): 量化网格参考, 与 kernel 完全相同的 int8 dequant 语义,
        fp32 精算, 代表"量化本身带来的误差";
      - cpu_benchmark 任务(`--bm_device cpu`): 高精度真值, bf16 输入直接 fp32 精算, 不做量化;
      - npu: 被测实现。

    判定: err(npu, 真值) <= RATIO_THD * max(err(量化参考, 真值), THD_FLOOR),
    即 kernel 额外引入的误差不应明显超过量化固有误差; 另加余弦相似度与 max_abs 兜底。
    内建 single_bm 只做纯相对误差(eps=1e-7), 注意力输出大量近零元素会把 MARE 放大到几百,
    不适用于 bf16 注意力。
    """

    COS_THD = 0.999
    COS_MARGIN = 5e-4
    NRMS_THD = 0.01
    MAX_ABS = 1.0
    THD_FLOOR = 1e-3
    RATIO_THD = 1.5

    @staticmethod
    def _metrics(x, ref):
        diff = (x - ref).abs()
        nrms = (
            diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)
        ).item()
        cos = torch.nn.functional.cosine_similarity(x, ref, dim=0).item()
        return nrms, cos, diff.max().item()

    def _load_truth(self, data_file):
        """加载 cpu_benchmark 任务产出的高精度真值; 缺失时返回 None。"""
        if not self.bm_path:
            return None
        candidates = [os.path.join(self.bm_path, data_file)]
        candidates += glob.glob(
            os.path.join(self.bm_path, "**", data_file), recursive=True
        )
        for path in candidates:
            if os.path.isfile(path):
                return torch.load(path, map_location="cpu").to(torch.float32).flatten()
        return None

    def compute_mere_mare(self, local_output, remote_output, data_file):
        npu = local_output.to(torch.float32).flatten()
        ref = remote_output.to(torch.float32).flatten()
        if npu.shape != ref.shape:
            return AccuracyConfig(
                result=False,
                filename=data_file,
                error_info=f"shape mismatch: npu {tuple(local_output.shape)} vs bm {tuple(remote_output.shape)}",
            )
        if not torch.isfinite(npu).all():
            return AccuracyConfig(
                result=False,
                filename=data_file,
                error_info=f"npu {data_file} contains nan/inf",
            )

        nrms_ref, cos_ref, max_abs = self._metrics(npu, ref)
        truth = self._load_truth(data_file)
        if truth is None or truth.shape != npu.shape:
            # 无法取得真值时退化为对量化参考的单标杆判定
            ok = (
                cos_ref >= self.COS_THD
                and nrms_ref <= self.NRMS_THD
                and max_abs <= self.MAX_ABS
            )
            return AccuracyConfig(
                result=ok,
                filename=data_file,
                mean_rel_err=nrms_ref,
                max_rel_err=max_abs,
                error_info=f"[single-bm] cos={cos_ref:.6f} (thd {self.COS_THD}), "
                f"nrms_vs_ref={nrms_ref:.4%} (thd {self.NRMS_THD:.2%}), "
                f"max_abs={max_abs:.5f} (thd {self.MAX_ABS})",
            )

        nrms_npu, cos_npu, max_npu = self._metrics(npu, truth)
        nrms_bm, cos_bm, _ = self._metrics(ref, truth)
        ratio = nrms_npu / max(nrms_bm, self.THD_FLOOR)
        # 余弦门槛同样按双标杆: 参考实现自身在该用例上的余弦更低时, 不能要求被测更高
        cos_thd = min(self.COS_THD, cos_bm - self.COS_MARGIN)
        ok = ratio <= self.RATIO_THD and cos_npu >= cos_thd and max_npu <= self.MAX_ABS
        return AccuracyConfig(
            result=ok,
            filename=data_file,
            mean_rel_err=ratio,
            max_rel_err=max_npu,
            error_info=f"ratio={ratio:.3f} (thd {self.RATIO_THD}): "
            f"npu_nrms={nrms_npu:.4%} / bm_nrms={nrms_bm:.4%} (floor {self.THD_FLOOR}); "
            f"cos_npu={cos_npu:.6f} (thd {cos_thd:.6f}, bm {cos_bm:.6f}), "
            f"max_abs={max_npu:.5f} (thd {self.MAX_ABS}), "
            f"npu_vs_ref_nrms={nrms_ref:.4%} (info only)",
        )


def _quantize(q_bf, q_rope_bf, k_bf, k_rope_bf, batch, kv_len):
    """按 INT8 全量化约定构造量化张量与 NZ cache 排布。"""
    q_scale = (q_bf.abs().amax(dim=-1) / 127.0).clamp_min(1e-6)
    q_int8 = torch.round(q_bf / q_scale[..., None]).clamp(-128, 127).to(torch.int8)
    # rope 与 nope 共用同一 dequant 网格, 以 bf16 承载量化后的整数值
    q_rope_q = torch.round(q_rope_bf / q_scale[..., None]).to(torch.bfloat16)

    k_scale = (k_bf.abs().amax() / 127.0).clamp_min(1e-6)
    k_int8 = torch.round(k_bf / k_scale).clamp(-128, 127).to(torch.int8)
    k_rope_q = torch.round(k_rope_bf / k_scale).to(torch.bfloat16)

    block_num = k_bf.shape[0]
    blocks_per_batch = kv_len // BLOCK
    assert block_num == batch * blocks_per_batch, (block_num, batch, blocks_per_batch)

    k_cache = (
        k_int8.view(block_num, BLOCK, D_NOPE // 32, 32)
        .permute(0, 2, 1, 3)
        .contiguous()
        .unsqueeze(1)
    )
    k_rope_nz = (
        k_rope_q.float()
        .view(block_num, BLOCK, D_ROPE // 16, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .unsqueeze(1)
        .to(torch.bfloat16)
    )
    block_table = torch.arange(block_num, dtype=torch.int32).reshape(
        batch, blocks_per_batch
    )
    cache_seqlens = torch.full((batch,), kv_len, dtype=torch.int64)
    return {
        "q_scale": q_scale.contiguous(),
        "q_int8": q_int8,
        "q_rope_q": q_rope_q,
        "k_scale": k_scale,
        "k_cache": k_cache,
        "k_rope_nz": k_rope_nz,
        "k_int8": k_int8,
        "k_rope_flat": k_rope_q,
        "block_table": block_table,
        "cache_seqlens": cache_seqlens,
    }


def _golden(
    qt, q_bf, q_rope_q, k_int8, k_rope_q, k_scale, block_table, batch, kv_len, layout
):
    """float32 golden: 复现 INT8 量化网格上的 attention。"""
    block_num = k_int8.shape[0]
    qd_int = torch.cat(
        [q_int8_for_golden(q_bf, qt["q_scale"]).float(), q_rope_q.float()], dim=-1
    )
    kd_int = torch.cat(
        [
            k_int8.view(block_num, BLOCK, D_NOPE).float(),
            k_rope_q.view(block_num, BLOCK, D_ROPE).float(),
        ],
        dim=-1,
    )
    v_full = k_int8.view(block_num, BLOCK, D_NOPE).float() * k_scale

    heads = q_bf.shape[-2]
    rows = []
    for b in range(batch):
        ids = block_table[b].tolist()
        ks = torch.cat([kd_int[i] for i in ids], dim=0)[:kv_len]
        vs = torch.cat([v_full[i] for i in ids], dim=0)[:kv_len]
        if layout == "TND":
            idx = slice(
                b * (q_bf.shape[0] // batch), (b + 1) * (q_bf.shape[0] // batch)
            )
            qd = qd_int[idx]
            scale = qt["q_scale"][idx]
        else:
            qd = qd_int[b]
            scale = qt["q_scale"][b]
        scores = torch.einsum("snd,kd->nsk", qd, ks)
        scores = scores * scale.transpose(0, 1)[:, :, None] * k_scale * SCALE
        rows.append(torch.einsum("nsk,kd->snd", torch.softmax(scores, dim=-1), vs))
    golden = torch.cat(rows, dim=0)
    if layout == "TND":
        return golden.reshape(-1, heads, D_NOPE)
    return golden.reshape(batch, -1, heads * D_NOPE)


def q_int8_for_golden(q_bf, q_scale):
    """量化到 int8 的整数值(与 kernel 使用同一网格)。"""
    return torch.round(q_bf / q_scale[..., None]).clamp(-128, 127)


def _truth(q_bf, q_rope_bf, k_bf, k_rope_bf, block_table, batch, kv_len, layout):
    """高精度真值(双标杆第一标杆): bf16 输入直接 fp32 精算, 不做 int8 量化。"""
    qd_all = torch.cat([q_bf.float(), q_rope_bf.float()], dim=-1)
    kd = torch.cat([k_bf.float(), k_rope_bf.float()], dim=-1)

    heads = q_bf.shape[-2]
    rows = []
    for b in range(batch):
        ids = block_table[b].tolist()
        ks = torch.cat([kd[i] for i in ids], dim=0)[:kv_len]
        if layout == "TND":
            seq = q_bf.shape[0] // batch
            qd = qd_all[b * seq : (b + 1) * seq]
        else:
            qd = qd_all[b]
        scores = torch.einsum("snd,kd->nsk", qd, ks) * SCALE
        rows.append(
            torch.einsum("nsk,kd->snd", torch.softmax(scores, dim=-1), ks[..., :D_NOPE])
        )
    truth = torch.cat(rows, dim=0)
    if layout == "TND":
        return truth.reshape(-1, heads, D_NOPE)
    return truth.reshape(batch, -1, heads * D_NOPE)


@register("function_quant_flash_mla")
class QuantFlashMla(BaseApi):
    """quant_mode=2 (MLA_INT8_FULLQUANT) 全量化端到端执行器。"""

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self._prep = None

    def init_by_input_data(self, input_data: InputDataset):
        OpsDataset.seed_everything()
        self._prep = self._prepare(input_data.kwargs)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        prep = (
            self._prep if self._prep is not None else self._prepare(input_data.kwargs)
        )
        if self.task_result.is_benchmark_task:
            # 双标杆: benchmark 任务产出高精度真值, 供 cpu 比较节点定位
            if "truth" not in prep:
                prep["truth"] = _truth(
                    prep["q_bf"],
                    prep["q_rope_bf"],
                    prep["k_bf"],
                    prep["k_rope_bf"],
                    prep["block_table"],
                    prep["batch"],
                    prep["kv_len"],
                    prep["layout"],
                )
            return prep["truth"]
        if self.name == "cpu":
            # cpu 节点产出量化网格参考(按需计算, npu/benchmark 任务不再重复算)
            if "golden" not in prep:
                prep["golden"] = _golden(
                    prep,
                    prep["q_bf"],
                    prep["q_rope_q"],
                    prep["k_int8"],
                    prep["k_rope_flat"],
                    prep["k_scale"],
                    prep["block_table"],
                    prep["batch"],
                    prep["kv_len"],
                    prep["layout"],
                )
            return prep["golden"]
        return self._run_npu(prep, input_data.kwargs)

    @staticmethod
    def _prepare(kw):
        layout = "TND" if int(kw.get("layout", 0)) == 1 else "BSH"
        batch = int(kw["batch"])
        kv_len = int(kw["kv_len"])
        q_bf = kw["q"].detach().float().cpu()
        q_rope_bf = kw["q_rope"].detach().float().cpu()
        k_bf = kw["k"].detach().float().cpu()
        k_rope_bf = kw["k_rope"].detach().float().cpu()

        qt = _quantize(q_bf, q_rope_bf, k_bf, k_rope_bf, batch, kv_len)

        if layout == "TND":
            seq_len = q_bf.shape[0] // batch
            cu_seqlens = torch.arange(0, q_bf.shape[0] + 1, seq_len, dtype=torch.int32)
            max_seqlen_q = q_bf.shape[0]
        else:
            seq_len = q_bf.shape[1]
            cu_seqlens = None
            max_seqlen_q = seq_len

        return {
            "layout": layout,
            "batch": batch,
            "kv_len": kv_len,
            "seq_len": seq_len,
            "max_seqlen_q": max_seqlen_q,
            "num_heads": q_bf.shape[-2],
            "cu_seqlens": cu_seqlens,
            "q_bf": q_bf,
            "q_rope_bf": q_rope_bf,
            "k_bf": k_bf,
            "k_rope_bf": k_rope_bf,
            **qt,
        }

    def _run_npu(self, prep, kw):
        from cann_ops_transformer_custom.ops.attention.quant_flash_mla import (
            quant_flash_mla,
            quant_flash_mla_metadata,
        )

        layout = prep["layout"]
        batch = prep["batch"]
        kv_len = prep["kv_len"]
        heads = prep["num_heads"]
        seq_len = prep["seq_len"]
        max_seqlen_q = prep["max_seqlen_q"]

        cache_seqlens = prep["cache_seqlens"].npu()
        meta_kwargs = dict(
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=kv_len,
            head_dim_qk=D_NOPE,
            mask_mode=0,
            layout_q=layout,
        )
        if layout == "TND":
            cu_seqlens = prep["cu_seqlens"]
            meta_kwargs.update(
                cu_seqlens_q=cu_seqlens.npu(),
                seqused_q=torch.full((batch,), seq_len, dtype=torch.int32).npu(),
            )
        meta = quant_flash_mla_metadata(
            cache_seqlens, heads, 1, QUANT_MODE_INT8, **meta_kwargs
        )

        # BSH 对外契约 q 为 (B, S, N*D) 3 维, q_rope 为 (B, S, N, 64); TND 为 (T, N, D)
        if layout == "BSH":
            q_in = prep["q_int8"].reshape(batch, seq_len, heads * D_NOPE).contiguous()
            q_rope_in = prep["q_rope_q"]
        else:
            q_in = prep["q_int8"]
            q_rope_in = prep["q_rope_q"]

        out, _ = quant_flash_mla(
            q_in.npu(),
            prep["k_cache"].npu(),
            prep["q_scale"].npu(),
            prep["k_scale"].view(1).npu(),
            prep["block_table"].npu(),
            cache_seqlens,
            QUANT_MODE_INT8,
            metadata=meta,
            q_rope=q_rope_in.npu(),
            k_rope=prep["k_rope_nz"].npu(),
            cu_seqlens_q=prep["cu_seqlens"].npu() if layout == "TND" else None,
            softmax_scale=SCALE,
            mask_mode=0,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=kv_len,
            layout_q=layout,
            layout_kv="PA_NZ",
            layout_out=layout,
        )
        return out
