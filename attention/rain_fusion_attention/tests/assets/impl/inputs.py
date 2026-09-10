#!/usr/bin/python
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

"""RFA (RainFusionAttention) customize_inputs — 从 ATK init_by_input_data 移植

参数签名与 aclnnRainFusionAttentionGetWorkspaceSize 一致（含输出 tensor 占位）。
in-place 修改 tensor，不返回值。
"""

import inspect as _inspect
import os as _os
import random

import numpy as np
import torch


# nd 分布哨兵范围 → (mean, std) 映射
_ND_SENTINEL_MAP = {
    (-100, 100): (0.0, 1.0),
    (-101, 101): (1.0, 1.0),
    (-102, 102): (0.0, 0.001),
}
_ND_TENSOR_NAMES = ["query", "key", "value"]


def _is_nd_sentinel(rv):
    """检测 input_data_ranges 中的哨兵范围"""
    if rv is None or not isinstance(rv, (tuple, list)):
        return None
    if len(rv) != 2:
        return None
    return _ND_SENTINEL_MAP.get((rv[0], rv[1]))


def _gen_nd_tensor(case_id, shape, dtype, mean, std):
    """用与 ATK 完全相同的逻辑生成正态分布 tensor

    RFA 的 nd 参数为固定值（非范围），直接使用 mean/std 生成。
    """
    data = torch.normal(mean, std, tuple(shape)).to(dtype=dtype)
    return data


def _parse_case_id(testcase_name):
    """从 testcase_name 中解析用例 id"""
    try:
        parts = testcase_name.rsplit("_", 1)
        if len(parts) == 2:
            return int(parts[1])
    except (ValueError, IndexError):
        pass
    return 0


def _gen_select_idx_data(
    q_seqlen_list, kv_seqlen_list, s_block_x, s_block_y, batch, num_heads, select_ratio
):
    """生成 selectIdx 数据，与 ATK gen_select_idx_data 等价"""
    select_idx_list = []
    select_num_idx_list = []

    total_q_blocks = 0
    max_kv_block_num = 0

    for b in range(batch):
        q_seqlen = q_seqlen_list[b]
        kv_seqlen = kv_seqlen_list[b]
        s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
        s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y
        total_q_blocks += s_block_num_q
        max_kv_block_num = max(max_kv_block_num, s_block_num_kv)

    q_block_offset = 0
    for b in range(batch):
        q_seqlen = q_seqlen_list[b]
        kv_seqlen = kv_seqlen_list[b]
        s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
        s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y

        for t_local in range(s_block_num_q):
            q_block_idx = t_local
            for head in range(num_heads):
                selected_kv_blocks = []
                if s_block_num_kv > 0:
                    random.seed(head * 1000 + q_block_idx)
                    if select_ratio == 1:
                        select_ratio = np.random.random()
                    num_select = max(1, int(s_block_num_kv * select_ratio))
                    selected_kv_blocks = random.sample(
                        range(s_block_num_kv), num_select
                    )
                else:
                    selected_kv_blocks = []
                selected_kv_blocks.sort()
                padded_blocks = selected_kv_blocks + [-1] * (
                    max_kv_block_num - len(selected_kv_blocks)
                )
                batch_select_idx = padded_blocks
                batch_select_num = len(selected_kv_blocks)
                select_idx_list.extend(batch_select_idx)
                select_num_idx_list.append(batch_select_num)

        q_block_offset += s_block_num_q

    return select_idx_list, select_num_idx_list, total_q_blocks, max_kv_block_num


def customize_inputs(
    query,
    key,
    value,
    selectIdx,
    selectNumIdx,
    blockShape,
    attenMaskOptional,
    actualSeqLengthsOptional,
    actualSeqLengthsKvOptional,
    blockTableOptional,
    qInputLayout,
    kvInputLayout,
    numKeyValueHeads,
    maskType,
    scaleValue,
    innerPrecise,
    blockSize,
    attentionOut,
    softmaxLseOptional,
    **kwargs,
):
    """定制输入修正

    参数签名与 aclnnRainFusionAttentionGetWorkspaceSize 一致:
      inputs: query, key, value, selectIdx, selectNumIdx, blockShape,
              attenMaskOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional,
              blockTableOptional
      attr:   qInputLayout, kvInputLayout, numKeyValueHeads, maskType,
              scaleValue (double), innerPrecise, blockSize
      outputs: attentionOut, softmaxLseOptional (占位，不处理)

    TTK 已生成随机 tensor，此函数做原地确定性修正。
    必须 in-place 修改（用 copy_()），不返回值。

    **kwargs: testcase_name, tensor_dtypes, tensor_formats, scalar_dtypes,
              use_torch, short_soc_version
    """
    testcase_name = kwargs.get("testcase_name", "")
    case_id = _parse_case_id(testcase_name)
    print(f"[CUSTOMIZE-RFA] case_id={case_id}, scaleValue={scaleValue}", flush=True)

    # scaleValue 随机参数处理: 如果是范围值 [min, max], 用 case_id 作为 seed 生成
    _caller = _inspect.currentframe().f_back
    _self_obj = _caller.f_locals.get("self") if _caller is not None else None
    _ctx_obj = getattr(_self_obj, "_ctx", None) if _self_obj is not None else None
    if _ctx_obj is not None and hasattr(_ctx_obj, "attributes"):
        _sv = _ctx_obj.attributes.get("scaleValue")
        if isinstance(_sv, (list, tuple)) and len(_sv) == 2:
            _rng = np.random.RandomState(case_id)
            _new_sv = float(_rng.uniform(_sv[0], _sv[1]))
            _ctx_obj.attributes["scaleValue"] = _new_sv
            print(
                f"[SCALAR-GEN-RFA] case_id={case_id}, scaleValue: {_sv} -> {_new_sv:.6f}",
                flush=True,
            )

    # selectIdx 重生成逻辑: 从 ATK .pt 文件加载或用确定性 seed 重新生成
    # 环境变量 RFA_USE_ATK_PT=1 时才走 .pt 加载分支，默认关闭
    _use_atk_pt = _os.environ.get("RFA_USE_ATK_PT", "0") == "1"
    atk_pt_path = _os.path.join(
        "/home/j00946653/ATK2TTK/temp/atk_golden", f"case_{case_id}.pt"
    )
    if not _use_atk_pt or not _os.path.exists(atk_pt_path):
        # ATK .pt 不存在，走原始 selectIdx 重生成逻辑
        np.random.seed(10)
        if selectIdx is None or selectNumIdx is None:
            return
        select_idx_value = selectIdx.cpu()
        q_seqlen_list = (
            list(actualSeqLengthsOptional)
            if actualSeqLengthsOptional is not None
            else []
        )
        kv_seqlen_list = (
            list(actualSeqLengthsKvOptional)
            if actualSeqLengthsKvOptional is not None
            else []
        )
        batch = len(q_seqlen_list)
        total_q_blocks = select_idx_value.shape[0]
        head_num = select_idx_value.shape[1]
        max_kv_block_num = select_idx_value.shape[2]
        select_num_value = selectNumIdx[0][0]
        sparsity_ratio = min(select_num_value.item() / 100.0, 1)
        select_idx_list, select_num_idx_list, _, _ = _gen_select_idx_data(
            q_seqlen_list,
            kv_seqlen_list,
            blockShape[0],
            blockShape[1],
            batch,
            head_num,
            1 - sparsity_ratio,
        )
        new_select_idx = torch.tensor(select_idx_list).view(
            total_q_blocks, head_num, max_kv_block_num
        )
        new_select_num_idx = torch.tensor(select_num_idx_list).view(
            total_q_blocks, head_num
        )
        selectIdx.copy_(new_select_idx.to(selectIdx.dtype).to(selectIdx.device))
        selectNumIdx.copy_(
            new_select_num_idx.to(selectNumIdx.dtype).to(selectNumIdx.device)
        )

        # nd 分布输入修正: 检测哨兵范围并重新生成正态分布数据
        if _ctx_obj is not None:
            input_ranges = getattr(_ctx_obj, "flat_input_data_ranges", None)
            flat_shapes = getattr(_ctx_obj, "flat_tensor_view_shapes", None)
            if input_ranges is not None and flat_shapes is not None:
                nd_tensors = [query, key, value]
                for idx, tensor in enumerate(nd_tensors):
                    if tensor is None or idx >= len(input_ranges):
                        continue
                    nd_params = _is_nd_sentinel(input_ranges[idx])
                    if nd_params is None:
                        continue
                    mean, std = nd_params
                    shape = (
                        flat_shapes[idx]
                        if idx < len(flat_shapes)
                        else list(tensor.shape)
                    )
                    orig_dtype = tensor.dtype
                    new_data = _gen_nd_tensor(case_id, shape, orig_dtype, mean, std)
                    tensor.copy_(new_data)
                    print(
                        f"[ND-GEN-RFA] case_id={case_id}, tensor_idx={idx} "
                        f"({_ND_TENSOR_NAMES[idx]}), mean={mean}, std={std}, "
                        f"shape={shape}, dtype={orig_dtype}",
                        flush=True,
                    )
        return

    # ATK .pt 存在：加载并覆写所有入参
    atk_data = torch.load(atk_pt_path, weights_only=False)

    # selectIdx shape 不匹配时，回退到 _gen_select_idx_data 重生成
    _need_regen_select = False
    if selectIdx is not None and atk_data.get("selectIdx") is not None:
        atk_si = atk_data["selectIdx"].to(selectIdx.dtype)
        if atk_si.shape == selectIdx.shape:
            selectIdx.copy_(atk_si.to(selectIdx.device))
        else:
            _need_regen_select = True
    else:
        _need_regen_select = True

    if _need_regen_select and selectIdx is not None and selectNumIdx is not None:
        np.random.seed(10)
        select_idx_value = selectIdx.cpu()
        q_seqlen_list = (
            list(actualSeqLengthsOptional)
            if actualSeqLengthsOptional is not None
            else []
        )
        kv_seqlen_list = (
            list(actualSeqLengthsKvOptional)
            if actualSeqLengthsKvOptional is not None
            else []
        )
        batch = len(q_seqlen_list)
        total_q_blocks = select_idx_value.shape[0]
        head_num = select_idx_value.shape[1]
        max_kv_block_num = select_idx_value.shape[2]
        select_num_value = selectNumIdx[0][0]
        sparsity_ratio = min(select_num_value.item() / 100.0, 1)
        select_idx_list, select_num_idx_list, _, _ = _gen_select_idx_data(
            q_seqlen_list,
            kv_seqlen_list,
            blockShape[0],
            blockShape[1],
            batch,
            head_num,
            1 - sparsity_ratio,
        )
        new_select_idx = torch.tensor(select_idx_list).view(
            total_q_blocks, head_num, max_kv_block_num
        )
        new_select_num_idx = torch.tensor(select_num_idx_list).view(
            total_q_blocks, head_num
        )
        selectIdx.copy_(new_select_idx.to(selectIdx.dtype).to(selectIdx.device))
        selectNumIdx.copy_(
            new_select_num_idx.to(selectNumIdx.dtype).to(selectNumIdx.device)
        )
        print(
            f"[SELECT-REGEN-RFA] case_id={case_id}, pt shape mismatch, "
            f"regenerated selectIdx: {selectIdx.shape}",
            flush=True,
        )
    elif selectNumIdx is not None and atk_data.get("selectNumIdx") is not None:
        atk_sni = atk_data["selectNumIdx"].to(selectNumIdx.dtype)
        if atk_sni.shape == selectNumIdx.shape:
            selectNumIdx.copy_(atk_sni.to(selectNumIdx.device))

    if query is not None and atk_data.get("query") is not None:
        atk_q = atk_data["query"].to(query.dtype)
        if atk_q.shape == query.shape:
            query.copy_(atk_q.to(query.device))
    if key is not None and atk_data.get("key") is not None:
        atk_k = atk_data["key"].to(key.dtype)
        if atk_k.shape == key.shape:
            key.copy_(atk_k.to(key.device))
    if value is not None and atk_data.get("value") is not None:
        atk_v = atk_data["value"].to(value.dtype)
        if atk_v.shape == value.shape:
            value.copy_(atk_v.to(value.device))

    # 输出 tensor (attentionOut, softmaxLseOptional) 是占位参数，不处理
    # 不返回任何值
