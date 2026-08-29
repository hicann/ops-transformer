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

"""Golden for aclnnQuantGroupedMatMulAlltoAllv.

Quant grouped matmul + alltoall: uses shared GMM golden logic.
"""

import importlib.util
from pathlib import Path


def _load_shared_utils():
    utils_path = (
        Path(__file__).resolve().parents[4]
        / "common"
        / "tests"
        / "assets"
        / "golden_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"mc2_common_golden_utils_{abs(hash(utils_path))}", utils_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quant_grouped_mat_mul_allto_allv_multi_device_golden(
    thread_contexts, device_ids, all_precision
):
    utils = _load_shared_utils()
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    ep_ws = int(attrs.get("epWorldSize", len(device_ids)))
    exp_per_card = int(attrs.get("expPerCard", 1))
    exp_token_nums = attrs.get("expTokenNums", None)

    if exp_token_nums is not None:
        rank_goldens = utils.golden_gmm_alltoallv(
            thread_contexts, device_ids, exp_token_nums, ep_ws, exp_per_card
        )
    else:
        rank_goldens = utils.golden_alltoallv_gmm(
            thread_contexts, device_ids, None, ep_ws, exp_per_card
        )

    # Cascade third_party
    rank_third_parties = None
    try:
        from ttk.core_modules.npu.op_api.hccl_cascade import run_gmm_alltoallv_cascade

        trans_gmm_weight = bool(attrs.get("transGmmWeight", False))
        trans_mm_weight = bool(attrs.get("transMmWeight", False))
        mm_out_flag = rank_goldens.get(device_ids[0], {}).get("mm") is not None
        if exp_token_nums is not None:
            cascade_outs = run_gmm_alltoallv_cascade(
                thread_contexts,
                device_ids,
                exp_token_nums,
                ep_ws,
                exp_per_card,
                trans_gmm_weight=trans_gmm_weight,
                trans_mm_weight=trans_mm_weight,
                mm_out_flag=mm_out_flag,
            )
        else:
            from ttk.core_modules.npu.op_api.hccl_cascade import (
                run_alltoallv_gmm_cascade,
            )

            permute_out_flag = bool(attrs.get("permuteOutFlag", False))
            cascade_outs = run_alltoallv_gmm_cascade(
                thread_contexts,
                device_ids,
                exp_token_nums,
                ep_ws,
                exp_per_card,
                trans_gmm_weight=trans_gmm_weight,
                trans_mm_weight=trans_mm_weight,
                permute_out_flag=permute_out_flag,
                mm_out_flag=mm_out_flag,
            )
        rank_third_parties = {}
        for did in device_ids:
            tp_list = [cascade_outs[did]["main"]]
            out_idxs = thread_contexts[did].output_tensor_indexes
            for oi in range(1, len(out_idxs)):
                if oi == 1:
                    if cascade_outs[did].get("mm") is not None:
                        tp_list.append(cascade_outs[did]["mm"])
                    elif cascade_outs[did].get("permute") is not None:
                        tp_list.append(cascade_outs[did]["permute"])
                    else:
                        tp_list.append(None)
                elif oi == 2:
                    tp_list.append(cascade_outs[did].get("permute"))
                else:
                    tp_list.append(None)
            rank_third_parties[did] = tp_list
    except Exception:
        import logging

        logging.exception("QuantGroupedMatMulAlltoAllv: cascade failed")
        rank_third_parties = None

    utils.apply_gmm_goldens(
        thread_contexts,
        device_ids,
        rank_goldens,
        all_precision,
        rank_third_parties=rank_third_parties,
    )


def quant_grouped_mat_mul_allto_allv_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    return [None] * len(cpu_inputs_per_rank)
