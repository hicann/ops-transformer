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

logger = logging.getLogger(__name__)

from chunk_gated_delta_rule_main import run_chunk_gated_delta_rule_eager


def run_precision_test(params, pt_path=""):
    # 解包参数
    (
        B,
        seqlen,
        nk,
        nv,
        dk,
        dv,
        chunk_size,
        data_type,
        state_data_type,
        has_g,
        is_contiguous,
    ) = params
    logger.info(f"params = {params}")
    ret = run_chunk_gated_delta_rule_eager(
        B,
        seqlen,
        nk,
        nv,
        dk,
        dv,
        chunk_size=chunk_size,
        data_type=data_type,
        state_data_type=state_data_type,
        has_g=has_g,
        is_contiguous=is_contiguous,
        pt_path=pt_path,
    )
    assert ret, "precision check failed"
