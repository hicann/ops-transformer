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

from recurrent_gated_delta_rule_golden import (
    run_recurrent_gated_delta_rule_eager,
    cpu_recurrent_gated_delta_rule,
)


class Generalized_operator:
    def forward(
        self,
        q,
        k,
        v,
        state,
        beta,
        scale_value,
        act_seq_len,
        ssm_state_indices,
        num_accepted_tokens=None,
        g=None,
        gk=None,
    ):
        return cpu_recurrent_gated_delta_rule(
            q,
            k,
            v,
            state,
            beta,
            scale_value,
            act_seq_len,
            ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            g=g,
            gk=gk,
        )


def output_operator(params):
    # 构造输入
    (
        batch_size,
        mtp,
        nk,
        nv,
        dk,
        dv,
        actual_seq_lengths,
        ssm_state_indices,
        has_gamma,
        has_gamma_k,
        has_num_accepted_tokens,
        scale_value,
        num_accepted_tokens,
        block_num,
        data_type,
        state_data_type,
        query_datarange,
        key_datarange,
        value_datarange,
        gamma_datarange,
        gamma_k_datarange,
        beta_datarange,
        state_datarange,
        state_non_contiguous,
    ) = params

    logger.info(f"params = {params}")

    run_recurrent_gated_delta_rule_eager(
        batch_size,
        mtp,
        nk,
        nv,
        dk,
        dv,
        actual_seq_lengths,
        ssm_state_indices,
        has_gamma=has_gamma,
        has_gamma_k=has_gamma_k,
        has_num_accepted_tokens=has_num_accepted_tokens,
        scale_value=scale_value,
        num_accepted_tokens=num_accepted_tokens,
        block_num=block_num,
        data_type=data_type,
        state_data_type=state_data_type,
        query_datarange=query_datarange,
        key_datarange=key_datarange,
        value_datarange=value_datarange,
        gamma_datarange=gamma_datarange,
        gamma_k_datarange=gamma_k_datarange,
        beta_datarange=beta_datarange,
        state_datarange=state_datarange,
        state_non_contiguous=state_non_contiguous,
    )
