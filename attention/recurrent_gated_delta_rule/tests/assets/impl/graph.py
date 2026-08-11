#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Custom torch.nn.Module for RecurrentGatedDeltaRule graph mode (torch.compile).

Explicitly returns (output, state) so torch.compile can trace the in-place state
modification through the graph. Without this, torchair backend may not connect the
input state to the in-place output, producing incorrect results.
"""

import torch
import torch_npu


class RgdrGraphModule(torch.nn.Module):
    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale

    def forward(
        self,
        query,
        key,
        value,
        state,
        beta=None,
        actual_seq_lengths=None,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        g=None,
        gk=None,
    ):
        output = torch_npu.npu_recurrent_gated_delta_rule(
            query,
            key,
            value,
            state,
            beta=beta,
            scale=self.scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            g=g,
            gk=gk,
        )
        return output, state
