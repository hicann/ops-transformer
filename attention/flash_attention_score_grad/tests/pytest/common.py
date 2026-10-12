#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import datetime

np.random.seed(0)
gtype = torch.float32
PER_BLOCK_SIZE = 128
PER_VCORE_BLOCK_SIZE = 64
EPSILON = 1e-8
DEVICE_ID = 0
pta_mode = "only_grad"


@dataclass
class CalculationContext:
    input_case: Dict[str, Any]

    cu_seqlens_q_npu: Optional[torch.Tensor] = None
    cu_seqlens_kv_npu: Optional[torch.Tensor] = None
    x_max_npu: Optional[torch.Tensor] = None
    x_sum_npu: Optional[torch.Tensor] = None
    q_npu: Optional[torch.Tensor] = None
    k_npu: Optional[torch.Tensor] = None
    v_npu: Optional[torch.Tensor] = None
    q_rope_npu: Optional[torch.Tensor] = None
    k_rope_npu: Optional[torch.Tensor] = None
    dx_npu: Optional[torch.Tensor] = None
    atten_masks_npu: Optional[torch.Tensor] = None
    prefix_npu: Optional[torch.Tensor] = None
    pse_npu: Optional[torch.Tensor] = None
    dscale_q_npu: Optional[torch.Tensor] = None
    dscale_k_npu: Optional[torch.Tensor] = None
    dscale_v_npu: Optional[torch.Tensor] = None
    dscale_dx_npu: Optional[torch.Tensor] = None
    dscale_o_npu: Optional[torch.Tensor] = None
    out_npu: Optional[torch.Tensor] = None


def log(text):
    print(text)
    log_file = open("run_log.txt", "a")
    log_file.write(str(datetime.datetime.now()) + "  ")
    log_file.write(text)
    log_file.write("\n")
    log_file.close()


def print_log(data=None):
    """
    print log
    :param data: value to print
    :param level: log level, include DEBUG, INFO(default), WARN and ERROR
    :return:
    """
    level = "INFO"
    log_info = "%s - [%s]-%s:%s - %s" % (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
        level,
        os.path.basename(sys._getframe().f_back.f_code.co_filename),
        str(sys._getframe().f_back.f_lineno).zfill(4),
        data,
    )
    log(log_info)
