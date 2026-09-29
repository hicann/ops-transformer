# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import torch

import quant_lightning_indexer_v2_golden


def test_qliv2_process(filepath, device_id=0):
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    cpu_result, npu_result = quant_lightning_indexer_v2_golden.run_qliv2_case_data(
        data, device_id=device_id
    )
    return (
        cpu_result,
        npu_result,
        cpu_result.get("_reference_scores"),
        cpu_result.get("sparse_values"),
        npu_result.get("sparse_values"),
        None,
        data.params,
    )


def test_qliv2_process_graph(filepath, device_id=0):
    raise NotImplementedError(
        "DSL QLI transfer suite does not expose an ACL graph entry"
    )
