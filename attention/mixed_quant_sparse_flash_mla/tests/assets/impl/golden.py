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

"""CPU golden adapter for MixedQuantSparseFlashMla TTK cases."""


class CaseDataStore:
    """Share pytest-generated case data between input and golden callbacks."""

    def __init__(self):
        self.case_data = {}

    def put(self, testcase_name, data):
        if testcase_name is not None:
            self.case_data[str(testcase_name)] = data

    def get(self, testcase_name):
        if testcase_name is None:
            return None
        return self.case_data.get(str(testcase_name))


CASE_DATA = CaseDataStore()


def cpu_mixed_quant_sparse_flash_mla(q, *, return_softmax_lse=False,
                                     testcase_name=None, **kwargs):
    del q, kwargs
    data = CASE_DATA.get(testcase_name)
    if data is None:
        raise RuntimeError(
            "MixedQuantSparseFlashMla TTK golden requires customize_inputs "
            "to generate pytest data first"
        )
    softmax_lse = data.get("cpu_lse") if bool(return_softmax_lse) else None
    return data["cpu_output"], softmax_lse
