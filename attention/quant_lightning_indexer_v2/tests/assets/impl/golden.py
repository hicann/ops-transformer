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

"""CPU golden adapter for QuantLightningIndexer V2 TTK cases."""

import torch


class CaseDataStore:
    """Share one pytest-generated case between input, golden, and compare callbacks."""

    def __init__(self):
        self.case_data = {}
        self.active_testcase_name = None

    def clear(self):
        self.case_data.clear()
        self.active_testcase_name = None

    def put(self, testcase_name, data):
        if testcase_name is not None:
            self.clear()
            self.case_data = {str(testcase_name): data}

    def activate(self, testcase_name):
        data = self.case_data.get(str(testcase_name)) if testcase_name is not None else None
        if data is None:
            raise RuntimeError(
                "QuantLightningIndexer V2 TTK golden requires customize_inputs "
                "to generate pytest data first"
            )
        self.active_testcase_name = str(testcase_name)
        return data


CASE_DATA = CaseDataStore()


def get_compare_data(testcase_name):
    """Return pytest comparison context for the active or replayed case."""
    if testcase_name is None:
        return None
    return CASE_DATA.case_data.get(str(testcase_name))


def set_compare_data(testcase_name, data):
    CASE_DATA.active_testcase_name = str(testcase_name)
    CASE_DATA.case_data = {str(testcase_name): data}


def cpu_quant_lightning_indexer_v2(query, key, weights, query_dequant_scale,
                                   key_dequant_scale, *, return_value=0,
                                   testcase_name=None, **kwargs):
    """Return the CPU outputs produced while generating this exact pytest case."""
    del query, key, weights, query_dequant_scale, key_dequant_scale, kwargs
    data = CASE_DATA.activate(testcase_name)
    if int(return_value):
        sparse_value = data["cpu_topk_value"]
    else:
        sparse_value = torch.zeros(0, dtype=data["topk_value"].dtype)
    return data["cpu_result"], sparse_value
