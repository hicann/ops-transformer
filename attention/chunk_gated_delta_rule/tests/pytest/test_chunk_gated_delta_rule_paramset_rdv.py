# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

DEFAULT_CHUNK_SIZE = 64


# 用例
# name, B, seqlen, Nv, Nk, Dv, Dk, has_g, scale, data_type, state_data_type, is_contiguous
# fmt: off
A5_REDLINE_CASES = [
    ("ARC-001", 1, 2048, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-002", 1, 4096, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-003", 1, 8192, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-004", 2, 2048, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-005", 4, 2048, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-006", 1, 32768, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-007", 1, 65536, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-008", 1, 131072, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-009", 1, 262144, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-010", 1, 2048, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-011", 1, 2048, 2, 2, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-012", 1, 2048, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-013", 1, 2048, 16, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-014", 1, 4096, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-015", 1, 8192, 16, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-016", 32, 8000, 8, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-017", 1, 6000, 8, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-018", 80, 5000, 16, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-019", 200, 256, 8, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-020", 1, 10240, 8, 2, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-021", 1, 10240, 16, 4, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-022", 1, 10240, 32, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-023", 1, 10240, 64, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-024", 1, 32768, 8, 2, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-025", 1, 32768, 16, 4, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-026", 1, 32768, 32, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-027", 1, 32768, 64, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-028", 4, 10240, 4, 2, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-029", 4, 10240, 8, 4, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-030", 4, 10240, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-031", 1, 32768, 4, 2, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-032", 1, 32768, 8, 4, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-033", 1, 32768, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-034", 1, 65535, 4, 2, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-035", 1, 65535, 8, 4, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-036", 1, 65535, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-037", 1, 32768, 48, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ARC-038", 1, 32768, 16, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
]
# fmt: on

# fmt: off
A5_STC_CASES = [
    ("ASC-001", 2, 4096, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-002", 4, 8192, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-003", 1, 32768, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-004", 2, 32768, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-005", 3, 32768, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-006", 4, 32768, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-007", 1, 65536, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-008", 2, 65536, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-009", 3, 65536, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-010", 4, 65536, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-011", 1, 131072, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-012", 1, 262144, 32, 32, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-013", 1, 2048, 16, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-014", 1, 2048, 2, 2, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-015", 1, 2048, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-016", 1, 2048, 16, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-017", 1, 4096, 32, 16, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-018", 1, 8192, 16, 8, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-019", 1, 128, 32, 32, 128, 64, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-020", 4, 512, 8, 8, 64, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-021", 2, 1024, 32, 4, 64, 64, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-022", 1, 4096, 64, 1, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-023", 3, 10000, 64, 16, 50, 50, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-024", 3, 10000, 32, 16, 8, 5, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-025", 2, 300, 8, 8, 64, 64, False, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-026", 4, 512, 8, 8, 1, 1, False, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-027", 4, 1024, 8, 8, 64, 64, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-028", 4, 512, 8, 4, 64, 64, False, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-029", 4, 512, 16, 4, 64, 64, False, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-030", 8, 4096, 64, 64, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-031", 4, 128, 16, 8, 31, 31, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-032", 8, 128, 2, 2, 16, 16, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-033", 1, 128, 3, 3, 128, 128, True, 1.0, torch.bfloat16, torch.bfloat16, True),
    ("ASC-034", 4, 2000, 2, 2, 16, 16, True, 1.0, torch.bfloat16, torch.bfloat16, True),
]
# fmt: on


def _gen_fp32_cases(cases, suffix="-FP32"):
    return [
        (name + suffix, B, s, Nv, Nk, Dv, Dk, g, scale, dt, torch.float32, ic)
        for name, B, s, Nv, Nk, Dv, Dk, g, scale, dt, _, ic in cases
    ]


def _gen_noncontiguous_cases(cases, suffix="-NC"):
    return [
        (name + suffix, B, s, Nv, Nk, Dv, Dk, g, scale, dt, sdt, False)
        for name, B, s, Nv, Nk, Dv, Dk, g, scale, dt, sdt, _ in cases
    ]


def _expand(cases):
    cases = cases + _gen_fp32_cases(cases)
    return cases + _gen_noncontiguous_cases(cases)


def _convert_cases(cases):
    result = []
    for (
        _name,
        B,
        seqlen,
        Nv,
        Nk,
        Dv,
        Dk,
        has_g,
        _scale,
        data_type,
        state_dtype,
        is_contiguous,
    ) in cases:
        result.append(
            {
                "_name": [_name],
                "B": [B],
                "seqlen": [list(seqlen)]
                if isinstance(seqlen, (list, tuple))
                else [seqlen],
                "nk": [Nk],
                "nv": [Nv],
                "dk": [Dk],
                "dv": [Dv],
                "chunk_size": [DEFAULT_CHUNK_SIZE],
                "data_type": [data_type],
                "state_data_type": [state_dtype],
                "has_g": [has_g],
                "is_contiguous": [is_contiguous],
                "pt_path": [""],
                "query_datarange": [[0, 1]],
                "key_datarange": [[0, 1]],
                "value_datarange": [[0, 1]],
                "gamma_datarange": [[-1, 0]],
                "beta_datarange": [[0, 1]],
                "state_datarange": [[0, 1]],
            }
        )
    return result


GROUP_REDLINE = _convert_cases(_expand(A5_REDLINE_CASES))
GROUP_STC = _convert_cases(_expand(A5_STC_CASES))

ENABLED_PARAMS_RDV = GROUP_REDLINE + GROUP_STC

# 如需为特定用例指定 pt 路径，在此按用例名注入，例如:
# _PT_PATH_MAP = {
#     "ARC-001": "output/pt/xxx.pt",
#     "ASC-025-NC": "output/pt/yyy.pt",
# }
# for _params in ENABLED_PARAMS_RDV:
#     _n = _params["_name"][0]
#     if _n in _PT_PATH_MAP:
#         _params["pt_path"] = [_PT_PATH_MAP[_n]]
