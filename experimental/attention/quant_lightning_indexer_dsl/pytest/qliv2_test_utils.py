# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from __future__ import annotations

import re
from pathlib import Path

PARAM_NAMES = (
    "batch_size",
    "q_seq",
    "k_seq",
    "q_t_size",
    "k_t_size",
    "q_head_num",
    "k_head_num",
    "head_dim",
    "block_size",
    "block_num",
    "qk_dtype",
    "weight_dtype",
    "dequant_dtype",
    "actual_seq_dtype",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "seqused_q",
    "seqused_k",
    "cmp_residual_k",
    "block_table",
    "max_seqlen_q",
    "quant_mode",
    "layout_query",
    "layout_key",
    "sparse_count",
    "sparse_mode",
    "query_datarange",
    "key_datarange",
    "weights_datarange",
    "q_scale_datarange",
    "k_scale_datarange",
    "cmp_ratio",
    "return_value",
    "output_idx_offset",
    "candidate_topk_blocks",
    "candidate_block_size",
    "storage_layout",
    "seed",
)


def ensure_comparison_passed(
    case_name,
    result,
    fulfill_percent,
    result_return_value="N/A",
    fulfill_percent_return_value=0,
):
    failures = []
    if result != "Pass":
        failures.append(f"index result={result}, fulfill_percent={fulfill_percent}")
    if result_return_value not in ("N/A", "Pass"):
        failures.append(
            "value result="
            f"{result_return_value}, fulfill_percent={fulfill_percent_return_value}"
        )
    if failures:
        raise AssertionError(
            f"accuracy comparison failed for {case_name}: " + "; ".join(failures)
        )


class QliV2CaseSelector:
    @staticmethod
    def natural_key(path):
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", Path(path).name)
        ]

    @staticmethod
    def parse_indexes(expression, total):
        indexes = []
        for token in str(expression or "").split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError(f"invalid descending case index range: {token}")
                indexes.extend(range(start, end + 1))
            else:
                indexes.append(int(token))
        invalid = [index for index in indexes if index < 1 or index > total]
        if invalid:
            raise ValueError(f"case indexes out of range 1..{total}: {invalid}")
        return indexes

    @classmethod
    def resolve(cls, pt_dir, explicit_files="", case_names="", case_indexes=""):
        if explicit_files:
            candidates = [
                Path(item.strip()) for item in explicit_files.split(",") if item.strip()
            ]
        else:
            directory = Path(pt_dir)
            if not directory.is_dir():
                raise ValueError(f"PT directory does not exist: {directory}")
            candidates = sorted(directory.glob("*.pt"), key=cls.natural_key)
        missing = [str(path) for path in candidates if not path.is_file()]
        if missing:
            raise ValueError(f"PT files do not exist: {missing}")
        if not candidates:
            raise ValueError(f"no PT cases found in: {pt_dir}")
        if case_names and case_indexes:
            raise ValueError("case names and case indexes cannot be specified together")
        if case_names:
            by_name = {path.stem: path for path in candidates}
            names = [
                Path(item.strip()).stem
                for item in case_names.split(",")
                if item.strip()
            ]
            unknown = [name for name in names if name not in by_name]
            if unknown:
                raise ValueError(f"unknown case names: {unknown}")
            candidates = [by_name[name] for name in names]
        elif case_indexes:
            candidates = [
                candidates[index - 1]
                for index in cls.parse_indexes(case_indexes, len(candidates))
            ]
        return [str(path) for path in candidates]


class QliV2ResultWriter:
    @staticmethod
    def case_name(params, explicit_name=None):
        if explicit_name:
            normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(explicit_name))
            normalized = normalized.strip("._-")
            if not normalized:
                raise ValueError("case name has no usable filename characters")
            return normalized
        values = dict(zip(PARAM_NAMES, params))
        return (
            f"QLI_B{values['batch_size']}_S1{values['q_seq']}_S2{values['k_seq']}_"
            f"R{values['cmp_ratio']}_{values['storage_layout']}"
        )

    @staticmethod
    def row(
        case_name,
        params,
        result,
        fulfill_percent,
        result_return_value="N/A",
        fulfill_percent_return_value=0,
    ):
        if len(params) != len(PARAM_NAMES):
            raise ValueError(
                f"QLI V2 parameter count mismatch: got {len(params)}, "
                f"expected {len(PARAM_NAMES)}"
            )
        row = {"case_name": case_name}
        row.update(dict(zip(PARAM_NAMES, params)))
        row.update(
            {
                "result": result,
                "fulfill_percent": fulfill_percent,
                "result_return_value": result_return_value,
                "fulfill_percent_return_value": fulfill_percent_return_value,
            }
        )
        return row

    @staticmethod
    def append(path, row):
        import pandas as pd

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.read_excel(output) if output.exists() else pd.DataFrame()
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        frame.to_excel(output, index=False)
