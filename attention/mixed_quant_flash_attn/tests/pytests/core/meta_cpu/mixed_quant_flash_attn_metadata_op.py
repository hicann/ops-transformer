#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import json
import os
import subprocess
from typing import Optional, List


class MixedQuantFlashAttnMetadataOp:
    def __init__(
        self,
        binary_name: str = "mixed_quant_flash_attn_metadata_test",
    ):
        work_dir = os.path.dirname(os.path.abspath(__file__))
        self.work_dir = work_dir
        self.binary_path = os.path.join(work_dir, binary_name)
        self.case_json_path = os.path.join(
            work_dir, "mixed_quant_flash_attn_metadata_case.json"
        )
        self.output_json_path = os.path.join(
            work_dir, "mixed_quant_flash_attn_metadata_case_output.json"
        )
        self.aic_core_num = 0
        self.aiv_core_num = 0

    def _build_input_json(
        self,
        cu_seqlens_q: Optional[list],
        cu_seqlens_kv: Optional[list],
        seqused_q: Optional[list],
        seqused_kv: Optional[list],
        batch_size: Optional[int],
        max_seqlen_q: Optional[int],
        max_seqlen_kv: Optional[int],
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        quant_compute_mode: int,
        mask_mode: Optional[int],
        win_left: Optional[int],
        win_right: Optional[int],
        layout_q: Optional[str],
        layout_kv: Optional[str],
        layout_out: Optional[str],
    ) -> dict:
        if batch_size is None:
            if cu_seqlens_q is not None:
                batch_size = len(cu_seqlens_q) - 1
            elif seqused_q is not None:
                batch_size = len(seqused_q)
            else:
                batch_size = 1

        if mask_mode is None:
            mask_mode = 0
        if win_left is None:
            win_left = -1
        if win_right is None:
            win_right = -1

        fd_on = quant_compute_mode != 0

        if (max_seqlen_q * num_heads_q // num_heads_kv) <= 32:
            m_base_size = 32
        elif (max_seqlen_q * num_heads_q // num_heads_kv) <= 48:
            m_base_size = 48
        elif (max_seqlen_q * num_heads_q // num_heads_kv) <= 64:
            m_base_size = 64
        else:
            m_base_size = 48
        return {
            "socVersion": "",
            "aicCoreNum": 64,
            "aivCoreNum": 64,
            "cuSeqlensQ": cu_seqlens_q,
            "cuSeqlensKv": cu_seqlens_kv,
            "sequsedQ": seqused_q,
            "sequsedKv": seqused_kv,
            "batchSize": batch_size,
            "maxSeqlenQ": max_seqlen_q,
            "maxSeqlenKv": max_seqlen_kv,
            "numHeadsQ": num_heads_q,
            "numHeadsKv": num_heads_kv,
            "quantMode": quant_compute_mode,
            "headDim": head_dim,
            "maskMode": mask_mode,
            "winLeft": win_left,
            "winRight": win_right,
            "layoutQ": layout_q if layout_q else "BSND",
            "layoutKv": layout_kv if layout_kv else "BSND",
            "layoutOut": layout_out if layout_out else "BSND",
            "l2Byte": 134217728,
            "fdOn": True,
            "mode": 0,
            "mBaseSize": m_base_size,
            "s2BaseSize": 512,
        }

    def _write_case_json(self, input_json: dict):
        with open(self.case_json_path, "w", encoding="utf-8") as f:
            json.dump(input_json, f, indent=2, ensure_ascii=False)

    def _run_binary(self) -> dict:
        if not os.path.isfile(self.binary_path):
            raise FileNotFoundError(f"Binary not found: {self.binary_path}")

        command = [
            self.binary_path,
            "-i",
            "mixed_quant_flash_attn_metadata_case.json",
            "-o",
            "./",
        ]
        result = subprocess.run(
            command,
            cwd=self.work_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Binary failed with code {result.returncode}: {result.stderr}"
            )

        with open(self.output_json_path, "r", encoding="utf-8") as f:
            output_json = json.load(f)

        if output_json.get("errorMessage", ""):
            raise RuntimeError(f"Operator error: {output_json['errorMessage']}")

        return output_json

    def _parse_output_to_metadata(self, output_json: dict) -> List[int]:
        output = output_json["output"]
        input_info = output_json["input"]
        section_num = output["sectionNum"]
        aic_core_num = input_info["aicCoreMaxNum"]
        aiv_core_num = input_info["aivCoreMaxNum"]
        m_base_size = input_info.get("mBaseSize", 0)
        s2_base_size = input_info.get("s2BaseSize", 0)

        total_size = 16 + section_num * (aic_core_num + aiv_core_num) * 16
        metadata = [0] * total_size

        metadata[0] = section_num

        section_fd = output["sectionFdResult"][0]
        metadata[1] = 1 if section_fd["usedVecNum"] > 0 else 0

        metadata[2] = m_base_size
        metadata[3] = s2_base_size
        metadata[4] = aic_core_num
        metadata[5] = aiv_core_num
        metadata[6] = 0

        for sec in range(section_num):
            fa_split_res = output["sectionFaResult"][sec]
            fd_split_res = output["sectionFdResult"][sec]
            used_core_num = fa_split_res["usedCoreNum"]
            used_vec_num = fd_split_res["usedVecNum"]

            fa_base = 16 + sec * aic_core_num * 16

            for i in range(used_core_num):
                offset = fa_base + i * 16

                if i > 0:
                    metadata[offset + 0] = fa_split_res["bN2End"][i - 1]
                    metadata[offset + 1] = fa_split_res["gS1End"][i - 1]
                    metadata[offset + 2] = fa_split_res["s2End"][i - 1]

                metadata[offset + 3] = fa_split_res["bN2End"][i]
                metadata[offset + 4] = fa_split_res["gS1End"][i]
                metadata[offset + 5] = fa_split_res["s2End"][i]
                metadata[offset + 6] = fa_split_res["firstFdDataWorkspaceIdx"][i]

            fa_offset = 16 + section_num * aic_core_num * 16
            fd_base = fa_offset + sec * aiv_core_num * 16

            for i in range(used_vec_num):
                offset = fd_base + i * 16
                cur_task_idx = fd_split_res["taskIdx"][i]

                metadata[offset + 0] = fd_split_res["bN2Idx"][cur_task_idx]
                metadata[offset + 1] = fd_split_res["gS1Idx"][cur_task_idx]
                metadata[offset + 2] = fd_split_res["workspaceIdx"][cur_task_idx]
                metadata[offset + 3] = fd_split_res["s2SplitNum"][cur_task_idx]
                metadata[offset + 4] = fd_split_res["mStart"][i]
                metadata[offset + 5] = fd_split_res["mLen"][i]

        return metadata

    def npu_mixed_quant_flash_attn_metadata(
        self,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        quant_compute_mode: int,
        cu_seqlens_q: Optional[list] = None,
        cu_seqlens_kv: Optional[list] = None,
        seqused_q: Optional[list] = None,
        seqused_kv: Optional[list] = None,
        batch_size: Optional[int] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        mask_mode: Optional[int] = None,
        win_left: Optional[int] = None,
        win_right: Optional[int] = None,
        layout_q: Optional[str] = None,
        layout_kv: Optional[str] = None,
        layout_out: Optional[str] = None,
        aic_core_num: int = 64,
        aiv_core_num: int = 64,
    ) -> List[int]:
        self.aic_core_num = aic_core_num
        self.aiv_core_num = aiv_core_num

        input_json = self._build_input_json(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            quant_compute_mode=quant_compute_mode,
            mask_mode=mask_mode,
            win_left=win_left,
            win_right=win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_out=layout_out,
        )

        self._write_case_json(input_json)

        output_json = self._run_binary()

        metadata = self._parse_output_to_metadata(output_json)

        return metadata
