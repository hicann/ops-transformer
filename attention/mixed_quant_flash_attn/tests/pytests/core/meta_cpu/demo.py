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
import os
import struct
from mixed_quant_flash_attn_metadata_op import MixedQuantFlashAttnMetadataOp


def main():
    op = MixedQuantFlashAttnMetadataOp()

    metadata = op.npu_mixed_quant_flash_attn_metadata(
        num_heads_q=16,
        num_heads_kv=1,
        head_dim=128,
        quant_mode=1,
        batch_size=1,
        cu_seqlens_q=[1],
        cu_seqlens_kv=[1],
        seqused_q=[1],
        seqused_kv=[1],
        max_seqlen_q=32768,
        max_seqlen_kv=32768,
        mask_mode=3,
        win_left=-1,
        win_right=-1,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
    )

    print(f"Metadata total size: {len(metadata)} uint32_t")
    print("Header:")
    print(f"  sectionNum  = {metadata[0]}")
    print(f"  isFd        = {metadata[1]}")
    print(f"  mBaseSize   = {metadata[2]}")
    print(f"  s2BaseSize  = {metadata[3]}")
    print(f"  aicCoreNum  = {metadata[4]}")
    print(f"  aivCoreNum  = {metadata[5]}")
    print(f"  isS1G       = {metadata[6]}")

    section_num = metadata[0]
    aic_core_num = metadata[4]
    aiv_core_num = metadata[5]

    for sec in range(section_num):
        print(f"\nFA Metadata (section {sec}):")
        fa_base = 16 + sec * aic_core_num * 16
        for core in range(aic_core_num):
            off = fa_base + core * 16
            vals = metadata[off : off + 7]
            if any(v != 0 for v in vals):
                print(
                    f"  core[{core}]: BN_START={vals[0]} M_START={vals[1]} "
                    f"S2_START={vals[2]} BN_END={vals[3]} M_END={vals[4]} "
                    f"S2_END={vals[5]} FD_WS_IDX={vals[6]}"
                )

        print(f"\nFD Metadata (section {sec}):")
        fa_offset = 16 + section_num * aic_core_num * 16
        fd_base = fa_offset + sec * aiv_core_num * 16
        for core in range(aiv_core_num):
            off = fd_base + core * 16
            vals = metadata[off : off + 6]
            if any(v != 0 for v in vals):
                print(
                    f"  core[{core}]: BN2_IDX={vals[0]} M_IDX={vals[1]} "
                    f"WS_IDX={vals[2]} WS_NUM={vals[3]} M_START={vals[4]} M_NUM={vals[5]}"
                )

    # packed = struct.pack(f"<{len(metadata)}I", *metadata)
    # bin_path = os.path.join(work_dir, "metadata_tensor.bin")
    # with open(bin_path, "wb") as f:
    #     f.write(packed)
    # print(f"\nMetadata binary written to {bin_path} ({len(packed)} bytes)")


if __name__ == "__main__":
    main()
