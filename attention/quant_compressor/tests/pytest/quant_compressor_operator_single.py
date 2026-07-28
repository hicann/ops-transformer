# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

from quant_compressor_golden import run_compressor_eager, cpu_compressor


class Generalized_operator:
    def forward(
        self,
        x,
        wkv,
        wgate,
        kv_state,
        score_state,
        ape,
        block_table,
        cu_seqlens,
        seqused,
        start_pos,
        cmp_ratio,
        coff,
        cache_mode,
    ):
        return cpu_compressor(
            x,
            wkv,
            wgate,
            kv_state,
            score_state,
            ape,
            block_table=block_table,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            start_pos=start_pos,
            cmp_ratio=cmp_ratio,
            coff=coff,
            cache_mode=cache_mode,
        )


def output_operator(params):
    # 构造输入
    (
        batch_size,
        hidden_size,
        Seq_len,
        head_dim,
        block_size,
        cmp_ratio,
        coff,
        start_p,
        cache_mode,
        layout_x,
        data_type,
        cu_seqlens,
        seqused,
        start_pos,
        x_datarange,
        wkv_datarange,
        wgate_datarange,
        ape_datarange,
        kv_state_datarange,
        score_state_datarange,
        x_descale_datarange,
        wkv_descale_datarange,
        wgate_descale_datarange,
        quant_mode,
    ) = params
    S_max = 0
    save_state_seqlens = None
    bs_combine_flag = False
    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32)
    if start_pos is not None:
        start_pos = torch.tensor(start_pos).to(torch.int32)
    elif start_p is not None:
        start_pos = torch.full((batch_size,), start_p, dtype=torch.int32)

    if layout_x == "TH":
        bs_combine_flag = True
        if cu_seqlens is None:
            T = batch_size * Seq_len
            if T != 0:
                cu_seqlens = torch.arange(0, T + 1, Seq_len, dtype=torch.int32)
            else:
                cu_seqlens = torch.zeros((batch_size + 1), dtype=torch.int32)
        else:
            cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32)
        for i in range(batch_size):
            if start_pos is not None:
                if start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i] > S_max:
                    S_max = start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i]
            else:
                if cu_seqlens[i + 1] - cu_seqlens[i] > S_max:
                    S_max = cu_seqlens[i + 1] - cu_seqlens[i]

    else:
        cu_seqlens = None
        if start_pos is not None:
            S_max = max(start_pos) + Seq_len
        else:
            S_max = Seq_len
    print(
        f"params: batch_size={batch_size}, hidden_size={hidden_size}, Seq_len={Seq_len}, head_dim={head_dim}, "
        f"block_size={block_size}, cmp_ratio={cmp_ratio}, coff={coff}, "
        f"start_p={start_p}, cache_mode={cache_mode}, "
        f"layout_x={layout_x}, data_type={data_type}, cu_seqlens={cu_seqlens}, seqused={seqused}, start_pos={start_pos}, "
        f"x_datarange={x_datarange}, wkv_datarange={wkv_datarange}, wgate_datarange={wgate_datarange}, "
        f"ape_datarange={ape_datarange}, "
        f"kv_state_datarange={kv_state_datarange}, score_state_datarange={score_state_datarange}, "
        f"x_descale_datarange={x_descale_datarange}, wkv_descale_datarange={wkv_descale_datarange}, "
        f"wgate_descale_datarange={wgate_descale_datarange}, quant_mode={quant_mode}"
    )

    run_compressor_eager(
        batch_size,
        S_max,
        head_dim,
        coff,
        cmp_ratio,
        bs_combine_flag,
        S=Seq_len,
        start_pos=start_pos,
        seqused=seqused,
        cu_seqlens=cu_seqlens,
        block_size=block_size,
        cache_mode=cache_mode,
        data_type=data_type,
        hidden_size=hidden_size,
        save_state_seqlens=save_state_seqlens,
        quant_mode=quant_mode,
        x_datarange=x_datarange,
        wkv_datarange=wkv_datarange,
        wgate_datarange=wgate_datarange,
        ape_datarange=ape_datarange,
        kv_state_datarange=kv_state_datarange,
        score_state_datarange=score_state_datarange,
        x_descale_datarange=x_descale_datarange,
        wkv_descale_datarange=wkv_descale_datarange,
        wgate_descale_datarange=wgate_descale_datarange,
    )
