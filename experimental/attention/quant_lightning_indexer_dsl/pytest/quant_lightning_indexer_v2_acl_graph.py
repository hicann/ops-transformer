# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


def qliv2_output_acl_graph(test_data):
    import torch
    import torch_npu
    import cann_ops_transformer.ops.attention.quant_lightning_indexer_dsl
    from quant_lightning_indexer_v2_golden import (
        generate_qliv2_test_data,
        generate_cpu_golden,
        _operator_kwargs,
        _params_dict,
    )

    data = generate_qliv2_test_data(test_data)
    golden = generate_cpu_golden(data)
    options = _operator_kwargs(_params_dict(data.params), data)
    options = {
        name: value.npu() if isinstance(value, torch.Tensor) else value
        for name, value in options.items()
    }
    positional = tuple(
        value.npu()
        for value in (data.q, data.k, data.w, data.q_descale, data.k_descale)
    )
    operator = torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer

    def execute():
        metadata = torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer_metadata(
            options["cu_seqlens_q"],
            options["cu_seqlens_k"],
            options["seqused_q"],
            options["seqused_k"],
            options["cmp_residual_k"],
            num_heads_q=32,
            num_heads_k=1,
            head_dim=128,
            **{
                name: options[name]
                for name in (
                    "topk",
                    "max_seqlen_q",
                    "mask_mode",
                    "cmp_ratio",
                    "layout_q",
                    "layout_k",
                    "candidate_topk_blocks",
                    "candidate_block_size",
                )
            },
        )
        return operator(*positional, metadata=metadata, **options)

    for warmup in range(3):
        execute()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = execute()
    graph.replay()
    torch.npu.synchronize()
    names = (
        "sparse_indices",
        "sparse_values",
        "candidate_block_indices",
        "candidate_block_length",
    )
    result = {name: value.cpu() for name, value in zip(names, outputs)}
    return (
        golden,
        result,
        golden.get("_reference_scores"),
        golden.get("sparse_values"),
        result["sparse_values"],
    )
