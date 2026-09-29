# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from qliv2_test_utils import QliV2CaseSelector, QliV2ResultWriter

import pytest


def test_parse_indexes():
    assert QliV2CaseSelector.parse_indexes("1,3-4", 5) == [1, 3, 4]


def test_case_name_is_stable():
    assert QliV2ResultWriter.case_name((), "decode B8") == "decode_B8"


def test_offset_comparison_uses_local_reference_scores():
    import torch
    from qli_test_utils.result_compare import compare_sparse_outputs

    expected = {
        "sparse_indices": torch.tensor([[[1000]]], dtype=torch.int32),
        "_reference_scores": torch.tensor([[[1.0, 1.0]]]),
        "_output_idx_offset": torch.tensor([[1000]], dtype=torch.int32),
    }
    actual = {"sparse_indices": torch.tensor([[[1001]]], dtype=torch.int32)}
    assert compare_sparse_outputs(expected, actual)["pass"]
    actual["sparse_indices"][0, 0, 0] = 1002
    assert not compare_sparse_outputs(expected, actual)["pass"]
    from qli_test_utils.golden import quant_lightning_indexer_golden

    result = quant_lightning_indexer_golden(
        torch.zeros((1, 32, 64), dtype=torch.uint8),
        torch.zeros((3, 1, 64), dtype=torch.uint8),
        torch.ones((1, 32)),
        torch.full((1, 32, 2, 2), 127, dtype=torch.uint8),
        torch.full((3, 1, 2, 2), 127, dtype=torch.uint8),
        topk=512,
        layout_k="TND",
        output_idx_offset=torch.tensor([[1000]], dtype=torch.int32),
    )
    assert set(result["sparse_indices"][0, 0, :3].tolist()) == {1000, 1001, 1002}
    assert torch.all(result["sparse_indices"][0, 0, 3:] == -1)


def test_qli_rejects_generalized_head_shape():
    import pytest
    import torch
    from ops.quant_lightning_indexer_dsl import quant_lightning_indexer
    from ops.quant_lightning_indexer_metadata_dsl import (
        quant_lightning_indexer_metadata,
    )

    for layout in ("TND", "PA_BBND"):
        with pytest.raises(ValueError, match="N1=32"):
            quant_lightning_indexer(
                torch.empty((1, 64, 64), dtype=torch.uint8),
                None,
                None,
                None,
                None,
                topk=512,
                quant_mode=1,
                layout_k=layout,
            )
        with pytest.raises(ValueError, match="N1=32"):
            quant_lightning_indexer_metadata(
                num_heads_q=64, num_heads_k=1, head_dim=128, topk=512, layout_k=layout
            )


def test_tnd_pa_metadata_offsets_and_tail_storage():
    import os
    import torch
    import torch_npu
    import cann_ops_transformer.ops.attention.quant_lightning_indexer_dsl
    from ops.quant_lightning_indexer_metadata_dsl import build_metadata, choose_splits

    quant_lightning_indexer = (
        torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer
    )
    quant_lightning_indexer_metadata = (
        torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer_metadata
    )

    torch_npu.npu.set_device(int(os.environ.get("QLIV2_DEVICE_ID", "0")))
    generator = torch.Generator().manual_seed(314159)
    query = torch.randint(
        0, 256, (3, 32, 64), dtype=torch.uint8, generator=generator
    ).npu()
    weights = torch.rand((3, 32), generator=generator).npu()
    qscale = torch.full((3, 32, 2, 2), 127, dtype=torch.uint8).npu()
    key_cpu = torch.randint(
        0, 256, (388, 1, 64), dtype=torch.uint8, generator=generator
    )
    scale_cpu = torch.full((388, 1, 2, 2), 127, dtype=torch.uint8)
    key_storage = torch.empty((780, 1, 64), dtype=torch.uint8, device=query.device)
    scale_storage = torch.empty((1170, 1, 2, 2), dtype=torch.uint8, device=query.device)
    key = key_storage[2:778:2]
    scale = scale_storage[3:1167:3]
    key.copy_(key_cpu)
    scale.copy_(scale_cpu)
    cu_q = torch.tensor([0, 2, 3], dtype=torch.int32).npu()
    cu_k = torch.tensor([0, 257, 388], dtype=torch.int32).npu()
    used_k = torch.tensor([257, 131], dtype=torch.int32).npu()
    meta_options = dict(
        max_seqlen_q=2,
        num_heads_q=32,
        num_heads_k=1,
        head_dim=128,
        topk=512,
        mask_mode=3,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
    )
    metadata = quant_lightning_indexer_metadata(cu_q, cu_k, **meta_options)
    pa_metadata = quant_lightning_indexer_metadata(
        cu_q, seqused_k=used_k, layout_k="PA_BBND", **meta_options
    )
    assert torch.equal(metadata.cpu(), pa_metadata.cpu())
    no_ld = build_metadata(
        query,
        cu_q,
        None,
        used_k,
        None,
        None,
        batch=2,
        max_tasks=1,
        capacity=0,
        splits=choose_splits(2, 0, 512),
        mask=3,
        ratio=1,
        query_rows=6,
        ld=False,
    )
    assert metadata.cpu()[288] != 0
    assert torch.count_nonzero(no_ld.cpu()[288:864]) == 0
    options = dict(
        topk=512,
        quant_mode=1,
        max_seqlen_q=2,
        mask_mode=3,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        return_value=True,
    )

    def run_tnd(meta, offset=None, used=None):
        return quant_lightning_indexer(
            query,
            key,
            weights,
            qscale,
            scale,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            layout_k="TND",
            seqused_k=used,
            output_idx_offset=offset,
            metadata=meta,
            **options,
        )

    ordinary = run_tnd(metadata)
    unsplit = run_tnd(no_ld, used=used_k)
    offsets = torch.full((3, 1), 1000, dtype=torch.int32, device=query.device)
    shifted = run_tnd(metadata, offsets)
    pa_key = torch.zeros((5, 128, 1, 64), dtype=torch.uint8)
    pa_scale = torch.full((5, 128, 1, 2, 2), 127, dtype=torch.uint8)
    pa_key[:3].view(-1, 1, 64)[:257].copy_(key_cpu[:257])
    pa_key[3:].view(-1, 1, 64)[:131].copy_(key_cpu[257:])
    table = torch.tensor([[0, 1, 2], [3, 4, 4]], dtype=torch.int32).npu()
    paged = quant_lightning_indexer(
        query,
        pa_key.npu(),
        weights,
        qscale,
        pa_scale.npu(),
        cu_seqlens_q=cu_q,
        seqused_k=used_k,
        block_table=table,
        metadata=pa_metadata,
        layout_k="PA_BBND",
        **options,
    )
    torch.npu.synchronize()
    for expected, actual in zip(ordinary, paged):
        assert torch.equal(expected.cpu(), actual.cpu())
    for expected, actual in zip(ordinary, unsplit):
        assert torch.equal(expected.cpu(), actual.cpu())
    expected_indices = torch.where(ordinary[0] >= 0, ordinary[0] + 1000, ordinary[0])
    assert torch.equal(shifted[0].cpu(), expected_indices.cpu())
    for expected, actual in zip(ordinary[1:], shifted[1:]):
        assert torch.equal(expected.cpu(), actual.cpu())


@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("candidate", [-1, 2048])
def test_qli_torch_extension_outputs(layout, return_value, candidate, monkeypatch):
    import os
    import torch
    import torch_npu
    import cann_ops_transformer.ops.attention.quant_lightning_indexer_dsl
    from ops.quant_lightning_indexer_dsl import quant_lightning_indexer as direct
    from ops.quant_lightning_indexer_metadata_dsl import (
        quant_lightning_indexer_metadata as direct_metadata,
    )

    torch.npu.set_device(int(os.environ.get("QLIV2_DEVICE_ID", "0")))
    generator = torch.Generator().manual_seed(4201)
    query = torch.randint(
        0, 256, (2, 32, 64), dtype=torch.uint8, generator=generator
    ).npu()
    key = torch.randint(
        0, 256, (512, 1, 64), dtype=torch.uint8, generator=generator
    ).npu()
    scale = torch.full((512, 1, 2, 2), 127, dtype=torch.uint8).to(query.device)
    positional = (
        query,
        key if layout == "TND" else key.view(4, 128, 1, 64),
        torch.rand((2, 32), generator=generator).npu(),
        torch.full((2, 32, 2, 2), 127, dtype=torch.uint8).to(query.device),
        scale if layout == "TND" else scale.view(4, 128, 1, 2, 2),
    )
    cu_q = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
    cu_k = torch.tensor([0, 256, 512], dtype=torch.int32).npu()
    used_k = torch.tensor([256, 256], dtype=torch.int32).npu()
    meta_options = dict(
        max_seqlen_q=1,
        max_seqlen_k=512,
        num_heads_q=32,
        num_heads_k=1,
        head_dim=128,
        topk=512,
        candidate_topk_blocks=candidate,
        candidate_block_size=8 if candidate > 0 else -1,
    )
    namespace = torch.ops.cann_ops_transformer.ds41
    metadata = namespace.quant_lightning_indexer_metadata(
        cu_q, cu_k, seqused_k=used_k, layout_k=layout, **meta_options
    )
    reference_metadata = direct_metadata(
        cu_q, cu_k, seqused_k=used_k, layout_k=layout, **meta_options
    )
    assert torch.equal(metadata.cpu(), reference_metadata.cpu())
    options = dict(
        topk=512,
        quant_mode=1,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        seqused_k=used_k,
        output_idx_offset=torch.full((2, 1), 1000, dtype=torch.int32).npu(),
        block_table=None
        if layout == "TND"
        else torch.arange(4, dtype=torch.int32).reshape(2, 2).npu(),
        max_seqlen_q=1,
        return_value=return_value,
        candidate_topk_blocks=candidate,
        candidate_block_size=8 if candidate > 0 else -1,
    )
    explicit = namespace.quant_lightning_indexer(
        *positional, metadata=metadata, layout_k=layout, **options
    )
    reference = direct(*positional, metadata=metadata, layout_k=layout, **options)
    for actual, expected in zip(explicit, reference):
        assert torch.equal(actual.cpu(), expected.cpu())
    assert explicit[1].shape == (explicit[0].shape if return_value else (0,))
    assert explicit[2].shape == ((2, 1, candidate) if candidate > 0 else (0,))
    assert explicit[3].shape == ((2, 1) if candidate > 0 else (0,))
    from unittest.mock import Mock

    metadata_call = Mock(
        side_effect=AssertionError("unexpected automatic metadata call")
    )
    with monkeypatch.context() as patch:
        patch.setattr(namespace, "quant_lightning_indexer_metadata", metadata_call)
        with pytest.raises(ValueError, match="metadata is required"):
            namespace.quant_lightning_indexer(*positional, layout_k=layout, **options)
    metadata_call.assert_not_called()


@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
def test_qli_metadata_without_tensor(layout):
    import os
    import torch
    import torch_npu
    import cann_ops_transformer.ops.attention.quant_lightning_indexer_dsl
    from ops.quant_lightning_indexer_metadata_dsl import (
        quant_lightning_indexer_metadata,
    )

    torch.npu.set_device(int(os.environ.get("QLIV2_DEVICE_ID", "0")))
    options = dict(
        max_seqlen_q=1,
        max_seqlen_k=512,
        num_heads_q=32,
        num_heads_k=1,
        head_dim=128,
        topk=512,
    )
    actual = torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer_metadata(
        layout_k=layout, **options
    )
    expected = quant_lightning_indexer_metadata(layout_k=layout, **options)
    assert actual.device == expected.device
    assert actual.dtype == torch.int32 and actual.shape == (1024,)
    assert torch.equal(actual.cpu(), expected.cpu())


def test_torch_extension_meta_and_fake(monkeypatch):
    import torch
    import torch_npu
    import importlib
    import cann_ops_transformer.ops.attention.quant_lightning_indexer_dsl
    import cann_ops_transformer.ops.attention.quant_sparse_lightning_indexer_dsl
    from torch._subclasses.fake_tensor import FakeTensorMode

    namespace = torch.ops.cann_ops_transformer.ds41
    names = ("quant_lightning_indexer", "quant_sparse_lightning_indexer")

    def forbidden(*args, **kwargs):
        raise AssertionError("abstract execution must not call DSL")

    for name in names:
        for suffix in ("", "_metadata"):
            module = importlib.import_module("ops." + name + suffix + "_dsl")
            monkeypatch.setattr(module, name + suffix, forbidden)
            assert torch._C._dispatch_has_kernel_for_dispatch_key(
                "cann_ops_transformer::ds41." + name + suffix, "PrivateUse1"
            )
            assert torch._C._dispatch_has_kernel_for_dispatch_key(
                "cann_ops_transformer::ds41." + name + suffix, "Meta"
            )

    def check(device):
        query = torch.empty((3, 32, 64), dtype=torch.uint8, device=device)
        weights = torch.empty((3, 32), device=device)
        qscale = torch.empty((3, 32, 2, 2), dtype=torch.uint8, device=device)
        lengths = torch.empty((3, 1), dtype=torch.int32, device=device)
        candidates = torch.empty((3, 1, 2048), dtype=torch.int32, device=device)
        metadata_options = dict(num_heads_q=32, num_heads_k=1, head_dim=128, topk=512)
        for layout in ("PA_BBND", "TND"):
            key = torch.empty(
                (4, 128, 1, 64) if layout == "PA_BBND" else (512, 1, 64),
                dtype=torch.uint8,
                device=device,
            )
            scale = torch.empty(
                (4, 128, 1, 2, 2) if layout == "PA_BBND" else (512, 1, 2, 2),
                dtype=torch.uint8,
                device=device,
            )
            for enabled in (False, True):
                outputs = namespace.quant_lightning_indexer(
                    query,
                    key,
                    weights,
                    qscale,
                    scale,
                    512,
                    1,
                    layout_k=layout,
                    return_value=enabled,
                    candidate_topk_blocks=2048 if enabled else -1,
                    candidate_block_size=8 if enabled else -1,
                )
                assert [tuple(value.shape) for value in outputs] == [
                    (3, 1, 512),
                    (3, 1, 512) if enabled else (0,),
                    (3, 1, 2048) if enabled else (0,),
                    (3, 1) if enabled else (0,),
                ]
                assert [value.dtype for value in outputs] == [
                    torch.int32,
                    torch.bfloat16,
                    torch.int32,
                    torch.int32,
                ]
                packed = (
                    torch.empty((4, 16, 544), dtype=torch.uint8, device=device)
                    if layout == "PA_BBND"
                    else key
                )
                sparse = namespace.quant_sparse_lightning_indexer(
                    query,
                    packed,
                    weights,
                    qscale,
                    candidates,
                    lengths,
                    512,
                    1,
                    8,
                    descale_k=scale if layout == "TND" else None,
                    layout_k=layout,
                    return_value=enabled,
                )
                assert sparse[0].shape == (3, 1, 512)
                assert sparse[1].shape == ((3, 1, 512) if enabled else (0,))
                assert [value.dtype for value in sparse] == [
                    torch.int32,
                    torch.bfloat16,
                ]
                assert all(
                    value.device == query.device for value in (*outputs, *sparse)
                )
            metadata = namespace.quant_lightning_indexer_metadata(
                seqused_q=lengths[:, 0], layout_k=layout, **metadata_options
            )
            sparse_metadata = namespace.quant_sparse_lightning_indexer_metadata(
                lengths,
                quant_mode=1,
                candidate_block_size=8,
                layout_k=layout,
                **metadata_options,
            )
            for value in (metadata, sparse_metadata):
                assert (
                    value.shape == (1024,)
                    and value.dtype == torch.int32
                    and value.device == query.device
                )

    check("meta")
    with FakeTensorMode():
        check("npu:1")
        no_tensor = namespace.quant_lightning_indexer_metadata(
            num_heads_q=32,
            num_heads_k=1,
            head_dim=128,
            topk=512,
            max_seqlen_q=1,
            max_seqlen_k=512,
        )
        assert no_tensor.shape == (1024,) and no_tensor.dtype == torch.int32
