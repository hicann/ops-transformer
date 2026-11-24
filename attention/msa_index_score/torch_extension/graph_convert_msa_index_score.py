# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode
# key layouts: PA BBND/BNBD and TND packed.
# 950 also accepts DT_HIFLOAT8 / DT_FLOAT8_E5M2 / DT_FLOAT8_E4M3FN.

try:
    import torch
    import torch_npu
    import torchair
    from typing import Optional
    from torchair.ge._ge_graph import (
        DataType,
        Tensor,
        TensorSpec,
        _ge_dtype_to_ge_proto_dtype,
    )
    from torchair.ge._ge_graph import auto_convert_to_tensor, get_default_ge_graph
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    class _ByteStorageInput:
        def __init__(self, fx_input_idx):
            self.fx_input_idx = fx_input_idx

        def __call__(self, *args):
            return args[self.fx_input_idx].view(torch.uint8).contiguous()

        def codegen(self, ge_idx, ge_inputs_name):
            return f"{ge_inputs_name}[{ge_idx}] = args[{self.fx_input_idx}].view(torch.uint8).contiguous()"

    def _bitcast(tensor, dtype):
        result = ge.Bitcast(tensor, type=dtype)
        result.desc.dtype = _ge_dtype_to_ge_proto_dtype(dtype)
        return result

    def _fp4_from_bytes(tensor, rank):
        factors = ge.Const([1] * (rank - 1) + [2])
        logical_shape = ge.Mul(ge.Shape(tensor), factors)
        return ge.Reshape(_bitcast(tensor, DataType.DT_FLOAT4_E2M1), logical_shape)

    def _native_storage(node, dtype, nodes):
        graph = get_default_ge_graph()
        byte_dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_UINT8)

        if node.type == "Bitcast":
            name, index = node.input[0].rsplit(":", 1)
            source = Tensor(nodes[name], int(index))
            return source if source.desc.dtype == byte_dtype else None
        if node.type not in ("Data", "StridedSliceV2"):
            return None
        if node.type == "StridedSliceV2":
            source = _native_storage(
                nodes[node.input[0].rsplit(":", 1)[0]], dtype, nodes
            )
            if source is None:
                return None

        # Keep Data nodes in FX argument order for TorchAir's input adapter.
        pos = next(i for i, op in enumerate(graph.op) if op.name == node.name)
        graph.op.insert(pos, node)
        raw = graph.op[pos]
        raw.name = graph.next_unique_name(node.name + "_bytes", node.type)
        raw.input_desc[0].dtype = byte_dtype
        raw.output_desc[0].dtype = byte_dtype
        nodes[raw.name] = raw
        if node.type == "Data":
            index = node.attr["index"].i
            graph.indexed_inputs[index] = raw
            info = graph.named_inputs_info.pop(node.name)
            info.func = _ByteStorageInput(info.func.fx_input_idx)
            graph.record_input_info(raw.name, info)
        else:
            raw.input[0] = source.tensor

        # Keep the original edge name and native dtype for other users. Only
        # the graph input and its storage-preserving slices operate on bytes.
        storage = Tensor(raw)
        alias = _bitcast(storage, dtype)
        name = node.name
        node.CopyFrom(alias.node)
        node.name = name
        graph.op.remove(alias.node)
        return storage

    def _native_fp4_operand(tensor, scale, nodes):
        storage = _native_storage(tensor.node, DataType.DT_FLOAT4_E2M1, nodes)
        if storage is not None:
            return _fp4_from_bytes(storage, tensor.rank)
        if tensor.rank != scale.rank:
            raise ValueError(
                "Native MXFP4 operands and scales must have the same rank."
            )
        # GE producers may expose an extra packing axis. Restore logical D
        # without taking the operand's token, head and page axes from the scale.
        prefix = ge.GatherV2(ge.Shape(tensor), list(range(tensor.rank - 1)), 0)
        groups = ge.GatherV2(ge.Shape(scale), [scale.rank - 1], 0)
        head_dim = ge.Mul(groups, ge.Const(32, dtype=DataType.DT_INT32))
        shape = ge.ConcatV2([prefix, head_dim], 0, N=2)
        return ge.Reshape(tensor, shape)

    def _unpack_mxfp4_inputs(query, key, query_scale, key_scale):
        fp4_dtype = getattr(DataType, "DT_FLOAT4_E2M1", None)
        if query.dtype not in (DataType.DT_UINT8, fp4_dtype):
            return query, key, query_scale, key_scale
        scale_dtype = getattr(DataType, "DT_FLOAT8_E8M0", None)
        if fp4_dtype is None or scale_dtype is None:
            raise RuntimeError(
                "MXFP4 graph conversion requires TorchAir FP4 and E8M0 support."
            )
        if query.dtype == fp4_dtype:
            if key.dtype != query.dtype or any(
                scale is None or scale.dtype != scale_dtype
                for scale in (query_scale, key_scale)
            ):
                raise TypeError("Native MXFP4 requires FP4 query/key and E8M0 scales.")
            nodes = {op.name: op for op in get_default_ge_graph().op}
            _native_storage(query_scale.node, scale_dtype, nodes)
            _native_storage(key_scale.node, scale_dtype, nodes)
            return (
                _native_fp4_operand(query, query_scale, nodes),
                _native_fp4_operand(key, key_scale, nodes),
                query_scale,
                key_scale,
            )
        if any(
            tensor is None or tensor.dtype != DataType.DT_UINT8
            for tensor in (key, query_scale, key_scale)
        ):
            raise TypeError(
                "Packed MXFP4 requires uint8 query, key, query_scale and key_scale."
            )
        return (
            _fp4_from_bytes(query, query.rank),
            _fp4_from_bytes(key, key.rank),
            _bitcast(query_scale, scale_dtype),
            _bitcast(key_scale, scale_dtype),
        )

    @auto_convert_to_tensor(
        [False, False, False, False, False, False, False, False, False],
        [False, False, False, True, True, True, True, True, True],
    )
    def MsaIndexScore(
        query: Tensor,
        key: Tensor,
        start_loc: Tensor,
        block_table: Optional[Tensor] = None,
        query_scale: Optional[Tensor] = None,
        key_scale: Optional[Tensor] = None,
        atten_mask: Optional[Tensor] = None,
        actual_seq_qlen: Optional[Tensor] = None,
        actual_seq_klen: Optional[Tensor] = None,
        *,
        layout_key: str = "BBND",
        sparse_mode: int = 3,
        init_blocks: int = 0,
        local_blocks: int = 1,
        compress_factor: int = 1,
        dependencies=None,
        node_name=None,
    ):
        if query.dtype in (torch.int8, DataType.DT_INT8):
            raise ValueError("A8C8 requires msa_index_score_v2")
        if compress_factor != 1:
            raise ValueError("compress_factor only supports 1")
        if dependencies is None:
            dependencies = []

        query, key, query_scale, key_scale = _unpack_mxfp4_inputs(
            query, key, query_scale, key_scale
        )
        inputs = {
            "query": query,
            "key": key,
            "start_loc": start_loc,
            "block_table": block_table,
            "query_scale": query_scale,
            "key_scale": key_scale,
            "atten_mask": atten_mask,
            "actual_seq_qlen": actual_seq_qlen,
            "actual_seq_klen": actual_seq_klen,
        }
        attrs = {
            "layout_key": attr.Str(layout_key),
            "sparse_mode": attr.Int(sparse_mode),
            "init_blocks": attr.Int(init_blocks),
            "local_blocks": attr.Int(local_blocks),
            "layout_query": attr.Str("TND"),
            "pool_block_size": attr.Int(128),
            "return_choice": attr.Int(0),
            "topk": attr.Int(16),
            "compress_factor": attr.Int(compress_factor),
        }
        outputs = ["score", "sparse_indices", "sparse_values"]
        return ge_op(
            op_type="MsaIndexScore",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MsaIndexScore")
            .input(
                "query",
                "DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1",
            )
            .input(
                "key",
                "DT_FLOAT16, DT_BF16, DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1",
            )
            .input("start_loc", "DT_INT32")
            .optional_input("block_table", "DT_INT32")
            .optional_input("query_scale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .optional_input("key_scale", "DT_FLOAT, DT_FLOAT8_E8M0, DT_FLOAT8_E4M3FN")
            .optional_input("atten_mask", "DT_INT8")
            .optional_input("actual_seq_qlen", "DT_INT32")
            .optional_input("actual_seq_klen", "DT_INT32")
            .attr("layout_key", attr.Str("BBND"))
            .attr("sparse_mode", attr.Int(3))
            .attr("init_blocks", attr.Int(0))
            .attr("local_blocks", attr.Int(1))
            .attr("layout_query", attr.Str("TND"))
            .attr("pool_block_size", attr.Int(128))
            .attr("return_choice", attr.Int(0))
            .attr("topk", attr.Int(16))
            .attr("compress_factor", attr.Int(1))
            .output("score", "DT_FLOAT, DT_FLOAT16")
            .output("sparse_indices", "DT_INT32")
            .output("sparse_values", "DT_FLOAT, DT_FLOAT16"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer._msa_index_score.default
    )
    def convert_msa_index_score(
        query: Tensor,
        key: Tensor,
        start_loc: Tensor,
        block_table: Optional[Tensor] = None,
        query_scale: Optional[Tensor] = None,
        key_scale: Optional[Tensor] = None,
        atten_mask: Optional[Tensor] = None,
        actual_seq_qlen: Optional[Tensor] = None,
        actual_seq_klen: Optional[Tensor] = None,
        *,
        layout_key: str = "BBND",
        sparse_mode: int = 3,
        init_blocks: int = 0,
        local_blocks: int = 1,
        compress_factor: int = 1,
        meta_outputs: TensorSpec = None,
    ):
        result = MsaIndexScore(
            query=query,
            key=key,
            block_table=block_table,
            query_scale=query_scale,
            key_scale=key_scale,
            atten_mask=atten_mask,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            start_loc=start_loc,
            layout_key=layout_key,
            sparse_mode=sparse_mode,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            compress_factor=compress_factor,
        )
        if isinstance(result, (tuple, list)):
            return result[0]
        return result

else:

    def convert_msa_index_score(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
