# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
try:
    import numpy as np
    import torch
    import torch_npu
    import torchair
    from typing import Optional
    from torchair.ge._ge_graph import Tensor, TensorSpec, Const, DataType
    from torchair.ge._ge_graph import auto_convert_to_tensor
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
    _DT_BF16_INT8_FP8_HIF8 = "DT_BF16, DT_INT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8"
    _DT_BF16_INT8 = "DT_BF16, DT_INT8"
    _DT_DEQUANT_SCALE = "DT_FLOAT, DT_FLOAT8_E8M0"

    def _resolve_do_rope_ge(
        rope_sin: Optional[Tensor], rope_cos: Optional[Tensor]
    ) -> bool:
        has_sin = rope_sin is not None
        has_cos = rope_cos is not None
        if has_sin != has_cos:
            raise ValueError(
                "rope_sin and rope_cos must both be provided or both be empty/None"
            )
        return has_sin

    def _empty_rope_bf16() -> Tensor:
        return Const(np.array([], dtype=np.float32), dtype=DataType.DT_BF16)

    # GE Bitcast 的 keep_dim 默认 false：源/目标字节宽度不同时会在末轴增维
    # （如 bf16 4D cache -> HIFLOAT8 会变成 5D），因此只对单字节存储做 dtype 重解释。
    _ONE_BYTE_GE_DTYPES = (
        DataType.DT_UINT8,
        DataType.DT_INT8,
        DataType.DT_HIFLOAT8,
        DataType.DT_FLOAT8_E4M3FN,
        DataType.DT_FLOAT8_E5M2,
        DataType.DT_FLOAT8_E8M0,
    )

    def _bitcast_one_byte(tensor: Optional[Tensor], ge_dtype: int) -> Optional[Tensor]:
        if tensor is None or tensor.dtype == ge_dtype:
            return tensor
        if tensor.dtype not in _ONE_BYTE_GE_DTYPES:
            return tensor
        return ge.Bitcast(tensor, type=ge_dtype)

    def _bitcast_hif8(tensor: Optional[Tensor]) -> Optional[Tensor]:
        return _bitcast_one_byte(tensor, DataType.DT_HIFLOAT8)

    def _bitcast_e8m0(tensor: Optional[Tensor]) -> Optional[Tensor]:
        return _bitcast_one_byte(tensor, DataType.DT_FLOAT8_E8M0)

    def _apply_quant_bitcasts(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_dkv_kr,
        kv_cache,
        dequant_scale_x,
        dequant_scale_w_dq,
        dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr,
        weight_quant_mode,
        kv_cache_quant_mode,
    ):
        if weight_quant_mode == 5:
            token_x = _bitcast_hif8(token_x)
            weight_dq = _bitcast_hif8(weight_dq)
            weight_uq_qr = _bitcast_hif8(weight_uq_qr)
            weight_dkv_kr = _bitcast_hif8(weight_dkv_kr)
            # 对齐 PTA force_kv_cache_hifloat8：仅 kv_cache_quant_mode 1/3 时 kv 以 HIFLOAT8 存储，
            # kv_cache_quant_mode=0/2 时 kv 仍是 bf16，不能做 HIFLOAT8 重解释
            if kv_cache_quant_mode in (1, 3):
                kv_cache = _bitcast_hif8(kv_cache)
        if weight_quant_mode == 3:
            dequant_scale_x = _bitcast_e8m0(dequant_scale_x)
            dequant_scale_w_dq = _bitcast_e8m0(dequant_scale_w_dq)
            dequant_scale_w_uq_qr = _bitcast_e8m0(dequant_scale_w_uq_qr)
            dequant_scale_w_dkv_kr = _bitcast_e8m0(dequant_scale_w_dkv_kr)
        return (
            token_x,
            weight_dq,
            weight_uq_qr,
            weight_dkv_kr,
            kv_cache,
            dequant_scale_x,
            dequant_scale_w_dq,
            dequant_scale_w_uq_qr,
            dequant_scale_w_dkv_kr,
        )

    def _resolve_graph_rope(rope_sin, rope_cos):
        do_rope = _resolve_do_rope_ge(rope_sin, rope_cos)
        if do_rope:
            return do_rope, rope_sin, rope_cos
        return do_rope, _empty_rope_bf16(), _empty_rope_bf16()

    def _pack_torch_outputs(outs, weight_quant_mode, functional):
        query, query_rope = outs[0], outs[1]
        kv_cache_out, kr_cache_out = outs[2], outs[3]
        dequant_scale_q_nope, query_norm, dequant_scale_q_norm = (
            outs[4],
            outs[5],
            outs[6],
        )
        if weight_quant_mode == 3:
            dequant_scale_q_norm = _bitcast_one_byte(
                dequant_scale_q_norm, DataType.DT_UINT8
            )
        if functional:
            return (
                query,
                query_rope,
                dequant_scale_q_nope,
                query_norm,
                dequant_scale_q_norm,
                kv_cache_out,
                kr_cache_out,
            )
        return query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm

    @auto_convert_to_tensor(
        [False] * 21,
        [False] * 11 + [True] * 10,
    )
    def MlaPrologV3(
        token_x: Tensor,
        weight_dq: Tensor,
        weight_uq_qr: Tensor,
        weight_uk: Tensor,
        weight_dkv_kr: Tensor,
        rmsnorm_gamma_cq: Tensor,
        rmsnorm_gamma_ckv: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        kv_cache: Tensor,
        kr_cache: Tensor,
        cache_index: Optional[Tensor] = None,
        dequant_scale_x: Optional[Tensor] = None,
        dequant_scale_w_dq: Optional[Tensor] = None,
        dequant_scale_w_uq_qr: Optional[Tensor] = None,
        dequant_scale_w_dkv_kr: Optional[Tensor] = None,
        quant_scale_ckv: Optional[Tensor] = None,
        quant_scale_ckr: Optional[Tensor] = None,
        smooth_scales_cq: Optional[Tensor] = None,
        actual_seq_len: Optional[Tensor] = None,
        k_nope_clip_alpha: Optional[Tensor] = None,
        *,
        rmsnorm_epsilon_cq: float = 1e-05,
        rmsnorm_epsilon_ckv: float = 1e-05,
        cache_mode: str = "PA_BSND",
        query_norm_flag: bool = False,
        weight_quant_mode: int = 0,
        kv_cache_quant_mode: int = 0,
        query_quant_mode: int = 0,
        ckvkr_repo_mode: int = 0,
        quant_scale_repo_mode: int = 0,
        tile_size: int = 128,
        qc_qr_scale: float = 1.0,
        kc_scale: float = 1.0,
        do_rope: bool = True,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MlaPrologV3) via ge_op, aligned with op_host/mla_prolog_v3_def.cpp."""
        if dependencies is None:
            dependencies = []

        inputs = {
            "token_x": token_x,
            "weight_dq": weight_dq,
            "weight_uq_qr": weight_uq_qr,
            "weight_uk": weight_uk,
            "weight_dkv_kr": weight_dkv_kr,
            "rmsnorm_gamma_cq": rmsnorm_gamma_cq,
            "rmsnorm_gamma_ckv": rmsnorm_gamma_ckv,
            "rope_sin": rope_sin,
            "rope_cos": rope_cos,
            "kv_cache": kv_cache,
            "kr_cache": kr_cache,
            "cache_index": cache_index,
            "dequant_scale_x": dequant_scale_x,
            "dequant_scale_w_dq": dequant_scale_w_dq,
            "dequant_scale_w_uq_qr": dequant_scale_w_uq_qr,
            "dequant_scale_w_dkv_kr": dequant_scale_w_dkv_kr,
            "quant_scale_ckv": quant_scale_ckv,
            "quant_scale_ckr": quant_scale_ckr,
            "smooth_scales_cq": smooth_scales_cq,
            "actual_seq_len": actual_seq_len,
            "k_nope_clip_alpha": k_nope_clip_alpha,
        }
        attrs = {
            "rmsnorm_epsilon_cq": attr.Float(rmsnorm_epsilon_cq),
            "rmsnorm_epsilon_ckv": attr.Float(rmsnorm_epsilon_ckv),
            "cache_mode": attr.Str(cache_mode),
            "query_norm_flag": attr.Bool(query_norm_flag),
            "weight_quant_mode": attr.Int(weight_quant_mode),
            "kv_cache_quant_mode": attr.Int(kv_cache_quant_mode),
            "query_quant_mode": attr.Int(query_quant_mode),
            "ckvkr_repo_mode": attr.Int(ckvkr_repo_mode),
            "quant_scale_repo_mode": attr.Int(quant_scale_repo_mode),
            "tile_size": attr.Int(tile_size),
            "qc_qr_scale": attr.Float(qc_qr_scale),
            "kc_scale": attr.Float(kc_scale),
            "do_rope": attr.Bool(do_rope),
        }
        outputs = [
            "query",
            "query_rope",
            "kv_cache",
            "kr_cache",
            "dequant_scale_q_nope",
            "query_norm",
            "dequant_scale_q_norm",
        ]
        return ge_op(
            op_type="MlaPrologV3",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("MlaPrologV3")
            .input("token_x", _DT_BF16_INT8_FP8_HIF8)
            .input("weight_dq", _DT_BF16_INT8_FP8_HIF8)
            .input("weight_uq_qr", _DT_BF16_INT8_FP8_HIF8)
            .input("weight_uk", "DT_BF16")
            .input("weight_dkv_kr", _DT_BF16_INT8_FP8_HIF8)
            .input("rmsnorm_gamma_cq", "DT_BF16")
            .input("rmsnorm_gamma_ckv", "DT_BF16")
            .input("rope_sin", "DT_BF16")
            .input("rope_cos", "DT_BF16")
            .input("kv_cache", _DT_BF16_INT8_FP8_HIF8)
            .input("kr_cache", _DT_BF16_INT8)
            .optional_input("cache_index", "DT_INT64")
            .optional_input("dequant_scale_x", _DT_DEQUANT_SCALE)
            .optional_input("dequant_scale_w_dq", _DT_DEQUANT_SCALE)
            .optional_input("dequant_scale_w_uq_qr", _DT_DEQUANT_SCALE)
            .optional_input("dequant_scale_w_dkv_kr", _DT_DEQUANT_SCALE)
            .optional_input("quant_scale_ckv", "DT_FLOAT")
            .optional_input("quant_scale_ckr", "DT_FLOAT")
            .optional_input("smooth_scales_cq", "DT_FLOAT")
            .optional_input("actual_seq_len", "DT_INT32")
            .optional_input("k_nope_clip_alpha", "DT_FLOAT")
            .attr("rmsnorm_epsilon_cq", attr.Float(1e-05))
            .attr("rmsnorm_epsilon_ckv", attr.Float(1e-05))
            .attr("cache_mode", attr.Str("PA_BSND"))
            .attr("query_norm_flag", attr.Bool(False))
            .attr("weight_quant_mode", attr.Int(0))
            .attr("kv_cache_quant_mode", attr.Int(0))
            .attr("query_quant_mode", attr.Int(0))
            .attr("ckvkr_repo_mode", attr.Int(0))
            .attr("quant_scale_repo_mode", attr.Int(0))
            .attr("tile_size", attr.Int(128))
            .attr("qc_qr_scale", attr.Float(1.0))
            .attr("kc_scale", attr.Float(1.0))
            .attr("do_rope", attr.Bool(True))
            .output("query", _DT_BF16_INT8_FP8_HIF8)
            .output("query_rope", "DT_BF16")
            .output("kv_cache", _DT_BF16_INT8_FP8_HIF8)
            .output("kr_cache", _DT_BF16_INT8)
            .output("dequant_scale_q_nope", "DT_FLOAT")
            .output("query_norm", _DT_BF16_INT8_FP8_HIF8)
            .output("dequant_scale_q_norm", _DT_DEQUANT_SCALE),
        )

    @register_fx_node_ge_converter(torch.ops.cann_ops_transformer.mla_prolog.default)
    def convert_mla_prolog(
        token_x: Tensor,
        weight_dq: Tensor,
        weight_uq_qr: Tensor,
        weight_uk: Tensor,
        weight_dkv_kr: Tensor,
        rmsnorm_gamma_cq: Tensor,
        rmsnorm_gamma_ckv: Tensor,
        kv_cache: Tensor,
        kr_cache: Tensor,
        *,
        rope_sin: Optional[Tensor] = None,
        rope_cos: Optional[Tensor] = None,
        cache_index: Optional[Tensor] = None,
        dequant_scale_x: Optional[Tensor] = None,
        dequant_scale_w_dq: Optional[Tensor] = None,
        dequant_scale_w_uq_qr: Optional[Tensor] = None,
        dequant_scale_w_dkv_kr: Optional[Tensor] = None,
        quant_scale_ckv: Optional[Tensor] = None,
        quant_scale_ckr: Optional[Tensor] = None,
        smooth_scales_cq: Optional[Tensor] = None,
        actual_seq_len: Optional[Tensor] = None,
        k_nope_clip_alpha: Optional[Tensor] = None,
        rmsnorm_epsilon_cq: float = 1e-05,
        rmsnorm_epsilon_ckv: float = 1e-05,
        cache_mode: str = "PA_BSND",
        query_norm_flag: bool = False,
        weight_quant_mode: int = 0,
        kv_cache_quant_mode: int = 0,
        query_quant_mode: int = 0,
        ckvkr_repo_mode: int = 0,
        quant_scale_repo_mode: int = 0,
        tile_size: int = 128,
        qc_qr_scale: float = 1.0,
        kc_scale: float = 1.0,
        token_x_dtype: Optional[int] = None,
        weight_dq_dtype: Optional[int] = None,
        weight_uq_qr_dtype: Optional[int] = None,
        weight_dkv_kr_dtype: Optional[int] = None,
        kv_cache_dtype: Optional[int] = None,
        meta_outputs: TensorSpec = None,
    ):
        """GE converter: cann_ops_transformer.mla_prolog -> MlaPrologV3.

        do_rope 由 rope_sin/rope_cos 是否成对传入推断，并下沉为 GE do_rope attr。
        torch schema 不含 do_rope，converter 签名与 schema 对齐。
        """
        do_rope, graph_rope_sin, graph_rope_cos = _resolve_graph_rope(
            rope_sin, rope_cos
        )
        (
            token_x,
            weight_dq,
            weight_uq_qr,
            weight_dkv_kr,
            kv_cache,
            dequant_scale_x,
            dequant_scale_w_dq,
            dequant_scale_w_uq_qr,
            dequant_scale_w_dkv_kr,
        ) = _apply_quant_bitcasts(
            token_x,
            weight_dq,
            weight_uq_qr,
            weight_dkv_kr,
            kv_cache,
            dequant_scale_x,
            dequant_scale_w_dq,
            dequant_scale_w_uq_qr,
            dequant_scale_w_dkv_kr,
            weight_quant_mode,
            kv_cache_quant_mode,
        )

        outs = MlaPrologV3(
            token_x,
            weight_dq,
            weight_uq_qr,
            weight_uk,
            weight_dkv_kr,
            rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv,
            graph_rope_sin,
            graph_rope_cos,
            kv_cache,
            kr_cache,
            cache_index=cache_index,
            dequant_scale_x=dequant_scale_x,
            dequant_scale_w_dq=dequant_scale_w_dq,
            dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
            dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr,
            quant_scale_ckv=quant_scale_ckv,
            quant_scale_ckr=quant_scale_ckr,
            smooth_scales_cq=smooth_scales_cq,
            actual_seq_len=actual_seq_len,
            k_nope_clip_alpha=k_nope_clip_alpha,
            rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
            rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
            cache_mode=cache_mode,
            query_norm_flag=query_norm_flag,
            weight_quant_mode=weight_quant_mode,
            kv_cache_quant_mode=kv_cache_quant_mode,
            query_quant_mode=query_quant_mode,
            ckvkr_repo_mode=ckvkr_repo_mode,
            quant_scale_repo_mode=quant_scale_repo_mode,
            tile_size=tile_size,
            qc_qr_scale=qc_qr_scale,
            kc_scale=kc_scale,
            do_rope=do_rope,
        )
        return _pack_torch_outputs(outs, weight_quant_mode, functional=False)

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.mla_prolog_functional.default
    )
    def convert_mla_prolog_functional(
        token_x: Tensor,
        weight_dq: Tensor,
        weight_uq_qr: Tensor,
        weight_uk: Tensor,
        weight_dkv_kr: Tensor,
        rmsnorm_gamma_cq: Tensor,
        rmsnorm_gamma_ckv: Tensor,
        kv_cache: Tensor,
        kr_cache: Tensor,
        *,
        rope_sin: Optional[Tensor] = None,
        rope_cos: Optional[Tensor] = None,
        cache_index: Optional[Tensor] = None,
        dequant_scale_x: Optional[Tensor] = None,
        dequant_scale_w_dq: Optional[Tensor] = None,
        dequant_scale_w_uq_qr: Optional[Tensor] = None,
        dequant_scale_w_dkv_kr: Optional[Tensor] = None,
        quant_scale_ckv: Optional[Tensor] = None,
        quant_scale_ckr: Optional[Tensor] = None,
        smooth_scales_cq: Optional[Tensor] = None,
        actual_seq_len: Optional[Tensor] = None,
        k_nope_clip_alpha: Optional[Tensor] = None,
        rmsnorm_epsilon_cq: float = 1e-05,
        rmsnorm_epsilon_ckv: float = 1e-05,
        cache_mode: str = "PA_BSND",
        query_norm_flag: bool = False,
        weight_quant_mode: int = 0,
        kv_cache_quant_mode: int = 0,
        query_quant_mode: int = 0,
        ckvkr_repo_mode: int = 0,
        quant_scale_repo_mode: int = 0,
        tile_size: int = 128,
        qc_qr_scale: float = 1.0,
        kc_scale: float = 1.0,
        token_x_dtype: Optional[int] = None,
        weight_dq_dtype: Optional[int] = None,
        weight_uq_qr_dtype: Optional[int] = None,
        weight_dkv_kr_dtype: Optional[int] = None,
        kv_cache_dtype: Optional[int] = None,
        meta_outputs: TensorSpec = None,
    ):
        """GE converter: mla_prolog_functional -> TensorMove + MlaPrologV3.

        对齐 PTA npu_mla_prolog_v3_functional：先 TensorMove 两份 cache，再调原地 GE 算子，
        按 torch schema 返回 7 个输出。
        """
        do_rope, graph_rope_sin, graph_rope_cos = _resolve_graph_rope(
            rope_sin, rope_cos
        )
        (
            token_x,
            weight_dq,
            weight_uq_qr,
            weight_dkv_kr,
            kv_cache,
            dequant_scale_x,
            dequant_scale_w_dq,
            dequant_scale_w_uq_qr,
            dequant_scale_w_dkv_kr,
        ) = _apply_quant_bitcasts(
            token_x,
            weight_dq,
            weight_uq_qr,
            weight_dkv_kr,
            kv_cache,
            dequant_scale_x,
            dequant_scale_w_dq,
            dequant_scale_w_uq_qr,
            dequant_scale_w_dkv_kr,
            weight_quant_mode,
            kv_cache_quant_mode,
        )
        kv_cache_copy = ge.TensorMove(kv_cache)
        kr_cache_copy = ge.TensorMove(kr_cache)
        outs = MlaPrologV3(
            token_x,
            weight_dq,
            weight_uq_qr,
            weight_uk,
            weight_dkv_kr,
            rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv,
            graph_rope_sin,
            graph_rope_cos,
            kv_cache_copy,
            kr_cache_copy,
            cache_index=cache_index,
            dequant_scale_x=dequant_scale_x,
            dequant_scale_w_dq=dequant_scale_w_dq,
            dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
            dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr,
            quant_scale_ckv=quant_scale_ckv,
            quant_scale_ckr=quant_scale_ckr,
            smooth_scales_cq=smooth_scales_cq,
            actual_seq_len=actual_seq_len,
            k_nope_clip_alpha=k_nope_clip_alpha,
            rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
            rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
            cache_mode=cache_mode,
            query_norm_flag=query_norm_flag,
            weight_quant_mode=weight_quant_mode,
            kv_cache_quant_mode=kv_cache_quant_mode,
            query_quant_mode=query_quant_mode,
            ckvkr_repo_mode=ckvkr_repo_mode,
            quant_scale_repo_mode=quant_scale_repo_mode,
            tile_size=tile_size,
            qc_qr_scale=qc_qr_scale,
            kc_scale=kc_scale,
            do_rope=do_rope,
        )
        return _pack_torch_outputs(outs, weight_quant_mode, functional=True)

    def _register_aclgraph_inplaceable():
        """ACL graph reinplace：mla_prolog_functional -> mla_prolog。

        torchair.register_inplaceable_npu_op 只认 torch.ops.npu，这里直接写入 inplaceable_npu_ops。
        mutated_arg 是 positional index：schema 里 kv_cache=7、kr_cache=8。
        """
        mutated_arg = [7, 8]
        modules = (
            "torch_npu.dynamo.torchair._acl_concrete_graph.graph_pass",
            "torch_npu.dynamo.npugraph_ex._acl_concrete_graph.graph_pass",
            "torchair._acl_concrete_graph.graph_pass",
        )
        functional_op = torch.ops.cann_ops_transformer.mla_prolog_functional.default
        inplace_op = torch.ops.cann_ops_transformer.mla_prolog.default
        for mod_name in modules:
            try:
                mod = __import__(
                    mod_name, fromlist=["inplaceable_npu_ops", "InplaceableNpuOp"]
                )
            except ImportError:
                continue
            extra_check = getattr(
                mod, "check_multi_stream_for_multi_reinplace", lambda node: True
            )
            mod.inplaceable_npu_ops[functional_op] = mod.InplaceableNpuOp(
                inplace_op=inplace_op,
                mutated_arg=mutated_arg,
                extra_check=extra_check,
            )

    _register_aclgraph_inplaceable()

else:

    def convert_mla_prolog(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )

    def convert_mla_prolog_functional(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
