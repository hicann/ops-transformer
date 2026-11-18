# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off
# 保持参数分组紧凑，与原有 FlashKDA 样例风格一致。


"""训练前向入口，根据 metadata 选择 staged 或 persistent 设备模板。"""

import os
import threading
from typing import Optional, Tuple

import cannbotdsl
from cannbotdsl import Tensor, dtypes
from cannbotdsl.aot import export
from cannbotdsl.aot.runtime import COLLECTOR
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_num
from cannbotdsl.ops.sync import onchip_rewind
import torch

from .flash_kda_metadata import (
    ABI_VERSION,
    HEADER_ABI_VERSION,
    HEADER_CORE_NUM,
    HEADER_MAGIC,
    HEADER_STATUS,
    HEADER_TEMPLATE,
    MAGIC,
    MAX_AIC_CORES,
    STATUS_OK,
    TEMPLATE_AUTO,
    TEMPLATE_PERSISTENT,
    TEMPLATE_STAGED,
    WORKSPACE_SLOTS,
    _capture_schedule,
    _device_block_num,
    decode_metadata,
)
from . import flash_kda_staged as staged
from . import flash_kda_persistent as persistent
from .flash_kda_validation import (
    CHUNK_SIZE, SUPPORTED_HEAD_DIM,
    require, validate_inputs, read_lengths, validate_schedule, validate_buffer, output_conditions,
)

LOW_DTYPE = torch.bfloat16
HIGH_DTYPE = torch.float32


def compile_environment():
    """使用框架完整的工厂环境标识和进程号作为缓存标识。"""
    environment = tuple(
        sorted(
            (
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("CANNBOTDSL_") or name == "ASCEND_HOME_PATH"
        ),
        ),
    )
    return os.getpid(), environment


@kernel(
    profile=cannbotdsl.ProfileSpec(
        name="flash_kda", op_type="FlashKDA",
        inputs=("gm_K", "gm_V", "gm_Q", "gm_beta", "gm_g", "gm_A_log", "gm_dt_bias", "gm_metadata",),
        outputs=(
            "gm_O",
            "gm_state_out",
            "gm_Aqk",
            "gm_Akk",
            "gm_saved_w",
            "gm_u",
            "gm_qg",
            "gm_kg",
            "gm_v_new",
            "gm_g_cumsum",
            "gm_h",
        ),
    ),
)
class flash_kda_fwd_kernel:
    def __call__(
        self,
        gm_K: Tensor,
        gm_V: Tensor,
        gm_Q: Tensor,
        gm_beta: Tensor,
        gm_g: Tensor,
        gm_qk_exchange: Tensor,
        gm_A_log: Tensor,
        gm_dt_bias: Tensor,
        gm_kg: Tensor,
        gm_gamma: Tensor,
        gm_q_decayed: Tensor,
        gm_u: Tensor,
        gm_saved_w: Tensor,
        gm_qg: Tensor,
        gm_g_cumsum: Tensor,
        gm_v_new: Tensor,
        gm_h: Tensor,
        gm_O: Tensor,
        gm_state_in: Tensor,
        gm_state_out: Tensor,
        gm_identity16: Tensor,
        gm_q_norm: Tensor,
        gm_k_norm: Tensor,
        gm_q_rstd: Tensor,
        gm_k_rstd: Tensor,
        gm_beta_activated: Tensor,
        gm_Aqk: Tensor,
        gm_Akk: Tensor,
        gm_metadata: Tensor,
        scale_value: float,
        lower_bound: float,
        use_qk_l2norm_in_kernel: int,
        use_gate_in_kernel: int,
        use_beta_sigmoid_in_kernel: int,
        allow_neg_eigval: int,
        has_lower_bound: int,
        state_is_compact: int,
        save_qg: int,
        save_g_cumsum: int,
        save_intermediate_state: int,
        save_matrices: int,
        has_initial_state: int,
        save_final_state: int,
    ):
        _block_num = get_block_num()
        if (
            gm_metadata[HEADER_MAGIC] == MAGIC
            and gm_metadata[HEADER_ABI_VERSION] == ABI_VERSION
            and (gm_metadata[HEADER_STATUS] == STATUS_OK)
            and (gm_metadata[HEADER_CORE_NUM] == get_block_num())
            and (
                gm_metadata[HEADER_TEMPLATE] == TEMPLATE_STAGED
                or gm_metadata[HEADER_TEMPLATE] == TEMPLATE_PERSISTENT
            )
        ):
            template = gm_metadata[HEADER_TEMPLATE]
            if template == TEMPLATE_STAGED:
                onchip_rewind(reset_sync_id=True)
                staged.run_staged_flash_kda_fwd(
                    gm_K, gm_V, gm_Q, gm_beta, gm_g, gm_qk_exchange, gm_A_log, gm_dt_bias, gm_kg, gm_gamma,
                    gm_q_decayed, gm_u, gm_saved_w, gm_qg, gm_g_cumsum, gm_v_new, gm_h, gm_O, gm_state_in, gm_state_out,
                    gm_identity16, gm_q_norm, gm_k_norm, gm_q_rstd, gm_k_rstd, gm_beta_activated, gm_Aqk, gm_Akk,
                    gm_metadata, scale_value, lower_bound, use_qk_l2norm_in_kernel, use_gate_in_kernel,
                    use_beta_sigmoid_in_kernel, allow_neg_eigval, has_lower_bound, state_is_compact, save_qg,
                    save_g_cumsum, save_intermediate_state, save_matrices, has_initial_state, save_final_state,
                )
            elif template == TEMPLATE_PERSISTENT:
                onchip_rewind(reset_sync_id=True)
                persistent.run_persistent_flash_kda_fwd(
                    gm_K, gm_V, gm_Q, gm_beta, gm_g, gm_qk_exchange, gm_A_log, gm_dt_bias, gm_kg, gm_gamma,
                    gm_q_decayed, gm_u, gm_saved_w, gm_qg, gm_g_cumsum, gm_v_new, gm_h, gm_O, gm_state_in, gm_state_out,
                    gm_identity16, gm_q_norm, gm_k_norm, gm_q_rstd, gm_k_rstd, gm_beta_activated, gm_Aqk, gm_Akk,
                    gm_metadata, scale_value, lower_bound, use_qk_l2norm_in_kernel, use_gate_in_kernel,
                    use_beta_sigmoid_in_kernel, allow_neg_eigval, has_lower_bound, state_is_compact, save_qg,
                    save_g_cumsum, save_intermediate_state, save_matrices, has_initial_state, save_final_state,
                )


@kernel(
    profile=cannbotdsl.ProfileSpec(
        name="staged_flash_kda", op_type="FlashKDA",
        inputs=("gm_K", "gm_V", "gm_Q", "gm_beta", "gm_g", "gm_A_log", "gm_dt_bias", "gm_metadata",),
        outputs=(
            "gm_O",
            "gm_state_out",
            "gm_Aqk",
            "gm_Akk",
            "gm_saved_w",
            "gm_u",
            "gm_qg",
            "gm_kg",
            "gm_v_new",
            "gm_g_cumsum",
            "gm_h",
        ),
    ),
)
class staged_flash_kda_fwd_kernel:
    def __call__(
        self,
        gm_K: Tensor,
        gm_V: Tensor,
        gm_Q: Tensor,
        gm_beta: Tensor,
        gm_g: Tensor,
        gm_qk_exchange: Tensor,
        gm_A_log: Tensor,
        gm_dt_bias: Tensor,
        gm_kg: Tensor,
        gm_gamma: Tensor,
        gm_q_decayed: Tensor,
        gm_u: Tensor,
        gm_saved_w: Tensor,
        gm_qg: Tensor,
        gm_g_cumsum: Tensor,
        gm_v_new: Tensor,
        gm_h: Tensor,
        gm_O: Tensor,
        gm_state_in: Tensor,
        gm_state_out: Tensor,
        gm_identity16: Tensor,
        gm_q_norm: Tensor,
        gm_k_norm: Tensor,
        gm_q_rstd: Tensor,
        gm_k_rstd: Tensor,
        gm_beta_activated: Tensor,
        gm_Aqk: Tensor,
        gm_Akk: Tensor,
        gm_metadata: Tensor,
        scale_value: float,
        lower_bound: float,
        use_qk_l2norm_in_kernel: int,
        use_gate_in_kernel: int,
        use_beta_sigmoid_in_kernel: int,
        allow_neg_eigval: int,
        has_lower_bound: int,
        state_is_compact: int,
        save_qg: int,
        save_g_cumsum: int,
        save_intermediate_state: int,
        save_matrices: int,
        has_initial_state: int,
        save_final_state: int,
    ):
        if (
            gm_metadata[HEADER_MAGIC] == MAGIC
            and gm_metadata[HEADER_ABI_VERSION] == ABI_VERSION
            and (gm_metadata[HEADER_STATUS] == STATUS_OK)
            and (gm_metadata[HEADER_CORE_NUM] == get_block_num())
            and (gm_metadata[HEADER_TEMPLATE] == TEMPLATE_STAGED)
        ):
            onchip_rewind(reset_sync_id=True)
            staged.run_staged_flash_kda_fwd(
                gm_K, gm_V, gm_Q, gm_beta, gm_g, gm_qk_exchange, gm_A_log, gm_dt_bias, gm_kg, gm_gamma, gm_q_decayed,
                gm_u, gm_saved_w, gm_qg, gm_g_cumsum, gm_v_new, gm_h, gm_O, gm_state_in, gm_state_out, gm_identity16,
                gm_q_norm, gm_k_norm, gm_q_rstd, gm_k_rstd, gm_beta_activated, gm_Aqk, gm_Akk, gm_metadata, scale_value,
                lower_bound, use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval,
                has_lower_bound, state_is_compact, save_qg, save_g_cumsum, save_intermediate_state, save_matrices, has_initial_state, save_final_state,
            )


@kernel(
    profile=cannbotdsl.ProfileSpec(
        name="persistent_flash_kda", op_type="FlashKDA",
        inputs=("gm_K", "gm_V", "gm_Q", "gm_beta", "gm_g", "gm_A_log", "gm_dt_bias", "gm_metadata",),
        outputs=(
            "gm_O",
            "gm_state_out",
            "gm_Aqk",
            "gm_Akk",
            "gm_saved_w",
            "gm_u",
            "gm_qg",
            "gm_kg",
            "gm_v_new",
            "gm_g_cumsum",
            "gm_h",
        ),
    ),
)
class persistent_flash_kda_fwd_kernel:
    def __call__(
        self,
        gm_K: Tensor,
        gm_V: Tensor,
        gm_Q: Tensor,
        gm_beta: Tensor,
        gm_g: Tensor,
        gm_qk_exchange: Tensor,
        gm_A_log: Tensor,
        gm_dt_bias: Tensor,
        gm_kg: Tensor,
        gm_gamma: Tensor,
        gm_q_decayed: Tensor,
        gm_u: Tensor,
        gm_saved_w: Tensor,
        gm_qg: Tensor,
        gm_g_cumsum: Tensor,
        gm_v_new: Tensor,
        gm_h: Tensor,
        gm_O: Tensor,
        gm_state_in: Tensor,
        gm_state_out: Tensor,
        gm_identity16: Tensor,
        gm_q_norm: Tensor,
        gm_k_norm: Tensor,
        gm_q_rstd: Tensor,
        gm_k_rstd: Tensor,
        gm_beta_activated: Tensor,
        gm_Aqk: Tensor,
        gm_Akk: Tensor,
        gm_metadata: Tensor,
        scale_value: float,
        lower_bound: float,
        use_qk_l2norm_in_kernel: int,
        use_gate_in_kernel: int,
        use_beta_sigmoid_in_kernel: int,
        allow_neg_eigval: int,
        has_lower_bound: int,
        state_is_compact: int,
        save_qg: int,
        save_g_cumsum: int,
        save_intermediate_state: int,
        save_matrices: int,
        has_initial_state: int,
        save_final_state: int,
    ):
        if (
            gm_metadata[HEADER_MAGIC] == MAGIC
            and gm_metadata[HEADER_ABI_VERSION] == ABI_VERSION
            and (gm_metadata[HEADER_STATUS] == STATUS_OK)
            and (gm_metadata[HEADER_CORE_NUM] == get_block_num())
            and (gm_metadata[HEADER_TEMPLATE] == TEMPLATE_PERSISTENT)
        ):
            onchip_rewind(reset_sync_id=True)
            persistent.run_persistent_flash_kda_fwd(
                gm_K, gm_V, gm_Q, gm_beta, gm_g, gm_qk_exchange, gm_A_log, gm_dt_bias, gm_kg, gm_gamma, gm_q_decayed,
                gm_u, gm_saved_w, gm_qg, gm_g_cumsum, gm_v_new, gm_h, gm_O, gm_state_in, gm_state_out, gm_identity16,
                gm_q_norm, gm_k_norm, gm_q_rstd, gm_k_rstd, gm_beta_activated, gm_Aqk, gm_Akk, gm_metadata, scale_value,
                lower_bound, use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval,
                has_lower_bound, state_is_compact, save_qg, save_g_cumsum, save_intermediate_state, save_matrices, has_initial_state, save_final_state,
            )


class FlashKDAFwd:
    """供 :func:`flash_kda_fwd` 使用的已编译 AICore 入口。"""

    @host
    def run_flash_kda(
        self,
        K: Tensor,
        V: Tensor,
        Q: Tensor,
        beta: Tensor,
        g: Tensor,
        qk_exchange: Tensor,
        A_log: Tensor,
        dt_bias: Tensor,
        kg: Tensor,
        gamma: Tensor,
        qd: Tensor,
        u: Tensor,
        saved_w: Tensor,
        qg: Tensor,
        g_cumsum: Tensor,
        v_new: Tensor,
        h: Tensor,
        out: Tensor,
        state_in: Tensor,
        state_out: Tensor,
        identity: Tensor,
        q_norm: Tensor,
        k_norm: Tensor,
        q_rstd: Tensor,
        k_rstd: Tensor,
        beta_activated: Tensor,
        Aqk: Tensor,
        Akk: Tensor,
        metadata: Tensor,
        scale: float,
        lower_bound: float,
        use_qk_l2norm_in_kernel: int,
        use_gate_in_kernel: int,
        use_beta_sigmoid_in_kernel: int,
        allow_neg_eigval: int,
        has_lower_bound: int,
        state_is_compact: int,
        save_qg: int,
        save_g_cumsum: int,
        save_intermediate_state: int,
        save_matrices: int,
        has_initial_state: int,
        save_final_state: int,
        core_num: int,
    ):
        op = flash_kda_fwd_kernel()
        op[core_num](
            K, V, Q, beta, g, qk_exchange, A_log, dt_bias, kg, gamma, qd, u, saved_w, qg, g_cumsum, v_new, h, out,
            state_in, state_out, identity, q_norm, k_norm, q_rstd, k_rstd, beta_activated, Aqk, Akk, metadata, scale,
            lower_bound, use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval,
            has_lower_bound, state_is_compact, save_qg, save_g_cumsum, save_intermediate_state, save_matrices, has_initial_state, save_final_state,
        )


class StagedFlashKDAFwd:
    """供 :func:`flash_kda_fwd` 使用的已编译 AICore 入口。"""

    @host
    def run_staged_flash_kda(
        self,
        K: Tensor,
        V: Tensor,
        Q: Tensor,
        beta: Tensor,
        g: Tensor,
        qk_exchange: Tensor,
        A_log: Tensor,
        dt_bias: Tensor,
        kg: Tensor,
        gamma: Tensor,
        qd: Tensor,
        u: Tensor,
        saved_w: Tensor,
        qg: Tensor,
        g_cumsum: Tensor,
        v_new: Tensor,
        h: Tensor,
        out: Tensor,
        state_in: Tensor,
        state_out: Tensor,
        identity: Tensor,
        q_norm: Tensor,
        k_norm: Tensor,
        q_rstd: Tensor,
        k_rstd: Tensor,
        beta_activated: Tensor,
        Aqk: Tensor,
        Akk: Tensor,
        metadata: Tensor,
        scale: float,
        lower_bound: float,
        use_qk_l2norm_in_kernel: int,
        use_gate_in_kernel: int,
        use_beta_sigmoid_in_kernel: int,
        allow_neg_eigval: int,
        has_lower_bound: int,
        state_is_compact: int,
        save_qg: int,
        save_g_cumsum: int,
        save_intermediate_state: int,
        save_matrices: int,
        has_initial_state: int,
        save_final_state: int,
        core_num: int,
    ):
        op = staged_flash_kda_fwd_kernel()
        op[core_num](
            K, V, Q, beta, g, qk_exchange, A_log, dt_bias, kg, gamma, qd, u, saved_w, qg, g_cumsum, v_new, h, out,
            state_in, state_out, identity, q_norm, k_norm, q_rstd, k_rstd, beta_activated, Aqk, Akk, metadata, scale,
            lower_bound, use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval,
            has_lower_bound, state_is_compact, save_qg, save_g_cumsum, save_intermediate_state, save_matrices, has_initial_state, save_final_state,
        )


class PersistentFlashKDAFwd:
    """供 :func:`flash_kda_fwd` 使用的已编译 AICore 入口。"""

    @host
    def run_persistent_flash_kda(
        self,
        K: Tensor,
        V: Tensor,
        Q: Tensor,
        beta: Tensor,
        g: Tensor,
        qk_exchange: Tensor,
        A_log: Tensor,
        dt_bias: Tensor,
        kg: Tensor,
        gamma: Tensor,
        qd: Tensor,
        u: Tensor,
        saved_w: Tensor,
        qg: Tensor,
        g_cumsum: Tensor,
        v_new: Tensor,
        h: Tensor,
        out: Tensor,
        state_in: Tensor,
        state_out: Tensor,
        identity: Tensor,
        q_norm: Tensor,
        k_norm: Tensor,
        q_rstd: Tensor,
        k_rstd: Tensor,
        beta_activated: Tensor,
        Aqk: Tensor,
        Akk: Tensor,
        metadata: Tensor,
        scale: float,
        lower_bound: float,
        use_qk_l2norm_in_kernel: int,
        use_gate_in_kernel: int,
        use_beta_sigmoid_in_kernel: int,
        allow_neg_eigval: int,
        has_lower_bound: int,
        state_is_compact: int,
        save_qg: int,
        save_g_cumsum: int,
        save_intermediate_state: int,
        save_matrices: int,
        has_initial_state: int,
        save_final_state: int,
        core_num: int,
    ):
        op = persistent_flash_kda_fwd_kernel()
        op[core_num](
            K, V, Q, beta, g, qk_exchange, A_log, dt_bias, kg, gamma, qd, u, saved_w, qg, g_cumsum, v_new, h, out,
            state_in, state_out, identity, q_norm, k_norm, q_rstd, k_rstd, beta_activated, Aqk, Akk, metadata, scale,
            lower_bound, use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval,
            has_lower_bound, state_is_compact, save_qg, save_g_cumsum, save_intermediate_state, save_matrices, has_initial_state, save_final_state,
        )


_COMPILED_KERNEL = {}
_COMPILED_KERNEL_LOCK = threading.Lock()
_COMPILED_KERNEL_ENVIRONMENT = None


def _sequence_spec(dtype, physical_batches, length, heads, width, stride_prefix):
    minimum_stride = 0 if stride_prefix in ("RSTD", "ACT", "G_CUMSUM") else 1
    stride_0 = cannbotdsl.Dim(f"{stride_prefix}_S0", min=minimum_stride)
    stride_1 = cannbotdsl.Dim(f"{stride_prefix}_S1", min=minimum_stride)
    stride_2 = cannbotdsl.Dim(f"{stride_prefix}_S2", min=minimum_stride)
    return cannbotdsl.TensorSpec(
        (physical_batches, heads, length, width), dtype, stride=(stride_0, stride_1, stride_2, 1),
    )


def _chunk_cache_spec(dtype, batches, heads, chunks, width, stride_prefix):
    outer_stride = tuple(cannbotdsl.Dim(f"{stride_prefix}_S{i}", min=1) for i in range(3))
    return cannbotdsl.TensorSpec(
        (batches, heads, chunks, width, width), dtype, stride=(*outer_stride, width, 1),
    )


def _get_compiled_kernel(template=TEMPLATE_AUTO):
    global _COMPILED_KERNEL, _COMPILED_KERNEL_ENVIRONMENT, _COMPILED_KERNEL_LOCK

    current_collection = COLLECTOR.get
    environment = compile_environment()
    if (
        _COMPILED_KERNEL_ENVIRONMENT is not None
        and _COMPILED_KERNEL_ENVIRONMENT[0] != environment[0]
    ):
        _COMPILED_KERNEL = {}
        _COMPILED_KERNEL_LOCK = threading.Lock()
    with _COMPILED_KERNEL_LOCK:
        if _COMPILED_KERNEL_ENVIRONMENT != environment:
            for cached in _COMPILED_KERNEL.values():
                cached.close()
            _COMPILED_KERNEL.clear()
            _COMPILED_KERNEL_ENVIRONMENT = environment
        collecting = current_collection() is not None
        if (
            not collecting
            and template in _COMPILED_KERNEL
            and not _COMPILED_KERNEL[template].closed
        ):
            return _COMPILED_KERNEL[template]
        physical_batches = cannbotdsl.Dim("P")
        storage_length = cannbotdsl.Dim("L")
        output_length = cannbotdsl.Dim("OL")
        batch = cannbotdsl.Dim("B")
        state_batches = cannbotdsl.Dim("STATE_B")
        state_chunks = cannbotdsl.Dim("STATE_CHUNKS")
        n_qk = cannbotdsl.Dim("NQK")
        n_v = cannbotdsl.Dim("NV")
        metadata_capacity = cannbotdsl.Dim("META")
        core_num = cannbotdsl.Dim("CORE_NUM", min=1, max=MAX_AIC_CORES)
        scratch_tokens = WORKSPACE_SLOTS * CHUNK_SIZE
        gamma_rows = WORKSPACE_SLOTS * SUPPORTED_HEAD_DIM
        fake = cannbotdsl.TensorSpec
        qk_spec = _sequence_spec(dtypes.bfloat16, physical_batches, storage_length, n_qk, SUPPORTED_HEAD_DIM, "QK")
        v_spec = _sequence_spec(dtypes.bfloat16, physical_batches, storage_length, n_v, SUPPORTED_HEAD_DIM, "V")
        # v2 接口的 gate 和缓存可采用与 V 不同的物理布局。
        # 分别使用独立的符号步长，兼容未启用缓存时传入的占位张量。
        def value_spec(prefix):
            return _sequence_spec(dtypes.bfloat16, physical_batches, storage_length,
                                  n_v, SUPPORTED_HEAD_DIM, prefix)
        gate_spec = value_spec("GATE")
        kg_spec, u_spec, w_spec, qg_spec, vnew_spec = (
            value_spec(prefix) for prefix in ("KG", "U", "W", "QG", "VNEW")
        )
        beta_spec = _sequence_spec(dtypes.float32, physical_batches, storage_length, n_v, 1, "BETA")
        output_spec = _sequence_spec(dtypes.bfloat16, physical_batches, output_length, n_v, SUPPORTED_HEAD_DIM, "OUT")
        state_spec = fake((batch, n_v, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), dtypes.float32)
        rstd_spec = _sequence_spec(dtypes.float32, physical_batches, storage_length, n_qk, 1, "RSTD")
        activated_spec = _sequence_spec(dtypes.float32, physical_batches, storage_length, n_v, 1, "ACT")
        matrix_spec = _sequence_spec(
            dtypes.bfloat16, cannbotdsl.Dim("MATRIX_B"), cannbotdsl.Dim("MATRIX_L"),
            cannbotdsl.Dim("MATRIX_N"), CHUNK_SIZE, "MATRIX",
        )
        gate_cumsum_spec = _sequence_spec(
            dtypes.float32, cannbotdsl.Dim("G_B"), cannbotdsl.Dim("G_L"), cannbotdsl.Dim("G_N"), SUPPORTED_HEAD_DIM, "G_CUMSUM",
        )
        h_spec = _chunk_cache_spec(dtypes.float32, state_batches, n_v, state_chunks, SUPPORTED_HEAD_DIM, "H")
        entry = (
            FlashKDAFwd().run_flash_kda,
            StagedFlashKDAFwd().run_staged_flash_kda,
            PersistentFlashKDAFwd().run_persistent_flash_kda,
        )[template]
        compiled_kernel = cannbotdsl.compile(
            entry, qk_spec, v_spec, qk_spec, beta_spec, gate_spec, fake((core_num, 2, 2, CHUNK_SIZE), dtypes.float32),
            fake((n_v,), dtypes.float32), fake((n_v, SUPPORTED_HEAD_DIM), dtypes.float32), kg_spec,
            fake((gamma_rows, 1), dtypes.float32), fake((scratch_tokens, SUPPORTED_HEAD_DIM), dtypes.bfloat16), u_spec,
            w_spec, qg_spec, gate_cumsum_spec, vnew_spec, h_spec, output_spec, state_spec, state_spec,
            fake((16, 16), dtypes.float16), qk_spec, qk_spec, rstd_spec, rstd_spec, activated_spec, matrix_spec,
            matrix_spec, fake((metadata_capacity,), dtypes.int32), dtypes.float32, dtypes.float32, dtypes.int64,
            dtypes.int64, dtypes.int64, dtypes.int64, dtypes.int64, dtypes.int64, dtypes.int64, dtypes.int64,
            dtypes.int64, dtypes.int64, dtypes.int64, dtypes.int64, dtypes.int64,
        )
        _COMPILED_KERNEL[template] = compiled_kernel
        return compiled_kernel


def _allocate_output(spec, device):
    """公开输出与内核视图共用按 chunk 对齐的存储，Meta 复用相同的步长。"""
    batch, _, n_v, storage_length, layout = spec
    dim = SUPPORTED_HEAD_DIM
    padded_length = (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE * CHUNK_SIZE
    if layout == "TND":
        out_storage = torch.empty((padded_length, n_v, dim), dtype=LOW_DTYPE, device=device)
        out = out_storage[:storage_length]
        out_stage = out_storage.unsqueeze(0).transpose(1, 2)
    else:
        # 定长输入统一返回 BSND，内核通过转置视图访问同一块存储。
        out_storage = torch.empty((batch, padded_length, n_v, dim), dtype=LOW_DTYPE, device=device)
        out = out_storage[:, :storage_length]
        out_stage = out_storage.transpose(1, 2)
    return out, out_stage


def _flash_kda_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    metadata: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[list[int]] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = SUPPORTED_HEAD_DIM ** (-0.5),
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
    internal_template: int = TEMPLATE_AUTO,
) -> Tuple[Optional[torch.Tensor], ...]:
    """前向返回固定的 16 个槽位，未启用的槽位为 None。

    模式由 torch.is_inference_mode_enabled() 决定，不依赖 requires_grad/no_grad。
    禁用 gate/beta 激活时使用预处理输入。转换或分配前检查静态输入，
    回读后检查 metadata，启动前检查内部缓冲区。"""
    is_meta = isinstance(q, torch.Tensor) and q.is_meta
    spec = validate_inputs(
        q, k, v, g, beta, initial_state, A_log, dt_bias,
        metadata=metadata, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, seqused=seqused, scale=scale,
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval, disable_recompute=disable_recompute,
        lower_bound=lower_bound, layout=layout,
        real_execution=not is_meta,
    )
    inference = torch.is_inference_mode_enabled()
    outputs = output_conditions(
        inference=inference, output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        disable_recompute=disable_recompute,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
    )
    if is_meta:
        return _meta_outputs(spec, outputs)
    core_num = _device_block_num(q)
    if torch.npu.is_current_stream_capturing():
        lengths, template = _capture_schedule(metadata, spec, cu_seqlens, seqused, q.device, core_num)
    else:
        lengths = read_lengths(cu_seqlens, seqused, spec)
        words = metadata.detach().cpu().tolist()
        decoded = decode_metadata(words)
        validate_schedule(decoded, spec, lengths, core_num,
                          cu_seqlens is not None or seqused is not None)
        template = words[HEADER_TEMPLATE]
    batch, n_qk, n_v, storage_length, layout = spec
    dim = SUPPORTED_HEAD_DIM
    physical_batches = 1 if layout == "TND" else batch
    if internal_template:
        require(template == internal_template, "metadata template does not match the selected entry")
    # persistent 共用一个存储开关控制预处理和递推缓存。
    # 独立请求 norm/beta 输出时仍需开启存储，再按输出条件筛选公开返回值。
    save_kernel_caches = outputs.caches or (
        template == TEMPLATE_PERSISTENT and not inference
        and (use_qk_l2norm_in_kernel or use_beta_sigmoid_in_kernel)
    )
    if scale is None:
        scale = dim ** -0.5
    has_initial_state = initial_state is not None
    if not has_initial_state:
        initial_state = torch.empty((batch, n_v, dim, dim), dtype=HIGH_DTYPE, device=q.device)
    # 转换前已检查原始输入的数据类型、设备和连续性。
    g = g.to(dtype=LOW_DTYPE)
    beta = beta.to(dtype=HIGH_DTYPE)

    def logical(value):
        if layout == "TND":
            return value.unsqueeze(0).transpose(1, 2)
        return value.transpose(1, 2) if layout == "BSND" else value

    q_logical, k_logical, v_logical = map(logical, (q, k, v))
    # 定长模式的 g/beta 固定使用 BSND 布局，不随 Q/K/V 的布局变化。
    g_logical = g.unsqueeze(0).transpose(1, 2) if layout == "TND" else g.transpose(1, 2)
    beta_logical = beta.unsqueeze(-1)
    beta_logical = beta_logical.unsqueeze(0).transpose(1, 2) if layout == "TND" else beta_logical.transpose(1, 2)
    out, out_stage = _allocate_output(spec, q.device)
    padded_length = (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE * CHUNK_SIZE
    gate_input_mode = staged.GATE_MODE_PREPROCESSED
    if use_gate_in_kernel:
        gate_input_mode = staged.GATE_MODE_FUSED_WITH_DT_BIAS
        kernel_lower_bound = 0.0 if lower_bound is None else lower_bound
        if dt_bias is None:
            gate_input_mode = staged.GATE_MODE_FUSED_WITHOUT_DT_BIAS
            dt_bias = initial_state.view(-1)[:n_v * dim].view(n_v, dim)
        else:
            # 两种公开 shape 统一为内核二维视图，不拷贝数据。
            dt_bias = dt_bias.view(n_v, dim)
    else:
        # 固定内核 ABI 即使不执行 gate 分支也需要张量占位。
        # 不检查或转换未使用的公开条件输入。
        kernel_lower_bound = -5.0
        state_flat = initial_state.view(-1)
        A_log = state_flat[:n_v]
        dt_bias = state_flat[:n_v * dim].view(n_v, dim)

    def empty_with_layout(ref, shape=None, dtype=None):
        shape = ref.shape if shape is None else shape
        stride = ref.stride()[:len(shape)]
        return torch.empty_strided(
            shape, stride, dtype=ref.dtype if dtype is None else dtype, device=q.device,
        )

    def empty_inverse_norm(ref):
        return torch.empty(ref.shape[:-1], dtype=HIGH_DTYPE, device=q.device)

    def empty_token_cache(dtype=LOW_DTYPE):
        return torch.empty(v_logical.shape, dtype=dtype, device=q.device)

    persistent = template == TEMPLATE_PERSISTENT
    save_tokens = not persistent or save_kernel_caches
    if outputs.norm:
        q_norm = empty_with_layout(q_logical, dtype=LOW_DTYPE)
        k_norm = empty_with_layout(k_logical, dtype=LOW_DTYPE)
        q_rstd = empty_inverse_norm(q_logical)
        k_rstd = empty_inverse_norm(k_logical)
    else:
        # 固定内核 ABI 接收形状兼容的视图；禁用路径不写入，公开返回值为 None。
        q_norm, k_norm = q_logical, k_logical
        q_rstd = empty_inverse_norm(q_logical)
        k_rstd = q_rstd
    beta_activated = (torch.empty(beta_logical.shape[:-1], dtype=HIGH_DTYPE, device=q.device)
                      if outputs.beta or (persistent and save_kernel_caches) else beta_logical.squeeze(-1))
    chunk_count = (
        staged._tnd_state_chunk_capacity(storage_length, initial_state.shape[0])
        if layout == "TND"
        else (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    )
    matrix_shape = (physical_batches, n_v, storage_length, CHUNK_SIZE)
    g_cumsum_shape = (physical_batches, n_v, storage_length, dim)
    # 矩阵直接按有效 token 行写入 GM，不为尾块或序列间预留填充行。
    # staged 的后续阶段读取 Aqk；persistent 在核内保留递推所需矩阵。
    if persistent and not outputs.matrices:
        matrix_shape = (1, 1, 1, CHUNK_SIZE)
    Aqk = torch.empty(matrix_shape, dtype=LOW_DTYPE, device=q.device)
    Akk = torch.empty(matrix_shape, dtype=LOW_DTYPE, device=q.device) if outputs.matrices else Aqk
    w = empty_token_cache() if save_tokens else v_logical
    u = empty_token_cache() if save_tokens else v_logical
    qg = empty_token_cache() if save_kernel_caches else v_logical
    kg = empty_token_cache() if save_tokens else v_logical
    v_new = empty_token_cache() if save_tokens else v_logical
    save_g_cumsum = int(save_kernel_caches)
    if not save_g_cumsum:
        g_cumsum_shape = (1, 1, 1, dim)
        g_cumsum = initial_state.view(-1)[:dim].view(g_cumsum_shape)
    else:
        g_cumsum = torch.empty(g_cumsum_shape, dtype=HIGH_DTYPE, device=q.device)
    state_is_compact = int(layout == "TND")
    save_qg = int(save_kernel_caches)
    if persistent and not outputs.chunk_states:
        h = initial_state[:1].unsqueeze(2)
    elif state_is_compact:
        state_chunks = staged._tnd_state_chunk_capacity(storage_length, batch)
        storage = torch.empty(
            (state_chunks, n_v, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
            dtype=HIGH_DTYPE,
            device=q.device,
        )
        h = storage.permute(1, 0, 2, 3).unsqueeze(0)
    else:
        h = torch.empty(
            (batch, n_v, (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE,
             SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
            dtype=HIGH_DTYPE,
            device=q.device,
        )

    workspace = staged.allocate_stage1_workspace(1, 1, WORKSPACE_SLOTS, SUPPORTED_HEAD_DIM, ref=q)
    qk_exchange = staged.allocate_qk_exchange_workspace(active_block_num=core_num, ref=q)
    final_state = torch.empty_like(initial_state) if outputs.final_state else initial_state
    # 编译或启动前，检查实际传入内核的张量视图。
    q_rstd_buffer = q_rstd.unsqueeze(-1)
    k_rstd_buffer = k_rstd.unsqueeze(-1)
    beta_buffer = beta_activated.unsqueeze(-1)
    q_shape = (physical_batches, n_qk, storage_length, dim)
    v_shape = (physical_batches, n_v, storage_length, dim)
    state_shape = (batch, n_v, dim, dim)
    chunk_state_shape = (physical_batches, n_v, chunk_count, dim, dim)
    if persistent and not outputs.chunk_states:
        chunk_state_shape = (1, n_v, 1, dim, dim)
    buffers = (
        ("k", k_logical, q_shape, LOW_DTYPE, False),
        ("v", v_logical, v_shape, LOW_DTYPE, False),
        ("q", q_logical, q_shape, LOW_DTYPE, False),
        ("beta", beta_logical, (physical_batches, n_v, storage_length, 1), HIGH_DTYPE, False),
        ("g", g_logical, v_shape, LOW_DTYPE, False),
        ("qk_exchange", qk_exchange, (core_num, 2, 2, CHUNK_SIZE), HIGH_DTYPE, True),
        ("A_log", A_log, (n_v,), HIGH_DTYPE, True),
        ("dt_bias", dt_bias, (n_v, dim), HIGH_DTYPE, True),
        ("kg", kg, v_shape, LOW_DTYPE, False),
        ("gamma_C", workspace.gamma_C, (WORKSPACE_SLOTS * dim, 1), HIGH_DTYPE, True),
        ("Q_decayed", workspace.Q_decayed, (WORKSPACE_SLOTS * CHUNK_SIZE, dim), LOW_DTYPE, True),
        ("u", u, v_shape, LOW_DTYPE, False),
        ("w", w, v_shape, LOW_DTYPE, False),
        ("qg", qg, v_shape, LOW_DTYPE, False),
        ("g_cumsum", g_cumsum, g_cumsum_shape, HIGH_DTYPE, False),
        ("v_new", v_new, v_shape, LOW_DTYPE, False),
        ("h", h, chunk_state_shape, HIGH_DTYPE, False),
        ("out_stage", out_stage, (physical_batches, n_v, padded_length, dim), LOW_DTYPE, False),
        ("initial_state", initial_state, state_shape, HIGH_DTYPE, True),
        ("final_state", final_state, state_shape, HIGH_DTYPE, True),
        ("identity16", workspace.identity16, (16, 16), torch.float16, True),
        ("q_norm", q_norm, q_shape, LOW_DTYPE, False),
        ("k_norm", k_norm, q_shape, LOW_DTYPE, False),
        ("q_rstd", q_rstd_buffer, (physical_batches, n_qk, storage_length, 1), HIGH_DTYPE, False),
        ("k_rstd", k_rstd_buffer, (physical_batches, n_qk, storage_length, 1), HIGH_DTYPE, False),
        ("beta_activated", beta_buffer, (physical_batches, n_v, storage_length, 1), HIGH_DTYPE, False),
        ("Aqk", Aqk, matrix_shape, LOW_DTYPE, False),
        ("Akk", Akk, matrix_shape, LOW_DTYPE, False),
    )
    for name, value, shape, dtype, contiguous in buffers:
        validate_buffer(name, value, shape, dtype, contiguous=contiguous)
        if len(shape) == 4 and name not in ("initial_state", "final_state"):
            require(value.stride(-1) == 1, f"{name} last stride must be 1")
    # BSND/TND 输出存储统一转换为内核使用的 BNSD 逻辑视图。
    expected_stride = (padded_length * n_v * dim, dim, n_v * dim, 1)
    for size, actual, expected in zip(out_stage.shape, out_stage.stride(), expected_stride):
        require(size == 1 or actual == expected, f"out_stage stride incompatible with {layout}")
    _get_compiled_kernel(template)(
        k_logical,
        v_logical,
        q_logical,
        beta_logical,
        g_logical,
        qk_exchange,
        A_log,
        dt_bias,
        kg,
        workspace.gamma_C,
        workspace.Q_decayed,
        u,
        w,
        qg,
        g_cumsum,
        v_new,
        h,
        out_stage,
        initial_state,
        final_state,
        workspace.identity16,
        q_norm,
        k_norm,
        q_rstd_buffer,
        k_rstd_buffer,
        beta_buffer,
        Aqk,
        Akk,
        metadata,
        float(scale),
        float(kernel_lower_bound),
        int(use_qk_l2norm_in_kernel),
        gate_input_mode,
        int(use_beta_sigmoid_in_kernel),
        int(allow_neg_eigval),
        int(use_gate_in_kernel and lower_bound is not None),
        state_is_compact,
        save_qg,
        save_g_cumsum,
        int(outputs.chunk_states),
        int(outputs.matrices),
        int(has_initial_state),
        int(outputs.final_state),
        core_num,
    )

    def token_public(value):
        # 内核缓冲区已采用公开接口要求的头维度在前的缓存布局。
        return value[0] if layout == "TND" else value

    def chunk_public(value):
        return value[0].transpose(0, 1).contiguous() if layout == "TND" else value.transpose(1, 2).contiguous()

    return (
        out,
        final_state if outputs.final_state else None,
        # 内部状态按 FP32 递推，公开的 chunk 入口状态保存为 BF16。
        chunk_public(h).to(dtype=LOW_DTYPE) if outputs.chunk_states else None,
        token_public(Aqk) if outputs.matrices else None,
        token_public(Akk) if outputs.matrices else None,
        token_public(w) if outputs.caches else None,
        token_public(u) if outputs.caches else None,
        token_public(qg) if outputs.caches else None,
        token_public(kg) if outputs.caches else None,
        token_public(v_new) if outputs.caches else None,
        token_public(g_cumsum) if outputs.caches else None,
        staged._public_bnsd_tensor(q_norm, layout) if outputs.norm else None,
        staged._public_bnsd_scalar(q_rstd, layout) if outputs.norm else None,
        staged._public_bnsd_tensor(k_norm, layout) if outputs.norm else None,
        staged._public_bnsd_scalar(k_rstd, layout) if outputs.norm else None,
        token_public(beta_activated) if outputs.beta else None,
    )


def flash_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    metadata: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[list[int]] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = SUPPORTED_HEAD_DIM ** (-0.5),
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
) -> Tuple[Optional[torch.Tensor], ...]:
    """按接口约定执行前向，返回固定的 16 个槽位。"""
    return _flash_kda_fwd_impl(
        q, k, v, g, beta, metadata, initial_state, A_log, dt_bias,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, seqused=seqused, scale=scale,
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval, disable_recompute=disable_recompute,
        lower_bound=lower_bound, layout=layout,
    )


def clear_caches():
    global _COMPILED_KERNEL, _COMPILED_KERNEL_ENVIRONMENT, _COMPILED_KERNEL_LOCK
    environment = compile_environment()
    if (
        _COMPILED_KERNEL_ENVIRONMENT is not None
        and _COMPILED_KERNEL_ENVIRONMENT[0] != environment[0]
    ):
        _COMPILED_KERNEL = {}
        _COMPILED_KERNEL_LOCK = threading.Lock()
        _COMPILED_KERNEL_ENVIRONMENT = environment
    with _COMPILED_KERNEL_LOCK:
        for compiled_kernel in _COMPILED_KERNEL.values():
            compiled_kernel.close()
        _COMPILED_KERNEL.clear()


@export("flash_kda_fwd")
def export_flash_kda_fwd():
    """Publish the same metadata-selected entries used by both public APIs."""
    export_flash_kda()



def _meta_outputs(spec, outputs):
    batch, n_qk, n_v, length, layout = spec
    chunks = (
        staged._tnd_state_chunk_capacity(length, batch)
        if layout == "TND" else (length + CHUNK_SIZE - 1) // CHUNK_SIZE
    )
    chunk_prefix = (chunks, n_v) if layout == "TND" else (batch, chunks, n_v)
    dim = SUPPORTED_HEAD_DIM
    norm_shape = ((length, n_qk, dim) if layout == "TND" else
                  ((batch, length, n_qk, dim) if layout == "BSND" else (batch, n_qk, length, dim)))
    rstd_shape = (n_qk, length) if layout == "TND" else (batch, n_qk, length)
    beta_shape = (n_v, length) if layout == "TND" else (batch, n_v, length)
    cache_shape = beta_shape + (dim,)
    state_shape = (batch, n_v, dim, dim)
    matrix_shape = beta_shape + (CHUNK_SIZE,)
    def empty(shape, dtype, enabled=True):
        return torch.empty(shape, dtype=dtype, device="meta") if enabled else None

    out, _ = _allocate_output(spec, "meta")
    return (
        out,
        empty(state_shape, HIGH_DTYPE, outputs.final_state),
        empty(chunk_prefix + (dim, dim), LOW_DTYPE, outputs.chunk_states),
        empty(matrix_shape, LOW_DTYPE, outputs.matrices),  # 因果 QK 矩阵 Aqk
        empty(matrix_shape, LOW_DTYPE, outputs.matrices),  # 块内逆矩阵 Akk
        empty(cache_shape, LOW_DTYPE, outputs.caches),  # 反向缓存 w
        empty(cache_shape, LOW_DTYPE, outputs.caches),  # 反向缓存 u
        empty(cache_shape, LOW_DTYPE, outputs.caches),  # 门控查询缓存 qg
        empty(cache_shape, LOW_DTYPE, outputs.caches),  # 恢复键缓存 kg
        empty(cache_shape, LOW_DTYPE, outputs.caches),  # 更新值缓存 v_new
        empty(cache_shape, HIGH_DTYPE, outputs.caches),
        empty(norm_shape, LOW_DTYPE, outputs.norm),
        empty(rstd_shape, HIGH_DTYPE, outputs.norm),
        empty(norm_shape, LOW_DTYPE, outputs.norm),
        empty(rstd_shape, HIGH_DTYPE, outputs.norm),
        empty(beta_shape, HIGH_DTYPE, outputs.beta),
    )


def flash_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    metadata: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[list[int]] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = SUPPORTED_HEAD_DIM ** (-0.5),
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run the shared forward and return (out, final_state, chunk_states)."""
    outputs = flash_kda_fwd(
        q, k, v, g, beta, metadata, initial_state, A_log, dt_bias,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, seqused=seqused, scale=scale,
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval, disable_recompute=disable_recompute,
        lower_bound=lower_bound, layout=layout,
    )
    return outputs[:3]


@export("flash_kda")
def export_flash_kda():
    """Collect both compute routes selected by the public metadata interface."""
    try:
        _get_compiled_kernel(TEMPLATE_STAGED)
        _get_compiled_kernel(TEMPLATE_PERSISTENT)
    finally:
        clear_caches()
