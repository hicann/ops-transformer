#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
import random
import copy
from einops import rearrange
from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


def input_funcs(bathc_size, seq_qkv, block_num, block_size):
    data = list(range(-1, block_num * block_size))
    if bathc_size * seq_qkv > block_num * block_size:
        sampled_data = random.sample(data, block_num * block_size)
        for _ in range(bathc_size * seq_qkv - block_num * block_size):
            sampled_data.append(-1)
    else:
        sampled_data = random.sample(data, bathc_size * seq_qkv)
    index = torch.Tensor(sampled_data)
    index = index.to(torch.int64)
    return index


@register("function_pyaclnn_qkv_rms_norm_rope_cache")
class Functionaclnnqkvrmsnormropecache(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        import random
        import ctypes
        from atk.tasks.backends.lib_interface.acl_wrapper import (
            nnopbase,
            AclDataType,
            AclFormat,
            AclTensor,
        )

        qkvSize = input_data.kwargs["qkvSize"]
        kCache = input_data.kwargs["kCache"]

        idx = self.task_result.case_config.id
        random.seed(idx)
        index = input_funcs(qkvSize[0], qkvSize[1], kCache.shape[0], kCache.shape[2])
        input_data.kwargs["index"] = index
        input_args, output_packages = super().init_by_input_data(input_data)

        AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        null_tensor_ptr = ctypes.cast(
            null_void_ptr, AclTensorPtr
        )  # 把这个空指针类型转换为tensor指针类型

        if type(input_args[9]).__name__ == "c_void_p":
            input_args[9] = null_tensor_ptr
        if type(input_args[10]).__name__ == "c_void_p":
            input_args[10] = null_tensor_ptr
        if type(input_args[11]).__name__ == "c_void_p":
            input_args[11] = null_tensor_ptr
        if type(input_args[12]).__name__ == "c_void_p":
            input_args[12] = null_tensor_ptr

        input_args.pop(17)
        input_args.pop(17)
        input_args.pop(17)

        output_packages[:] = [
            input_args[6],
            input_args[7],
            input_args[8],
            input_args[17],
            input_args[18],
            input_args[19],
        ]
        return input_args, output_packages

    def after_call(self, output_packages):
        q_out, kCache, vCache, q_out_proto, kCache_proto, vCache_proto = (
            super().after_call(output_packages)
        )
        return q_out, kCache, vCache, q_out_proto, kCache_proto, vCache_proto


@register("function_qkv_rms_norm_rope_cache")
class FunctionAdanceStep(BaseApi):
    def qkvRmsNormRopeCache_golden(
        self,
        qkv,
        gamma_q,
        gamma_k,
        cos,
        sin,
        index,
        q_out_ori,
        k_cache_ori,
        v_cache_ori,
        k_scale,
        v_scale,
        k_offset,
        v_offset,
        qkv_size,
        headNums,
        eps,
        cache_mode,
    ):
        q_out = copy.deepcopy(q_out_ori)  # q_out_ori为np，
        k_cache = copy.deepcopy(k_cache_ori)
        v_cache = copy.deepcopy(v_cache_ori)

        qkv_dtype = qkv.dtype

        qkv = qkv.to(torch.float32)
        gamma_q = gamma_q.to(torch.float32)
        gamma_k = gamma_k.to(torch.float32)
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

        Bqkv = qkv_size[0]
        Sqkv = qkv_size[1]
        Nqkv = qkv_size[2]
        Dqkv = qkv_size[3]

        Nq = headNums[0]
        Nk = headNums[1]
        Nv = headNums[2]

        assert Nqkv == Nq + Nk + Nv, "Inconsistent total head num!"
        assert Nk == Nv, "Inconsistent KV pair!"

        q, k, v = qkv.split([Nq * Dqkv, Nk * Dqkv, Nv * Dqkv], dim=-1)  # 2维

        q_4 = q.view(Bqkv, Sqkv, Nq, Dqkv)  # 4维[B, S, Nq, D]
        k_4 = k.view(Bqkv, Sqkv, Nk, Dqkv)  # 4维[B, S, Nk, D]
        v_4 = v.view(Bqkv, Sqkv, Nv, Dqkv)  # 4维[B, S, Nv, D]

        # parse Rope term sizes
        assert cos.shape[0] % Bqkv == 0, "Rope terms MUST matches batch size of qkv!"
        Srope = cos.shape[0] // Bqkv
        Nrope = cos.shape[1] // Dqkv
        cos = cos.view(Bqkv, Srope, Nrope, Dqkv)
        sin = sin.view(Bqkv, Srope, Nrope, Dqkv)

        # q计算
        rope_range = [0, Dqkv]
        q_4 = self.RmsNorm(q_4, eps, gamma_q)
        q_4 = self.Rope(q_4, cos, sin, rope_range)
        q_out_res = rearrange(q_4, "b s n d -> (b s)  (n d)")
        q_proto_out = q_out_res.clone()

        # k计算
        k_4 = self.RmsNorm(k_4, eps, gamma_k)
        real = k_4[..., : k_4.shape[-1] // 2]
        imag = k_4[..., k_4.shape[-1] // 2 :]
        k_4 = self.Rope(k_4, cos, sin, rope_range)
        k_cache_proto = k_4.clone()
        k_cache_proto = rearrange(k_cache_proto, "b s n d -> (b s)  (n d)")
        k_4 = self.Quant(k_4, k_scale, k_offset)
        k_cache_res = self.Scatter(k_4, k_cache, cache_mode, index)

        # v计算
        v_cache_proto = v_4.clone()
        v_cache_proto = rearrange(v_cache_proto, "b s n d -> (b s)  (n d)")
        v_4 = self.Quant(v_4, v_scale, v_offset)
        v_cache_res = self.Scatter(v_4, v_cache, cache_mode, index)

        # 输出
        return (
            q_out_res.to(qkv_dtype),
            k_cache_res,
            v_cache_res,
            q_proto_out.to(qkv_dtype),
            k_cache_proto.to(qkv_dtype),
            v_cache_proto.to(qkv_dtype),
        )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        import random
        import ctypes

        qkv = input_data.kwargs["qkv"]
        qGamma = input_data.kwargs["qGamma"]
        kGamma = input_data.kwargs["kGamma"]
        cos = input_data.kwargs["cos"]
        sin = input_data.kwargs["sin"]
        index = input_data.kwargs["index"]
        qOut = input_data.kwargs["qOut"]
        kCache = input_data.kwargs["kCache"]
        vCache = input_data.kwargs["vCache"]
        kScaleOptional = input_data.kwargs["kScaleOptional"]
        vScaleOptional = input_data.kwargs["vScaleOptional"]
        kOffsetOptional = input_data.kwargs["kOffsetOptional"]
        vOffsetOptional = input_data.kwargs["vOffsetOptional"]
        qkv_size = input_data.kwargs["qkvSize"]
        num_heads = input_data.kwargs["headNums"]
        epslison = input_data.kwargs["epsilon"]
        cacheModeOptional = input_data.kwargs["cacheModeOptional"]

        idx = self.task_result.case_config.id
        random.seed(idx)
        index = input_funcs(qkv_size[0], qkv_size[1], kCache.shape[0], kCache.shape[2])

        qOut, kCache, vCache, qOutBeforeQuant, kOutBeforeQuant, vOutBeforeQuant = (
            self.qkvRmsNormRopeCache_golden(
                qkv,
                qGamma,
                kGamma,
                cos,
                sin,
                index,
                qOut,
                kCache,
                vCache,
                kScaleOptional,
                vScaleOptional,
                kOffsetOptional,
                vOffsetOptional,
                qkv_size,
                num_heads,
                epslison,
                cacheModeOptional,
            )
        )
        return qOut, kCache, vCache, qOutBeforeQuant, kOutBeforeQuant, vOutBeforeQuant

    def RmsNorm(self, process, eps, gamma):
        rms_res = process / torch.sqrt(
            torch.mean(process**2, dim=-1, keepdim=True) + eps
        )
        rms_res = rms_res * gamma
        return rms_res

    def Rope(self, rms_res, cos, sin, rope_range):
        # rms_res为4维
        B = rms_res.shape[0]
        S = rms_res.shape[1]
        Nqkv = rms_res.shape[2]

        rope_dim = rope_range[1] - rope_range[0]
        assert rope_dim == cos.shape[-1], "Inconsistent Rope length!"
        # shape alignment
        Srope = cos.shape[1]
        Nrope = cos.shape[2]

        if Srope == 1:
            # align sequence length under MTP
            cos = cos.repeat(1, S, 1, 1)
            sin = sin.repeat(1, S, 1, 1)
        else:
            assert S == Srope, "Sequence length NOT ALIGNED!"
        if Nrope == 1:
            # expand shared rope factors shared among multiple heads
            cos = cos.repeat(1, 1, Nqkv, 1)
            sin = sin.repeat(1, 1, Nqkv, 1)
        else:
            assert Nqkv == Nrope, "Head num NOT ALIGNED!"

        rope_in = rms_res[..., rope_range[0] : rope_range[1]]  # rms_res[..., :rope_dim]
        rope_tmp1 = rope_in[..., : rope_in.shape[-1] // 2]
        rope_tmp2 = rope_in[..., rope_in.shape[-1] // 2 :]

        rotate_half = torch.cat((-rope_tmp2, rope_tmp1), dim=-1)
        rope_embed = (rope_in * cos) + (rotate_half * sin)
        # concatenation rope part with the remnants
        out = torch.cat(
            [rms_res[..., : rope_range[0]], rope_embed, rms_res[..., rope_range[1] :]],
            dim=-1,
        )
        return out

    def Quant(self, out, scale, offset):
        if scale is not None:
            out = out / scale
        if offset is not None:
            out = out + offset
        if scale is not None:
            out = torch.round(out).clamp(-128, 127)
        return out

    def Scatter(self, k_embed, k_cache, cache_mode, index):
        Bqkv, Sqkv, N, Dqkv = k_embed.shape
        k_cache_shape = k_cache.shape
        bn = k_cache_shape[0]
        block_size = k_cache_shape[2]
        dk0 = k_cache_shape[-1]
        dk1 = Dqkv // dk0
        num_head = N
        k_cache = k_cache.reshape(bn, num_head, dk1, block_size, dk0)
        k_embed = rearrange(k_embed, "b s n d -> (b s)  n d")
        for batch in range(len(index)):
            index_value = index[batch]
            if index_value < 0:
                continue
            bn_id = index_value // block_size
            block_offset = index_value % block_size
            for i in range(dk1):
                k_cache[bn_id, :, i, block_offset, :] = k_embed[
                    batch, :, i * dk0 : (i + 1) * dk0
                ].to(k_cache.dtype)
        k_cache = k_cache.reshape(k_cache_shape)
        return k_cache
