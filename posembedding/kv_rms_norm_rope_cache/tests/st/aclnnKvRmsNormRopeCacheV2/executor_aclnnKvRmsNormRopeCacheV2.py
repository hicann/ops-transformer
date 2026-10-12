#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
import copy
import einops

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.configs.results_config import TaskResult

import random

Method_mode = 1  # V2 mode


K_CACHE_REF_IDX = 5
V_CACHE_REF_IDX = 6
USE_PRINT = False


@register("aclnn_function")
class FunctionaclnnKvRmsNormRopeCacheV2(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        import random
        import ctypes
        from atk.tasks.backends.lib_interface.acl_wrapper import (
            nnopbase,
            AclDataType,
            AclFormat,
            AclTensor,
        )

        idx = self.task_result.case_config.id
        # torch.random.manual_seed(idx)
        kv = input_data.kwargs["kv"]
        gamma = input_data.kwargs["gamma"]
        cos_ = input_data.kwargs["cos"]
        sin_ = input_data.kwargs["sin"]
        index = input_data.kwargs["index"]
        k_cache = input_data.kwargs["k_cache"]  # 5
        ckv_cache = input_data.kwargs["ckv_cache"]  # 6
        k_scale = input_data.kwargs["k_scale"]
        v_scale = input_data.kwargs["v_scale"]
        k_offset = input_data.kwargs["k_offset"]
        v_offset = input_data.kwargs["v_offset"]
        v_optional = input_data.kwargs["vOptional"]
        eps = input_data.kwargs["eps"]
        cache_mode = input_data.kwargs["cacheMode"]
        is_output_kv = input_data.kwargs["isOutputKv"]

        index_shape = index.shape
        kv_shape = kv.shape
        Bkv = kv_shape[0]
        Nkv = kv_shape[1]
        Skv = kv_shape[2]
        Dkv = kv_shape[3]

        g = torch.Generator()
        g.manual_seed(self.task_result.case_config.id)

        # PA BLK场景
        if "BLK" in cache_mode:
            S = k_cache.shape[1]
            B = k_cache.shape[0]
            CeilDivS = (Skv + S - 1) // S
            data = list(range(0, Bkv * CeilDivS * S, S))
            index = torch.Tensor(data)
            index = index.to(torch.int64)
        # PA非BLK场景
        elif "PA" in cache_mode:
            # 该场景index的值表示BS维度上的偏移
            S = k_cache.shape[1]
            B = k_cache.shape[0]
            data = list(range(-1, B * S))
            data = torch.Tensor(data)
            if Bkv * Skv > B * S:
                random_indices = torch.randperm(len(data), generator=g)
                sampled_data = data[random_indices[: B * S]]
                for _ in range(Bkv * Skv - B * S):
                    sampled_data.append(-1)
            else:
                random_indices = torch.randperm(len(data), generator=g)
                sampled_data = data[random_indices[: Bkv * Skv]]
            index = sampled_data
            index = index.to(torch.int64)
        # Norm场景
        else:
            S = k_cache.shape[2]
            data = list(range(-1, S))
            data = torch.Tensor(data)
            index_list = []
            index = torch.zeros(Bkv, Skv).to(torch.int64)
            for i in range(Bkv):
                # 输入S大于CacheS，防止同地址写，多余S需要丢弃设置index为-1
                if Skv > S:
                    random_indices = torch.randperm(len(data), generator=g)
                    sampled_data = data[random_indices[:S]]
                    for _ in range(Skv - S):
                        sampled_data.append(-1)
                else:
                    random_indices = torch.randperm(len(data), generator=g)
                    sampled_data = data[random_indices[:Skv]]
                sub_index = sampled_data.to(torch.int64)
                index[i, :] = sub_index

        input_data.kwargs["index"] = index

        input_args, output_packages = super().init_by_input_data(input_data)

        AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        null_tensor_ptr = ctypes.cast(
            null_void_ptr, AclTensorPtr
        )  # 把这个空指针类型转换为tensor指针类型

        if type(input_args[7]).__name__ == "c_void_p":
            input_args[7] = null_tensor_ptr
        if type(input_args[8]).__name__ == "c_void_p":
            input_args[8] = null_tensor_ptr
        if type(input_args[9]).__name__ == "c_void_p":
            input_args[9] = null_tensor_ptr
        if type(input_args[10]).__name__ == "c_void_p":
            input_args[10] = null_tensor_ptr
        if type(input_args[11]).__name__ == "c_void_p":
            input_args[11] = null_tensor_ptr

        input_args.pop(15)
        input_args.pop(15)

        self.is_output_kv = input_data.kwargs["isOutputKv"]
        output_packages[:] = [
            input_args[5],
            input_args[6],
            input_args[15],
            input_args[16],
        ]

        return input_args, output_packages

    def after_call(self, output_packages):
        k_cache, ckv_cache, k_embed, v_embed = super().after_call(output_packages)
        if not self.is_output_kv:
            k_embed.zero_()
            v_embed.zero_()
        # print(f'pyaclnn k_cache = {k_cache}')
        # print(f'pyaclnn ckv_cache = {ckv_cache}')
        # print(f'pyaclnn k_embed = {k_embed}')
        # print(f'pyaclnn k_embed shape = {k_embed.shape}')

        # print(f'pyaclnn v_embed = {v_embed}')
        return k_cache, ckv_cache, k_embed, v_embed


@register("function_kvrnrcV2")
class FunctionKvRmsNormRopeCache(BaseApi):
    def KvRmsNormRopeCacheV2(
        self,
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        k_offset,
        v_offset,
        v_optional,
        eps,
        cache_mode,
        is_output_kv,
    ):
        from einops import rearrange

        ori_dtype = kv.dtype
        ori_k_cache_dtype = k_cache.dtype
        ori_v_cache_dtype = v_cache.dtype
        tensor_v = v_optional

        method_mode = 0 if tensor_v is None else 1
        kv_dtype = kv.dtype
        # print(method_mode)
        if method_mode == 1:
            tensor_v = tensor_v.to(torch.float32)

        kv = kv.to(torch.float32)
        gamma = gamma.to(torch.float32)
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

        kv_shape = kv.shape
        Bkv = kv_shape[0]

        # Nkv = 1
        Nkv = kv_shape[1]
        Skv = kv_shape[2]
        Dkv = kv_shape[3]

        v_dim = gamma.shape[-1] if method_mode == 0 else tensor_v.shape[3]
        k_dim = cos.shape[-1] if method_mode == 0 else Dkv

        kv = rearrange(kv, "b n s d -> b s n d")
        cos = rearrange(cos, "b n s d -> b s n d")
        sin = rearrange(sin, "b n s d -> b s n d")

        if method_mode == 0:
            rms_in, rope_in = kv.split([v_dim, k_dim], dim=-1)
            # do rms
            v = rms_in / torch.sqrt(torch.mean(rms_in**2, dim=-1, keepdim=True) + eps)
            v = v * gamma
            v_out = rearrange(v, "b s n d -> b n s d")
            if v_scale is not None:
                v = v * v_scale
            if v_offset is not None:
                v = v + v_offset
            if v_scale is not None:
                v = torch.round(v).clamp(-128, 127)

            # do rope
            k = (
                rope_in.view(Bkv, Skv, Nkv, k_dim // 2, 2)
                .transpose(-1, -2)
                .reshape(Bkv, Skv, Nkv, k_dim)
            )
            k1 = k[..., : k.shape[-1] // 2]
            k2 = k[..., k.shape[-1] // 2 :]
            rotate_half_k = torch.cat((-k2, k1), dim=-1)
            k_embed = (k * cos) + (rotate_half_k * sin)
            k_embed_out = rearrange(k_embed, "b s n d -> b n s d")
            if k_scale is not None:
                k_embed = k_embed * k_scale
            if k_offset is not None:
                k_embed = k_embed + k_offset
            if k_scale is not None:
                k_embed = torch.round(k_embed).clamp(-128, 127)

        else:
            if cache_mode == "Norm" and k_scale is None and v_scale is None:
                is_output_kv = False
            rms_in = kv
            # print("rmsNorm前结果", rms_in)
            v_in = tensor_v
            v_in = rearrange(v_in, "b n s d -> b s n d")
            # do rms

            v = rms_in / torch.sqrt(torch.mean(rms_in**2, dim=-1, keepdim=True) + eps)
            v = v * gamma
            # print("rmsNorm后，rope前结果(fp32)", v)
            # do rope
            rope_dim = cos.shape[-1]
            rope_in = v[..., :rope_dim]
            k = (
                rope_in.view(Bkv, Skv, Nkv, rope_dim // 2, 2)
                .transpose(-1, -2)
                .reshape(Bkv, Skv, Nkv, rope_dim)
            )
            k1 = k[..., : k.shape[-1] // 2]
            k2 = k[..., k.shape[-1] // 2 :]
            # print("GatherMask 偶数", k1)
            # print("k1.shape=",k1.shape)
            # print("GatherMask 奇数", k2)
            # print("k2.shape=",k2.shape)
            rotate_half_k = torch.cat((-k2, k1), dim=-1)
            k_embed = (k * cos) + (rotate_half_k * sin)

            # print("k_embed.shape=", k_embed.shape)
            # print("v[...,rope_dim:].shape=", v[...,rope_dim:].shape)
            kv_out = torch.cat([k_embed, v[..., rope_dim:]], dim=-1)
            # print("kv_out.shape=", kv_out.shape)
            # print("RMS Rope后结果,fp32", kv_out)
            # print("Rms Norm 和 Rope计算后结果(bf16/half)", kv_out.to(torch.float16))
            k_embed_out = rearrange(kv_out, "b s n d -> b n s d").to(kv_dtype)
            # print("Rms Norm 和 Rope计算后结果(bf16/half)", k_embed_out)
            # print("Kv 可选输出", k_embed_out)
            if k_scale is not None:
                kv_out = kv_out * k_scale
            if k_offset is not None:
                kv_out = kv_out + k_offset
            # print("2 int8 之前",kv_out)
            if k_scale is not None:
                kv_out = torch.round(kv_out).clamp(-128, 127)
            k_embed = kv_out
            # print("kScale.shape=", k_scale.shape)
            # print("kScale:", k_scale)
            # print("kOffset.shape=", k_offset.shape)
            # print("kOffset:", k_offset)

            # print("kv Quant之后的结果", rearrange(kv_out, 'b s n d -> b n s d'))
            # do tensor v
            v_out = rearrange(v_in, "b s n d -> b n s d").to(kv_dtype)
            # print("v Quant之前的结果", v_out)
            # print("Tensor v初始值", v_out)
            if v_scale is not None:
                v_in = v_in * v_scale
            if v_offset is not None:
                v_in = v_in + v_offset
            if v_scale is not None:
                v_in = torch.round(v_in).clamp(-128, 127)
            v = v_in
            # print("v Quant之后的结果", rearrange(v_in, 'b s n d -> b n s d'))

        # PA场景
        if cache_mode == "PA_BNSD" or cache_mode == "PA":
            # print("isPA_BNSD")
            k_cache_shape = k_cache.shape
            v_cache_shape = v_cache.shape
            k_cache = rearrange(k_cache, "b s n d -> (b s)  n d")
            v_cache = rearrange(v_cache, "b s n d -> (b s)  n d")
            k_embed = rearrange(k_embed, "b s n d -> (b s)  n d")
            v = rearrange(v, "b s n d -> (b s) n d")
            for batch in range(len(index)):
                if index[batch] == -1:
                    continue

                k_cache[index[batch], :, :] = k_embed[batch, :, :].to(k_cache.dtype)
                v_cache[index[batch], :, :] = v[batch, :, :].to(v_cache.dtype)
            k_cache = rearrange(k_cache, "(b s) n d -> b s n d", b=k_cache_shape[0])
            v_cache = rearrange(v_cache, "(b s) n d -> b s n d", b=v_cache_shape[0])

        elif cache_mode == "PA_NZ":
            # print("isPA_NZ")
            k_cache_shape = k_cache.shape
            v_cache_shape = v_cache.shape
            bn = k_cache_shape[0]
            block_size = k_cache_shape[1]
            dk = k_cache_shape[-1]
            dv = v_cache_shape[-1]
            dk0 = 32 if k_cache.dtype == torch.int8 else 16
            dv0 = 32 if v_cache.dtype == torch.int8 else 16
            dk1 = dk // dk0
            dv1 = dv // dv0
            num_head = k_cache_shape[2]
            k_cache = k_cache.reshape(bn, num_head, dk1, block_size, dk0)
            v_cache = v_cache.reshape(bn, num_head, dv1, block_size, dv0)
            k_embed = rearrange(k_embed, "b s n d -> (b s)  n d")
            v = rearrange(v, "b s n d -> (b s) n d")
            # batch 是 B * S 的index
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
                for i in range(dv1):
                    v_cache[bn_id, :, i, block_offset, :] = v[
                        batch, :, i * dv0 : (i + 1) * dv0
                    ].to(v_cache.dtype)
            k_cache = k_cache.reshape(k_cache_shape)
            v_cache = v_cache.reshape(v_cache_shape)

        elif cache_mode == "PA_BLK_BNSD":
            # print("isPA_BLK_BNSD")
            k_cache_shape = k_cache.shape
            v_cache_shape = v_cache.shape
            block_size = k_cache_shape[1]
            block_num = k_cache_shape[0]
            ceil_div_s = (Skv + block_size - 1) // block_size
            for batch in range(Bkv):
                for seq_id in range(ceil_div_s):
                    seq_start = seq_id * block_size
                    seq_end = (
                        Skv if seq_id == (ceil_div_s - 1) else (seq_id + 1) * block_size
                    )
                    copy_len = seq_end - seq_start
                    index_value = index[batch * ceil_div_s + seq_id]
                    cache_b = index_value // block_size
                    if index_value == -1:
                        continue
                    k_cache[cache_b, :copy_len, :, :] = k_embed[
                        batch, seq_start:seq_end, :, :
                    ].to(k_cache.dtype)
                    v_cache[cache_b, :copy_len, :, :] = v[
                        batch, seq_start:seq_end, :, :
                    ].to(v_cache.dtype)

        elif cache_mode == "PA_BLK_NZ":
            # print("isPA_BLK_NZ")
            k_cache_shape = k_cache.shape
            v_cache_shape = v_cache.shape
            bn = k_cache_shape[0]
            block_size = k_cache_shape[1]
            dk = k_cache_shape[-1]  # 192
            dv = v_cache_shape[-1]  # 128
            dk0 = 32 if k_cache.dtype == torch.int8 else 16
            dv0 = 32 if v_cache.dtype == torch.int8 else 16
            dk1 = dk // dk0
            dv1 = dv // dv0
            num_head = k_cache_shape[2]
            k_cache = k_cache.reshape(bn, num_head, dk1, block_size, dk0)
            v_cache = v_cache.reshape(bn, num_head, dv1, block_size, dv0)
            ceil_div_s = (Skv + block_size - 1) // block_size
            # print("v.shape=", v.shape)
            # print("v.value=", v)
            for batch in range(Bkv):
                for seq_id in range(ceil_div_s):
                    seq_start = seq_id * block_size
                    seq_end = (
                        Skv if seq_id == (ceil_div_s - 1) else (seq_id + 1) * block_size
                    )
                    copy_len = seq_end - seq_start
                    index_value = index[batch * ceil_div_s + seq_id]
                    cache_b = index_value // block_size
                    if index_value == -1:
                        continue
                    for n_idx in range(num_head):
                        for i in range(dk1):
                            # k_cache -> blksize, dk0    k_embed->blksize, dk0
                            k_cache[cache_b, n_idx, i, :copy_len, :] = k_embed[
                                batch, seq_start:seq_end, n_idx, i * dk0 : (i + 1) * dk0
                            ].to(k_cache.dtype)
                        for i in range(dv1):
                            v_cache[cache_b, n_idx, i, :copy_len, :] = v[
                                batch, seq_start:seq_end, n_idx, i * dv0 : (i + 1) * dv0
                            ].to(v_cache.dtype)
            # print("v_cache.shape=", v_cache.shape)
            # print("v_cache.value=", v_cache)
            k_cache = k_cache.reshape(k_cache_shape)
            v_cache = v_cache.reshape(v_cache_shape)
            # print("reshape 之后的")
            # print("v_cache.shape=", v_cache.shape)
            # print("v_cache.value=", v_cache)
        else:
            v_cache = rearrange(v_cache, "b n s d -> b s n d")
            k_cache = rearrange(k_cache, "b n s d -> b s n d")
            for batch in range(index.shape[0]):
                for sdx in range(index.shape[1]):
                    if index[batch][sdx] == -1:
                        continue
                    v_cache[batch, index[batch][sdx], :, :] = v[batch, sdx, :, :].to(
                        v_cache.dtype
                    )
                    k_cache[batch, index[batch][sdx], :, :] = k_embed[
                        batch, sdx, :, :
                    ].to(k_cache.dtype)
            v_cache = rearrange(v_cache, "b s n d -> b n s d")
            k_cache = rearrange(k_cache, "b s n d -> b n s d")
        # package outputs
        if is_output_kv:
            output_data = [
                k_cache.to(ori_k_cache_dtype),
                v_cache.to(ori_v_cache_dtype),
                k_embed_out.to(ori_dtype),
                v_out.to(ori_dtype),
            ]
        else:
            output_data = [
                k_cache.to(ori_k_cache_dtype),
                v_cache.to(ori_v_cache_dtype),
                torch.zeros_like(k_embed_out).to(ori_dtype),
                torch.zeros_like(v_out).to(ori_dtype),
            ]
        return output_data

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kv = input_data.kwargs["kv"]
        gamma = input_data.kwargs["gamma"]
        cos_ = input_data.kwargs["cos"]
        sin_ = input_data.kwargs["sin"]
        index = input_data.kwargs["index"]
        k_cache = input_data.kwargs["k_cache"]  # 5
        ckv_cache = input_data.kwargs["ckv_cache"]  # 6
        k_scale = input_data.kwargs["k_scale"]
        v_scale = input_data.kwargs["v_scale"]
        k_offset = input_data.kwargs["k_offset"]
        v_offset = input_data.kwargs["v_offset"]
        v_optional = input_data.kwargs["vOptional"]
        eps = input_data.kwargs["eps"]
        cache_mode = input_data.kwargs["cacheMode"]
        is_output_kv = input_data.kwargs["isOutputKv"]

        self.is_output_kv = is_output_kv

        k_cache, ckv_cache, k_embed, v_embed = self.KvRmsNormRopeCacheV2(
            kv,
            gamma,
            cos_,
            sin_,
            index,
            k_cache,
            ckv_cache,
            k_scale,
            v_scale,
            k_offset,
            v_offset,
            v_optional,
            eps,
            cache_mode,
            is_output_kv,
        )

        # k_embed = k_embed.permute(0,2,1,3).view(16,8,128,192)
        # print(f'golden k_cache = {k_cache}')
        # print(f'golden ckv_cache = {ckv_cache}')
        # print(f'golden k_embed = {k_embed}')
        # print(f'golden k_embed shape = {k_embed.shape}')
        # print(f'golden v_embed = {v_embed}')
        return k_cache, ckv_cache, k_embed.contiguous(), v_embed.contiguous()

    def init_by_input_data(self, input_data: InputDataset):
        kv = input_data.kwargs["kv"]
        gamma = input_data.kwargs["gamma"]
        cos_ = input_data.kwargs["cos"]
        sin_ = input_data.kwargs["sin"]
        index = input_data.kwargs["index"]
        k_cache = input_data.kwargs["k_cache"]  # 5
        ckv_cache = input_data.kwargs["ckv_cache"]  # 6
        k_scale = input_data.kwargs["k_scale"]
        v_scale = input_data.kwargs["v_scale"]
        k_offset = input_data.kwargs["k_offset"]
        v_offset = input_data.kwargs["v_offset"]
        v_optional = input_data.kwargs["vOptional"]
        eps = input_data.kwargs["eps"]
        cache_mode = input_data.kwargs["cacheMode"]
        is_output_kv = input_data.kwargs["isOutputKv"]

        idx = self.task_result.case_config.id
        # torch.random.manual_seed(idx)

        index_shape = index.shape
        kv_shape = kv.shape
        Bkv = kv_shape[0]
        Nkv = kv_shape[1]
        Skv = kv_shape[2]
        Dkv = kv_shape[3]

        g = torch.Generator()
        g.manual_seed(self.task_result.case_config.id)

        # PA BLK场景
        if "BLK" in cache_mode:
            S = k_cache.shape[1]
            B = k_cache.shape[0]
            CeilDivS = (Skv + S - 1) // S
            data = list(range(0, Bkv * CeilDivS * S, S))
            index = torch.Tensor(data)
            index = index.to(torch.int64)
        # PA非BLK场景
        elif "PA" in cache_mode:
            # 该场景index的值表示BS维度上的偏移
            S = k_cache.shape[1]
            B = k_cache.shape[0]
            data = list(range(-1, B * S))
            data = torch.Tensor(data)
            if Bkv * Skv > B * S:
                random_indices = torch.randperm(len(data), generator=g)
                sampled_data = data[random_indices[: B * S]]
                for _ in range(Bkv * Skv - B * S):
                    sampled_data.append(-1)
            else:
                random_indices = torch.randperm(len(data), generator=g)
                sampled_data = data[random_indices[: Bkv * Skv]]
            index = sampled_data
            index = index.to(torch.int64)
        # Norm场景
        else:
            S = k_cache.shape[2]
            data = list(range(-1, S))
            data = torch.Tensor(data)
            index_list = []
            index = torch.zeros(Bkv, Skv).to(torch.int64)
            for i in range(Bkv):
                # 输入S大于CacheS，防止同地址写，多余S需要丢弃设置index为-1
                if Skv > S:
                    random_indices = torch.randperm(len(data), generator=g)
                    sampled_data = data[random_indices[:S]]
                    for _ in range(Skv - S):
                        sampled_data.append(-1)
                else:
                    random_indices = torch.randperm(len(data), generator=g)
                    sampled_data = data[random_indices[:Skv]]
                sub_index = sampled_data.to(torch.int64)
                index[i, :] = sub_index

        input_data.kwargs["index"] = index
