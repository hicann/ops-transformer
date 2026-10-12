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

from atk.configs.dataset_config import InputDataset

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("function_aclnn_dequant_rope_quant_kvcache")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        QUANTMAX = 127
        QUANTMIN = -128

        def tof(x_npu):
            if x_npu != None:
                if x_npu.dtype == torch.bfloat16:
                    return x_npu.to(torch.float)
            return x_npu

        def dequant(input, bias, weight_scale, activation_scale):
            input_tensor = input
            ws_cpu = weight_scale.reshape([1, 1, -1])
            if activation_scale != None:
                as_cpu = activation_scale.reshape(
                    [input_tensor.shape[0], input_tensor.shape[1], 1]
                )
            else:
                as_cpu = None

            if bias != None:
                bias_cpu = bias.reshape([1, 1, -1])
                if bias.dtype == torch.int32:
                    input_tensor = torch.add(input_tensor, bias_cpu)
                    input_tensor = torch.mul(input_tensor, ws_cpu)
                    if as_cpu != None:
                        input_tensor = torch.mul(input_tensor, as_cpu)
                else:
                    input_tensor = torch.mul(input_tensor, ws_cpu)
                    if as_cpu != None:
                        input_tensor = torch.mul(input_tensor, as_cpu)
                    input_tensor = torch.add(input_tensor, bias_cpu)
            else:
                input_tensor = torch.mul(input_tensor, ws_cpu)
                if as_cpu != None:
                    input_tensor = torch.mul(input_tensor, as_cpu)
            return input_tensor

        def rope(x, cos, sin):
            dd = x.dtype
            x1 = x
            sin1 = sin
            cos1 = cos
            if dd == torch.bfloat16:
                x1 = x
                cos1 = cos
                sin1 = sin
            axis_d = x.shape[-1]
            rotary_x = torch.concat(
                (-x1[..., axis_d // 2 :], x1[..., : axis_d // 2]), dim=-1
            )
            return x1 * cos1 + rotary_x * sin1

        def quant_update_scatter(key_cache, key, scale, indice, offset, ifpa=False):
            # quant
            quant_out = []
            scale = scale.reshape([-1, key.shape[-1]])
            offset = offset.reshape([-1, key.shape[-1]])
            if offset != None:
                quant_out = key.float() * scale + offset
            else:
                quant_out = key.float() * scale

            quant_out = quant_out.round()
            d0 = key_cache.shape[0]
            d1 = key_cache.shape[1]
            d2 = key_cache.shape[2]
            d3 = key_cache.shape[3]

            # scatter
            quant_out1 = torch.clamp(
                torch.round(quant_out.float()), min=QUANTMIN, max=QUANTMAX
            )
            quant_out2 = quant_out1.reshape(
                -1, quant_out1.shape[-2], quant_out1.shape[-1]
            )
            if ifpa:
                key_cachepa = key_cache.reshape(
                    -1, key_cache.shape[-2], key_cache.shape[-1]
                )

                for b in range(indice.shape[0]):
                    indice_value = indice[b]
                    key_cachepa[indice_value] = quant_out2[b]
                key_cache = key_cachepa.reshape([d0, d1, d2, d3])
            else:
                for b in range(indice.shape[0]):
                    indice_value = indice[b]

                    key_cache[b][indice_value : indice_value + quant_out.shape[1]][:][
                        :
                    ] = quant_out1[b][:][:][:]

        def drqk(
            x_npu,
            cos_npu,
            sin_npu,
            k_cache_npu,
            v_cache_npu,
            indices_npu,
            scale_k_npu,
            scale_v_npu,
            size_splits,
            offset_k,
            offset_v,
            weight_scale,
            activation_scale,
            bias,
            ifpa="contiguous",
        ):
            if x_npu.ndimension() == 2:
                x_npu = x_npu.reshape(x_npu.shape[-2], 1, x_npu.shape[-1])
                cos_npu = cos_npu.reshape(cos_npu.shape[-2], 1, 1, cos_npu.shape[-1])
                sin_npu = sin_npu.reshape(sin_npu.shape[-2], 1, 1, sin_npu.shape[-1])
            out_type = cos_npu.dtype

            if bias != None:
                if bias.dtype != torch.int32:
                    bias = bias.to(torch.float32)
            # cos_npu = tof(cos_npu)
            # sin_npu = tof(sin_npu)
            activation_scale = tof(activation_scale)
            weight_scale = tof(weight_scale)

            if x_npu.dtype == torch.int32:
                x_npu = dequant(x_npu, bias, weight_scale, activation_scale)
                # if out_type == torch.float16:
                #     x_npu = x_npu.to(out_type)
                x_npu = x_npu.to(out_type)

            # x_npu = tof(x_npu)

            b = x_npu.shape[0]
            # h = scale_k_npu.shape[-1]
            h = k_cache_npu.shape[-1]
            if x_npu.ndimension() == 2:
                s = 1
            else:
                s = x_npu.shape[1]
            q, k, v = x_npu.split(size_splits, dim=-1)
            print("*****************************************************************")
            print("q.shape:", q.shape)
            print("b:", b)
            print("s:", s)
            print("h:", h)
            q1 = q.reshape([b, s, -1, h])
            k1 = k.reshape([b, s, -1, h])
            v1 = v.reshape([b, s, -1, h])
            ropek = rope(k1, cos_npu, sin_npu)
            ropeq = rope(q1, cos_npu, sin_npu)

            quant_update_scatter(
                k_cache_npu,
                ropek,
                1 / scale_k_npu.to(torch.float),
                indices_npu,
                offset_k,
                (ifpa != "contiguous"),
            )
            quant_update_scatter(
                v_cache_npu,
                v1,
                1 / scale_v_npu.to(torch.float),
                indices_npu,
                offset_v,
                (ifpa != "contiguous"),
            )
            return ropeq, ropek, v1, k_cache_npu, v_cache_npu

        qout, kout, vout, k_cache_npu, v_cache_npu = drqk(
            input_data.kwargs["x"],
            input_data.kwargs["cos"],
            input_data.kwargs["sin"],
            input_data.kwargs["kCacheRef"],
            input_data.kwargs["vCacheRef"],
            input_data.kwargs["indices"],
            input_data.kwargs["scaleK"],
            input_data.kwargs["scaleV"],
            input_data.kwargs["sizeSplits"],
            input_data.kwargs["offsetKOptional"],
            input_data.kwargs["offsetVOptional"],
            input_data.kwargs["weightScaleOptional"],
            input_data.kwargs["activationScaleOptional"],
            input_data.kwargs["biasOptional"],
            input_data.kwargs["cacheModeOptional"],
        )

        return qout, kout, vout

    def init_by_input_data(self, input_data: InputDataset):
        if input_data.kwargs["cacheModeOptional"] == "page":
            indices_shape = input_data.kwargs["indices"].shape[0]
            high = (
                input_data.kwargs["vCacheRef"].shape[0]
                * input_data.kwargs["vCacheRef"].shape[1]
            )
            indices = torch.randint(low=0, high=high, size=(high,)).to(torch.int32)
            indices = torch.argsort(indices).to(torch.int32)
            indices = indices[:indices_shape]
            input_data.kwargs["indices"] = indices

    # def get_cpp_func_signature_type(self):
    #         return "aclnnStatus aclnnMoeTokenUnpermuteGradGetWorkspaceSize( \
    #         const aclTensor* permuteTokens, const aclTensor* unpermutedTokensGrad, const aclTensor* sortedIndices, \
    #         const aclTensor* probsOptional, bool paddedMode, const aclIntArray* restoreShapeOptional, \
    #         aclTensor* permutedTokensGradOut, aclTensor* probsGradOut, uint64_t* workspaceSize, aclOpExecutor** executor)"
