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
import numpy
from atk.configs.dataset_config import InputDataset

from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult


def image_rotary_emb(hidden_states, rope_sin, rope_cos, mode=1):
    out = torch.empty_like(hidden_states)
    if mode == 1:
        x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
        x1, x2 = x[..., 0], x[..., 1]
        rotated_x = torch.stack([-x2, x1], dim=-1).flatten(3)

        out = hidden_states.float() * rope_cos + rotated_x.float() * rope_sin
        return out.type_as(hidden_states)
    else:
        x1, x2 = hidden_states.reshape(*hidden_states.shape[:-1], 2, -1).unbind(-2)

        rotated_x = torch.cat([-x2, x1], dim=-1)

        out = hidden_states.float() * rope_cos + rotated_x.float() * rope_sin
        return out.type_as(hidden_states)


def layer_norm(x, weight=None, bias=None, eps=1e-5, is_training=True):
    x = x.float()
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, correction=0, keepdim=True)
    rstd = 1 / torch.sqrt(var + eps)
    out = (x - mean) * rstd
    if weight is not None:
        weight = weight.float()
        bias = bias.float()
        out = out * weight + bias
    if not is_training:
        mean = torch.zeros(1, dtype=torch.float32, device=x.device)
        rstd = torch.zeros(1, dtype=torch.float32, device=x.device)

    return [out, mean, rstd]


def rms_norm(x, weight=None, eps=1e-5):
    x = x.float()
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    out = x / rms
    if weight is not None:
        weight = weight.float()
        out = out * weight
    return out


@register("torch_norm_rope_concat")
class TorchNormRopeConcatApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        query = input_data.kwargs.get("query")
        key = input_data.kwargs.get("key")
        value = input_data.kwargs.get("value")
        encoder_query = input_data.kwargs.get("encoder_query")
        encoder_key = input_data.kwargs.get("encoder_key")
        encoder_value = input_data.kwargs.get("encoder_value")

        norm_query_weight = input_data.kwargs.get("norm_query_weight")
        norm_query_bias = input_data.kwargs.get("norm_query_bias")
        norm_key_weight = input_data.kwargs.get("norm_key_weight")
        norm_key_bias = input_data.kwargs.get("norm_key_bias")
        norm_added_query_weight = input_data.kwargs.get("norm_added_query_weight")
        norm_added_query_bias = input_data.kwargs.get("norm_added_query_bias")
        norm_added_key_weight = input_data.kwargs.get("norm_added_key_weight")
        norm_added_key_bias = input_data.kwargs.get("norm_added_key_bias")

        rope_sin = input_data.kwargs.get("rope_sin")
        rope_cos = input_data.kwargs.get("rope_cos")

        norm_type = input_data.kwargs.get("norm_type")
        added_norm_type = input_data.kwargs.get("norm_added_type")
        rope_type = input_data.kwargs.get("rope_type")
        eps = input_data.kwargs.get("eps")
        if not eps:
            eps = 1e-5
        is_training = input_data.kwargs.get("is_training")
        self.is_training = is_training
        concat_order = input_data.kwargs.get("concat_order")

        origin_dtype = query.dtype
        head_dim = query.shape[-1]

        q_mean = torch.zeros(1, dtype=torch.float32, device=query.device)
        k_mean = torch.zeros(1, dtype=torch.float32, device=query.device)
        eq_mean = torch.zeros(1, dtype=torch.float32, device=query.device)
        ek_mean = torch.zeros(1, dtype=torch.float32, device=query.device)
        q_rstd = torch.zeros(1, dtype=torch.float32, device=query.device)
        eq_rstd = torch.zeros(1, dtype=torch.float32, device=query.device)
        k_rstd = torch.zeros(1, dtype=torch.float32, device=query.device)
        ek_rstd = torch.zeros(1, dtype=torch.float32, device=query.device)

        if norm_type == 2:
            query, q_mean, q_rstd = layer_norm(
                query,
                norm_query_weight,
                norm_query_bias,
                eps=eps,
                is_training=is_training,
            )
            key, k_mean, k_rstd = layer_norm(
                key, norm_key_weight, norm_key_bias, eps=eps, is_training=is_training
            )
        elif norm_type == 1:
            query, q_mean, q_rstd = layer_norm(query, eps=eps, is_training=is_training)
            key, k_mean, k_rstd = layer_norm(key, eps=eps, is_training=is_training)
        elif norm_type == 3:
            query = rms_norm(query, eps=eps)
            key = rms_norm(key, eps=eps)
        elif norm_type == 4:
            query = rms_norm(query, norm_query_weight, eps=eps)
            key = rms_norm(key, norm_key_weight, eps=eps)
        if added_norm_type == 2:
            encoder_query, eq_mean, eq_rstd = layer_norm(
                encoder_query,
                norm_added_query_weight,
                norm_added_query_bias,
                eps=eps,
                is_training=is_training,
            )
            encoder_key, ek_mean, ek_rstd = layer_norm(
                encoder_key,
                norm_added_key_weight,
                norm_added_key_bias,
                eps=eps,
                is_training=is_training,
            )
        elif added_norm_type == 1:
            encoder_query, eq_mean, eq_rstd = layer_norm(
                encoder_query, eps=eps, is_training=is_training
            )
            encoder_key, ek_mean, ek_rstd = layer_norm(
                encoder_key, eps=eps, is_training=is_training
            )
        elif added_norm_type == 3:
            encoder_query = rms_norm(encoder_query, eps=eps)
            encoder_key = rms_norm(encoder_key, eps=eps)
        elif added_norm_type == 4:
            encoder_query = rms_norm(encoder_query, norm_added_query_weight, eps=eps)
            encoder_key = rms_norm(encoder_key, norm_added_key_weight, eps=eps)
        if concat_order == 0:
            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)
        else:
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        query_res = torch.permute(query, (0, 2, 1, 3)).contiguous()
        key_res = torch.permute(key, (0, 2, 1, 3)).contiguous()
        value_res = torch.permute(value, (0, 2, 1, 3)).contiguous()

        rope_sin_shape = rope_sin.shape
        if rope_type != 0:
            query_res[:, :, : rope_sin_shape[0]] = image_rotary_emb(
                query_res[:, :, : rope_sin_shape[0]], rope_sin, rope_cos, mode=rope_type
            )
            key_res[:, :, : rope_sin_shape[0]] = image_rotary_emb(
                key_res[:, :, : rope_sin_shape[0]], rope_sin, rope_cos, mode=rope_type
            )

        return (
            query_res.to(origin_dtype),
            key_res.to(origin_dtype),
            value_res.to(origin_dtype),
            q_mean,
            q_rstd,
            k_mean,
            k_rstd,
            eq_mean,
            eq_rstd,
            ek_mean,
            ek_rstd,
        )


@register("aclnn_norm_rope_concat")
class AclnnNormRopeConcatApi(AclnnBaseApi):
    def init(self, task_result: TaskResult, backend):
        super(AclnnNormRopeConcatApi, self).init(task_result, backend)

    def init_by_input_data(self, input_data):
        self.is_training = input_data.kwargs.get("is_training")
        self.norm_type = input_data.kwargs.get("norm_type")
        self.added_norm_type = input_data.kwargs.get("norm_added_type")
        return super().init_by_input_data(input_data)

    def after_call(self, output_packages):
        output = [self.acl_tensor_to_torch(x) for x in output_packages]
        if self.is_training:
            if self.norm_type == 0:
                for i in range(3, 7):
                    output[i].zero_()
            if self.added_norm_type == 0:
                for i in range(7, 11):
                    output[i].zero_()

            return output

        for i in range(3, 11):
            output[i].zero_()

        return output

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnNormRopeConcatGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *encoderQuery,const aclTensor *encoderKey, const aclTensor *encoderValue, const aclTensor *normQueryWeight,const aclTensor *normQueryBias, const aclTensor *normKeyWeight, const aclTensor *normKeyBias,const aclTensor *normAddedQueryWeight, const aclTensor *normAddedQueryBias, const aclTensor *normAddedKeyWeight,const aclTensor *normAddedKeyBias, const aclTensor *ropeSin, const aclTensor *ropeCos, int64_t normType,int64_t normAddedType, int64_t ropeType, int64_t concatOrder, double eps, bool isTraining, const aclTensor *queryOutput,  const aclTensor *keyOutput,const  aclTensor *valueOutput,const aclTensor *normQueryMean,const  aclTensor *normQueryRstd,const  aclTensor *normKeyMean,const aclTensor *normKeyRstd,const  aclTensor *normAddedQueryMean,const aclTensor *normAddedQueryRstd,const aclTensor *normAddedKeyMean, const aclTensor *normAddedKeyRstd, uint64_t *workspaceSize, aclOpExecutor **executor)"
