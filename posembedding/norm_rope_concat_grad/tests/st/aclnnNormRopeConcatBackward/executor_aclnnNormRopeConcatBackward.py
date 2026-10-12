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
import json
import time
import torch
import torch_npu
import torch.nn as nn
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


def image_rotary_emb(hidden_states, ropeSin, ropeCos, mode=1):
    out = torch.empty_like(hidden_states)
    if mode == 1:
        x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
        x1, x2 = x[..., 0], x[..., 1]
        rotated_x = torch.stack([-x2, x1], dim=-1).flatten(3)

        out = hidden_states.float() * ropeCos + rotated_x.float() * ropeSin
        return out.type_as(hidden_states)
    else:
        x1, x2 = hidden_states.reshape(*hidden_states.shape[:-1], 2, -1).unbind(-2)

        rotated_x = torch.cat([-x2, x1], dim=-1)

        out = hidden_states.float() * ropeCos + rotated_x.float() * ropeSin
        return out.type_as(hidden_states)


def norm(x, mean, rstd, modeType, weight=None, bias=None, eps=1e-5):
    headdim = x.shape[-1]

    x = x.float()
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, correction=0, keepdim=True)
    rstd = 1 / torch.sqrt(var + eps)
    out = (x - mean) * rstd
    if modeType == 2:
        out = weight.float() * out + bias.float()
    return out


@register("ascend_method_torch_nn_NormRopeConcatGrad")
class MethodTorchNnNormRopeConcatGradApi(BaseApi):
    def normropeconcat_forward(self, input_data):
        self.origin_dtype = input_data.kwargs["gradQueryOutput"].dtype
        self.gradQueryOutput = (
            input_data.kwargs["gradQueryOutput"].float().clone().requires_grad_(False)
        )
        self.gradKeyOutput = (
            input_data.kwargs["gradKeyOutput"].float().clone().requires_grad_(False)
        )
        self.gradValueOutput = (
            input_data.kwargs["gradValueOutput"].float().clone().requires_grad_(False)
        )
        self.query = input_data.kwargs["query"].float().clone().requires_grad_(True)
        self.key = input_data.kwargs["key"].float().clone().requires_grad_(True)
        self.value = input_data.kwargs["key"].float().clone().requires_grad_(True)
        self.encoderQuery = (
            input_data.kwargs["encoderQuery"].float().clone().requires_grad_(True)
        )
        self.encoderKey = (
            input_data.kwargs["encoderKey"].float().clone().requires_grad_(True)
        )
        self.encoderValue = (
            input_data.kwargs["encoderKey"].float().clone().requires_grad_(True)
        )
        self.normQueryWeight = nn.Parameter(
            input_data.kwargs["normQueryWeight"].float().clone(), requires_grad=True
        )
        self.normQueryBias = nn.Parameter(
            input_data.kwargs["normQueryWeight"].float().clone(), requires_grad=True
        )
        self.normQueryMean = (
            input_data.kwargs["normQueryMean"].float().clone().requires_grad_(False)
        )
        self.normQueryRstd = (
            input_data.kwargs["normQueryRstd"].float().clone().requires_grad_(False)
        )
        self.normKeyWeight = nn.Parameter(
            input_data.kwargs["normKeyWeight"].float().clone(), requires_grad=True
        )
        self.normKeyBias = nn.Parameter(
            input_data.kwargs["normKeyWeight"].float().clone(), requires_grad=True
        )
        self.normKeyMean = (
            input_data.kwargs["normKeyMean"].float().clone().requires_grad_(False)
        )
        self.normKeyRstd = (
            input_data.kwargs["normKeyRstd"].float().clone().requires_grad_(False)
        )
        self.normAddedQueryWeight = nn.Parameter(
            input_data.kwargs["normAddedQueryWeight"].float().clone(),
            requires_grad=True,
        )
        self.normAddedQueryBias = nn.Parameter(
            input_data.kwargs["normAddedQueryWeight"].float().clone(),
            requires_grad=True,
        )
        self.normAddedQueryMean = (
            input_data.kwargs["normAddedQueryMean"]
            .float()
            .clone()
            .requires_grad_(False)
        )
        self.normAddedQueryRstd = (
            input_data.kwargs["normAddedQueryRstd"]
            .float()
            .clone()
            .requires_grad_(False)
        )
        self.normAddedKeyWeight = nn.Parameter(
            input_data.kwargs["normAddedKeyWeight"].float().clone(), requires_grad=True
        )
        self.normAddedKeyBias = nn.Parameter(
            input_data.kwargs["normAddedKeyWeight"].float().clone(), requires_grad=True
        )
        self.normAddedKeyMean = (
            input_data.kwargs["normAddedKeyMean"].float().clone().requires_grad_(False)
        )
        self.normAddedKeyRstd = (
            input_data.kwargs["normAddedKeyRstd"].float().clone().requires_grad_(False)
        )
        self.ropeSin = (
            input_data.kwargs["ropeSin"].float().clone().requires_grad_(False)
        )
        self.ropeCos = (
            input_data.kwargs["ropeCos"].float().clone().requires_grad_(False)
        )
        self.normType = input_data.kwargs["normType"]
        self.normAddedType = input_data.kwargs["normAddedType"]
        self.ropeType = input_data.kwargs["ropeType"]
        self.concatOrder = input_data.kwargs["concatOrder"]

        # 获取shape
        batchSize, headNum, _, headDim = self.gradQueryOutput.shape
        querySeq = self.query.shape[1]
        keySeq = self.key.shape[1]
        valueSeq = keySeq
        encoderQuerySeq = self.gradQueryOutput.shape[2] - querySeq
        encoderKeySeq = self.gradKeyOutput.shape[2] - keySeq
        encoderValueSeq = encoderKeySeq
        ropeSeq = self.ropeSin.shape[0]
        main_dtype = self.query.dtype
        is_training = True
        self.is_training = is_training
        eps = 1e-5

        # 开始计算
        if self.normType != 0:
            queryNorm = norm(
                self.query,
                self.normQueryMean,
                self.normQueryRstd,
                self.normType,
                self.normQueryWeight,
                self.normQueryBias,
            )
            keyNorm = norm(
                self.key,
                self.normKeyMean,
                self.normKeyRstd,
                self.normType,
                self.normKeyWeight,
                self.normKeyBias,
            )
        else:
            queryNorm = self.query
            keyNorm = self.key

        if self.normAddedType != 0:
            encoderQueryNorm = norm(
                self.encoderQuery,
                self.normAddedQueryMean,
                self.normAddedQueryRstd,
                self.normAddedType,
                self.normAddedQueryWeight,
                self.normAddedQueryBias,
            )
            encoderKeyNorm = norm(
                self.encoderKey,
                self.normAddedKeyMean,
                self.normAddedKeyRstd,
                self.normAddedType,
                self.normAddedKeyWeight,
                self.normAddedKeyBias,
            )
        else:
            encoderQueryNorm = self.encoderQuery
            encoderKeyNorm = self.encoderKey

        if self.concatOrder == 0:
            queryConcat = torch.cat([queryNorm, encoderQueryNorm], dim=1)
            keyConcat = torch.cat([keyNorm, encoderKeyNorm], dim=1)
            valueConcat = torch.cat([self.value, self.encoderValue], dim=1)
        else:
            queryConcat = torch.cat([encoderQueryNorm, queryNorm], dim=1)
            keyConcat = torch.cat([encoderKeyNorm, keyNorm], dim=1)
            valueConcat = torch.cat([self.encoderValue, self.value], dim=1)

        self.queryConcat = torch.permute(queryConcat, (0, 2, 1, 3))
        self.keyConcat = torch.permute(keyConcat, (0, 2, 1, 3))
        self.valueConcat = torch.permute(valueConcat, (0, 2, 1, 3))

        if self.ropeType != 0:
            self.queryConcat[:, :, :ropeSeq] = image_rotary_emb(
                self.queryConcat[:, :, :ropeSeq],
                self.ropeSin,
                self.ropeCos,
                mode=self.ropeType,
            )
            self.keyConcat[:, :, :ropeSeq] = image_rotary_emb(
                self.keyConcat[:, :, :ropeSeq],
                self.ropeSin,
                self.ropeCos,
                mode=self.ropeType,
            )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        results = []
        if self.device == "cpu" or self.device == "npu":
            # 反向求导
            self.queryConcat.backward(self.gradQueryOutput, retain_graph=True)
            self.keyConcat.backward(self.gradKeyOutput, retain_graph=True)
            self.valueConcat.backward(self.gradValueOutput, retain_graph=True)

            # 获取梯度值
            gradQuery = self.query.grad.clone()
            gradKey = self.key.grad.clone()
            gradValue = self.value.grad.clone()
            gradencoderQuery = self.encoderQuery.grad.clone()
            gradEncoderKey = self.encoderKey.grad.clone()
            gradEncoderValue = self.encoderValue.grad.clone()

            if self.normType != 2:
                gradNormQueryWeight = torch.empty([]).zero_()
                gradNormQueryBias = torch.empty([]).zero_()
                gradNormKeyWeight = torch.empty([]).zero_()
                gradNormKeyBias = torch.empty([]).zero_()
            else:
                gradNormQueryWeight = self.normQueryWeight.grad.clone()
                gradNormQueryBias = self.normQueryBias.grad.clone()
                gradNormKeyWeight = self.normKeyWeight.grad.clone()
                gradNormKeyBias = self.normKeyBias.grad.clone()

            if self.normAddedType != 2:
                gradNormAddedQueryWeight = torch.empty([]).zero_()
                gradNormAddedQueryBias = torch.empty([]).zero_()
                gradNormAddedKeyWeight = torch.empty([]).zero_()
                gradNormAddedKeyBias = torch.empty([]).zero_()
            else:
                gradNormAddedQueryWeight = self.normAddedQueryWeight.grad.clone()
                gradNormAddedQueryBias = self.normAddedQueryBias.grad.clone()
                gradNormAddedKeyWeight = self.normAddedKeyWeight.grad.clone()
                gradNormAddedKeyBias = self.normAddedKeyBias.grad.clone()

            results = [
                gradQuery,
                gradKey,
                gradValue,
                gradencoderQuery,
                gradEncoderKey,
                gradEncoderValue,
                gradNormQueryWeight,
                gradNormQueryBias,
                gradNormKeyWeight,
                gradNormKeyBias,
                gradNormAddedQueryWeight,
                gradNormAddedQueryBias,
                gradNormAddedKeyWeight,
                gradNormAddedKeyBias,
            ]
            for i in range(len(results)):
                results[i] = results[i].to(self.origin_dtype)

        return tuple(results)

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND

    def init_by_input_data(self, input_data: InputDataset, with_output: bool = False):
        normType = input_data.kwargs["normType"]
        normAddedType = input_data.kwargs["normAddedType"]
        ropeType = input_data.kwargs["ropeType"]

        eps = 1e-5
        if normType is not None and normType != 0:
            mean = torch.mean(input_data.kwargs["query"].float(), -1, keepdim=True)
            var = torch.var(
                input_data.kwargs["query"].float(), -1, correction=0, keepdim=True
            )
            rstd = torch.rsqrt(var + eps)
            input_data.kwargs["normQueryMean"] = mean.to(torch.float32)
            input_data.kwargs["normQueryRstd"] = rstd.to(torch.float32)

            mean2 = torch.mean(input_data.kwargs["key"].float(), -1, keepdim=True)
            var2 = torch.var(
                input_data.kwargs["key"].float(), -1, correction=0, keepdim=True
            )
            rstd2 = torch.rsqrt(var2 + eps)
            input_data.kwargs["normKeyMean"] = mean2.to(torch.float32)
            input_data.kwargs["normKeyRstd"] = rstd2.to(torch.float32)

        if normAddedType is not None and normAddedType != 0:
            mean3 = torch.mean(
                input_data.kwargs["encoderQuery"].float(), -1, keepdim=True
            )
            var3 = torch.var(
                input_data.kwargs["encoderQuery"].float(),
                -1,
                correction=0,
                keepdim=True,
            )
            rstd3 = torch.rsqrt(var3 + eps)
            input_data.kwargs["normAddedQueryMean"] = mean3.to(torch.float32)
            input_data.kwargs["normAddedQueryRstd"] = rstd3.to(torch.float32)

            mean4 = torch.mean(
                input_data.kwargs["encoderKey"].float(), -1, keepdim=True
            )
            var4 = torch.var(
                input_data.kwargs["encoderKey"].float(), -1, correction=0, keepdim=True
            )
            rstd4 = torch.rsqrt(var4 + eps)
            input_data.kwargs["normAddedKeyMean"] = mean4.to(torch.float32)
            input_data.kwargs["normAddedKeyRstd"] = rstd4.to(torch.float32)

        if self.device == "cpu" or self.device == "npu":
            self.normropeconcat_forward(input_data)

        time.sleep(2)


@register("aclnn_norm_rope_concat_grad")
class NormRopeConcatGradAclnnApi(AclnnBaseApi):
    def call(self):
        super().call()

    def init_by_input_data(self, input_data):
        """参数处理"""
        self.my_data = input_data
        return super().init_by_input_data(input_data)

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))

        (
            gradQuery,
            gradKey,
            gradValue,
            gradencoderQuery,
            gradEncoderKey,
            gradEncoderValue,
            gradNormQueryWeight,
            gradNormQueryBias,
            gradNormKeyWeight,
            gradNormKeyBias,
            gradNormAddedQueryWeight,
            gradNormAddedQueryBias,
            gradNormAddedKeyWeight,
            gradNormAddedKeyBias,
        ) = output
        if self.my_data.kwargs["normType"] != 2:
            gradNormQueryWeight.zero_()
            gradNormQueryBias.zero_()
            gradNormKeyWeight.zero_()
            gradNormKeyBias.zero_()

        if self.my_data.kwargs["normAddedType"] != 2:
            gradNormAddedQueryWeight.zero_()
            gradNormAddedQueryBias.zero_()
            gradNormAddedKeyWeight.zero_()
            gradNormAddedKeyBias.zero_()

        output = (
            gradQuery,
            gradKey,
            gradValue,
            gradencoderQuery,
            gradEncoderKey,
            gradEncoderValue,
            gradNormQueryWeight,
            gradNormQueryBias,
            gradNormKeyWeight,
            gradNormKeyBias,
            gradNormAddedQueryWeight,
            gradNormAddedQueryBias,
            gradNormAddedKeyWeight,
            gradNormAddedKeyBias,
        )
        return output
