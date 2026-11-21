# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import torch
import ctypes
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute.base_api import BaseApi
import numpy as np
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import torch.nn as nn


@register("function_ScatterPaKvCache")
class FunctionScatterPaKvCache(BaseApi):
    def golden_nd(self, key, keyCacheRef, slotMapping, value, valueCacheRef):
        block_size = keyCacheRef.shape[1]
        for i, slot in enumerate(slotMapping):
            if slot < 0:
                continue
            block_index = slot // block_size
            block_offset = slot % block_size

            token_key = key[i]
            token_v = value[i]
            keyCacheRef[block_index][block_offset] = token_key
            valueCacheRef[block_index][block_offset] = token_v
        return keyCacheRef, valueCacheRef

    def golden_nz(self, key, keyCacheRef, slotMapping, value, valueCacheRef):
        block_size = keyCacheRef.shape[2]
        k_last_dim = keyCacheRef.shape[3]
        v_last_dim = valueCacheRef.shape[3]
        num_heads = key.shape[1]
        k_head_size = key.shape[2]
        v_head_size = value.shape[2]
        for i, slot in enumerate(slotMapping):
            if slot < 0:
                continue
            block_index = slot // block_size
            block_offset = slot % block_size

            token_key = key[i]
            token_v = value[i]
            token_key = token_key.reshape(num_heads * k_head_size)
            token_v = token_v.reshape(num_heads * v_head_size)
            for k in range(num_heads * k_head_size // k_last_dim):
                keyCacheRef[block_index][k][block_offset][:] = token_key[
                    k * k_last_dim : k * k_last_dim + k_last_dim
                ]
            for k in range(num_heads * v_head_size // v_last_dim):
                valueCacheRef[block_index][k][block_offset][:] = token_v[
                    k * v_last_dim : k * v_last_dim + v_last_dim
                ]
        return keyCacheRef, valueCacheRef

    def golden_alibi(
        self,
        key,
        keyCacheRef,
        slotMapping,
        value,
        valueCacheRef,
        compressLeqLensOptional,
        seqLensOptional,
    ):
        new_seq = seqLensOptional.clone()
        new_seq[0] = seqLensOptional[0]
        for n in range(1, len(seqLensOptional)):
            new_seq[n] = seqLensOptional[n] + new_seq[n - 1]

        block_size = keyCacheRef.shape[1]
        num_heads = key.shape[1]
        for i, slot in enumerate(slotMapping):
            if slot < 0:
                continue
            curSlot = slot
            win = compressLeqLensOptional[i]
            for j in range(win):
                block_index = curSlot // block_size
                block_offset = curSlot % block_size
                curSlot += 1

                curBatch = i // num_heads
                bsID = new_seq[curBatch] - win + j
                headID = i % num_heads

                token_key = key[bsID][headID]
                token_v = value[bsID][headID]
                keyCacheRef[block_index][block_offset] = token_key
                valueCacheRef[block_index][block_offset] = token_v
        return keyCacheRef, valueCacheRef

    def golden_rope(
        self,
        key,
        keyCacheRef,
        slotMapping,
        value,
        valueCacheRef,
        compressLensOptional,
        compressSeqOffsetOptional,
        seqLensOptional,
    ):
        block_size = keyCacheRef.shape[1]
        num_heads = key.shape[1]
        head_size = key.shape[2]

        keyCacheRef_fp32 = keyCacheRef.clone().to(torch.float32)
        valueCacheRef_fp32 = valueCacheRef.clone().to(torch.float32)

        new_seq = seqLensOptional.clone()
        new_seq[0] = seqLensOptional[0]
        for n in range(1, len(seqLensOptional)):
            new_seq[n] = seqLensOptional[n] + new_seq[n - 1]
        new_seq = torch.cat((torch.zeros(1, dtype=torch.int32), new_seq), dim=0)

        for i, slot in enumerate(slotMapping):
            if slot < 0:
                continue
            win = compressLensOptional[i].clone()
            idxOffset = compressSeqOffsetOptional[i]

            bsID = i // num_heads
            headID = i % num_heads
            headStartIdx = new_seq[bsID]
            headEndIdx = idxOffset + compressLensOptional[i]
            bs = new_seq[bsID]

            sum_key = torch.zeros(head_size, dtype=torch.float32)
            sum_value = torch.zeros(head_size, dtype=torch.float32)
            for j in range(seqLensOptional[bsID]):
                block_index = torch.div(slot, block_size, rounding_mode="trunc")
                block_offset = slot % block_size

                token_key = key[bs + j][headID]
                token_v = value[bs + j][headID]
                if idxOffset == -1 or (j < idxOffset and idxOffset != -1):
                    keyCacheRef_fp32[block_index][block_offset] = token_key
                    valueCacheRef_fp32[block_index][block_offset] = token_v
                    slot += 1

                if j >= idxOffset and idxOffset != -1 and win > 0:
                    k = bs + j
                    while win > 0:
                        rope_key = torch.clone(key[k][headID]).to(torch.float32)
                        rope_v = torch.clone(value[k][headID]).to(torch.float32)
                        sum_key += rope_key
                        sum_value += rope_v
                        win -= 1
                        k += 1

                    keyCacheRef_fp32[block_index][block_offset] = (
                        sum_key / (compressLensOptional[i])
                    )
                    valueCacheRef_fp32[block_index][block_offset] = (
                        sum_value / (compressLensOptional[i])
                    )
                    slot += 1

                if j >= headEndIdx and idxOffset != -1:
                    keyCacheRef_fp32[block_index][block_offset] = token_key.to(
                        torch.float32
                    )
                    valueCacheRef_fp32[block_index][block_offset] = token_v.to(
                        torch.float32
                    )
                    slot += 1
        return keyCacheRef_fp32, valueCacheRef_fp32

    def golden_omni(
        self,
        key,
        keyCacheRef,
        slotMapping,
        value,
        valueCacheRef,
        compressLensOptional,
        compressSeqOffsetOptional,
        seqLensOptional,
    ):
        block_size = keyCacheRef.shape[1]
        num_heads = key.shape[1]
        head_size = key.shape[2]

        keyCacheRef_fp32 = keyCacheRef.clone().to(torch.float32)
        valueCacheRef_fp32 = valueCacheRef.clone().to(torch.float32)

        new_seq = seqLensOptional.clone()
        new_seq[0] = seqLensOptional[0]
        for n in range(1, len(seqLensOptional)):
            new_seq[n] = seqLensOptional[n] + new_seq[n - 1]
        new_seq = torch.cat((torch.zeros(1, dtype=torch.int32), new_seq), dim=0)

        for i, slot in enumerate(slotMapping):
            if slot < 0:
                continue
            win = compressLensOptional[i].clone()
            idxOffset = compressSeqOffsetOptional[i]

            bsID = i // num_heads
            headID = i % num_heads
            headStartIdx = new_seq[bsID]
            headEndIdx = idxOffset + compressLensOptional[i]
            bs = new_seq[bsID]

            sum_key = torch.zeros(head_size, dtype=torch.float32)
            sum_value = torch.zeros(head_size, dtype=torch.float32)
            for j in range(seqLensOptional[bsID]):
                block_index = torch.div(slot, block_size, rounding_mode="trunc")
                block_offset = slot % block_size

                token_key = key[bs + j][headID]
                token_v = value[bs + j][headID]
                if idxOffset == -1 or (j < idxOffset and idxOffset != -1):
                    keyCacheRef_fp32[block_index][block_offset] = token_key
                    valueCacheRef_fp32[block_index][block_offset] = token_v
                    slot += 1

                if j >= headEndIdx and idxOffset != -1:
                    keyCacheRef_fp32[block_index][block_offset] = token_key.to(
                        torch.float32
                    )
                    valueCacheRef_fp32[block_index][block_offset] = token_v.to(
                        torch.float32
                    )
                    slot += 1
        return keyCacheRef_fp32, valueCacheRef_fp32

    def golden_siso(self, key, keyCacheRef, slotMapping, value, valueCacheRef):
        block_size = keyCacheRef.shape[1]
        for i, slot in enumerate(slotMapping):
            if slot < 0:
                continue
            block_index = slot // block_size
            block_offset = slot % block_size

            token_key = key[i]
            keyCacheRef[block_index][block_offset] = token_key
        return keyCacheRef, valueCacheRef

    def init_by_input_data(self, input_data: InputDataset):
        key = input_data.kwargs["key"]
        keyCacheRef = input_data.kwargs["keyCacheRef"]
        slotMapping = input_data.kwargs["slotMapping"]
        value = input_data.kwargs["value"]
        valueCacheRef = input_data.kwargs["valueCacheRef"]
        compressLensOptional = input_data.kwargs["compressLensOptional"]
        compressSeqOffsetOptional = input_data.kwargs["compressSeqOffsetOptional"]
        seqLensOptional = input_data.kwargs["seqLensOptional"]
        cacheMode = input_data.kwargs["cacheMode"]

        if compressLensOptional.shape == torch.Size([]):
            num_tokens = key.shape[0]
            slot_mapping = torch.arange(start=0, end=num_tokens, step=1).to(torch.int32)
            input_data.kwargs["slotMapping"] = slot_mapping

        if compressLensOptional.shape != torch.Size(
            []
        ) and compressSeqOffsetOptional.shape == torch.Size([]):
            batch = seqLensOptional.shape[0]
            num_heads = key.shape[1]
            block_size = keyCacheRef.shape[1]
            value = []
            sumX = 0
            for i in range(batch * num_heads):
                x = (compressLensOptional[i] + block_size - 1) // block_size
                value.append(sumX * block_size)
                sumX += x

            slot_mapping = torch.Tensor(value).to(torch.int32)
            input_data.kwargs["slotMapping"] = slot_mapping

        if compressSeqOffsetOptional.shape != torch.Size([]):
            batch = seqLensOptional.shape[0]
            num_heads = key.shape[1]
            cacheStart = []
            cacheOffset = 0
            for i in range(batch * num_heads):
                cacheStart.append(cacheOffset)
                if compressSeqOffsetOptional[i] != -1:
                    cacheOffset += (
                        seqLensOptional[int(i / num_heads)].item()
                        - compressLensOptional[i].item()
                        + 1
                    )
                else:
                    cacheOffset += seqLensOptional[int(i / num_heads)].item()
            slot_mapping = torch.Tensor(cacheStart).to(torch.int32)
            input_data.kwargs["slotMapping"] = slot_mapping

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        key = input_data.kwargs["key"]
        keyCacheRef = input_data.kwargs["keyCacheRef"]
        slotMapping = input_data.kwargs["slotMapping"]
        value = input_data.kwargs["value"]
        valueCacheRef = input_data.kwargs["valueCacheRef"]
        compressLensOptional = input_data.kwargs["compressLensOptional"]
        compressSeqOffsetOptional = input_data.kwargs["compressSeqOffsetOptional"]
        seqLensOptional = input_data.kwargs["seqLensOptional"]
        cacheMode = input_data.kwargs["cacheMode"]
        scatterMode = input_data.kwargs["scatterMode"]

        if value.shape == torch.Size([]):
            return self.golden_siso(key, keyCacheRef, slotMapping, value, valueCacheRef)
        if cacheMode == "PA_NZ":
            return self.golden_nz(key, keyCacheRef, slotMapping, value, valueCacheRef)
        if compressLensOptional.shape == torch.Size([]):
            return self.golden_nd(key, keyCacheRef, slotMapping, value, valueCacheRef)
        if scatterMode == "Alibi":
            return self.golden_alibi(
                key,
                keyCacheRef,
                slotMapping,
                value,
                valueCacheRef,
                compressLensOptional,
                seqLensOptional,
            )
        elif scatterMode == "Rope":
            return self.golden_rope(
                key,
                keyCacheRef,
                slotMapping,
                value,
                valueCacheRef,
                compressLensOptional,
                compressSeqOffsetOptional,
                seqLensOptional,
            )
        else:
            return self.golden_omni(
                key,
                keyCacheRef,
                slotMapping,
                value,
                valueCacheRef,
                compressLensOptional,
                compressSeqOffsetOptional,
                seqLensOptional,
            )

    # # 函数入参签名校验
    # def get_cpp_func_signature_type(self):
    #     return "aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(const aclTensor *key, const aclTensor *keyCacheRef,\
    #             const aclTensor *slotMapping, const aclTensor *value, const aclTensor *valueCacheRef,\
    #             const aclTensor *compressLensOptional, const aclTensor *compressSeqOffsetOptional,\
    #             const aclTensor *seqLensOptional, char *cacheMode, uint64_t *workspaceSize, aclOpExecutor **executor)"


@register("function_aclnnScatterPaKvCache")
class FunctionaclnnScatterPaKvCache(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

        key = input_data.kwargs["key"]
        keyCacheRef = input_data.kwargs["keyCacheRef"]
        slotMapping = input_data.kwargs["slotMapping"]
        value = input_data.kwargs["value"]
        valueCacheRef = input_data.kwargs["valueCacheRef"]
        compressLensOptional = input_data.kwargs["compressLensOptional"]
        compressSeqOffsetOptional = input_data.kwargs["compressSeqOffsetOptional"]
        seqLensOptional = input_data.kwargs["seqLensOptional"]
        cacheMode = input_data.kwargs["cacheMode"]
        scatterMode = input_data.kwargs["scatterMode"]

        if compressLensOptional.shape == torch.Size([]):
            num_tokens = key.shape[0]
            slot_mapping = torch.arange(start=0, end=num_tokens, step=1).to(torch.int32)
            input_data.kwargs["slotMapping"] = slot_mapping.npu()

        if compressLensOptional.shape != torch.Size(
            []
        ) and compressSeqOffsetOptional.shape == torch.Size([]):
            batch = seqLensOptional.shape[0]
            num_heads = key.shape[1]
            block_size = keyCacheRef.shape[1]
            value = []
            sumX = 0
            for i in range(batch * num_heads):
                x = (compressLensOptional[i] + block_size - 1) // block_size
                value.append(sumX * block_size)
                sumX += x

            slot_mapping = torch.Tensor(value).to(torch.int32)
            input_data.kwargs["slotMapping"] = slot_mapping.npu()

        if compressSeqOffsetOptional.shape != torch.Size([]):
            batch = seqLensOptional.shape[0]
            num_heads = key.shape[1]
            cacheStart = []
            cacheOffset = 0
            for i in range(batch * num_heads):
                cacheStart.append(cacheOffset)
                if compressSeqOffsetOptional[i] != -1:
                    cacheOffset += (
                        seqLensOptional[int(i / num_heads)].item()
                        - compressLensOptional[i].item()
                        + 1
                    )
                else:
                    cacheOffset += seqLensOptional[int(i / num_heads)].item()
            slot_mapping = torch.Tensor(cacheStart).to(torch.int32)
            input_data.kwargs["slotMapping"] = slot_mapping.npu()

        # if value.shape == torch.Size([]):
        #     input_data.kwargs["value"] = TensorPtr()
        # if valueCacheRef.shape == torch.Size([]):
        #     input_data.kwargs["valueCacheRef"] = TensorPtr()
        if compressLensOptional.shape == torch.Size([]):
            input_data.kwargs["compressLensOptional"] = TensorPtr()
        if compressSeqOffsetOptional.shape == torch.Size([]):
            input_data.kwargs["compressSeqOffsetOptional"] = TensorPtr()
        if seqLensOptional.shape == torch.Size([]):
            input_data.kwargs["seqLensOptional"] = TensorPtr()
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        input_args.pop()
        output_packages[0] = input_args[1]
        output_packages[1] = input_args[4]

        return input_args, output_packages
