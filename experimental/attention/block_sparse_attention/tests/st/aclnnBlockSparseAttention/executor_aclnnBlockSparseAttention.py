# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
import numpy as np
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray
import ctypes
from ml_dtypes import bfloat16

def inspect_kwargs(kwargs):
    """
    遍历打印 kwargs 中的 key 和 value。
    如果是 Tensor/Array，只打印 shape 和 dtype。
    如果是标量或普通列表，打印具体值。
    """
    print(f"\n{'='*20} DEBUG: Inspect Kwargs {'='*20}")
    
    if not isinstance(kwargs, dict):
        print(f"Error: Expected dict, got {type(kwargs)}")
        return

    for key, value in kwargs.items():
        # 尝试获取类型名称
        type_name = type(value).__name__
        
        # 1. 处理 PyTorch Tensor
        if isinstance(value, torch.Tensor):
            print(f"[Tensor]  Key: {key:<25} | Shape: {list(value.shape)} | Dtype: {value.dtype} | Device: {value.device}")
            
        # 2. 处理 Numpy Array
        elif isinstance(value, np.ndarray):
            print(f"[Numpy]   Key: {key:<25} | Shape: {value.shape} | Dtype: {value.dtype}")
            
        # 3. 处理包含 Tensor 的列表 (常见于 List[Tensor])
        elif isinstance(value, (list, tuple)) and len(value) > 0 and (isinstance(value[0], torch.Tensor) or isinstance(value[0], np.ndarray)):
            shape_info = list(value[0].shape) if hasattr(value[0], 'shape') else "N/A"
            print(f"[List_T]  Key: {key:<25} | Len: {len(value)} | Item0_Shape: {shape_info}")
            
        # 4. 处理自定义对象 (如果有 shape 属性)
        elif hasattr(value, 'shape') and hasattr(value, 'dtype'):
             print(f"[Custom]  Key: {key:<25} | Shape: {value.shape} | Dtype: {value.dtype}")

        # 5. 其他标量或普通对象，直接打印值
        else:
            # 如果值太长，截断一下
            str_val = str(value)
            if len(str_val) > 100:
                str_val = str_val[:100] + "..."
            print(f"[Scalar]  Key: {key:<25} | Value: {str_val}")

    print(f"{'='*62}\n")

def safe_to_tensor(arr):
    if arr is None: 
        return None
    # 如果是 list，先转为 numpy array
    if isinstance(arr, list):
        arr = np.array(arr)
    # 如果是 bfloat16，先转 float32 再转 torch
    if arr.dtype.name == 'bfloat16':
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    # 其他类型直接转换
    return torch.from_numpy(arr)

@register("aclnn_blocksparseattentioninputprocess")
class BlockSparseAttentionInputProcess(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(BlockSparseAttentionInputProcess, self).__init__(task_result, backend)
        self.qSeqlenList = []
        
    @classmethod
    def change_bnsd_to_tnd(self, tensor, seqlenList):
        """
        把BNSD格式的tensor转换成TND格式
        """
        headDim = tensor.shape[-1]
        headNum = tensor.shape[1]
        tokenNum = sum(seqlenList)
        batch = len(seqlenList)
        res = torch.zeros((tokenNum, headNum, headDim), dtype=tensor.dtype)
        count = 0
        for i in range(batch):
            res[count:count + seqlenList[i], :, :] = tensor[i,:,:seqlenList[i], :].permute(1, 0, 2)
            count = count + seqlenList[i]
        return res
    
    def init_by_input_data(self, input_data):
        np.random.seed(10)
        torch.npu.synchronize()
        input_args = []  # 算子的入参列表
        output_packages = []  # 算子的出参数据包列表

        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        for name, kwarg in input_data.kwargs.items():
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)

        for index, output_data in enumerate(self.task_result.output_info_list):
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)

        input_args.extend(output_packages)

        AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)  # 把这个空指针类型转换为tensor指针类型

        # 将tensor重新to到npu上，防止流同步失败
        query = self.acl_tensor_to_torch(input_args[0]).to("npu")
        input_args[0] = self.torch_tensor_to_acl(query)
        key = self.acl_tensor_to_torch(input_args[1]).to("npu")
        input_args[1] = self.torch_tensor_to_acl(key)
        value = self.acl_tensor_to_torch(input_args[2]).to("npu")
        input_args[2] = self.torch_tensor_to_acl(value)
        blockSparseMask = self.acl_tensor_to_torch(input_args[3]).to("npu")
        input_args[3] = self.torch_tensor_to_acl(blockSparseMask)
        output_packages = []
        input_args[4] = null_tensor_ptr # attenMask
        self.qSeqlenList = input_data.kwargs['actualSeqLengths']
        input_args[8] = null_tensor_ptr # blockTable
        input_args[20] = null_tensor_ptr

        input_args.pop()
        output_packages.append(input_args[-2])

        return input_args, output_packages

    def __call__(self):
        """算子调用逻辑"""
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            temp_output_pack = self.acl_tensor_to_torch(output_pack).to(dtype=torch.float32)
            output.append(temp_output_pack)
        batch = len(self.qSeqlenList)
        count = 0
        tokenNum = sum(self.qSeqlenList)

        if output[0].dim() == 4:
            outputTemp = torch.zeros((tokenNum, output[0].shape[1], output[0].shape[-1]), dtype=output[0].dtype)
            outputTensor = output[0]
            for i in range(batch):
                outputTemp[count:count+self.qSeqlenList[i], :, :] = outputTensor[i, :, :self.qSeqlenList[i], :].permute(1, 0, 2)
                count += self.qSeqlenList[i]
            return [outputTemp]
        else:
            return output

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnBlockSparseAttentionGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *blockSparseMask, const aclTensor *attenMaskOptional, const aclIntArray *blockShape, const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *blockTableOptional, char *qInputLayout, char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t innerPrecise, int64_t blockSize, int64_t preTokens, int64_t nextTokens, int64_t softmaxLseFlag, const aclTensor *attentionOut, const aclTensor *softmaxLseOptional, uint64_t *workspaceSize, aclOpExecutor **executor)"

class TestBlockSparseAttentionTorch():
    @classmethod
    def online_softmax_attention_torch_high(cls, q_block, kv_blocks, scale):
        # 确保在 CPU 上运行
        q_block = q_block.cpu()
        device = torch.device('cpu')
        q_len = q_block.shape[1]
        head_size = q_block.shape[2]
        
        # 初始化状态量（确保在 CPU 上）
        m_i = torch.full((1, q_len, 1), -float('inf'), dtype=torch.float32, device=device)  # running max
        l_i = torch.zeros((1, q_len, 1), dtype=torch.float32, device=device)  # running sum
        O_i = torch.zeros((1, q_len, head_size), dtype=torch.float32, device=device)  # running output
        is_first = 1
        # 逐块处理
        for k_block, v_block in kv_blocks:
            # 确保 kv blocks 在 CPU 上
            k_block = k_block.cpu()
            v_block = v_block.cpu()
            # 1. 计算注意力分数 S_i = Q @ K_i^T * scale
            S_i = torch.matmul(q_block.to(torch.float32), k_block.to(torch.float32))  # (1, q_len, k_len)

            S_i = S_i * scale
            
            # 2. 计算当前块的最大值
            m_block, _ = torch.max(S_i, dim=-1, keepdim=True)  # (1, q_len, 1)
            
            # 3. 计算新的全局最大值
            m_new = torch.maximum(m_i, m_block)  # (1, q_len, 1)
            
            # 4. 计算修正因子
            # alpha: 旧输出的修正系数 = exp(m_old - m_new)
            # beta: 当前块的修正系数（用于 softmax）= exp(m_block - m_new)
            alpha = torch.exp(m_i - m_new) # (1, q_len, 1)
            
            # 5. 计算当前块的稳定 softmax 分子
            P_i = torch.exp(S_i - m_new)  # (1, q_len, k_len)
            
            # 6. 更新 running sum
            l_i = alpha * l_i + torch.sum(P_i, dim=-1, keepdim=True)  # (1, q_len, 1)
            
            # 7. 更新 running output
            # O_new = alpha * O_old + P_i @ V_i
            O_i = alpha * O_i + torch.matmul(P_i.to(torch.float32), v_block.to(torch.float32))  # (1, q_len, head_size)
            
            # 8. 更新 running max
            m_i = m_new

        # 最终归一化
        O_final = O_i / l_i  # (1, q_len, head_size)
        
        return O_final

    @classmethod
    def online_softmax_attention_torch(cls, q_block, kv_blocks, scale, torch_dtype, input_dtype, inner_precise):
        # 确保在 CPU 上运行
        q_block = q_block.cpu()
        device = torch.device('cpu')
        q_len = q_block.shape[1]
        head_size = q_block.shape[2]
        
        # 初始化状态量（确保在 CPU 上）
        m_i = torch.full((1, q_len, 1), -float('inf'), dtype=torch.float32, device=device)  # running max
        l_i = torch.zeros((1, q_len, 1), dtype=torch.float32, device=device)  # running sum
        O_i = torch.zeros((1, q_len, head_size), dtype=torch.float32, device=device)  # running output
        is_first = 1
        # 逐块处理
        for k_block, v_block in kv_blocks:
            # 确保 kv blocks 在 CPU 上
            k_block = k_block.cpu()
            v_block = v_block.cpu()
            # 1. 计算注意力分数 S_i = Q @ K_i^T * scale
            # CPU不支持half精度matmul，需要转换为float32计算
            S_i = torch.matmul(q_block.float(), k_block.float()).to(torch_dtype)  # (1, q_len, k_len)

            S_i = S_i * scale
            
            # 2. 计算当前块的最大值
            m_block, _ = torch.max(S_i, dim=-1, keepdim=True)  # (1, q_len, 1)
            
            # 3. 计算新的全局最大值
            m_new = torch.maximum(m_i, m_block)  # (1, q_len, 1)
            
            # 4. 计算修正因子
            # alpha: 旧输出的修正系数 = exp(m_old - m_new)
            # beta: 当前块的修正系数（用于 softmax）= exp(m_block - m_new)
            alpha = torch.exp(m_i - m_new).to(torch_dtype)  # (1, q_len, 1)
            
            # 5. 计算当前块的稳定 softmax 分子
            P_i = torch.exp(S_i - m_new).to(torch_dtype)  # (1, q_len, k_len)
            
            # 6. 更新 running sum
            l_i = alpha * l_i + torch.sum(P_i, dim=-1, keepdim=True)  # (1, q_len, 1)
            
            # 7. 更新 running output
            # CPU不支持half精度matmul，需要转换为float32计算
            O_i = alpha * O_i + torch.matmul(P_i.float(), v_block.float()).to(torch_dtype)  # (1, q_len, head_size)
            
            # 8. 更新 running max
            m_i = m_new
        
        # 最终归一化
        O_final = O_i / l_i  # (1, q_len, head_size)
        
        return O_final


    def ref_select_idx_attention_torch(self,
            query,
            key, 
            value,
            scale: float,
            select_idx_list: list,
            select_num_idx_list: list,
            s_block_x: int,
            s_block_y: int,
            total_q_blocks: int,
            max_kv_block_num: int,
            q_seqlen_list: list,
            kv_seqlen_list: list,
            batch: int,
            torch_dtype,
            inner_precise
    ):
        """
        PyTorch 版本的 ref_select_idx_attention
        确保所有计算在 CPU 上进行
        """
        # 确保输入在 CPU 上
        if query.device.type != 'cpu':
            query = query.cpu()
        if key.device.type != 'cpu':
            key = key.cpu()
        if value.device.type != 'cpu':
            value = value.cpu()
        
        device = torch.device('cpu')

        # 转置操作
        query = query.permute(1, 0, 2)  # (total_q_tokens, num_heads, head_size) -> (num_heads, total_q_tokens, head_size)
        key = key.permute(1, 2, 0)      # (total_kv_tokens, kv_heads, head_size) -> (kv_heads, head_size, total_kv_tokens)
        value = value.permute(1, 0, 2)   # (total_kv_tokens, kv_heads, head_size) -> (kv_heads, total_kv_tokens, head_size)
        num_heads = query.shape[0]
        kv_heads = key.shape[0]
        
        # 初始化输出 - 注意这里应该是total_q_tokens而不是max_q_seqlen
        total_q_tokens = query.shape[1]
        head_size = query.shape[2]
        out_high = torch.zeros((num_heads, total_q_tokens, head_size), dtype=torch.float32, device=device)
        out = torch.zeros((num_heads, total_q_tokens, head_size), dtype=query.dtype, device=device)
        
        # 【关键修复】：添加batch级别的累计偏移量
        q_token_offset = 0   # Q方向token累计偏移
        kv_token_offset = 0  # KV方向token累计偏移
        q_block_offset = 0   # Q块累计偏移（用于selectIdx索引）
        
        for batch_idx in range(batch):
            q_seqlen = q_seqlen_list[batch_idx]
            kv_seqlen = kv_seqlen_list[batch_idx]

            # 计算当前batch的分块数量
            s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
            s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y
            
            for t_local in range(s_block_num_q):
                t_global = q_block_offset + t_local
                q_block_idx = t_local  # 当前batch内的Q块索引
                
                # 【关键修复】：batch内的相对位置
                q_start_local = q_block_idx * s_block_x
                q_end_local = min((q_block_idx + 1) * s_block_x, q_seqlen)
                
                # 【关键修复】：加上batch偏移得到全局位置
                q_start_global = q_token_offset + q_start_local
                q_end_global = q_token_offset + q_end_local
                
                for head in range(num_heads):
                    # 获取该头对应的selectIdx
                    select_idx_offset = t_global * num_heads * max_kv_block_num + head * max_kv_block_num
                    select_num_offset = t_global * num_heads + head
                    
                    select_num = select_num_idx_list[select_num_offset]
                    selected_kv_blocks = select_idx_list[select_idx_offset:select_idx_offset + max_kv_block_num]
                    
                    # 【GQA修复】：在循环之前提取 q_block 和计算 GQA 参数（只需要一次）
                    q_block = query[head:head+1, q_start_global:q_end_global, :]  # (1, q_block_size, head_size)
                    
                    # 处理 group attention 的情况
                    group_size = num_heads // kv_heads
                    kv_head_idx = head // group_size
                    
                    # 收集所有选中的KV块数据（作为 (K, V) 元组列表）
                    kv_blocks = []
                    k_blocks = []
                    v_blocks = []
                    if select_num == 0:
                        continue

                    for kv_block_idx in selected_kv_blocks[:select_num]:
                        if kv_block_idx == -1:  # 跳过填充的-1
                            continue
                        
                        # 【关键修复】：batch内的相对位置
                        k_start_local = kv_block_idx * s_block_y
                        k_end_local = min((kv_block_idx + 1) * s_block_y, kv_seqlen)
                        
                        # 【关键修复】：加上batch偏移得到全局位置
                        k_start_global = kv_token_offset + k_start_local
                        k_end_global = kv_token_offset + k_end_local
                        
                        # 【修复】：使用全局位置访问key和value
                        k_block = key[kv_head_idx:kv_head_idx+1, :, k_start_global:k_end_global]    # (1, head_size, k_block_size)
                        v_block = value[kv_head_idx:kv_head_idx+1, k_start_global:k_end_global, :]  # (1, k_block_size, head_size)

                        k_blocks.append(k_block)
                        v_blocks.append(v_block)

                    k_block_com = torch.cat(k_blocks, dim=2)
                    v_block_com = torch.cat(v_blocks, dim=1)

                    k_total_len = k_block_com.shape[2]
                    v_total_len = v_block_com.shape[1]

                    if k_total_len <= 512:
                        kv_blocks.append((k_block_com, v_block_com))
                    else:
                        num_chunks = k_total_len // 512
                        remainder = k_total_len % 512
                        for i in range(num_chunks):
                            start = i * 512
                            end = start + 512
                            k_chunk = k_block_com[ : , : , start : end]
                            v_chunk = v_block_com[ : , start : end , : ]
                            kv_blocks.append((k_chunk, v_chunk))
                        if remainder > 0:
                            k_last_chunk = k_block_com[ : , : , -remainder : ]
                            v_last_chunk = v_block_com[ : , -remainder : , : ]
                            kv_blocks.append((k_last_chunk, v_last_chunk))
                    # 使用 Online Softmax 计算注意力（FlashAttention 风格）
                    if query.dtype == torch.float32:
                        out_block = self.online_softmax_attention_torch_high(q_block, kv_blocks, scale)  # (1, q_block_size, head_size)
                    else:
                        out_block = self.online_softmax_attention_torch(q_block, kv_blocks, scale, torch_dtype, query.dtype, inner_precise)
                    # 【修复】：输出到全局位置
                    # out_high[head:head+1, q_start_global:q_end_global, :] = out_block_high
                    out[head:head+1, q_start_global:q_end_global, :] = out_block
            # 【关键修复】：更新累计偏移量，为下一个batch做准备
            q_token_offset += q_seqlen
            kv_token_offset += kv_seqlen
            q_block_offset += s_block_num_q
        
        out = out.permute(1, 0, 2)
        return out

    @classmethod
    def change_bnsd_to_tnd(self, tensor, seqlenList):
        """
        把BNSD格式的tensor转换成TND格式
        """
        headDim = tensor.shape[-1]
        headNum = tensor.shape[1]
        tokenNum = sum(seqlenList)
        batch = len(seqlenList)
        res = torch.zeros((tokenNum, headNum, headDim), dtype=tensor.dtype)
        count = 0
        for i in range(batch):
            res[count:count + seqlenList[i], :, :] = tensor[i,:,:seqlenList[i], :].permute(1, 0, 2)
            count = count + seqlenList[i]
        return res
    
    @classmethod
    def change_tnd_to_bnsd(self, tensor, seqlenList, maxQSeqlen):
        """
        把TND格式的tensor转换成BNSD格式
        """
        headDim = tensor.shape[-1]
        headNum = tensor.shape[1]
        batch = len(seqlenList)
        res = torch.zeros((batch, headNum, maxQSeqlen, headDim), dtype=tensor.dtype)
        count = 0
        for i in range(batch):
            res[i, :, :seqlenList[i], :] = tensor[count:count+seqlenList[i], :, :].permute(1, 0, 2)
            count = count + seqlenList[i]
        return res
        
    def calc_data(self, query_dtype, query, key, value, select_idx, select_num_idx, block_shape, q_seqlen_list, kv_seqlen_list, scale_value, q_input_layout, kv_input_layout, inner_precise):
        """
        PyTorch 版本的 calc_data
        确保所有计算在 CPU 上进行
        """
        # 确保输入是 torch.Tensor，如果是 numpy 数组则转换
        if not isinstance(query, torch.Tensor):
            query = safe_to_tensor(query).cpu()
        if not isinstance(key, torch.Tensor):
            key = safe_to_tensor(key).cpu()
        if not isinstance(value, torch.Tensor):
            value = safe_to_tensor(value).cpu()
        if not isinstance(select_idx, torch.Tensor):
            select_idx = safe_to_tensor(select_idx).cpu()
        if not isinstance(select_num_idx, torch.Tensor):
            select_num_idx = safe_to_tensor(select_num_idx).cpu()

        # 确保在 CPU 上
        query = query.cpu()
        key = key.cpu()
        value = value.cpu()
        select_idx = select_idx.cpu()
        select_num_idx = select_num_idx.cpu()
        maxQSeqlen = 0
        if q_input_layout == "BNSD":
            maxQSeqlen = query.shape[-2]
            query = self.change_bnsd_to_tnd(query, q_seqlen_list)
            key = self.change_bnsd_to_tnd(key, kv_seqlen_list)
            value = self.change_bnsd_to_tnd(value, kv_seqlen_list)
        embedding_size = query.shape[2]
        num_heads = query.shape[1]
        batch_size = len(q_seqlen_list)

        if inner_precise == 1 :
            scale_value = np.float16(scale_value)

        # if q_input_layout == 'TND' and kv_input_layout == 'TND':
        # 使用 torch 计算总和
        if isinstance(q_seqlen_list, (list, tuple)):
            num_tokens = sum(q_seqlen_list)
        else:
            num_tokens = torch.tensor(q_seqlen_list).sum().item()
        head_size_vo = embedding_size

        shape_out = (num_tokens, num_heads, head_size_vo)
        # 根据 query_dtype 确定 torch dtype
        if isinstance(query_dtype, torch.dtype):
            torch_dtype = query_dtype
        elif query_dtype == np.float32 or str(query_dtype) == 'float32':
            torch_dtype = torch.float32
        elif query_dtype == np.float16 or str(query_dtype) == 'float16':
            torch_dtype = torch.float16
        elif query_dtype == np.bfloat16 or str(query_dtype) == 'bfloat16':
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        input_type = torch_dtype
        if inner_precise == 1 :
            torch_dtype = torch.float16

        ref_output = torch.zeros(shape_out, dtype=torch_dtype, device=torch.device('cpu'))
        ref_output_high = torch.zeros(shape_out, dtype=torch.float32, device=torch.device('cpu'))

        total_q_blocks = select_idx.shape[0]
        max_kv_block_num = select_idx.shape[2]
        s_block_x = block_shape[0]
        s_block_y = block_shape[1]
        select_idx_list = select_idx.flatten().tolist()
        select_num_idx_list = select_num_idx.flatten().tolist()
        ref_output = self.ref_select_idx_attention_torch(
            query, key, value, scale_value,
            select_idx_list, select_num_idx_list,
            s_block_x, s_block_y,
            total_q_blocks, max_kv_block_num,
            q_seqlen_list, kv_seqlen_list, batch_size, torch_dtype, inner_precise
        )
        if query.dtype != torch.float32:
            ref_output_h = ref_output.to(torch.float32)
            return ref_output_h
        else:
            return ref_output

@register("aclnn_blocksparseattention")
class BlockSparseAttentionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(BlockSparseAttentionApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        query = input_data.kwargs["query"]
        query_dtype = query.dtype
        key = input_data.kwargs["key"]
        value = input_data.kwargs["value"]
        block_shape = input_data.kwargs["blockShape"]
        block_sparse_mask = input_data.kwargs["blockSparseMask"]
        atten_mask = input_data.kwargs["attenMask"]
        q_seqlen_list = input_data.kwargs["actualSeqLengths"]
        kv_seqlen_list = input_data.kwargs["actualSeqLengthsKv"]
        block_table = input_data.kwargs["blockTable"]
        q_input_layout = input_data.kwargs["qInputLayout"]
        kv_input_layout = input_data.kwargs["kvInputLayout"]
        kv_heads = input_data.kwargs["numKeyValueHeads"]
        mask_type = input_data.kwargs["maskType"]
        scale_value = input_data.kwargs["scaleValue"]
        inner_precise = input_data.kwargs["innerPrecise"]
        block_size = input_data.kwargs["blockSize"]
        q_input_value = query.cpu()
        k_input_value = key.cpu()
        v_input_value = value.cpu()
        # 从blockSparseMask转换为selectIdx和selectNumIdx
        batch = len(q_seqlen_list)
        select_idx_input, select_num_idx_input = self.change_block_sparsemask_to_selectidx_selctnumidx(
            block_sparse_mask, q_seqlen_list, kv_seqlen_list, block_shape, batch
        )
        testObj = TestBlockSparseAttentionTorch()
        atten_out_golden = testObj.calc_data(query_dtype, q_input_value, k_input_value, v_input_value, select_idx_input, select_num_idx_input, block_shape, q_seqlen_list, kv_seqlen_list, scale_value, q_input_layout, kv_input_layout, inner_precise)
        return atten_out_golden

    def change_block_sparsemask_to_selectidx_selctnumidx(self, block_sparse_mask, q_seqlen_list, kv_seqlen_list, block_shape, batch):
        """
        把blocksparseMask转为selectIdx和selectNumIdx
        其中blockSparseMask为[b, headNum, maxQBlockNum, maxKvBlockNum]
        selectIdx为[QBlockNum, headNum, maxKvBlockNum]维度，其中第一维表示每个batch块的合轴
        selectNumIdx为[QBlockNum, headNum]维度表示每个Q方向块有效的KV块数
        """
        import numpy as np
        s_block_x = block_shape[0]
        s_block_y = block_shape[1]
        # blockSparseMask shape: [b, headNum, maxQBlockNum, maxKvBlockNum]
        num_heads = block_sparse_mask.shape[1]
        max_q_block_num = block_sparse_mask.shape[2]
        max_kv_block_num = block_sparse_mask.shape[3]
        # 计算总的Q块数
        total_q_blocks = 0
        for b in range(batch):
            q_seqlen = q_seqlen_list[b]
            s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
            total_q_blocks += s_block_num_q
        # 初始化selectIdx和selectNumIdx列表
        select_idx_list = []
        select_num_idx_list = []
        # 遍历每个batch
        q_block_offset = 0
        for b in range(batch):
            q_seqlen = q_seqlen_list[b]
            kv_seqlen = kv_seqlen_list[b]
            # 计算当前batch的分块数量
            s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
            s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y
            # 遍历当前batch的每个Q块
            for q_block_idx in range(s_block_num_q):
                # 遍历每个head
                for head in range(num_heads):
                    # 从blockSparseMask中提取当前位置的mask: [maxKvBlockNum]
                    mask_row = block_sparse_mask[b, head, q_block_idx, :]
                    # 找出所有非零的KV块索引（稀疏mask中非零表示该KV块被选中）
                    selected_kv_blocks = []
                    for kv_block_idx in range(s_block_num_kv):
                        if kv_block_idx < max_kv_block_num and mask_row[kv_block_idx] != 0:
                            selected_kv_blocks.append(kv_block_idx)
                    # 填充到max_kv_block_num长度，不足的用-1填充
                    padded_blocks = selected_kv_blocks + [-1] * (max_kv_block_num - len(selected_kv_blocks))
                    # 添加到selectIdx
                    select_idx_list.extend(padded_blocks)
                # 为当前Q块的每个head记录选中的KV块数量
                for head in range(num_heads):
                    mask_row = block_sparse_mask[b, head, q_block_idx, :]
                    num_selected = 0
                    for kv_block_idx in range(s_block_num_kv):
                        if kv_block_idx < max_kv_block_num and mask_row[kv_block_idx] != 0:
                            num_selected += 1
                    select_num_idx_list.append(num_selected)
            q_block_offset += s_block_num_q
        # 将列表转换为tensor并reshape为正确的shape
        select_idx_tensor = torch.tensor(select_idx_list, dtype=torch.int32).view(total_q_blocks, num_heads, max_kv_block_num)
        select_num_idx_tensor = torch.tensor(select_num_idx_list, dtype=torch.int32).view(total_q_blocks, num_heads)
        return select_idx_tensor, select_num_idx_tensor

    def gen_select_idx_data(self, q_seqlen_list, kv_seqlen_list, s_block_x, s_block_y, batch, num_heads, select_ratio):
        """
        @brief 生成新的selectIdx数据格式，支持变长序列
        @param q_seqlen_list  每个batch的Q序列长度列表
        @param kv_seqlen_list 每个batch的KV序列长度列表
        @param s_block_x      Q方向分块大小
        @param s_block_y      KV方向分块大小
        @param batch          batch数
        @param num_heads      注意力头数
        @param sparsity_ratio 稀疏度比例，控制每个Q块选择多少比例的KV块 (默认0.3表示30%的KV块)
        @return select_idx_list, select_num_idx_list, total_q_blocks, max_kv_block_num
        """
        select_idx_list = []
        select_num_idx_list = []
        # 计算T: 所有batch中Q方向切块的总数 (直接按照x基本块计算)
        total_q_blocks = 0
        max_kv_block_num = 0
        # 首先计算总的Q块数和最大KV块数
        for b in range(batch):
            q_seqlen = q_seqlen_list[b]
            kv_seqlen = kv_seqlen_list[b]
            # Q方向分块计算: 直接按照x基本块计算
            s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
            s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y
            total_q_blocks += s_block_num_q
            max_kv_block_num = max(max_kv_block_num, s_block_num_kv)
        # 为每个Q块和每个头生成selectIdx
        q_block_offset = 0
        for b in range(batch):
            q_seqlen = q_seqlen_list[b]
            kv_seqlen = kv_seqlen_list[b]
            # 计算当前batch的分块数量
            s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
            s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y
            for t_local in range(s_block_num_q):
                t_global = q_block_offset + t_local
                q_block_idx = t_local  # 当前batch内的Q块索引
                batch_select_idx = []
                batch_select_num = []
                for head in range(num_heads):
                    # 为每个头生成稀疏的KV块选择（模拟实际稀疏注意力模式）
                    selected_kv_blocks = []
                    # 全部采用随机稀疏，随机选择若干KV块，不区分对角线/额外块
                    if s_block_num_kv > 0:
                        # 为不同头和Q块设置不同的随机种子，确保多样性
                        random.seed(head * 1000 + q_block_idx)
                        # 随机选择1~int(s_block_num_kv * select_ratio)个KV块
                        if select_ratio == 1:
                            select_ratio = np.random.random()
                        num_select = max(1, int(s_block_num_kv * select_ratio))
                        selected_kv_blocks = random.sample(range(s_block_num_kv), num_select)
                    else:
                        selected_kv_blocks = []
                    selected_kv_blocks.sort()
                    # 填充到max_kv_block_num长度
                    padded_blocks = selected_kv_blocks + [-1] * (max_kv_block_num - len(selected_kv_blocks))
                    batch_select_idx.extend(padded_blocks)
                    batch_select_num.append(len(selected_kv_blocks))
                select_idx_list.extend(batch_select_idx)
                select_num_idx_list.extend(batch_select_num)

            q_block_offset += s_block_num_q

        return select_idx_list, select_num_idx_list, total_q_blocks, max_kv_block_num

    def init_by_input_data(self, input_data: InputDataset):
        np.random.seed(10)
        q_seqlen_list = input_data.kwargs["actualSeqLengths"]
        kv_seqlen_list = input_data.kwargs["actualSeqLengthsKv"]
        block_shape = input_data.kwargs["blockShape"]
        blockSparseMask = input_data.kwargs["blockSparseMask"]
        batch = blockSparseMask.shape[0]
        head_num = blockSparseMask.shape[1]
        max_q_block_num = blockSparseMask.shape[2]
        max_kv_block_num = blockSparseMask.shape[3]
        blockSparseMask = torch.zeros((batch, head_num, max_q_block_num, max_kv_block_num), dtype=torch.int8)
        for i in range(batch):
            for j in range(head_num):
                q_block_num = (q_seqlen_list[i] + block_shape[0] - 1) // block_shape[0]
                for k in range(q_block_num):
                    kv_block_num = (kv_seqlen_list[i] + block_shape[1] - 1) // block_shape[1]
                    for l in range(kv_block_num):
                        blockSparseMask[i, j, k, l] = np.random.randint(0, 2)
        if self.device == "cpu":
            input_data.kwargs["blockSparseMask"] = blockSparseMask
        if self.device == "pyaclnn" or self.device == "npu":
            input_data.kwargs["blockSparseMask"] = blockSparseMask.npu()