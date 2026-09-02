/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file common_utils.h
 * \brief
 */

#ifndef OP_KERNEL_ATTENTION_WORKER_COMBINE_COMMON_UTILS_H
#define OP_KERNEL_ATTENTION_WORKER_COMBINE_COMMON_UTILS_H

#define GET_OFFSET_BYTE(_type, member) ((unsigned long)(&((_type *)0)->member))
#define GET_OFFSET_B32(_type, member) ((unsigned long)(&((_type *)0)->member) / 4)
#define GET_OFFSET_B64(_type, member) ((unsigned long)(&((_type *)0)->member) / 8)

struct ScheduleContext {
    struct CommonArea {
        uint32_t session_num;         // attention 节点数
        uint32_t micro_batch_num;     // 拆分几个 batch
        uint32_t micro_batch_size;    // batch_size/micro_batch_num
        uint32_t selected_expert_num; // topK 个数 + 1
        uint32_t expert_num;          // 每层专家个数，含路由专家和共享专家
        uint32_t attn_to_ffn_token_size; // 每个 token 在 ffn window 数据区存储占用的大小，对齐到 512，存储数据格式为
                                         // struct Data {int8 hidden_states[hidden_size]; float32 quant_scale;}
                                         // quant_scale 只有量化时存在，hidden_states 类型根据量化类型变化
        uint32_t ffn_to_attn_token_size; // 每个 token 在 attention window 数据区存储占用的大小，对齐到
                                         // 512，存储数据格式为 struct Data {float16 hidden_states[hidden_size];}
        int32_t schedule_mode; // 0 表示只调度FFN，1 表示只调度 Attention，2 表示 FFN 和 Attention 同时调度
        int8_t reserve0[96]; // padding to 128 bytes
    };
    struct ControlArea {
        int32_t run_flag; // scheduler 使用，控制循环退出
        int8_t reserve1[4];
        uint64_t schedule_wait_buf; // point to a int64 dev mem
                                    // 预留，scheduler使用，待确认：如何aicpu算子可以使用，入图如何使用
        uint64_t ffn_wait_buf;       // point to a int64 dev mem
        uint64_t attention_wait_buf; // point to a int64 dev mem
        int8_t reserve2[96];         // padding to 128 bytes
    };
    struct FfnArea {
        // ffn area
        uint64_t token_info_buf; // point to window dev mem [A, M, F]
                                 // F = sizeof(DataDesc)
                                 // struct DataDesc {int32 flag; int32 layer_id; int32 expert_ids[batch_size][topk+1];}
        uint64_t token_info_buf_size;
        uint64_t token_data_buf; // point to window dev mem [A, M, BS, K+1, HS] HS = attn_to_ffn_token_size, 单位字节
        uint64_t token_data_buf_size;
        uint64_t polling_index; // ffn scheduler 内部使用
        int8_t reserve3[88];    // padding to 128 bytes

        // ffn out area
        uint64_t
            layer_ids_buf; // [session_num]  -- 目前和 sync_group_size 配置无关，可能多个组一起ready，按照最大可能申请
        uint64_t layer_ids_buf_size;       // session_num * sizeof(int32)
        uint64_t session_ids_buf;          // [session_num]
        uint64_t session_ids_buf_size;     // session_num * sizeof(int32)
        uint64_t micro_batch_ids_buf;      // [session_num]
        uint64_t micro_batch_ids_buf_size; // session_num * sizeof(int32)
        // uint64_t expert_masks_buf;       // [session_num, BS, K+1]
        // uint64_t expert_masks_buf_size;  // session_num * BS * K+1 * sizeof(int32)
        uint64_t expert_ids_buf;      // [session_num, BS, K+1]
        uint64_t expert_ids_buf_size; // session_num * BS * K+1 * sizeof(int32)
        uint32_t out_num;             // 实际收齐的 session 个数
        int8_t reserve4[60];          // padding to 256 bytes
    };

    struct AttentionArea {
        // attention area
        uint64_t
            token_info_buf; // point to window dev mem [M, DataDesc] struct DataDesc {int32 flags[batch_size][topk+1];}
        uint64_t token_info_buf_size;
        uint64_t token_data_buf; // point to window dev mem [M, BS, K+1, HS]
        uint64_t token_data_buf_size;
        uint32_t micro_batch_id; // inner use for polling
        int8_t reserve5[92];     // padding to 128 bytes
    };

    // common area
    CommonArea common;
    ControlArea control;
    AttentionArea attention;
    FfnArea ffn;
    // reserve area
    int8_t reserve6[384]; // padding to 1024 bytes
}; // 结构体大小 1024

#endif // OP_KERNEL_ATTENTION_WORKER_COMBINE_COMMON_UTILS_H
