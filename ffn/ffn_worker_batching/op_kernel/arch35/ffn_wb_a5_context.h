/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ffn_wb_a5_context.h
 * \brief arch35(A5) 自有基础层:输入契约解析 + 工具函数。A5 不再经桥接头引用 A2 的
 *        ../ffn_wb_common.h 与 ../ffn_wb_get_schedule_context.h。
 *
 * 本文件内**不写任何裸常量**:
 *   · schedule_context 的字段偏移与结构体大小,取自公共契约头 attention_ffn_schedule.h,
 *     由编译器 offsetof/sizeof 推出(A2 侧那份 valLocal[528] 之类的手写偏移是待淘汰写法);
 *   · 数据块字节数用 AscendC 的 ONE_BLK_SIZE,不自定义 32;
 *   · proposal 对宽度由"fp32 键 + uint32 索引"两个类型宽度相加得到,不写 8;
 *   · fp32 下界由编译器内建 __FLT_MAX__ 取反得到,不写 -3.4e38;
 *   · 切分/workspace 段偏移一律由 host tiling 下发,kernel 不持有任何布局常量;
 *   · A/M/K 的上界由 host 侧 CheckInputParam 统一校验,kernel 不再重复声明限值。
 */
#ifndef OP_KERNEL_ARCH35_FFN_WB_A5_CONTEXT_H
#define OP_KERNEL_ARCH35_FFN_WB_A5_CONTEXT_H
#include "kernel_operator.h"
#include "attention_ffn_schedule.h" // 与 Attention 侧约定的 schedule_context 权威定义

namespace FfnWbBatchingArch35 {
using namespace AscendC;

// 契约字段在 schedule_context 中的字节偏移:由权威结构体推出,不手写。
#define FFN_WB_CTX_OFFSET(field) static_cast<int32_t>(__builtin_offsetof(aicpu::ScheduleContext, field))
// schedule_context 的字节长度同样取自权威结构体(该结构体自带 static_assert 保证为约定值)。
constexpr int64_t SCHEDULE_CONTEXT_BYTES = static_cast<int64_t>(sizeof(aicpu::ScheduleContext));

// 排序中间表示:region proposal 对 = fp32 键 + uint32 索引。
constexpr int64_t SORT_PAIR_BYTES = static_cast<int64_t>(sizeof(float) + sizeof(uint32_t));
// 降序排序中代表"最小"的填充值:取 fp32 可表示的最小有限值。
constexpr float SORT_FILL_VALUE = -__FLT_MAX__;
// 以下两个是**硬件指令粒度**,无法由其他量推导:
//   · Sort32 单次排序元素数:SDK 定义在 impl/basic_api/dav_3510/kernel_operator_proposal_impl.h
//     (singleSortElementCountArch3510 = 32),该头为内部实现头、禁止直接 include,故此处按同值定义;
//   · MrgSort 单轮归并路数:由 MrgSortSrcList 的 4 个入参与 validBit 的 4 个有效位决定。
// 二者若随架构变化,以 SDK 上述定义为准。
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t MRG_LIST_NUM = 4;

// schedule_context 内存放的是设备地址(二级指针),真数据需二次解引用后使用。
struct BufferInfo {
    uint64_t tokenInfoBuf = 0;
    uint64_t tokenDataBuf = 0;
    uint64_t sessionIdsBuf = 0;
    uint64_t microBatchIdsBuf = 0;
    uint64_t expertIdsBuf = 0;
};

struct ScheduleContextInfo {
    uint32_t A = 0;               // attention session num
    uint32_t M = 0;               // micro batch num
    uint32_t BS = 0;              // micro batch size
    uint32_t K = 0;               // selected expert num(topK+1)
    uint32_t HS = 0;              // attn_to_ffn_token_size,单位字节
    uint32_t H = 0;               // hidden size(attr)
    uint32_t Y = 0;               // A*BS*K
    uint64_t curMicroBatchID = 0; // RECV:当前 expert id 已就绪的 micro batch
    uint32_t outNum = 0;          // NORM:FfnArea 中有效的 session 数
    uint32_t tokenDtype = 0;      // 0:FP16 1:BF16 2:int8 + dynamic scale 连续排布
    uint32_t expertNum = 0;
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t validGatherIdxLength = 0; // 排序后剔除无效值的有效长度,≤ A*BS*K
    int64_t BsKPaddingCount = 0;      // RECV:BS*K 按 block 对齐需补的个数

    BufferInfo bufferPtr;
};

// ---------------- 工具函数 ----------------
template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

// 与 Ceil 同义,保留此名以贴合"向上取整除"的调用点语义。
__aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return Ceil(a, b);
}

// 数据块字节数别名:沿用 AscendC 的 ONE_BLK_SIZE,不另立常量。
constexpr int64_t BLOCK_BYTES = static_cast<int64_t>(ONE_BLK_SIZE);
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_THREE = 3;
constexpr int64_t NUM_FOUR = 4;

// 返回按一个数据块(ONE_BLK_SIZE)对齐后的**元素个数**,bytes 为单元素字节数。
__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    return (bytes == 0) ? 0 : (elementNum * bytes + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE / bytes;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return (a < b) ? a : b;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return (a > b) ? a : b;
}

// proposal 对表示下,count 个元素占用的 T 元素个数。
template <typename T>
__aicore__ inline int64_t GetSortLen(int64_t count)
{
    return count * SORT_PAIR_BYTES / static_cast<int64_t>(sizeof(T));
}

// ---------------- schedule_context 解析 ----------------
// isRecv=false(NORM):expert_id 取自 FfnArea.expert_ids_buf,并读 session/micro_batch ids 与 out_num;
// isRecv=true (RECV):expert_id 藏在 FfnArea.token_info_buf 的 FfnDataDesc 内,
//                    另读 polling_index 定位当前 micro batch。
template <bool isRecv = false, typename TilingT>
__aicore__ inline void ScheduleContextParse(GM_ADDR schedule_context, const TilingT *tilingData,
                                            ScheduleContextInfo &ctx, TPipe *pipe)
{
    GlobalTensor<int8_t> ctxGm;
    TBuf<TPosition::VECIN> buffer;
    ctxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(schedule_context), SCHEDULE_CONTEXT_BYTES);
    pipe->InitBuffer(buffer, SCHEDULE_CONTEXT_BYTES);
    LocalTensor<int8_t> val = buffer.Get<int8_t>();
    DataCopy(val, ctxGm, SCHEDULE_CONTEXT_BYTES);
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    ctx.A = val[FFN_WB_CTX_OFFSET(common.session_num)].template ReinterpretCast<uint32_t>().GetValue(0);
    ctx.M = val[FFN_WB_CTX_OFFSET(common.micro_batch_num)].template ReinterpretCast<uint32_t>().GetValue(0);
    ctx.BS = val[FFN_WB_CTX_OFFSET(common.micro_batch_size)].template ReinterpretCast<uint32_t>().GetValue(0);
    ctx.K = val[FFN_WB_CTX_OFFSET(common.selected_expert_num)].template ReinterpretCast<uint32_t>().GetValue(0);
    ctx.HS = val[FFN_WB_CTX_OFFSET(common.attn_to_ffn_token_size)].template ReinterpretCast<uint32_t>().GetValue(0);

    ctx.H = tilingData->H;
    ctx.Y = tilingData->Y;
    ctx.tokenDtype = tilingData->tokenDtype;
    ctx.expertNum = tilingData->expertNum;
    ctx.coreNum = tilingData->coreNum;
    ctx.ubSize = tilingData->ubSize;

    ctx.bufferPtr.tokenDataBuf =
        val[FFN_WB_CTX_OFFSET(ffn.token_data_buf)].template ReinterpretCast<uint64_t>().GetValue(0);

    if constexpr (isRecv) {
        ctx.bufferPtr.tokenInfoBuf =
            val[FFN_WB_CTX_OFFSET(ffn.token_info_buf)].template ReinterpretCast<uint64_t>().GetValue(0);
        ctx.curMicroBatchID =
            val[FFN_WB_CTX_OFFSET(ffn.polling_index)].template ReinterpretCast<uint64_t>().GetValue(0);
        ASSERT_MSG(ctx.curMicroBatchID < ctx.M, "curMicroBatchID:%lu should be less than micro_batch_num:%u",
                   ctx.curMicroBatchID, ctx.M);
        const int64_t bsk = static_cast<int64_t>(ctx.BS) * ctx.K;
        ctx.BsKPaddingCount = Align(bsk, sizeof(int32_t)) - bsk;
    } else {
        ctx.bufferPtr.sessionIdsBuf =
            val[FFN_WB_CTX_OFFSET(ffn.session_ids_buf)].template ReinterpretCast<uint64_t>().GetValue(0);
        ctx.bufferPtr.microBatchIdsBuf =
            val[FFN_WB_CTX_OFFSET(ffn.micro_batch_ids_buf)].template ReinterpretCast<uint64_t>().GetValue(0);
        ctx.bufferPtr.expertIdsBuf =
            val[FFN_WB_CTX_OFFSET(ffn.expert_ids_buf)].template ReinterpretCast<uint64_t>().GetValue(0);
        ctx.outNum = val[FFN_WB_CTX_OFFSET(ffn.out_num)].template ReinterpretCast<uint32_t>().GetValue(0);
    }
}

} // namespace FfnWbBatchingArch35
#endif // OP_KERNEL_ARCH35_FFN_WB_A5_CONTEXT_H
