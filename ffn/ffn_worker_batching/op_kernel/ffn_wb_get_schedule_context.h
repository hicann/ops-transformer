/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file ffn_wb_get_schedule_context.h
 * \brief
 */

#ifndef OP_KERNEL_FFN_WB_GET_SCHEDULE_CONTEXT_H
#define OP_KERNEL_FFN_WB_GET_SCHEDULE_CONTEXT_H

#include "ffn_wb_common.h"
#include "kernel_log.h"

namespace FfnWbBatching {

using namespace AscendC;

constexpr int64_t STRUCT_LENGTH = 1024;
constexpr int64_t TEST_MAGIC_OFFSET = 640;
constexpr int64_t TEST_MAGIC = 0x54455354;
template <bool isScanMode = false>
__aicore__ inline void ScheduleContextInfoCompute(GM_ADDR schedule_context,
                                                  const FfnWorkerBatchingTilingData *tilingData,
                                                  ScheduleContextInfo &contextInfo, TPipe *pipe)
{
    GlobalTensor<int8_t> scheduleContext;
    TBuf<TPosition::VECIN> buffer;

    scheduleContext.SetGlobalBuffer((__gm__ int8_t *)schedule_context, STRUCT_LENGTH);

    pipe->InitBuffer(buffer, STRUCT_LENGTH); // 检验数据是否正确；
    LocalTensor<int8_t> valLocal = buffer.Get<int8_t>();
    DataCopy(valLocal, scheduleContext, STRUCT_LENGTH);

    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    // 读取 session_num（uint32_t） // 576
    LocalTensor<uint32_t> value_session_num = valLocal[0].template ReinterpretCast<uint32_t>();
    contextInfo.A = value_session_num.GetValue(0);
    ASSERT_MSG(contextInfo.A <= 1024, "contextInfo.A greater than 1024 error");

    // 读取 micro_batch_num（uint32_t）
    LocalTensor<uint32_t> value_micro_batch_num = valLocal[4].template ReinterpretCast<uint32_t>();
    contextInfo.M = value_micro_batch_num.GetValue(0);
    ASSERT_MSG(contextInfo.M <= 64, "contextInfo.M greater than 64 error");

    // 读取 micro_batch_size（uint32_t）
    LocalTensor<uint32_t> value_micro_batch_size = valLocal[8].template ReinterpretCast<uint32_t>();
    contextInfo.BS = value_micro_batch_size.GetValue(0);

    // 读取 selected_expert_num（uint32_t）
    LocalTensor<uint32_t> value_selected_expert_num = valLocal[12].template ReinterpretCast<uint32_t>();
    contextInfo.K = value_selected_expert_num.GetValue(0);
    ASSERT_MSG(contextInfo.K <= 64, "contextInfo.K greater than 64 error");

    // 读取 attn_to_ffn_token_size（uint32_t）
    LocalTensor<uint32_t> value_attn_to_ffn_token_size = valLocal[20].template ReinterpretCast<uint32_t>();
    contextInfo.HS = value_attn_to_ffn_token_size.GetValue(0); // 单位: 字节

    contextInfo.H = tilingData->H;
    contextInfo.Y = tilingData->Y;
    contextInfo.tokenDtype = tilingData->tokenDtype;
    contextInfo.expertNum = tilingData->expertNum;
    contextInfo.coreNum = tilingData->coreNum;
    contextInfo.ubSize = tilingData->ubSize;
    contextInfo.sortLoopMaxElement = tilingData->sortLoopMaxElement;
    contextInfo.sortNumWorkSpace = tilingData->sortNumWorkSpace;

    LocalTensor<uint32_t> magicField = valLocal[TEST_MAGIC_OFFSET].template ReinterpretCast<uint32_t>();
    uint32_t magicValue = magicField.GetValue(0);
    uint64_t scBase = (magicValue == TEST_MAGIC) ? reinterpret_cast<uint64_t>(schedule_context) : 0;

    // token_data_buf: offset = 272  // 400
    LocalTensor<uint64_t> localTokenDataBuf = valLocal[400].template ReinterpretCast<uint64_t>();
    uint64_t tokenDataBufValue = localTokenDataBuf.GetValue(0);
    contextInfo.bufferPtr.tokenDataBuf = scBase + tokenDataBufValue;
    // 验证 contextInfo.bufferPtr.tokenDataBuf = reinterpret_cast<uint64_t>(tokenDataBufValue + schedule_context);
    // 不扫描数据的逻辑
    if constexpr (isScanMode == false) {
        // session_ids_buf: offset = 400  // 528
        LocalTensor<uint64_t> localSessionIdsBuf = valLocal[528].template ReinterpretCast<uint64_t>();
        uint64_t sessionIdsBufValue = localSessionIdsBuf.GetValue(0);
        contextInfo.bufferPtr.sessionIdsBuf = scBase + sessionIdsBufValue;
        // 验证 contextInfo.bufferPtr.sessionIdsBuf = reinterpret_cast<uint64_t>(sessionIdsBufValue + schedule_context);

        // micro_batch_ids_buf: offset = 416  // 544
        LocalTensor<uint64_t> localMicroBatchIdsBuf = valLocal[544].template ReinterpretCast<uint64_t>();
        uint64_t microBatchIdsBufValue = localMicroBatchIdsBuf.GetValue(0);
        contextInfo.bufferPtr.microBatchIdsBuf = scBase + microBatchIdsBufValue;
        // 验证 contextInfo.bufferPtr.microBatchIdsBuf = reinterpret_cast<uint64_t>(microBatchIdsBufValue +
        // schedule_context);

        // expert_ids_buf: offset = 432  // 560
        LocalTensor<uint64_t> localExpertIdsBuf = valLocal[560].template ReinterpretCast<uint64_t>();
        uint64_t expertIdsBufValue = localExpertIdsBuf.GetValue(0);
        contextInfo.bufferPtr.expertIdsBuf = scBase + expertIdsBufValue;
        // 验证 contextInfo.bufferPtr.expertIdsBuf = reinterpret_cast<uint64_t>(expertIdsBufValue + schedule_context);

        // 读取 out_num (uint32_t)  // 576
        LocalTensor<uint32_t> localOutNum = valLocal[576].template ReinterpretCast<uint32_t>();
        contextInfo.outNum = localOutNum.GetValue(0);
    } else {
        LocalTensor<uint64_t> tokenInfoTensor = valLocal[384].template ReinterpretCast<uint64_t>();
        uint64_t tokenInfoValue = tokenInfoTensor.GetValue(0);
        contextInfo.bufferPtr.tokenInfoBuf = scBase + tokenInfoValue;
        // 验证 contextInfo.bufferPtr.tokenInfoBuf = reinterpret_cast<uint64_t>(tokenInfoValue + schedule_context);

        LocalTensor<uint64_t> pollIdxBuf = valLocal[416].template ReinterpretCast<uint64_t>();
        contextInfo.curMicroBatchID = (uint64_t)(pollIdxBuf.GetValue(0));
        ASSERT_MSG(contextInfo.curMicroBatchID < contextInfo.M,
                   "curMicroBatchID:%lu should be less than micro_batch_num:%lu", contextInfo.curMicroBatchID,
                   contextInfo.M);

        contextInfo.BsKPaddingCount =
            Align(contextInfo.BS * contextInfo.K, sizeof(int32_t)) - contextInfo.BS * contextInfo.K;
    }
}

} // namespace FfnWbBatching
#endif // OP_KERNEL_FFN_WB_GET_SCHEDULE_CONTEXT_H
