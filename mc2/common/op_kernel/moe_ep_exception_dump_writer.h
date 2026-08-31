/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_EP_EXCEPTION_DUMP_WRITER_H
#define MOE_EP_EXCEPTION_DUMP_WRITER_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "mc2_moe_context.h"
#include "moe_ep_exception_dump_defs.h"

namespace MoeEpExceptionDump {

class MoeEpCoreDiagWriter {
public:
    __aicore__ inline void Init(GM_ADDR context, MoeEpCoreDiagOpIndex opIndex, AscendC::TPipe *tpipe)
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        if (context == nullptr || tpipe == nullptr) {
            return;
        }

        tpipe_ = tpipe;
        auto commContext = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
        uint32_t aivId = AscendC::GetBlockIdx();
        GM_ADDR localWin = reinterpret_cast<GM_ADDR>(commContext->epHcclBuffer[commContext->epRankId]);
        recordAddr_ = localWin + MOE_EP_DUMP_METADATA_BYTES +
                      static_cast<uint64_t>(aivId) * MOE_EP_PER_CORE_DIAG_SLOT_BYTES +
                      static_cast<uint64_t>(opIndex) * MOE_EP_CORE_DIAG_RECORD_BYTES;

        AscendC::TBuf<> recordBuf;
        tpipe_->InitBuffer(recordBuf, static_cast<uint32_t>(MOE_EP_CORE_DIAG_RECORD_BYTES));
        AscendC::LocalTensor<uint8_t> recordLocal = recordBuf.Get<uint8_t>();
        AscendC::GlobalTensor<uint8_t> recordGlobal;
        recordGlobal.SetGlobalBuffer(recordAddr_, MOE_EP_CORE_DIAG_RECORD_BYTES);
        AscendC::DataCopyPad(recordLocal, recordGlobal,
                             {1U, static_cast<uint16_t>(MOE_EP_CORE_DIAG_RECORD_BYTES), 0U, 0U, 0U},
                             {false, 0U, 0U, 0U});
        AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();

        AscendC::LocalTensor<uint64_t> recordLocalU64 = recordLocal.template ReinterpretCast<uint64_t>();
        AscendC::LocalTensor<uint32_t> recordLocalU32 = recordLocal.template ReinterpretCast<uint32_t>();
        recordLocalU64.SetValue(0U, recordLocalU64.GetValue(0U) + 1UL);
        recordLocalU32.SetValue(2U, 0U);
        recordLocalU32.SetValue(3U, commContext->epRankId);
        recordLocalU32.SetValue(4U, aivId);

        AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(recordGlobal, recordLocal,
                             {1U, static_cast<uint16_t>(MOE_EP_CORE_DIAG_RECORD_BYTES), 0U, 0U, 0U});
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

    __aicore__ inline void RunPosRecord(uint32_t runPosition)
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        if (recordAddr_ == nullptr || tpipe_ == nullptr) {
            return;
        }

        // Hybrid dispatch may reset TPipe between markers, so allocate fresh scratch UB for every record.
        AscendC::TBuf<> runPositionBuf;
        tpipe_->InitBuffer(runPositionBuf, DIAG_UB_ALIGN_BYTES);
        AscendC::LocalTensor<uint32_t> runPositionLocal = runPositionBuf.Get<uint32_t>();
        AscendC::GlobalTensor<uint8_t> runPositionGlobal;
        runPositionGlobal.SetGlobalBuffer(recordAddr_ + offsetof(MoeEpCoreDiagRecord, runPosition),
                                          sizeof(runPosition));
        runPositionLocal.SetValue(0U, runPosition);
        AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(runPositionGlobal, runPositionLocal.template ReinterpretCast<uint8_t>(),
                             {1U, static_cast<uint16_t>(sizeof(runPosition)), 0U, 0U, 0U});
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

private:
    static constexpr uint32_t DIAG_UB_ALIGN_BYTES = 32U;
    GM_ADDR recordAddr_{nullptr};
    AscendC::TPipe *tpipe_{nullptr};
};

__aicore__ inline void WriteMetadata(GM_ADDR context, GM_ADDR metadataSource)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    if (AscendC::GetBlockIdx() != 0U || context == nullptr || metadataSource == nullptr) {
        return;
    }

    auto commContext = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    GM_ADDR localWin = reinterpret_cast<GM_ADDR>(commContext->epHcclBuffer[commContext->epRankId]);
    AscendC::LocalTensor<uint8_t> metadataLocal{AscendC::TPosition::LCM, 0,
                                                static_cast<uint32_t>(sizeof(MoeEpDumpMetadata))};
    AscendC::GlobalTensor<uint8_t> metadataSourceGlobal;
    AscendC::GlobalTensor<uint8_t> metadataWinGlobal;
    metadataSourceGlobal.SetGlobalBuffer(metadataSource, sizeof(MoeEpDumpMetadata));
    metadataWinGlobal.SetGlobalBuffer(localWin, sizeof(MoeEpDumpMetadata));

    AscendC::DataCopyPad(metadataLocal, metadataSourceGlobal,
                         {1U, static_cast<uint16_t>(sizeof(MoeEpDumpMetadata)), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    AscendC::SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    AscendC::DataCopyPad(metadataWinGlobal, metadataLocal,
                         {1U, static_cast<uint16_t>(sizeof(MoeEpDumpMetadata)), 0U, 0U, 0U});
    AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>();
}

} // namespace MoeEpExceptionDump

#endif // MOE_EP_EXCEPTION_DUMP_WRITER_H
