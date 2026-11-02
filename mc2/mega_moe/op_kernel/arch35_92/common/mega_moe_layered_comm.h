/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_LAYERED_COMM_H
#define MEGA_MOE_LAYERED_COMM_H

#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
#include "adv_api/hcomm/hcomm.h"
#include "../../../../common/op_kernel/mc2_moe_context.h"

namespace MegaMoeImpl {

using namespace AscendC;

// 两种 Layered 通信策略共用的 URMA 写配置。
static constexpr UrmaWqeEntry LAYERED_RELAXED_WRITE_WQE_CONFIG = {
    .odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
static constexpr UrmaWqeEntry LAYERED_STRONG_COMPLETION_WRITE_WQE_CONFIG = {
    .odr = 6, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};

// 封装 Layered 路径使用的 URMA 通信接口。
class LayeredUrmaComm {
public:
    using Channel = AscendC::ChannelHandle;
    using BatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;

    __aicore__ inline explicit LayeredUrmaComm(__gm__ Mc2Aclnn::Mc2MoeContext* context)
        : context_(context),
          rankId_(context->epRankId)
    {}

    __aicore__ inline ChannelHandle GetChannel(uint32_t peer) const
    {
        const uint32_t index = peer > rankId_ ? peer - 1U : peer;
        return context_->hcommHandle_[index];
    }

    __aicore__ inline uint32_t GetBulkWriteChunkBytes(ChannelHandle) const
    {
        // URMA WriteNbi 要求长度严格小于 256 MiB，分块时保持记录对齐。
        return 256U * 1024U * 1024U - 512U;
    }
    __aicore__ inline int32_t Init(const LocalTensor<uint8_t>& scratch, uint32_t bytes)
    {
        return hcomm_.Init(scratch, bytes);
    }

    __aicore__ inline int32_t Lock(ChannelHandle channel)
    {
        return hcomm_.Lock(channel);
    }
    __aicore__ inline int32_t Unlock(ChannelHandle channel)
    {
        return hcomm_.Unlock(channel);
    }

    /*
     * 绑定本核固定的批量描述符工作区；与 Init 使用的通信临时区相互独立。
     */
    __aicore__ inline void BindBatchBuffer(const LocalTensor<uint8_t>& buffer, uint32_t bytes)
    {
        batchBuffer_ = buffer;
        batchBufferBytes_ = bytes;
    }

    /*
     * 复用已绑定的 UB 工作区，为目标通道及远端注册内存新建批量句柄。
     */
    __aicore__ inline BatchHandle MakeBatchHandle(ChannelHandle channel, GM_ADDR remoteAddr)
    {
        return hcomm_.MakeBatchHandle(channel, batchBuffer_, batchBufferBytes_, remoteAddr);
    }

    // 使用 Hcomm 默认写配置，生成 CQE。
    __aicore__ inline int32_t WriteNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t bytes)
    {
        return hcomm_.WriteNbi(channel, dst, src, bytes);
    }

    __aicore__ inline int32_t WriteNbi(BatchHandle& batch, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        return hcomm_.WriteNbi(batch, dst, src, bytes);
    }

    __aicore__ inline int32_t WriteWithNotifyNbi(BatchHandle& batch, GM_ADDR dst, GM_ADDR src, uint32_t bytes,
                                                 GM_ADDR notifyAddr, uint64_t notifyValue)
    {
        return hcomm_.WriteWithNotifyNbi(batch, dst, src, bytes, notifyAddr, notifyValue);
    }

    __aicore__ inline int32_t ReadNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t bytes)
    {
        return hcomm_.ReadNbi(channel, dst, src, bytes);
    }

    __aicore__ inline int32_t ReadNbi(BatchHandle& batch, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        return hcomm_.ReadNbi(batch, dst, src, bytes);
    }

    // 弱执行保序写（ODR=5），配置完成保序，不生成 CQE。
    __aicore__ inline int32_t WriteRelaxedNbi(BatchHandle& batch, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        return hcomm_.WriteNbi<LAYERED_RELAXED_WRITE_WQE_CONFIG>(batch, dst, src, bytes);
    }

    /*
     * 整批发送的尾通知同时生成 CQE；若全部写入均为 cqe=0，
     * Drain 无法确认这些写入已全部完成。
     */
    __aicore__ inline int32_t WriteStrongNbiWithCqe(BatchHandle& batch, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        return hcomm_.WriteNbi<LAYERED_STRONG_COMPLETION_WRITE_WQE_CONFIG>(batch, dst, src, bytes);
    }

    __aicore__ inline int32_t BatchCommit(BatchHandle& batch)
    {
        return hcomm_.BatchCommit(batch);
    }
    __aicore__ inline int32_t Drain(BatchHandle& batch)
    {
        return hcomm_.Drain(batch);
    }
    __aicore__ inline int32_t Drain(ChannelHandle channel)
    {
        return hcomm_.Drain(channel);
    }

private:
    __gm__ Mc2Aclnn::Mc2MoeContext* const context_;
    const uint32_t rankId_;
    LocalTensor<uint8_t> batchBuffer_;
    uint32_t batchBufferBytes_ = 0U;
    Hcomm<COMM_PROTOCOL_UB_CTP> hcomm_;
};

// The communication library owns SQ/CQ reclamation for consecutive commits.
// Explicit completion waits follow the existing business dependencies.
// The scheduling/peermem layout remains the real Layered path.
// Calls forward native return codes, like LayeredUrmaComm.
class LayeredUbxComm {
public:
    struct Channel {
        ChannelHandle native{};
        bool roce = false;
    };
    struct BatchHandle {
        AscendC::BatchHandle<ChannelHandle> native{};
        bool roce = false;
    };
    __aicore__ inline LayeredUbxComm(__gm__ Mc2Aclnn::Mc2MoeContext* context)
        : ctx_(context),
          rank_(context->epRankId),
          rankNumPerServer_(context->rankSizePerServer),
          serverId_(rank_ / rankNumPerServer_)
    {}
    __aicore__ inline int32_t Init(const LocalTensor<uint8_t>& scratch, uint32_t bytes)
    {
        const int32_t urmaRet = urma_.Init(scratch, bytes);
        const int32_t roceRet = roce_.Init(scratch, bytes);
        return urmaRet != HCOMM_SUCCESS ? urmaRet : roceRet;
    }
    __aicore__ inline Channel GetChannel(uint32_t peer) const
    {
        // Retain peer routing with the native handle; later calls need no reverse lookup.
        return {ctx_->hcommHandle_[peer > rank_ ? peer - 1 : peer], peer / rankNumPerServer_ != serverId_};
    }
    __aicore__ inline uint32_t GetBulkWriteChunkBytes(Channel channel) const
    {
        // RoCE 单次写入长度须严格小于 2 GiB，URMA 须严格小于 256 MiB。
        return channel.roce ? (uint32_t(1) << 31U) - 512U : 256U * 1024U * 1024U - 512U;
    }
    __aicore__ inline int32_t Lock(Channel ch)
    {
        return ch.roce ? roce_.Lock(ch.native) : urma_.Lock(ch.native);
    }
    __aicore__ inline int32_t Unlock(Channel ch)
    {
        return ch.roce ? roce_.Unlock(ch.native) : urma_.Unlock(ch.native);
    }
    /*
     * 两种协议共用本核固定的批量描述符工作区，不改变各自的 Init 临时区。
     */
    __aicore__ inline void BindBatchBuffer(const LocalTensor<uint8_t>& buffer, uint32_t bytes)
    {
        batchBuffer_ = buffer;
        batchBufferBytes_ = bytes;
    }

    /*
     * 为目标通道新建批量句柄，并保持 RoCE 所需的本端注册内存选择。
     */
    __aicore__ inline BatchHandle MakeBatchHandle(Channel ch, GM_ADDR remote)
    {
        BatchHandle b{};
        b.roce = ch.roce;
        GM_ADDR local = reinterpret_cast<GM_ADDR>(ctx_->epHcclBuffer_[rank_]);
        b.native = b.roce ? roce_.MakeBatchHandle(ch.native, batchBuffer_, batchBufferBytes_, remote, local) :
                            urma_.MakeBatchHandle(ch.native, batchBuffer_, batchBufferBytes_, remote);
        return b;
    }
    __aicore__ inline int32_t WriteNbi(Channel ch, GM_ADDR dst, GM_ADDR src, uint64_t bytes)
    {
        return ch.roce ? roce_.WriteNbi(ch.native, dst, src, bytes) : urma_.WriteNbi(ch.native, dst, src, bytes);
    }
    __aicore__ inline int32_t WriteNbi(BatchHandle& b, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        return b.roce ? roce_.WriteNbi(b.native, dst, src, bytes) : urma_.WriteNbi(b.native, dst, src, bytes);
    }
    // Server combine contributions target peers inside the local UB domain.
    __aicore__ inline int32_t WriteWithNotifyNbi(BatchHandle& b, GM_ADDR dst, GM_ADDR src, uint32_t bytes,
                                                 GM_ADDR notifyAddr, uint64_t notifyValue)
    {
        return urma_.WriteWithNotifyNbi(b.native, dst, src, bytes, notifyAddr, notifyValue);
    }
    __aicore__ inline int32_t WriteRelaxedNbi(BatchHandle& b, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        return b.roce ? roce_.WriteNbi(b.native, dst, src, bytes) :
                        urma_.WriteNbi<LAYERED_RELAXED_WRITE_WQE_CONFIG>(b.native, dst, src, bytes);
    }
    __aicore__ inline int32_t WriteStrongNbiWithCqe(BatchHandle& b, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        // RoCE 提交使用同一 QP/SQ，写入按 FIFO 排序；尾通知设置 fence 并生成 CQE，
        // 发布就绪状态，同时使最终 Drain 能够回收全部写入。
        return b.roce ? roce_.WriteNbi(b.native, dst, src, bytes) :
                        urma_.WriteNbi<LAYERED_STRONG_COMPLETION_WRITE_WQE_CONFIG>(b.native, dst, src, bytes);
    }
    __aicore__ inline int32_t ReadNbi(Channel ch, GM_ADDR dst, GM_ADDR src, uint64_t bytes)
    {
        return ch.roce ? roce_.ReadNbi(ch.native, dst, src, bytes) : urma_.ReadNbi(ch.native, dst, src, bytes);
    }
    __aicore__ inline int32_t ReadNbi(BatchHandle& b, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        return b.roce ? roce_.ReadNbi(b.native, dst, src, bytes) : urma_.ReadNbi(b.native, dst, src, bytes);
    }
    __aicore__ inline int32_t BatchCommit(BatchHandle& b)
    {
        return b.roce ? roce_.BatchCommit(b.native) : urma_.BatchCommit(b.native);
    }
    __aicore__ inline int32_t Drain(BatchHandle& b)
    {
        return b.roce ? roce_.Drain(b.native) : urma_.Drain(b.native);
    }

    __aicore__ inline int32_t Drain(Channel ch)
    {
        return ch.roce ? roce_.Drain(ch.native) : urma_.Drain(ch.native);
    }

private:
    __gm__ Mc2Aclnn::Mc2MoeContext* ctx_;
    uint32_t rank_;
    const uint32_t rankNumPerServer_;
    const uint32_t serverId_;
    LocalTensor<uint8_t> batchBuffer_;
    uint32_t batchBufferBytes_ = 0U;
    Hcomm<COMM_PROTOCOL_UB_CTP> urma_;
    Hcomm<COMM_PROTOCOL_ROCE> roce_;
};

} // namespace MegaMoeImpl
#endif
#endif
