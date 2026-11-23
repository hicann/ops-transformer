/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_sum.h
 * \brief MRS AIV 归约组件, 对 inputSegNum 份输入求和
 *
 * 与 PUT 重叠的轮次 (下称在途) 和最后一个 tile (末 tile) 统一按线性分段;
 * 在途串行处理各份输入, 末 tile 逐份预取.
 * 输入为 IOType, 使用不同的 SumType 累加后转换回 IOType.
 * 各输入基址由调用方计算, 本组件不处理通信.
 */

#pragma once

#include "kernel_basic_intf.h"
#include "blaze/gemm/utils/common_utils.h"
#include "include/tensor_api/tensor.h"
#include "apace/core/aiv_compute/asc_tensor_api_vector_compute.h"

namespace Apace {
namespace ReduceSumImpl {

// 4 个事件按方向各用一个 id, 避免两个方向共用 id 互相误等; 4~7 避开通信组件占用的 id 1.
constexpr int32_t EVT_MTE2_V = 4; // MTE2 → V : 输入已搬到 UB
constexpr int32_t EVT_V_MTE2 = 5; // V → MTE2 : UB 输入槽可被覆写
constexpr int32_t EVT_V_MTE3 = 6; // V → MTE3 : 输出已就绪
constexpr int32_t EVT_MTE3_V = 7; // MTE3 → V : 输出已搬出 (输出槽可复用)

template <typename IOType, typename SumType = float>
class ReduceSum {
    static_assert(!AscendC::IsSameType<IOType, SumType>::value,
                  "ReduceSum supports heterogeneous IOType/SumType only (e.g. bf16 in, fp32 accumulate)");

    // 在途与末 tile 统一线性分段, 输入两槽交替; 在途串行不预取, 末 tile 逐份预取.
    // 两条路径都有 sum 和 scratch, 输出复用 in[0].
    static constexpr uint32_t MAX_INPUT_SEG_NUM = 16;

    // 输入槽数, 双槽交替预取
    static constexpr uint64_t PINGPONG_NUM = 2;

    // 段长按 32B 对齐, 对齐元素数取 IOType/SumType 中较小的字节数折算
    static constexpr uint64_t SEG_ALIGN_ELEMS =
        AscendC::ONE_BLK_SIZE / ((sizeof(IOType) < sizeof(SumType)) ? sizeof(IOType) : sizeof(SumType));

    // 在途段长上限: 过大会加剧与并发 PUT 的 GM 争用, 过小则段数开销上升
    static constexpr uint64_t MAX_SEG_LEN_ELEMS = 4096U;

    // 各分段共用的 UB 字节偏移
    struct SlotLayout {
        uint64_t sumByteOff;
        uint64_t inByteOff[MAX_INPUT_SEG_NUM]; // 在途与末 tile 均按份交替使用 in[0]/in[1]
        uint64_t scratchByteOff;
        uint64_t outByteOff; // 输出区, 复用第一个输入槽
    };

    // 生成 (1, n) 形状的一维布局, 供 asc::te::copy 在 GM 与 UB 间搬运连续数据
    template <typename T>
    using MakeFrame = asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<T>>;

public:
    __aicore__ inline ReduceSum() {}

    struct RsParams {
        GM_ADDR inputSegBaseAddrs[MAX_INPUT_SEG_NUM]; // 每份输入的 GM 基址, 由调用方计算
        GM_ADDR outputAddr{nullptr};
        uint64_t inputSegNum{0};     // 输入份数, 调用方需保证为 1~MAX_INPUT_SEG_NUM
        uint64_t aivNum{0};          // 参与归约的 AIV 核数, 调用核需满足 GetBlockIdx() < aivNum
        uint64_t nSize{0};           // 每行元素数
        uint64_t ubReservedBytes{0}; // UB 低地址保留区, 归约缓冲区从其之后开始布局
        // 末 tile 标记: true 时逐份预取, false 时串行处理 (在途与 PUT 重叠)
        bool tailCall{false};
    };

    __aicore__ inline void Init(const RsParams& params)
    {
        rsParams_ = params;
    }

    // 将 tileM 行分配给 AIV 核, 本核行区间按线性分段处理.
    // tailCall=false (在途) 时串行处理各份输入, 段长设上限控制单次 GM 访问量;
    // tailCall=true (末 tile) 时逐份预取, 段长只受 UB 容量限制
    __aicore__ inline void Process(uint64_t tileM)
    {
        if (rsParams_.inputSegNum > MAX_INPUT_SEG_NUM) {
            return;
        }

        if (tileM == 0U || rsParams_.aivNum == 0U) {
            return;
        }

        uint32_t startRowIdx = 0;
        uint32_t rowNumPerCore = 0;
        SplitToCore(static_cast<uint32_t>(tileM), static_cast<uint32_t>(rsParams_.aivNum),
                    static_cast<uint32_t>(AscendC::GetBlockIdx()), startRowIdx, rowNumPerCore);

        ProcessLinearBatch(startRowIdx, rowNumPerCore);
    }

private:
    // UB 一维向量视图, 用于类型转换 (cast_reg_loop) 和累加 (compute_add)
    template <typename T>
    static __aicore__ inline auto MakeUbVec(uint64_t byteOff, uint64_t elemNum)
    {
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, T>(byteOff),
                                    asc::te::make_layout(asc::te::make_shape(elemNum)));
    }

    template <typename T>
    static __aicore__ inline auto MakeGmFrame(__gm__ T* ptr, uint64_t elemNum)
    {
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(ptr), MakeFrame<T>{}(1, elemNum));
    }

    // 本核行区间按线性分段, 段可跨行, 尾段按实际长度处理; 在途与末 tile 共用, 仅各份输入的处理方式不同
    __aicore__ inline void ProcessLinearBatch(uint32_t startRowIdx, uint32_t rowNumPerCore)
    {
        // 段长由可用 UB 决定; 输入槽交替使用, 段长与份数无关
        const uint64_t ubBytesPerElem = 2 * sizeof(SumType) + PINGPONG_NUM * sizeof(IOType);
        constexpr uint64_t ubBudgetBytes = AscendC::TOTAL_UB_SIZE;
        uint64_t budgetBytes =
            (ubBudgetBytes > rsParams_.ubReservedBytes) ? (ubBudgetBytes - rsParams_.ubReservedBytes) : 0UL;
        uint64_t maxSegLen = budgetBytes / ubBytesPerElem;
        // 仅在途设段长上限: 过大的段加剧与并发 PUT 的 GM 争用; 末 tile 无并发 PUT 不设上限
        if (!rsParams_.tailCall && maxSegLen > MAX_SEG_LEN_ELEMS) {
            maxSegLen = MAX_SEG_LEN_ELEMS;
        }
        maxSegLen = maxSegLen / SEG_ALIGN_ELEMS * SEG_ALIGN_ELEMS;
        if (maxSegLen == 0U) {
            maxSegLen = SEG_ALIGN_ELEMS;
        }

        SlotLayout layout;
        BuildLinearSlotLayout(maxSegLen, layout);

        const uint64_t coreBase = static_cast<uint64_t>(startRowIdx) * rsParams_.nSize;
        const uint64_t totalElems = static_cast<uint64_t>(rowNumPerCore) * rsParams_.nSize;
        for (uint64_t segStart = 0; segStart < totalElems; segStart += maxSegLen) {
            const uint64_t segLen = (segStart + maxSegLen <= totalElems) ? maxSegLen : (totalElems - segStart);

            // 段首等上一段计算与输出转换结束再覆写输入槽; MTE3 写出已在 Store 末尾等其完成
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVT_V_MTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVT_V_MTE2);

            if (rsParams_.tailCall) {
                // 末 tile 无并发 PUT, 份 i 就位后先预取份 i+1 再计算份 i (加载与计算重叠).
                // 预写槽在上一轮被读过, 预取前等其计算结束, 防止覆写未读完的数据
                LoadLinearSegInput(layout, 0, coreBase + segStart, segLen);
                for (uint64_t segIdx = 0; segIdx < rsParams_.inputSegNum; ++segIdx) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVT_MTE2_V);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVT_MTE2_V);
                    if (segIdx + 1 < rsParams_.inputSegNum) {
                        if (segIdx >= 1U) {
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVT_V_MTE2);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVT_V_MTE2);
                        }
                        LoadLinearSegInput(layout, segIdx + 1, coreBase + segStart, segLen);
                    }
                    // 第一份输入转换后写入 sum, 后续输入经 scratch 转换后累加
                    AccumulateSegInput(layout, segIdx, segLen);
                }
            } else {
                // 在途与 PUT 重叠时不预取下一份输入, 避免预取拷贝与 PUT 争抢 GM 带宽
                for (uint64_t segIdx = 0; segIdx < rsParams_.inputSegNum; ++segIdx) {
                    LoadLinearSegInput(layout, segIdx, coreBase + segStart, segLen);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVT_MTE2_V);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVT_MTE2_V);
                    AccumulateSegInput(layout, segIdx, segLen);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVT_V_MTE2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVT_V_MTE2);
                }
            }

            StoreLinearSegResult(layout, coreBase + segStart, segLen);
        }
    }

    // 线性布局: sum | in[0..PINGPONG_NUM) | scratch, out 复用 in[0]
    __aicore__ inline void BuildLinearSlotLayout(uint64_t maxSegLen, SlotLayout& layout)
    {
        const uint64_t base = rsParams_.ubReservedBytes;
        const uint64_t sumBytes = maxSegLen * sizeof(SumType);
        const uint64_t ioBytes = maxSegLen * sizeof(IOType);
        layout.sumByteOff = base;
        for (uint64_t slot = 0; slot < PINGPONG_NUM; ++slot) {
            layout.inByteOff[slot] = base + sumBytes + slot * ioBytes;
        }
        layout.scratchByteOff = base + sumBytes + PINGPONG_NUM * ioBytes;
        layout.outByteOff = layout.inByteOff[0];
    }

    __aicore__ inline void LoadLinearSegInput(const SlotLayout& layout, uint64_t segIdx, uint64_t segStart,
                                              uint64_t segLen)
    {
        __gm__ IOType* gmPtr =
            reinterpret_cast<__gm__ IOType*>(rsParams_.inputSegBaseAddrs[segIdx] + segStart * sizeof(IOType));
        auto ubInTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, IOType>(layout.inByteOff[segIdx % PINGPONG_NUM]),
            MakeFrame<IOType>{}(1, segLen));
        asc::te::copy(copyGM2UB_, ubInTensor, MakeGmFrame<IOType>(gmPtr, segLen));
    }

    __aicore__ inline void AccumulateSegInput(const SlotLayout& layout, uint64_t segIdx, uint64_t segLen)
    {
        const uint64_t slot = segIdx % PINGPONG_NUM;
        if (segIdx == 0) {
            cast_reg_loop(MakeUbVec<IOType>(layout.inByteOff[slot], segLen),
                          MakeUbVec<SumType>(layout.sumByteOff, segLen), static_cast<uint32_t>(segLen));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            cast_reg_loop(MakeUbVec<IOType>(layout.inByteOff[slot], segLen),
                          MakeUbVec<SumType>(layout.scratchByteOff, segLen), static_cast<uint32_t>(segLen));
            AscendC::PipeBarrier<PIPE_V>();
            compute_add(MakeUbVec<SumType>(layout.sumByteOff, segLen),
                        MakeUbVec<SumType>(layout.scratchByteOff, segLen),
                        MakeUbVec<SumType>(layout.sumByteOff, segLen), static_cast<uint32_t>(segLen));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreLinearSegResult(const SlotLayout& layout, uint64_t segStart, uint64_t segLen)
    {
        AscendC::PipeBarrier<PIPE_V>();
        __gm__ IOType* outPtr = reinterpret_cast<__gm__ IOType*>(rsParams_.outputAddr + segStart * sizeof(IOType));
        auto gmOutTensor = MakeGmFrame<IOType>(outPtr, segLen);
        cast_reg_loop(MakeUbVec<SumType>(layout.sumByteOff, segLen), MakeUbVec<IOType>(layout.outByteOff, segLen),
                      static_cast<uint32_t>(segLen));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVT_V_MTE3);
        auto ubOutTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, IOType>(layout.outByteOff),
                                                MakeFrame<IOType>{}(1, segLen));
        asc::te::copy(copyUB2GM_, gmOutTensor, ubOutTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVT_MTE3_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    // 行均分到各 AIV 核: 前 remainderNum 个核各多分 1 行
    __aicore__ inline void SplitToCore(uint32_t totalRowNum, uint32_t totalAivNum, uint32_t coreId,
                                       uint32_t& startRowIdx, uint32_t& rowNumPerCore)
    {
        if (totalAivNum == 0U) {
            startRowIdx = 0;
            rowNumPerCore = 0;
            return;
        }
        rowNumPerCore = totalRowNum / totalAivNum;
        uint32_t remainderNum = totalRowNum % totalAivNum;
        startRowIdx = rowNumPerCore * coreId;
        if (coreId < remainderNum) {
            rowNumPerCore += 1;
            startRowIdx += coreId;
        } else {
            startRowIdx += remainderNum;
        }
    }

    using CopyMakerGM2UB = decltype(asc::te::make_copy(asc::te::copy_gm_to_ub{}));
    using CopyMakerUB2GM = decltype(asc::te::make_copy(asc::te::copy_ub_to_gm{}));
    CopyMakerGM2UB copyGM2UB_ = asc::te::make_copy(asc::te::copy_gm_to_ub{});
    CopyMakerUB2GM copyUB2GM_ = asc::te::make_copy(asc::te::copy_ub_to_gm{});
    RsParams rsParams_{};
};

} // namespace ReduceSumImpl
} // namespace Apace
