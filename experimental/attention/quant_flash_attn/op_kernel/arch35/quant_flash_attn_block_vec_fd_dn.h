/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
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
 * \file quant_flash_attn_block_vec_fd_dn.h
 * \brief MxFP4 DN FlashDecode 第二阶段归并: 各 split 的 partial 槽按 M 行重标定后合成最终 out
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_FD_DN_H_
#define QUANT_FLASH_ATTN_BLOCK_VEC_FD_DN_H_

#include "kernel_tensor.h"
#include "quant_flash_attn_common_def.h"
#include "vf/vf_common_def.h"
#include "vf/vf_attenOut_dn_mxfp4.h"

using namespace AscendC;
using namespace AscendC::Reg;

#include "../../common/op_kernel/memcopy/fa_gm_tensor.h"
#include "../../common/op_kernel/memcopy/fa_ub_tensor.h"
#include "../../common/op_kernel/memcopy/copy_ub_to_gm.h"

namespace QFA_KERNEL {

// out GM/UB 布局映射
template <QFA_LAYOUT LAYOUT_OUT>
__aicore__ inline constexpr GmFormat FdOutGmFormat()
{
    if constexpr (LAYOUT_OUT == QFA_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT_OUT == QFA_LAYOUT::BNSD) {
        return GmFormat::BNGSD;
    } else {
        return GmFormat::TNGD;
    }
}

template <QFA_LAYOUT LAYOUT_OUT>
__aicore__ inline constexpr UbFormat FdOutUbFormat()
{
    if constexpr (LAYOUT_OUT == QFA_LAYOUT::TND) {
        return UbFormat::S1G; // TND: M 行 t-major (r=t*G+g)
    } else {
        return UbFormat::GS1;
    }
}

// 槽 (一条 (b,n2,m-tile) 的一个 S2 核间切分) 布局: acc [slot][d=128][m=128] fp32 未归一化局部分子,
// sum、max 各 [slot][m=128] 1 个 fp32; 归约 M = max_i m_i, e_i = exp(m_i - M), out = (Σ_i e_i·O_i) / (Σ_i l_i·e_i)
template <typename QFAT>
class QuantFlashAttnBlockVecFdDn {
public:
    using OUT_T = typename QFAT::outputType;
    static constexpr QFA_LAYOUT LAYOUT_OUT = QFAT::outLayout;

    static constexpr GmFormat OUT_FORMAT = FdOutGmFormat<LAYOUT_OUT>();
    static constexpr UbFormat UB_FORMAT = FdOutUbFormat<LAYOUT_OUT>();
    static constexpr bool OUT_IS_TND = (LAYOUT_OUT == QFA_LAYOUT::TND);

    struct TaskInfo {
        uint32_t bIdx;
        uint32_t n2Idx;
        uint32_t gS1Idx;
        uint32_t actualCombineLoopSize;
    };

    // FD 业务区字节数: A 段 (归约分子/分批 max·sum/M/S), B 段 (acc 暂存)
    // 两段 overlay 用于避让 cube 侧 UB 写入区与常驻索引表, 具体偏移见 kernel 侧定义
    static __aicore__ inline constexpr uint32_t GetFdAUbBytes()
    {
        return FD_RES_BYTES + 2U * FD_CHUNK_BYTES + 2U * FD_TMP_BYTES;
    }

    static __aicore__ inline constexpr uint32_t GetFdBUbBytes()
    {
        return FD_ACC_BYTES;
    }

private:
    static constexpr uint32_t FD_TMP_BYTES = 256U;
    static constexpr uint32_t FD_D = 128U;
    static constexpr uint32_t FD_M = 64U;
    static constexpr uint32_t FD_CHUNK = 16U; // 单批 split 数; split 总数不受限, 按批滚动
    static constexpr uint32_t FD_RES_BYTES = FD_D * FD_M * sizeof(float);
    static constexpr uint32_t FD_ACC_BYTES = (FD_D / 2U) * FD_M * sizeof(float);
    static constexpr uint32_t FD_CHUNK_BYTES = FD_CHUNK * FD_M * sizeof(float);
    static constexpr uint32_t mBaseSize = 128U;
    static constexpr float FD_NEG_INIT = -3.402823466e+38F; // M 初值, 保证全掩蔽行差值为有限数

    // Mutex 协议: 同 id 上生产者 Unlock 与消费者 Lock 构成跨 pipe 依赖 (信号量初值 0)
    static constexpr uint32_t SYNC_FD_LSE_ID = 8;
    static constexpr uint32_t SYNC_FD_ACC_ID = 10;
    static constexpr uint32_t SYNC_FD_OUT_ID = 2;

    const ConstInfo& constInfo_;
    TaskInfo taskInfo_{};

    GlobalTensor<float> accumOutGm_; // [slot][d=128][m=128] fp32 未归一化局部分子
    GlobalTensor<float> lseSumFdGm_; // [slot][m] fp32 局部分母 (行连续)
    GlobalTensor<float> lseMaxFdGm_; // [slot][m] fp32 局部行 max (ln 域, 行连续)
    FaGmTensor<OUT_T, OUT_FORMAT, uint32_t, OUT_IS_TND> outGmTensor_;
    CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, UB_FORMAT> AttenOutUbToGmFd_;

    LocalTensor<float> fdRes_; // [FD_D][FD_M] 归约分子 (行距 FD_M)
    LocalTensor<float> fdAcc_; // [FD_D/2][FD_M] 单 d 半区 acc 暂存 (行距 FD_M)
    LocalTensor<float> fdMc_;  // [FD_CHUNK][FD_M]: 批次内各行 max, 原地覆写为重标定因子 e_i
    LocalTensor<float> fdLc_;  // [FD_CHUNK][FD_M] 批次内各行 sum
    LocalTensor<float> fdMx_;  // [FD_M] 全局行 max M
    LocalTensor<float> fdSum_; // [FD_M] 全局分母 S

public:
    __aicore__ inline QuantFlashAttnBlockVecFdDn(const ConstInfo& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitGlobalTensor(__gm__ uint8_t* workspace, __gm__ uint8_t* attentionOut,
                                            SeqLensTool<QFAT::qLayout, uint32_t>& qSeqLensTool)
    {
        this->accumOutGm_.SetGlobalBuffer((__gm__ float*)workspace);
        this->lseSumFdGm_.SetGlobalBuffer((__gm__ float*)(workspace + constInfo_.accumOutSize));
        this->lseMaxFdGm_.SetGlobalBuffer(
            (__gm__ float*)(workspace + constInfo_.accumOutSize + constInfo_.logSumExpSize));
        this->outGmTensor_.gmTensor.SetGlobalBuffer((__gm__ OUT_T*)attentionOut);
        if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            outGmTensor_.offsetCalculator.Init(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size,
                                               constInfo_.dSizeV, qSeqLensTool.seqUsedParser);
        } else if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            outGmTensor_.offsetCalculator.Init(constInfo_.n2Size, constInfo_.gSize, constInfo_.dSizeV,
                                               qSeqLensTool.cuSeqLensParser);
        }
    }

    template <uint32_t UB_FD_A_OFFSET, uint32_t UB_FD_B_OFFSET>
    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIV {
            // FD 业务区 overlay 在 FA 静态布局的 AIV 独占区, FA 与 FD 分时复用同一 UB, 不走 TPipe 分配器
            constexpr uint32_t OFF_RES = UB_FD_A_OFFSET;
            constexpr uint32_t OFF_MC = OFF_RES + FD_RES_BYTES;
            constexpr uint32_t OFF_LC = OFF_MC + FD_CHUNK_BYTES;
            constexpr uint32_t OFF_MX = OFF_LC + FD_CHUNK_BYTES;
            constexpr uint32_t OFF_SUM = OFF_MX + FD_TMP_BYTES;
            constexpr uint32_t OFF_ACC = UB_FD_B_OFFSET;

            fdRes_ = LocalTensor<float>(TPosition::VECIN, OFF_RES, FD_RES_BYTES / sizeof(float));
            fdMc_ = LocalTensor<float>(TPosition::VECIN, OFF_MC, FD_CHUNK_BYTES / sizeof(float));
            fdLc_ = LocalTensor<float>(TPosition::VECIN, OFF_LC, FD_CHUNK_BYTES / sizeof(float));
            fdMx_ = LocalTensor<float>(TPosition::VECIN, OFF_MX, FD_TMP_BYTES / sizeof(float));
            fdSum_ = LocalTensor<float>(TPosition::VECIN, OFF_SUM, FD_TMP_BYTES / sizeof(float));
            fdAcc_ = LocalTensor<float>(TPosition::VECIN, OFF_ACC, FD_ACC_BYTES / sizeof(float));
        }
    }

private:
    __aicore__ inline static __ubuf__ float* BufPtr(LocalTensor<float> t)
    {
        return (__ubuf__ float*)t.GetPhyAddr();
    }

    // mx = max(mx, mc); FD_M lanes
    __simd_vf__ static inline void FdMaxVf(__ubuf__ float* mxUb, __ubuf__ float* mcUb)
    {
        RegTensor<float> vmx;
        RegTensor<float> vmc;
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        LoadAlign(vmx, mxUb);
        LoadAlign(vmc, mcUb);
        Max(vmx, vmx, vmc, pregAll);
        StoreAlign(mxUb, vmx, pregAll);
    }

    // e = exp(mc - mx) 原地覆写 mc; sum += l·e; FD_M lanes
    __simd_vf__ static inline void FdExpAccVf(__ubuf__ float* eUb, __ubuf__ float* lUb, __ubuf__ float* sumUb,
                                              __ubuf__ float* mcUb, __ubuf__ float* mxUb)
    {
        RegTensor<float> ve;
        RegTensor<float> vl;
        RegTensor<float> vs;
        RegTensor<float> vmc;
        RegTensor<float> vmx;
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        LoadAlign(vmc, mcUb);
        LoadAlign(vmx, mxUb);
        LoadAlign(vl, lUb);
        LoadAlign(vs, sumUb);
        Sub(ve, vmc, vmx, pregAll);
        Exp(ve, ve, pregAll);
        StoreAlign(eUb, ve, pregAll);
        Mul(vl, vl, ve, pregAll);
        Add(vs, vs, vl, pregAll);
        StoreAlign(sumUb, vs, pregAll);
    }

    // res[r][:] += e[:] · acc[r][:]; 行距 FD_M, 行数编译期
    template <uint32_t ROWS>
    __simd_vf__ static inline void FdAccAddVf(__ubuf__ float* resUb, __ubuf__ float* accUb, __ubuf__ float* eUb)
    {
        RegTensor<float> ve;
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        LoadAlign(ve, eUb);
        for (uint32_t r = 0; r < ROWS; ++r) {
            RegTensor<float> vAcc;
            RegTensor<float> vRes;
            LoadAlign(vAcc, accUb + r * FD_M);
            LoadAlign(vRes, resUb + r * FD_M);
            Mul(vAcc, vAcc, ve, pregAll);
            Add(vRes, vRes, vAcc, pregAll);
            StoreAlign(resUb + r * FD_M, vRes, pregAll);
        }
    }

    // 从 GM [slot][rowLen] 布局批量取 count 个 slot 的 [rowStart, rowStart+sz8) 段到 dst [count][FD_M]
    __aicore__ inline void CopyStridedRows(LocalTensor<float> dst, const GlobalTensor<float>& srcGm, uint64_t startOff,
                                           uint32_t count, uint32_t sz8, uint32_t rowLen)
    {
        DataCopyExtParams params;
        params.blockCount = static_cast<uint16_t>(count);
        params.blockLen = static_cast<uint32_t>(sz8 * sizeof(float));
        params.srcStride = static_cast<uint32_t>((rowLen - sz8) * sizeof(float)); // GM 侧单位: 字节
        params.dstStride = (FD_M - sz8) / 8U;                                     // UB 侧单位: 32B
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(dst, srcGm[startOff], params, padParams);
    }

    // 取 slot 段 [slotBase, slotBase+count) 的行向量 (max 或 sum): 槽内行连续, 槽 stride = mBaseSize
    __aicore__ inline void CopyLseRows(LocalTensor<float> dst, const GlobalTensor<float>& srcGm, uint64_t slotBase,
                                       uint32_t count, uint32_t rowStart, uint32_t sz8)
    {
        CopyStridedRows(dst, srcGm, slotBase * mBaseSize + rowStart, count, sz8, mBaseSize);
    }

    // 取 acc 的 [d0, d0+64) 行、m 列 [blkStart, blkStart+sz8) 到紧凑 [64][64] (行距 256B)
    __aicore__ inline void CopyAccumBlock(uint64_t slot, uint32_t dHalf, uint32_t blkStart, uint32_t sz8)
    {
        uint64_t startOff = slot * mBaseSize * FD_D + (uint64_t)dHalf * (FD_D / 2U) * mBaseSize + blkStart;
        CopyStridedRows(fdAcc_, accumOutGm_, startOff, FD_D / 2U, sz8, mBaseSize);
    }

    // 单块归约: slot 段 [slotBase, slotBase+splitNum) 内 m 行 [blkStart, blkStart+blkRows) (blkRows <= FD_M)
    __aicore__ inline void ReduceBlock(uint64_t slotBase, uint32_t blkStart, uint32_t blkRows,
                                       LocalTensor<float> transUb)
    {
        uint32_t sz8 = ((blkRows + 7U) / 8U) * 8U;
        uint32_t splitNum = taskInfo_.actualCombineLoopSize;

        // ① M = max_i m_i, 分块滚动: 每批读入 FD_CHUNK 个 split 的行 max 折进 fdMx_
        Mutex::Lock<PIPE_V>(SYNC_FD_LSE_ID);
        Duplicate(fdMx_, FD_NEG_INIT, FD_M);
        Mutex::Unlock<PIPE_V>(SYNC_FD_LSE_ID);
        for (uint32_t base = 0; base < splitNum; base += FD_CHUNK) {
            uint32_t cnt = (splitNum - base) < FD_CHUNK ? (splitNum - base) : FD_CHUNK;
            Mutex::Lock<PIPE_MTE2>(SYNC_FD_LSE_ID);
            CopyLseRows(fdMc_, lseMaxFdGm_, slotBase + base, cnt, blkStart, sz8);
            Mutex::Unlock<PIPE_MTE2>(SYNC_FD_LSE_ID);
            Mutex::Lock<PIPE_V>(SYNC_FD_LSE_ID);
            for (uint32_t k = 0; k < cnt; ++k) {
                FdMaxVf(BufPtr(fdMx_), BufPtr(fdMc_[k * FD_M]));
            }
            Mutex::Unlock<PIPE_V>(SYNC_FD_LSE_ID);
        }

        // ② 分块重标定并累加: e_i = exp(m_i - M) 原地覆写 fdMc_, S = Σ_i l_i·e_i,
        //    res = Σ_i e_i·O_i (acc 按 d 半区暂存 [64][64])
        Mutex::Lock<PIPE_V>(SYNC_FD_LSE_ID);
        Duplicate(fdSum_, 0.0f, FD_M);
        Duplicate(fdRes_, 0.0f, FD_D * FD_M);
        Mutex::Unlock<PIPE_V>(SYNC_FD_LSE_ID);
        for (uint32_t base = 0; base < splitNum; base += FD_CHUNK) {
            uint32_t cnt = (splitNum - base) < FD_CHUNK ? (splitNum - base) : FD_CHUNK;
            Mutex::Lock<PIPE_MTE2>(SYNC_FD_LSE_ID);
            CopyLseRows(fdMc_, lseMaxFdGm_, slotBase + base, cnt, blkStart, sz8);
            CopyLseRows(fdLc_, lseSumFdGm_, slotBase + base, cnt, blkStart, sz8);
            Mutex::Unlock<PIPE_MTE2>(SYNC_FD_LSE_ID);
            Mutex::Lock<PIPE_V>(SYNC_FD_LSE_ID);
            for (uint32_t k = 0; k < cnt; ++k) {
                FdExpAccVf(BufPtr(fdMc_[k * FD_M]), BufPtr(fdLc_[k * FD_M]), BufPtr(fdSum_), BufPtr(fdMc_[k * FD_M]),
                           BufPtr(fdMx_));
            }
            Mutex::Unlock<PIPE_V>(SYNC_FD_LSE_ID);

            for (uint32_t k = 0; k < cnt; ++k) {
                for (uint32_t dHalf = 0; dHalf < 2U; ++dHalf) {
                    Mutex::Lock<PIPE_MTE2>(SYNC_FD_ACC_ID);
                    CopyAccumBlock(slotBase + base + k, dHalf, blkStart, sz8);
                    Mutex::Unlock<PIPE_MTE2>(SYNC_FD_ACC_ID);
                    Mutex::Lock<PIPE_V>(SYNC_FD_ACC_ID);
                    FdAccAddVf<FD_M>(BufPtr(fdRes_[(dHalf * (FD_D / 2U)) * FD_M]), BufPtr(fdAcc_),
                                     BufPtr(fdMc_[k * FD_M]));
                    Mutex::Unlock<PIPE_V>(SYNC_FD_ACC_ID);
                }
            }
        }

        // ③ 除分母 + [m][d] 转置 + bf16 打包: 复用 FA 侧 processOut (除数取 S)
        Mutex::Lock<PIPE_V>(SYNC_FD_OUT_ID);
        Mxfp4Api::processOut(transUb, fdRes_, fdSum_);
        Mutex::Unlock<PIPE_V>(SYNC_FD_OUT_ID);

        // ④ 写出 blkRows 行
        Mutex::Lock<PIPE_MTE3>(SYNC_FD_OUT_ID);
        LocalTensor<OUT_T> outUb = transUb.template ReinterpretCast<OUT_T>();
        uint32_t columnCnt = constInfo_.dSizeV + AttentionCommon::BYTE_BLOCK / sizeof(OUT_T);
        FaUbTensor<OUT_T> ubTensor{.tensor = outUb, .rowCount = blkRows, .colCount = columnCnt};
        GmCoord gmCoord{.bIdx = taskInfo_.bIdx,
                        .n2Idx = taskInfo_.n2Idx,
                        .gS1Idx = taskInfo_.gS1Idx + blkStart,
                        .dIdx = 0,
                        .gS1DealSize = blkRows,
                        .dDealSize = constInfo_.dSizeV};
        AttenOutUbToGmFd_(outGmTensor_, ubTensor, gmCoord);
        Mutex::Unlock<PIPE_MTE3>(SYNC_FD_OUT_ID);
    }

public:
    __aicore__ inline void FlashDecode(FDparams& fd, LocalTensor<float> transUb)
    {
        if (!fd.fdCoreEnable) {
            return;
        }
        uint32_t splitNum = fd.fdS2SplitNum;
        if (splitNum == 0U) {
            return;
        }
        taskInfo_.bIdx = fd.fdBN2Idx / constInfo_.n2Size;
        taskInfo_.n2Idx = fd.fdBN2Idx % constInfo_.n2Size;
        taskInfo_.gS1Idx = fd.fdMIdx * mBaseSize;
        taskInfo_.actualCombineLoopSize = splitNum;

        uint32_t mDone = 0;
        while (mDone < fd.mLen) {
            uint32_t blkStart = fd.mStart + mDone;
            uint32_t blkRows = fd.mLen - mDone;
            if (blkRows > FD_M) {
                blkRows = FD_M;
            }
            if (blkStart + blkRows > mBaseSize) {
                blkRows = mBaseSize - blkStart;
            }
            ReduceBlock(fd.fdWorkspaceIdx, blkStart, blkRows, transUb);
            mDone += blkRows;
        }
    }
};

} // namespace QFA_KERNEL
#endif // QUANT_FLASH_ATTN_BLOCK_VEC_FD_DN_H_
