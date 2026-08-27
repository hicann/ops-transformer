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
 * \file matmul_all_reduce_based_all_reduce.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_BASED_ALL_REDUCE_H
#define MATMUL_ALL_REDUCE_BASED_ALL_REDUCE_H
#include "matmul_all_reduce_base.h"
namespace MatmulAllReduceImpl {
template <typename XType, typename YType, Mc2CoreType CoreType, int commMode>
class MatmulAllReduceBase<XType, YType, CoreType, false, commMode> {
public:
    __aicore__ inline MatmulAllReduceBase(MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs, ArnGmAddrs *arnAddrs,
                                          MC2TilingHeader *tilingData, TPipe *tPipe)
        : addrs_(addrs),
          quantAddrs_(quantAddrs),
          arnAddrs_(arnAddrs),
          tilingData_(tilingData),
          tPipe_(tPipe)
    {
        if constexpr (CoreType == Mc2CoreType::ON_CUBE) {
            notifyFlag_ = (GetBlockIdx() == 0);
        } else {
            notifyFlag_ = (g_coreType == AscendC::AIV && GetBlockIdx() == 0);
        }
        paramInTiling_ = &tilingData->param;
        biasUbCnt_ = 0U; // 默认无 bias，由非量子类构造时覆盖
        hasBias_ = false;
    }

    __aicore__ inline void Init()
    {
        hccl_.InitV2(GetHcclContext<0>(), tilingData_);
        hccl_.SetCcTilingV2(offsetof(MC2TilingHeader, mc2CcTiling));
        __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
        OOMInit(context);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        addrs_->cGM = addrs_->workspaceGM + paramInTiling_->nd2NzWorkLen + paramInTiling_->biasLen;
#else
        if (msgInTiling_->useBufferType == MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_WINDOW_IN &&
            context->config.determinism != 1) {
            addrs_->cGM = hccl_.GetWindowsInAddr(hccl_.GetRankId());
        }
#endif
        addFlag_ = (paramInTiling_->isAdd != 0U);
        tailFlag_ = (paramInTiling_->tailCnt != 0U);
        isOneTileFlag_ = (paramInTiling_->tileCnt == 1U) && (paramInTiling_->tailCnt == 0U);
        const uint64_t mVal = isOneTileFlag_ ? ((uint64_t)paramInTiling_->rankM) : (uint64_t)tileInfo_.mmTiling->M;

        tileInfo_.aOffset = mVal * (uint64_t)tileInfo_.mmTiling->Ka;
        if (AscendC::IsSameType<XType, fp4x2_e2m1_t>::value) {
            // In 4-bits scenario, the data length is 0.5, the size of Xtype is 1, it should be divided by 2.
            tileInfo_.aAddrOffset = tileInfo_.aOffset * sizeof(XType) / 2;
        } else {
            tileInfo_.aAddrOffset = tileInfo_.aOffset * sizeof(XType);
        }
        tileInfo_.cOffset = mVal * (uint64_t)tileInfo_.mmTiling->N;
        tileInfo_.cAddrOffset = tileInfo_.cOffset * sizeof(YType);
        if (tailFlag_) {
            tailInfo_.aOffset = (uint64_t)tailInfo_.mmTiling->M * (uint64_t)tailInfo_.mmTiling->Ka;
            if (AscendC::IsSameType<XType, fp4x2_e2m1_t>::value || AscendC::IsSameType<XType, fp4x2_e1m2_t>::value) {
                // In 4-bits scenario, the data length is 0.5, the size of Xtype is 1, it should be divided by 2.
                tailInfo_.aAddrOffset = tailInfo_.aOffset * sizeof(XType) / 2;
            } else {
                tailInfo_.aAddrOffset = tailInfo_.aOffset * sizeof(XType);
            }
            tailInfo_.cOffset = (uint64_t)tailInfo_.mmTiling->M * (uint64_t)tailInfo_.mmTiling->N;
            tailInfo_.cAddrOffset = tailInfo_.cOffset * sizeof(YType);
        }

        if (notifyFlag_) {
            tileInfo_.hcclHandleId =
                hccl_.template AllReduce(addrs_->cGM, addrs_->outputGM, tileInfo_.cOffset, HCCL_DATA_TYPE,
                                         AscendC::HCCL_REDUCE_SUM, paramInTiling_->tileCnt);
            if (tailFlag_) {
                const uint64_t offset = tileInfo_.cAddrOffset * paramInTiling_->tileCnt;
                tailInfo_.hcclHandleId =
                    hccl_.template AllReduce(addrs_->cGM + offset, addrs_->outputGM + offset, tailInfo_.cOffset,
                                             HCCL_DATA_TYPE, AscendC::HCCL_REDUCE_SUM, paramInTiling_->tailCnt);
            }
        }
    }

protected:
#if (ORIG_DTYPE_X1 == DT_BF16)
    __aicore__ inline void PreProcForBiasOnVector()
    {
        if (paramInTiling_->biasLen == 0U) {
            return;
        }

        TBuf<TPosition::VECCALC> tmpBuf;
        tPipe_->InitBuffer(tmpBuf, TOTAL_UB_SIZE);
        CastBFtoFloatOnAiv0(addrs_->workspaceGM, addrs_->biasGM, paramInTiling_->rankN, tmpBuf);
        SyncAll<false>();
        addrs_->biasGM = addrs_->workspaceGM;
        addrs_->workspaceGM += paramInTiling_->biasLen;
    }
#endif

    __aicore__ inline void PostProcEachTurn(AscendC::HcclHandle handleId, uint64_t aOffset, uint64_t cOffset,
                                            uint64_t index = 0, uint64_t mTileValue = 0)
    {
        if (hasBias_) {
            // 有 bias：走新的二维 epilogue（bias + 可选 x3）
            // 用 cOffset / sizeof(YType) / rankN 精确计算实际 M 行数（rankN 必然 > 0）
            uint64_t actualM = cOffset / sizeof(YType) / paramInTiling_->rankN;
            Mc2SyncAll<CoreType>();
            MatmulAllReduceEpilogueKernel<YType>(addrs_->cGM,    // mmOutput (in/out)
                                                 addrs_->biasGM, // bias (不前移)
                                                 addrs_->addGM,  // x3 (前移，无 x3 时 hasAdd=false 不访问)
                                                 actualM,        // 本 tile 的 M 行数（精确计算）
                                                 paramInTiling_->rankN, // N 值
                                                 biasUbCnt_,            // bias UB 切分（基类成员）
                                                 addFlag_,              // 是否有 x3
                                                 tPipe_);
            if (addFlag_) {
                addrs_->addGM += cOffset;
            }
            // biasGM 不前移
        } else if (addFlag_ && addrs_->cGM != addrs_->addGM) {
            // 无 bias 有 x3：保留当前一维 add_x3 路径不变
            Mc2SyncAll<CoreType>();
            MatmulAllReduceAddX3Kernel<YType>(addrs_->cGM, addrs_->addGM, cOffset / sizeof(YType),
                                              paramInTiling_->addX3UbCnt, tPipe_);
            addrs_->addGM += cOffset;
        }

        addrs_->aGM += aOffset;
        addrs_->cGM += cOffset;

        Mc2SyncAll<CoreType>();
        if (notifyFlag_) {
            hccl_.Commit(handleId);
        }
    }

    __aicore__ inline void HcclFinalize()
    {
        if (notifyFlag_) {
            for (uint32_t i = 0; i < paramInTiling_->tileCnt; ++i) {
                hccl_.Wait(tileInfo_.hcclHandleId);
            }

            if (tailFlag_) {
                for (uint32_t i = 0; i < paramInTiling_->tailCnt; ++i) {
                    hccl_.Wait(tailInfo_.hcclHandleId);
                }
            }
        }

        Mc2SyncAll<CoreType>();
        if (notifyFlag_) {
            hccl_.Finalize();
        }
    }

    MC2GmAddrs *addrs_;
    QuantGmAddrs *quantAddrs_;
    ArnGmAddrs *arnAddrs_;
    Mc2Tiling::Mc2Msg *msgInTiling_;
    Mc2Tiling::RCSTiling *paramInTiling_;
    MC2TileInfo tileInfo_, tailInfo_;
    MC2TilingHeader *tilingData_;
    TPipe *tPipe_;
    typename HcclTypeSelector<commMode>::type hccl_;
    bool notifyFlag_;
    bool addFlag_;
    bool tailFlag_;
    bool isOneTileFlag_;
    uint32_t biasUbCnt_; // bias epilogue 的 N 方向 UB 切分粒度（子类构造时从 tiling struct 读取）
    bool hasBias_;       // bias 是否存在（子类构造时从 tiling struct 读取）
};
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_BASED_ALL_REDUCE_H
