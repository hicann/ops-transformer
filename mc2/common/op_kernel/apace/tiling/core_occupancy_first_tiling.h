/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file core_occupancy_first_tiling.h
 * \brief 核利用率优先的通信切分策略：根据 matmul tiling 结果推导通信切分方案
 *
 * 以核利用率为第一优先级，通信切分直接跟随 matmul 的最优分核结果——每轮通信的 M 跨度
 * 等于一个计算波（headMSize = ceil(usedCoreNum / nTile) * baseM），不引入时延模型。
 *
 *
 * 接口分层:
 *   核心推导 DeriveCommCut()——(m, n, baseM, baseN, usedCoreNum) → head/tail 切分, 纯几何;
 *   SWAT 形态包装 GetCommTilingData()——QuantMatmulTilingData 版(data/scale 双通道)
 *   统一入口 GetCoreOccupancyFirstTiling()——SWAT tiling → 通信切分 → fp4 K 轴修正的
 *   全序列封装。
 */

#pragma once

#include <cstdint>
#include "tiling_data_struct_define.h"
#include "matmul_tiling/quant_matmul_tiling.h"
#include "../utils/apace_constant.h"

// 核心推导: 每轮通信 M 跨度 = 一个计算波 headMSize = ceil(usedCoreNum/nTile)×baseM,
// 纯几何无时延模型。baseM/baseN/usedCoreNum 任一为 0(matmul tiling 异常)返回 false,
// 由调用方决定回退行为(校验与日志在 op_tiling)。
inline bool DeriveCommCut(uint64_t m, uint64_t n, uint64_t baseM, uint64_t baseN, uint64_t usedCoreNum,
                          uint64_t& headMSize, uint64_t& headTileCnt, uint64_t& tailMSize, uint64_t& tailTileCnt)
{
    if (baseM == 0UL || baseN == 0UL || usedCoreNum == 0UL) {
        return false;
    }
    const uint64_t nTile = (n + baseN - 1) / baseN;
    headMSize = ((usedCoreNum + nTile - 1) / nTile) * baseM;
    if (headMSize == 0UL) {
        return false;
    }
    headTileCnt = m / headMSize;
    tailMSize = m % headMSize;
    tailTileCnt = (tailMSize > 0) ? 1 : 0;
    return true;
}

class CoreOccupancyFirstTiling {
public:
    /*!
     * \brief 计算数据通道的通信 tiling（SWAT 形态: QuantMatmulTilingData 输入, data/scale 双通道）
     *
     * \param m         M_per_rank（单卡输出 M）
     * \param n         N
     * \param ka        per-rank K
     * \param mmTiling  已完成 matmul tiling 的结果（需填充 baseM/baseN/usedCoreNum）
     * \param dataTiling   [out] 数据通信切分
     * \param scaleTiling  [out] Scale 通信切分（nonSplitAxisSize = ka / 32）
     */
    static void GetCommTilingData(uint64_t m, uint64_t n, uint64_t ka, const QuantMatmulTilingData& mmTiling,
                                  CommTilingData& dataTiling, CommTilingData& scaleTiling)
    {
        uint64_t headMSize = 0;
        uint64_t headTileCnt = 0;
        uint64_t tailMSize = 0;
        uint64_t tailTileCnt = 0;
        if (!DeriveCommCut(m, n, mmTiling.baseM, mmTiling.baseN, mmTiling.usedCoreNum, headMSize, headTileCnt,
                           tailMSize, tailTileCnt)) {
            // 防御性零值守卫: matmul tiling 异常时返回零值 tiling(与原实现一致)
            dataTiling = {};
            scaleTiling = {};
            return;
        }

        dataTiling.splitAxisTileSize = headMSize;
        dataTiling.splitAxisTileCnt = headTileCnt;
        dataTiling.splitAxisTailSize = tailMSize;
        dataTiling.splitAxisTailCnt = tailTileCnt;
        dataTiling.nonSplitAxisSize = ka;

        scaleTiling = dataTiling;
        scaleTiling.nonSplitAxisSize = ka / MX_GROUP_SIZE;
    }
};

/*!
 * \brief 统一入口: SWAT matmul tiling → 核利用率优先通信切分 → fp4 K 轴修正
 *
 * \param platformInfo  平台信息(context_->GetPlatformInfo())
 * \param m/n/ka        输入shape维度
 * \param isFp4         x1 为 MXFP4 打包存储(2 fp4/字节): SWAT 按 fp4 推导 baseK 与 L1 布局,
 *                      数据通信 K 轴字节数减半(scale 为 e8m0, 32 元素/字节, 不受影响)
 * \param hasBias       是否带 bias 输入
 * \param[out] mmTiling     matmul tiling 结果(含 usedCoreNum, 调用方校验/SetBlockDim 用)
 * \param[out] dataTiling   数据通道通信切分
 * \param[out] scaleTiling  Scale 通道通信切分
 */
inline void GetCoreOccupancyFirstTiling(fe::PlatFormInfos* platformInfo, uint64_t m, uint64_t n, uint64_t ka,
                                        bool isFp4, bool hasBias, QuantMatmulTilingData& mmTiling,
                                        CommTilingData& dataTiling, CommTilingData& scaleTiling)
{
    // ---- matmul tiling(SWAT): MXFP4 按 fp4 推导 baseK(256 vs 128)与 L1 布局, 其余按 fp8 ----
    if (isFp4) {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT4_E2M1, mm::DataType::DT_FLOAT4_E2M1, mm::BiasDataType::DT_FLOAT>
            tilingEngine;
        tilingEngine.SetPlatformInfoPtr(platformInfo);
        tilingEngine.SetBiasInfo(hasBias);
        tilingEngine.EnableBaseMHalving(true);
        tilingEngine.SetOptimizeEnable(false);
        tilingEngine.GetTilingData(m, n, ka, false, true, mmTiling);
    } else {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN,
                              mm::BiasDataType::DT_FLOAT>
            tilingEngine;
        tilingEngine.SetPlatformInfoPtr(platformInfo);
        tilingEngine.SetBiasInfo(hasBias);
        tilingEngine.EnableBaseMHalving(true);
        tilingEngine.SetOptimizeEnable(false);
        tilingEngine.GetTilingData(m, n, ka, false, true, mmTiling);
    }

    // ---- 通信切分: 核利用率优先 ----
    CoreOccupancyFirstTiling::GetCommTilingData(m, n, ka, mmTiling, dataTiling, scaleTiling);

    // ---- MXFP4 打包存储(2 fp4/字节): 数据通信的 K 轴字节数需减半 ----
    if (isFp4) {
        dataTiling.nonSplitAxisSize = ka / 2;
    }
}
