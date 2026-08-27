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
 * \file matmul_all_reduce_tiling_base.h
 * \brief
 */
#ifndef MC2_MM_ALLREDUCE_TILING_ARCH35_H
#define MC2_MM_ALLREDUCE_TILING_ARCH35_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "all_reduce_formulaic_tiling.h"
#include "weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_adaptive_split_tiling.h"
#include "weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_reg_base_tiling.h"
#include "weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling.h"
#include "weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_weight_nz_tiling.h"
#include "quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.h"
#include "quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_tiling.h"
#include "op_kernel/mc2_tiling_struct.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "context_transfer.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/unquant_matmul_all_reduce_tiling_data.h"
#include "../../op_kernel/weight_quant_matmul_all_reduce_tiling_data.h"
#include "../../op_kernel/quant_matmul_all_reduce_tiling_data.h"

namespace optiling {

using AntiQuantType = Mc2QuantType;

constexpr uint8_t MC2_DEBUG_ONLY_AICPU = 4;
constexpr uint8_t DIM_NUM_THREE = 3;
constexpr uint8_t DIM_NUM_FOUR = 4;
constexpr uint8_t DIM_NUM_TWO = 2;
constexpr uint8_t DIM_NUM_ONE = 1;
constexpr int64_t SUPPORTED_BLOCK_SIZE = 128;
constexpr uint64_t MX_GROUP_SIZE = 64;
constexpr uint64_t MX_GROUP_SIZE_K = 32;
constexpr uint64_t MX_GROUP_SIZE_M = 1;
constexpr uint64_t MX_GROUP_SIZE_N = 1;
constexpr uint64_t MM_UN_ALINGNED_TILING_KEY_A2 = 65535UL;
constexpr uint64_t MM_ALINGNED_TILING_KEY_A2 = 65536UL;
constexpr uint64_t TILING_KEY_BASE_VALUE = 10000000000000000000UL;
constexpr uint64_t MM_TRANSB_TILING_KEY = 10000000000000002001UL;

struct L2TilePara {
    uint32_t mTile;
    uint32_t nTile;
    uint32_t mTileBlock;
    uint32_t nTileBlock;
    uint32_t mBlockCntTail;
    uint32_t nBlockCntTail;
};
enum class MatmulAllReduceTiling {
    ALL_REDUCE_GENERAL_910 = 1,
    ALL_REDUCE_GENERAL_310 = 2,
    ALL_REDUCE_A16W8 = 3,
    ALL_REDUCE_A16W4 = 4,
    ALL_REDUCE_A8W8 = 5,
};

const std::map<ge::DataType, uint64_t> D_TYPE_SIZE_MAP = {
    {ge::DT_BF16, 2},
    {ge::DT_FLOAT16, 2},
    {ge::DT_FLOAT, 4},
    {ge::DT_INT8, 1},
    {ge::DT_INT32, 4},
    {ge::DT_HIFLOAT8, 1},
    {ge::DT_FLOAT8_E4M3FN, 1},
    {ge::DT_FLOAT8_E5M2, 1},
    {ge::DT_INT4, ge::GetSizeByDataType(ge::DT_INT4)},
    {ge::DT_FLOAT4_E2M1, ge::GetSizeByDataType(ge::DT_FLOAT4_E2M1)},
};

const std::map<matmul_tiling::DataType, uint64_t> D_MTYPE_SIZE_MAP = {
    {matmul_tiling::DataType::DT_BFLOAT16, 2},
    {matmul_tiling::DataType::DT_FLOAT16, 2},
    {matmul_tiling::DataType::DT_FLOAT, 4},
    {matmul_tiling::DataType::DT_INT8, 1},
    {matmul_tiling::DataType::DT_INT32, 4},
    {matmul_tiling::DataType::DT_HIFLOAT8, 1},
    {matmul_tiling::DataType::DT_FLOAT8_E4M3FN, 1},
    {matmul_tiling::DataType::DT_FLOAT8_E5M2, 1},
    // DT_INT4的key值代码里面保证有效
    {matmul_tiling::DataType::DT_INT4, D_TYPE_SIZE_MAP.at(ge::DT_INT4)},
    {matmul_tiling::DataType::DT_FLOAT4_E2M1, D_TYPE_SIZE_MAP.at(ge::DT_FLOAT4_E2M1)},
};

enum class AllReduceScenario {
    A8W8 = 0,
    A16W8 = 1,
    A16W4 = 2,
    FP8HIF8 = 3,
    MXFP4 = 4,
    MXFP8 = 5,
    INVALID,
};

// currently 310p use
enum class ParamValue {
    INPUT = 0,
    WEIGHT = 1,
    BIAS = 2,
    X3 = 3,
    ANTIQUANT_SCALE = 4,
    ANTIQUANT_OFFSET = 5,
    DEQUANT = 6,
};

class MatmulAllReduceTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MatmulAllReduceTilingBase(gert::TilingContext *context)
        : TilingBaseClass(context),
          mmrCtxInfo_(mmrCtxInfoSelf_),
          tilingData_(tilingDataSelf_)
    {
        Reset();
        // tilingdata作为类的成员随构造函数实例化，使用引用来统一数据访问接口，同时用实体对象来管理数据的生命周期
        // 在PostTiling中memcpy到device
        OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                 0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                        OP_LOGE(opName_, "fail to memset tiling data"), return;);
    }
    MatmulAllReduceTilingBase(gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo)
        : TilingBaseClass(context),
          mmrCtxInfo_(*mmrCtxInfo),
          tilingData_(tilingDataSelf_)
    {
        Reset();
    }
    MatmulAllReduceTilingBase(gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo,
                              Mc2Tiling::MatmulAllReduceTilingData *tilingData)
        : TilingBaseClass(context),
          mmrCtxInfo_(*mmrCtxInfo),
          tilingData_(*tilingData)
    {
        Reset();
    }
    ~MatmulAllReduceTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    // every subclass need do IsCapable() DoOpTiling() and GetTilingKey()
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    virtual Mc2Tiling::RCSTiling &MutableRCSTilingData()
    {
        return tilingData_.param;
    }
    virtual Mc2Tiling::Mc2Msg &MutableMc2MsgData()
    {
        return tilingData_.msg;
    }
    virtual AscendC::tiling::TCubeTiling &MutableTCubeTileTilingData()
    {
        return tilingData_.matmulTiling;
    }
    virtual AscendC::tiling::TCubeTiling &MutableTCubeTailTilingData()
    {
        return tilingData_.tailTiling;
    }
    ge::graphStatus AnalyzeShapeAttr();
    void PrintTilingData();
    virtual void PrintExtendMatmulTiling([[maybe_unused]] bool isTail) {};
    // tiling
    void DoAllReduceTiling(bool useHcclApi = false);
    void CalcAllReduceSendRecvParams();
    void DoRCSTiling();
    void SetMCutSocVersion(SocVersion &inputSocVersion);
    void DoSplitMTiling();
    virtual CutResult GetTilingResult();
    ge::graphStatus DoMatmulTiling(matmul_tiling::MultiCoreMatmulTiling &mm1, AscendC::tiling::TCubeTiling &cubeTiling);
    void DoL2CacheTiling(Mc2Tiling::Mc2L2cacheTilePara &l2cacheTiling);
    void setUseBufferType();
    uint8_t CalcBufferTypeByWindowSize();
    void CalcSendOffWithPadM(uint64_t &tileSendOff, uint64_t &tailSendOff);

    void Reset();
    bool AnalyzeAttrs();
    void SetQuantData();
    void SetAntiQuantData();
    void SetCommQuantScale();
    void GetAtomicAddData();
    bool CheckBiasShape(const uint64_t nValue) const;
    virtual bool IsPerTensorDequantScale(const gert::Shape &dequantShape) const;
    bool CheckDequantScaleShape(const uint64_t nValue) const;
    bool CheckDequantScalePerBlock(const gert::StorageShape *dequantScaleShape) const;
    bool CheckPerblockShape(const uint64_t mValue, const uint64_t kValue) const;
    bool CheckPertokenScaleShape(const uint64_t mValue, const uint64_t kValue) const;
    bool CheckCommQuantScaleShape(const uint64_t nValue) const;
    bool CheckAntiQuantScaleShape(const uint64_t kValue, const uint64_t nValue);
    bool CheckAntiQuantOffsetValid() const;
    bool CheckAntiQuantOffsetShape(const gert::StorageShape *scale) const;
    bool CheckA16W4Shape(const uint64_t kValue, const uint64_t nValue);
    bool CheckPlatformInfo() const;
    bool CheckMXScenarioScaleShape(const uint64_t dimZeroValue, const uint64_t kValue,
                                   const gert::StorageShape *scaleShape, const bool isPertoken,
                                   const bool isMXfp4) const;
    void GetScaleMNAndKIdx(const gert::StorageShape *scaleShape, bool isPertoken, uint64_t scaleDimNum, uint64_t &MN,
                           uint64_t &K) const;
    AllReduceScenario GetAllReduceScenario(const ge::DataType aType, const ge::DataType bType,
                                           const gert::StorageShape *dequantScale,
                                           const gert::StorageShape *antiQuantScale) const;
    bool IsA16W8Scenario(const ge::DataType aType, const ge::DataType bType,
                         const gert::StorageShape *antiQuantScale) const;
    bool AnalyzeInputs();
    bool SetArgs(ge::DataType aType, ge::DataType bType, ge::DataType cType, ge::DataType biasType, bool isBias);
    uint64_t GetNValue() const;
    uint64_t GetKValue() const;
    uint64_t GetMValue() const;
    uint64_t GetBatchValue() const;
    virtual ge::graphStatus CheckInput();
    ge::graphStatus CheckInputBias() const;
    ge::graphStatus CheckA16W16();
    ge::graphStatus CheckA8W8();
    ge::graphStatus CheckA16W8();
    ge::graphStatus CheckEmptyTensor();
    ge::graphStatus CheckQuantEmptyTensor();
    ge::graphStatus CheckWeightQuantEmptyTensor();
    mc2tiling::HcclDataType GetDataType(ge::DataType type);
    virtual void DoEmptyTensorTiling() {};
    virtual void GetL2CacheParm(uint64_t &l2CacheSize, uint64_t &singleMatrixSize, uint32_t &tileSize,
                                uint32_t &tileLimit, bool useNewPara)
    {
        (void)l2CacheSize;
        (void)singleMatrixSize;
        (void)tileSize;
        (void)tileLimit;
        (void)useNewPara;
    };
    bool CalL2TilePara(L2TilePara &tileL2, uint64_t mValue, uint64_t kValue, uint64_t nValue, uint32_t cubeCoreNum);
    void CalcL2TileBlock(L2TilePara &tileL2, uint64_t mValue, uint64_t nValue, uint64_t kValue, uint64_t blockBaseM,
                         uint64_t blockBaseN, uint32_t tileSize, uint32_t tileLimit, uint64_t sizeA, uint64_t sizeB,
                         uint32_t cubeCoreNum) const;
    AntiQuantType GetAntiQuantType();
    bool HasAntiQuantOffset() const;
    void CalcUbTiling();
    ge::graphStatus CheckRanksizePlatformSupported() const;
    uint64_t tileMValue_{0U};
    uint64_t tailMValue_{0U};
    bool isQuantKey_{false};
    bool isPerTensor_{false};
    bool isPerBlock_{false};
    AntiQuantType antiQuantType_{AntiQuantType::NONE};
    Mc2QuantType quantType_{Mc2QuantType::PER_TENSOR};
    uint64_t antiGroupSize_{0UL}; // anti quant per group info
    bool isUbQuant_{false};
    bool enableL2Cache_{false};
    bool enableBiasConvert_{false};
    const char *opName_;
    const char *reduceOp_;
    uint32_t rankSize_{0U};
    uint32_t libApiWorkSpaceSize_{0U};
    platform_ascendc::SocVersion socVersion_;
    NpuArch npuArch_;
    bool supportL0c2Out_{false};
    mc2tiling::TilingArgs args_;
    bool isWeightNz_{false};
    bool isKZero_{false};
    bool isA8W8_{false};
    bool isA16W8_{false};
    bool isA16W4_{false};
    bool biasOnVec_{
        false}; // arch35 非量化场景 bias 移至 vec epilogue 累加（仅 arch35 子类 DoOpTiling 设置，其余架构保持 false）
    uint32_t biasUbCnt_{0U}; // bias epilogue 的 N 方向 UB 切分粒度（CalcUbTiling 计算后暂存，子类写入 extTiling）
    AllReduceScenario scenario_{AllReduceScenario::INVALID};
    MMRCtxInfo mmrCtxInfoSelf_{};
    MMRCtxInfo &mmrCtxInfo_;
    Mc2Tiling::MatmulAllReduceTilingData tilingDataSelf_{};
    Mc2Tiling::MatmulAllReduceTilingData &tilingData_;
    ge::graphStatus GetAndConvertCommMode(uint8_t &commMode) const;
    uint8_t GetCommMode() const;
    ge::graphStatus CheckCommModeA2() const;
};
} // namespace optiling
#endif // MC2_MM_ALLREDUCE_TILING_ARCH35_H
