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
 * \file qkv_rms_norm_rope_cache_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_QKV_RMS_NORM_ROPE_CACHE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_QKV_RMS_NORM_ROPE_CACHE_H_

#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include <unordered_map>
namespace optiling {
// tiling结构体
BEGIN_TILING_DATA_DEF(QkvRmsNormRopeCacheTilingData)
TILING_DATA_FIELD_DEF(int64_t, batchSize); // Bqkv
TILING_DATA_FIELD_DEF(int64_t, seqLength); // Sqkv
TILING_DATA_FIELD_DEF(int64_t, numHead);   // Nqkv
TILING_DATA_FIELD_DEF(int64_t, qkvDim);    // D
TILING_DATA_FIELD_DEF(int64_t, ropeRange); // D
TILING_DATA_FIELD_DEF(int64_t, numHeadQ);  // Nq
TILING_DATA_FIELD_DEF(int64_t, numHeadK);  // Nk
TILING_DATA_FIELD_DEF(int64_t, numHeadV);  // Nv
TILING_DATA_FIELD_DEF(int64_t, blockNum);  // k_cache/v_cache的blockNum
TILING_DATA_FIELD_DEF(int64_t, blockSize); // k_cache/v_cache的blockSize
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockFactorQ);
TILING_DATA_FIELD_DEF(int64_t, blockFactorK);
TILING_DATA_FIELD_DEF(int64_t, blockFactorV);
TILING_DATA_FIELD_DEF(int64_t, blockDim);
TILING_DATA_FIELD_DEF(int64_t, blockDimQ);
TILING_DATA_FIELD_DEF(int64_t, blockDimK);
TILING_DATA_FIELD_DEF(int64_t, blockDimV);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, ubFactorQ);
TILING_DATA_FIELD_DEF(int64_t, ubFactorK);
TILING_DATA_FIELD_DEF(int64_t, ubFactorV);
TILING_DATA_FIELD_DEF(float, reciprocal);
TILING_DATA_FIELD_DEF(int64_t, isOutputQkv);
TILING_DATA_FIELD_DEF(int64_t, isKQuant);
TILING_DATA_FIELD_DEF(int64_t, isVQuant);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QkvRmsNormRopeCache, QkvRmsNormRopeCacheTilingData)

// ---------------------------------------------------------------------------
// arch35(Ascend950)regbase tiling 结构体。
// 与 A2 结构体不共用:用独立 tiling key(10000)注册,避免 A5 结构与 A2 的
// REGISTER_TILING_DATA_CLASS(QkvRmsNormRopeCache, ...) 默认注册相互覆盖。
// 同族先例:kv_rms_norm_rope_cache(legacy 1000~5011 / regbase 10000、20000)。
// ---------------------------------------------------------------------------
BEGIN_TILING_DATA_DEF(QkvRmsNormRopeCacheRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, batchSize);   // Bqkv
TILING_DATA_FIELD_DEF(int64_t, seqLength);   // Sqkv
TILING_DATA_FIELD_DEF(int64_t, numHead);     // Nqkv
TILING_DATA_FIELD_DEF(int64_t, qkvDim);      // D(恒 128)
TILING_DATA_FIELD_DEF(int64_t, numHeadQ);    // Nq
TILING_DATA_FIELD_DEF(int64_t, numHeadK);    // Nk
TILING_DATA_FIELD_DEF(int64_t, numHeadV);    // Nv
TILING_DATA_FIELD_DEF(int64_t, blockSize);   // k_cache/v_cache 的 blockSize
TILING_DATA_FIELD_DEF(int64_t, blockNum);    // 同上;index 的合法上界 = blockNum * blockSize
TILING_DATA_FIELD_DEF(int64_t, blockFactor); // 每核负责的 token 数(末核吸收余数)
TILING_DATA_FIELD_DEF(int64_t, ubFactor);    // 每次 UB 迭代处理的 token 数
TILING_DATA_FIELD_DEF(int64_t, isOutputQkv); // 是否输出 *_before_quant
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, reciprocal); // 1 / D
// UB buffer 字节数一律由 host 一次算准、内核只透传,不在内核里重算
// (否则 host 预算与内核实际分配是两套算法,少算一个通道就越界)
TILING_DATA_FIELD_DEF(int64_t, inUbBytes);
TILING_DATA_FIELD_DEF(int64_t, cosSinUbBytes);
TILING_DATA_FIELD_DEF(int64_t, qOutUbBytes);
TILING_DATA_FIELD_DEF(int64_t, kProtoUbBytes);
TILING_DATA_FIELD_DEF(int64_t, vProtoUbBytes);
TILING_DATA_FIELD_DEF(int64_t, kCacheUbBytes);
TILING_DATA_FIELD_DEF(int64_t, vCacheUbBytes);
TILING_DATA_FIELD_DEF(int64_t, gammaUbBytes);
TILING_DATA_FIELD_DEF(int64_t, quantUbBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QkvRmsNormRopeCache_10000, QkvRmsNormRopeCacheRegbaseTilingData)

constexpr int32_t TEMPLATE_DS_PRIORITY = 1000;
// arch35 模板优先级必须高于 A2 模板:D2 模板的 IsCapable() 在 regbase SoC 上返回 false,
// 两个模板并存,由优先级升序取第一个 capable 的模板。
constexpr int32_t TEMPLATE_REGBASE_PRIORITY = 2000;
// arch35 的 tiling key(cache_mode 恒 PA_NZ)
constexpr int64_t TILING_KEY_REGBASE_PA_NZ = 10000;

struct QkvRmsNormRopeCacheCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    // 本编译目标是否为 regbase(arch35)SoC。
    // 与 coreNum/ubSize 同理:TilingPrepare 阶段 platformInfo 保证非空,那时算好存下,
    // tiling 阶段 platformInfo 为空时才有得可用 —— 否则 isRegbase_ 会停在默认 false,
    // 让 arch35 误选 A2 的 DS 模板。框架自己的 CompileInfoCommon.socVersion 就是这个用法。
    bool isRegbase = false;
};

enum CacheMode {
    Norm = 0,
    PA = 1,
    PA_NZ = 2,
    PA_BLK_BNSD = 3,
    PA_BLK_NZ = 4
};

constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t ONE_BUFFER = 1;
constexpr int64_t DIM_NUM_ONE = 1;
// Tensor Indices
constexpr int64_t QKV_INDEX = 0;
constexpr int64_t GAMMA_Q_INDEX = 1;
constexpr int64_t GAMMA_K_INDEX = 2;
constexpr int64_t COS_INDEX = 3;
constexpr int64_t SIN_INDEX = 4;
constexpr int64_t INDEX_INDEX = 5;
// In-place inputs
constexpr int64_t Q_OUT_INDEX = 6;
constexpr int64_t K_CACHE_INDEX = 7;
constexpr int64_t V_CACHE_INDEX = 8;
// Quant Scale
constexpr int64_t K_SCALE_IDX = 9;
constexpr int64_t V_SCALE_IDX = 10;
// Quant Offset
constexpr int64_t K_OFFSET_IDX = 11;
constexpr int64_t V_OFFSET_IDX = 12;
constexpr float FAC_K = 0.6; // K路和V路计算量的比例
constexpr float NEAR_ONE = 0.999999f;

// Attr Indices
constexpr int64_t QKV_SIZE_IDX = 0;
constexpr int64_t HEAD_NUMS_IDX = 1;
constexpr int64_t EPSILON_IDX = 2;
constexpr int64_t CACHE_MODE_IDX = 3;
constexpr int64_t IS_OUTPUT_QKV_IDX = 4;

constexpr int64_t SHAPE_IDX_B = 0;
constexpr int64_t SHAPE_IDX_S = 1;
constexpr int64_t SHAPE_IDX_N = 2;
constexpr int64_t SHAPE_IDX_D = 3;
constexpr int64_t SHAPE_IDX_BS = 0;
constexpr int64_t SHAPE_IDX_ND = 1;
constexpr int64_t SHAPE_IDX_BLOCK_NUM = 0;
constexpr int64_t SHAPE_IDX_BLOCK_SIZE = 2;

constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t FLOAT16_BYTES = 2;
constexpr int64_t INT8_BYTES = sizeof(int8_t);
constexpr int64_t FP32_BLOCK_ALIGN_NUM = 8;
constexpr int64_t FP16_BLOCK_ALIGN_NUM = 16;
constexpr int64_t INT8_BLOCK_ALIGN_NUM = 32;
constexpr int64_t BASE_BLOCK_SIZE = 32;

constexpr int64_t DIM_SIZE = 4;
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t NUM_ONE = 1;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_THREE = 3;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t NUM_HUNDRED = 100;
constexpr int64_t NUM_CACHE_MODE_UNIT = 10;

constexpr int64_t BYTES_PER_KILO_BYTE = 1024;
static constexpr int64_t UB_RESERVED_BYTES = 1 * BYTES_PER_KILO_BYTE; // 多留1K

class QkvRmsNormRopeCacheTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit QkvRmsNormRopeCacheTilingBase(gert::TilingContext *tillingContext)
        : TilingBaseClass(tillingContext)
    {}
    ~QkvRmsNormRopeCacheTilingBase() override {}
    uint64_t tilingKey_{0};
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    bool isRegbase_{false};
    int64_t rope_seq_ = 0; // cos/sin的S
    CacheMode currentCacheMode_ = CacheMode::PA_NZ;
    int64_t quantMode_ = 0; // 是否量化
    int64_t batchSize_ = 0;
    int64_t seqLength_ = 0;
    int64_t numHead_ = 0;
    int64_t qkvDim_ = 0;
    int64_t ropeRange_ = 0;
    int64_t numHeadQ_ = 0;
    int64_t numHeadK_ = 0;
    int64_t numHeadV_ = 0;
    int64_t blockNum_ = 0;
    int64_t blockSize_ = 0;
    float epsilon_ = 0.0;
    int64_t blockFactor_ = 0;
    int64_t blockFactorQ_ = 0;
    int64_t blockFactorK_ = 0;
    int64_t blockFactorV_ = 0;
    int64_t blockDim_ = 0;
    int64_t blockDimQ_ = 0;
    int64_t blockDimK_ = 0;
    int64_t blockDimV_ = 0;
    int64_t ubFactor_ = 0;
    int64_t ubFactorQ_ = 0;
    int64_t ubFactorK_ = 0;
    int64_t ubFactorV_ = 0;
    float reciprocal_ = 0.0;
    int64_t isOutputQkv_ = 0;
    int64_t isKQuant_ = 1;
    int64_t isVQuant_ = 1;

    ge::DataType qkvDtype_{ge::DataType::DT_FLOAT16};
    int64_t qkvDtypeSize_{0}; // qkv数据类型所占字节数

protected:
    ge::graphStatus GetShapeAttrsInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override
    {
        return false;
    }
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override {}

protected:
    std::tuple<int64_t, int64_t, int64_t, int64_t> GetShapeTuple(const gert::TilingContext *context,
                                                                 const int64_t index = 0);
    std::tuple<int64_t, int64_t> GetShapeTupleOfTH(const gert::TilingContext *context, const int64_t index = 0);
};

class QkvRmsNormRopeCacheTilingDs : virtual public QkvRmsNormRopeCacheTilingBase {
public:
    explicit QkvRmsNormRopeCacheTilingDs(gert::TilingContext *tillingContext)
        : QkvRmsNormRopeCacheTilingBase(tillingContext)
    {}
    ~QkvRmsNormRopeCacheTilingDs() {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

protected:
    ge::graphStatus GetShapeAttrsInfoInner();
    void CalUbTiling();
    ge::graphStatus CheckQkvValid();
    ge::graphStatus CheckGammaValid(int64_t gammaIdx);
    ge::graphStatus CheckCosSinValid();
    ge::graphStatus CheckIndexValid();
    ge::graphStatus CheckQOutValid();
    ge::graphStatus CheckKCacheValid();
    ge::graphStatus CheckVCacheValid();
    ge::graphStatus CheckKScaleValid();
    ge::graphStatus CheckVScaleValid();
    ge::graphStatus CheckKOffsetValid();
    ge::graphStatus CheckVOffsetValid();

private:
    QkvRmsNormRopeCacheTilingData tilingData_;
};

// arch35(Ascend950 / DAV_3510)regbase tiling 模板。
// 派生自 A2 的 Ds 模板,复用其 GetShapeAttrsInfoInner() 与全部 Check*Valid()
// (shape/dtype/attr 支持面与 A2 严格一致),只重写切核/UB 反推与 tiling key。
class QkvRmsNormRopeCacheTilingRegbase : public QkvRmsNormRopeCacheTilingDs {
public:
    // 基类是虚继承,最派生类必须显式初始化虚基类
    explicit QkvRmsNormRopeCacheTilingRegbase(gert::TilingContext *tillingContext)
        : QkvRmsNormRopeCacheTilingBase(tillingContext),
          QkvRmsNormRopeCacheTilingDs(tillingContext)
    {}
    ~QkvRmsNormRopeCacheTilingRegbase() {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

private:
    void CalBlockTiling();
    ge::graphStatus CalUbTiling();
    // A5 侧收紧:cos/sin 第一维必须恰为 B*S(原委见 regbase_tiling.cpp 实现处)。
    // 基类 CheckCosSinValid 是 A2/A3 共享的,额外放行了 [B,D] 广播形态;
    // 本模板只在 regbase 内再校一次,不改动公共逻辑。
    ge::graphStatus CheckCosSinExact();

private:
    QkvRmsNormRopeCacheRegbaseTilingData tilingData_;
    // host 侧一次算准的 UB buffer 字节数,内核只透传
    int64_t perTokenUbBytes_{0};
    int64_t cosSinUbBytes_{0};
    int64_t qOutUbBytes_{0};
    int64_t kProtoUbBytes_{0};
    int64_t vProtoUbBytes_{0};
    int64_t kCacheUbBytes_{0};
    int64_t vCacheUbBytes_{0};
    int64_t inUbBytes_{0};
    int64_t gammaUbBytes_{0};
    int64_t quantUbBytes_{0};
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_QKV_RMS_NORM_ROPE_CACHE_H_
