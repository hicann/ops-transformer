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
 * \file qkv_rms_norm_rope_cache_tiling_arch35.h
 * \brief QkvRmsNormRopeCache 的 arch35(Ascend950 / DAV_3510) tiling 声明
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_QKV_RMS_NORM_ROPE_CACHE_ARCH35_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_QKV_RMS_NORM_ROPE_CACHE_ARCH35_H_

#include "qkv_rms_norm_rope_cache_tiling.h"

namespace optiling {

// ---------------------------------------------------------------------------
// arch35(Ascend950)regbase tiling 结构体。
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

// arch35 模板优先级必须高于 A2 模板:D2 模板的 IsCapable() 在 regbase SoC 上返回 false,
// 两个模板并存,由优先级升序取第一个 capable 的模板。
constexpr int32_t TEMPLATE_REGBASE_PRIORITY = 2000;
// arch35 的 tiling key(cache_mode 恒 PA_NZ)
constexpr int64_t TILING_KEY_REGBASE_PA_NZ = 10000;

// arch35(Ascend950 / DAV_3510)regbase tiling 模板。
class QkvRmsNormRopeCacheTilingRegbase : virtual public QkvRmsNormRopeCacheTilingBase {
public:
    // 基类是虚继承,最派生类必须显式初始化虚基类
    explicit QkvRmsNormRopeCacheTilingRegbase(gert::TilingContext* tillingContext)
        : QkvRmsNormRopeCacheTilingBase(tillingContext)
    {}
    ~QkvRmsNormRopeCacheTilingRegbase() {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

protected:
    // 参数校验(实现在 regbase_tiling.cpp)
    ge::graphStatus GetShapeAttrsInfoInner();
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
    void CalBlockTiling();
    ge::graphStatus CalUbTiling();
    // A5 侧收紧:cos/sin 第一维必须恰为 B*S(原委见 regbase_tiling.cpp 实现处)。
    ge::graphStatus CheckCosSinExact();
    // cache BlockSize 按 dtype 判 32 Byte 对齐
    ge::graphStatus CheckBlockSizeAlign();

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

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_QKV_RMS_NORM_ROPE_CACHE_ARCH35_H_
