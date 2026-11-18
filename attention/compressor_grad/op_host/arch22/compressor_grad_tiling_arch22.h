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
 * \file compressor_grad_tiling_arch22.h
 * \brief CompressorGrad arch22 tiling 声明（索引/常量/context 复用 compressor_grad_tiling.h）。
 *        BSH 与 TH 双布局：TH 的逐 batch 块数/路由由 kernel 运行时读
 *        cu_seqlens/seqused/start_pos 决定，host 只做 shape 派生。
 */

#ifndef COMPRESSOR_GRAD_TILING_ARCH22_H
#define COMPRESSOR_GRAD_TILING_ARCH22_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "../compressor_grad_tiling.h"
#include "../../op_kernel/arch22/compressor_grad_template_tiling_key.h"
#include "../../op_kernel/arch22/compressor_grad_tiling_data.h"

#ifdef ASCENDC_OP_TEST
#define CMP_GRAD_EXTERN_C extern "C"
#else
#define CMP_GRAD_EXTERN_C
#endif

namespace optiling {
namespace {

// scatter UB budget: arch2201 TBuf ceiling (192KB UB minus TPipe reserve)
constexpr uint64_t CG_A3_SCATTER_UB_LIMIT = 184 * 1024;
// blocked-NZ tile edge of the cast/gemm dataflow
constexpr int64_t CG_A3_NZ_TILE = 128;

class CompressorGradTilingA3 {
public:
    explicit CompressorGradTilingA3(CompressorGradContext* context)
        : context_(context)
    {}
    ~CompressorGradTilingA3() = default;

    static ge::graphStatus ConvertContext(gert::TilingContext& context, CompressorGradContext& compressorGradContext);
    ge::graphStatus RunBigKernelTiling(CompressorGradTilingDataA3* tilingData);

private:
    static ge::graphStatus ConvertRequiredParams(gert::TilingContext& context,
                                                 CompressorGradContext& compressorGradContext);
    static void ConvertOptionalParams(gert::TilingContext& context, CompressorGradContext& compressorGradContext);
    ge::graphStatus GetNpuInfo();
    ge::graphStatus SetBaseInfo();
    ge::graphStatus SetSeqStartFlags();
    ge::graphStatus CalcSchedule();
    ge::graphStatus CalcWorkSpace();
    ge::graphStatus SetTilingData();
    ge::graphStatus GenTilingKey() const;
    ge::graphStatus CheckSinglePara() const;
    ge::graphStatus CheckLayoutSupport() const;
    ge::graphStatus CheckDimNumInLayoutSupport(const std::string& layout, const gert::StorageShape* shape,
                                               const std::string& name) const;
    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc* desc, const std::string& name) const;
    ge::graphStatus CheckDimNumSupport(const gert::StorageShape* shape, const std::string& name) const;
    ge::graphStatus CheckSingleParaX() const;
    ge::graphStatus CheckSingleParaWkv() const;
    ge::graphStatus CheckSingleParaWgate() const;
    ge::graphStatus CheckSingleParaDCmpKv() const;
    ge::graphStatus CheckSingleParaSoftmaxScore() const;
    ge::graphStatus CheckSingleParaKV() const;
    ge::graphStatus CheckSingleParaCuSeqlens() const;
    ge::graphStatus CheckSingleParaSeqused() const;
    ge::graphStatus CheckSingleParaStartPos() const;
    ge::graphStatus CheckSingleParaDX() const;
    ge::graphStatus CheckSingleParaDWkv() const;
    ge::graphStatus CheckSingleParaDWgate() const;
    ge::graphStatus CheckSingleParaDApe() const;
    ge::graphStatus CheckSingleParaCmpRatio() const;
    ge::graphStatus CheckSingleParaCoff() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckFeature() const;
    ge::graphStatus CheckShapeConsistency() const;
    ge::graphStatus CheckDtypeConsistencyX(const gert::CompileTimeTensorDesc* desc, const std::string& name) const;
    ge::graphStatus CheckDtypeConsistency() const;
    ge::graphStatus CheckMultiParaConsistency() const;
    ge::graphStatus CheckDimNumConsistency() const;
    ge::graphStatus CheckEmptyTensor() const;
    ge::graphStatus CheckBlockDimConstrain() const;
    ge::graphStatus CheckA3Constraints() const;

    size_t ubSize_ = 0;
    size_t l1Size_ = 0;
    size_t l0cSize_ = 0;
    size_t l0bSize_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    size_t libapiSize_ = 0;
    size_t workspaceSize_ = 0;
    uint8_t coff_ = 1;
    // arch22 derived shape/schedule info
    int64_t tPad_ = 0;
    int64_t nblk_ = 0;
    int64_t nblkIn_ = 0;
    int64_t seqUsedValid_ = 0;  // seqused 值可读（张量存在且 host 数据已喂入）
    int64_t startPosValid_ = 0; // start_pos 值可读
    int64_t dnPlanes_ = 1;
    int64_t segLen_ = 64;
    int64_t castCoreNum_ = 0;
    int64_t scatterCoreNum_ = 0;
    int64_t blocksPerCore_ = 0;
    int64_t gemmDxCores_ = 0;
    int64_t xF16WsOff_ = 0;
    int64_t wkvF16WsOff_ = 0;
    int64_t wgateF16WsOff_ = 0;
    int64_t dnKvWsOff_ = 0;
    int64_t dnScWsOff_ = 0;
    int64_t apePartWsOff_ = 0;
    int64_t apePartElems_ = 0;

    CompressorGradBaseParams baseParams_;
    CompressorGradContext* context_ = nullptr;
    CompressorGradTilingDataA3* tilingData_ = nullptr;
};

} // namespace

CMP_GRAD_EXTERN_C ge::graphStatus TilingCompressorGradArch22(gert::TilingContext* context);
} // namespace optiling

#endif // COMPRESSOR_GRAD_TILING_ARCH22_H
