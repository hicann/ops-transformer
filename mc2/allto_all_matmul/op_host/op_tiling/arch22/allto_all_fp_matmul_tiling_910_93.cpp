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
 * \file allto_all_fp_matmul_tiling_910_93.cpp
 * \brief
 */
#include "allto_all_fp_matmul_tiling_910_93.h"

#include <string>
#include <vector>

#include "common/utils/op_mc2.h"
#include "hccl/hccl_types.h"
#include "mc2_log.h"
#include "platform/platform_infos_def.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace MC2Tiling {

/**
 * @brief AlltoAllMatmul非量化的准入条件，当前仅实现非量化，所以直接return true
 * 后续支持更多量化再进行修改
 *
 * @return true
 */
bool AllToAllFpMatmulTilingBaseA3::IsCapable()
{
    fe::PlatFormInfos *platformInfoPtr = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfoPtr == nullptr, OP_LOGE(opName_, "fail to get platform info"), return false);
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersionStr_);
    OP_LOGD(opName_, "Current SocVersion is : %s", socVersionStr_.c_str());
    if (socVersionStr_ == "Ascend910_93") {
        OP_LOGI(opName_, "Start with A3 FpMatmulAllToAll tiling.");
        return true;
    }
    OP_LOGI(opName_, "Skip FpMatmulAllToAll tiling when it is not NON_QUANT or the SocVersion is unsupported.");
    return false;
}

/**
 * @brief 工具函数：判断指定value是否存在于list中
 *
 * @param list: 有效值列表
 * @param value: 给定值
 * @return
 */
static bool IsContains(const std::vector<uint32_t> &list, uint32_t value)
{
    return std::count(list.begin(), list.end(), value) > 0;
}

/**
 * @brief 校验输入Dtype信息是否合规
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllFpMatmulTilingBaseA3::CheckA3NonQuantTensorDataType(const gert::TilingContext *context,
                                                                            const char *opName)
{
    // 获取并校验输入张量描述符
    auto x1TensorDesc = context->GetInputDesc(INPUT_X1_INDEX);
    OP_TILING_CHECK((x1TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1"), return ge::GRAPH_FAILED);
    auto x2TensorDesc = context->GetInputDesc(INPUT_X2_INDEX);
    OP_TILING_CHECK((x2TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2"), return ge::GRAPH_FAILED);
    // 获取数据类型并校验一致性与范围
    ge::DataType x1Dtype = x1TensorDesc->GetDataType();
    ge::DataType x2Dtype = x2TensorDesc->GetDataType();
    OP_TILING_CHECK((x1Dtype != x2Dtype),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        opName, "x1 and x2",
                        (Ops::Base::ToString(x1Dtype) + " and " + Ops::Base::ToString(x2Dtype)).c_str(),
                        "The dtypes of x1 and x2 must be the same"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !IsContains(NON_QUANT_X_DTYPE_LIST, x1Dtype),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "x1 and x2", Ops::Base::ToString(x1Dtype).c_str(),
                                              "The dtype of x1 and x2 must be float16 or bf16 in non-quant range"),
        return ge::GRAPH_FAILED);

    // 校验 bias 数据类型（如果存在）
    auto biasTensorDesc = context->GetOptionalInputDesc(INPUT_BIAS_INDEX);
    if (biasTensorDesc != nullptr) {
        ge::DataType biasDtype = biasTensorDesc->GetDataType();
        if (x1Dtype == ge::DT_BF16) {
            OP_TILING_CHECK((biasDtype != ge::DT_FLOAT),
                            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "bias",
                                                                  Ops::Base::ToString(biasDtype).c_str(),
                                                                  "When x1 is BF16, bias dtype must be FLOAT32"),
                            return ge::GRAPH_FAILED);
        } else if (x1Dtype == ge::DT_FLOAT16) {
            OP_TILING_CHECK(
                (x1Dtype != biasDtype),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "bias", Ops::Base::ToString(biasDtype).c_str(),
                                                      "When x1 is FLOAT16, the dtype of bias must be same as x"),
                return ge::GRAPH_FAILED);
        } else {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "bias", Ops::Base::ToString(biasDtype).c_str(),
                                                  "The dtype of bias must be FLOAT16 or BF16 in non-quant mode");
            return ge::GRAPH_FAILED;
        }
    }

    // 校验 scale 张量为空（非量化场景）
    auto x1ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
    auto x2ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK(
        (x1ScaleTensorDesc != nullptr || x2ScaleTensorDesc != nullptr),
        OP_LOGE_FOR_INVALID_VALUE(opName, "x1Scale and x2Scale", "non-null", "should be null in non-quant mode"),
        return ge::GRAPH_FAILED);

    // 校验输出张量数据类型
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "y"), return ge::GRAPH_FAILED);
    ge::DataType yDtype = yDesc->GetDataType();
    OP_TILING_CHECK((yDtype != x1Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "y", Ops::Base::ToString(yDtype).c_str(),
                                                          "The dtype of y must be same as input x"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验输入信息是否合规:attr,Dtype,shape等，使用通用校验util中的check方法
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllFpMatmulTilingBaseA3::CheckOpInputInfo()
{
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckAttrsInfo(context_, opName_, ALLTOALL_MATMUL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckTensorFormat(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check format failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckA3NonQuantTensorDataType(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Dtype failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckShapeInfo(context_, opName_, ALLTOALL_MATMUL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMatrixMulShapes(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check shape input and output shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAlltoAllOut(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check allToAllOut failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 根据输入设置tiling参数
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllFpMatmulTilingBaseA3::InitTilingContextParameters()
{
    MC2_CHECK_LOG_RET(
        opName_, MatmulAlltoAllTilingUtil::SetAttrsInfo(context_, opName_, contextInfo_, ALLTOALL_MATMUL_INDEX_SCHEMA));
    MC2_CHECK_LOG_RET(opName_, MatmulAlltoAllTilingUtil::SetDataTypeInfo(context_, opName_, contextInfo_));
    MC2_CHECK_LOG_RET(opName_, SetAlltoAllMatmulShapeInfo(context_, contextInfo_));
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 主要处理逻辑，设置hccl参数；进行通算切分, 获取mm tiling等
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllFpMatmulTilingBaseA3::DoOpTiling()
{
    // 输入参数的校验:Attrs,Dtype,Shape等
    MC2_CHECK_LOG_RET(opName_, CheckOpInputInfo());
    // 参数校验通过后赋值给全局上下文变量
    MC2_CHECK_LOG_RET(opName_, InitTilingContextParameters());
    // 进行通算切分
    MC2_CHECK_LOG_RET(opName_, TileCommAndCompute());
    // 调用非量化Matmul的tiling方法进行切分
    MC2_CHECK_LOG_RET(opName_, DoMMTiling());
    // hccl的tiling参数赋值处理
    MC2_CHECK_LOG_RET(opName_, SetHcclTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 非量化MatmulTiling进行通算切分之后执行MM Tiling
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllFpMatmulTilingBaseA3::DoMMTiling()
{
    contextInfo_.args_.mValue = inferredInfo_.tileM;
    AllToAllFpMatmulHelper mmTile(*this, localTilingData_.mc2MmV3TileTilingData);
    MC2_CHECK_LOG_RET(opName_, mmTile.DoTiling());
    if (inferredInfo_.tailCnt == 0) {
        return ge::GRAPH_SUCCESS;
    }
    contextInfo_.args_.mValue = inferredInfo_.tailM;
    AllToAllFpMatmulHelper mmTail(*this, localTilingData_.mc2MmV3TailTilingData);
    auto res = mmTail.DoTiling();
    return res;
}

/**
 * @brief 设置hccl的config,进行hccl对应的通信任务设置
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllFpMatmulTilingBaseA3::SetHcclTiling()
{
    OP_TILING_CHECK(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo_.args_.geCType) ==
                        mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
                    OP_LOGE(opName_, "Cannot find HcclDataType according to ge datatype = %d.",
                            static_cast<int32_t>(contextInfo_.args_.geCType)),
                    return ge::GRAPH_FAILED;);
    auto commMode = context_->GetAttrs()->GetAttrPointer<char>(ALLTOALLMATMUL_ATTR_COMM_MODE_INDEX);
    OP_TILING_CHECK(commMode == nullptr,
                    OP_LOGE(context_->GetNodeName(), "GetAttrPointer for ALLTOALLMATMUL_ATTR_COMM_MODE_INDEX failed"),
                    return ge::GRAPH_FAILED);
    const size_t maxLength = 7UL;
    OP_TILING_CHECK((strncmp(commMode, "ai_cpu", maxLength) != 0),
                    OP_LOGE(context_->GetNodeName(), "comm_mode only support 'ai_cpu'."), return ge::GRAPH_FAILED);
    Mc2CcTilingConfigBuilder allToAllBuilder =
        Mc2CcTilingConfigBuilder::create(contextInfo_.group, mc2tiling::AicpuComType::HCCL_CMD_ALLTOALL,
                                         Mc2CcTilingConfigBuilder::AlgConfigType::ALL_TO_ALL);

    // reducetype接口附带的数据类型优先于调用通信接口传入的数据类型，因此这里需要设置
    AscendC::Mc2CcTilingConfig allToAllTilingConfig =
        allToAllBuilder
            .withReduceType(opName_, mc2tiling::HcclReduceOp::HCCL_REDUCE_SUM, contextInfo_.args_.geAType,
                            contextInfo_.args_.geAType)
            .build();
    if (!allToAllBuilder.isSuccess()) {
        return ge::GRAPH_FAILED;
    }
    allToAllTilingConfig.GetTiling(localTilingData_.mc2InitTiling);
    allToAllTilingConfig.GetTiling(localTilingData_.mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取对应的tilingKey
 * 使用QUANT_MODE来区分tilingKey,此处的QUANT_MODE指的也是量化组合，对于非量化的场景，QUANT_MODE=0
 *
 * @return uint64_t tilingKey结果
 */
uint64_t AllToAllFpMatmulTilingBaseA3::GetTilingKey() const
{
    // 按照量化组合模式，是否转置，bias数据类型进行展开
    // 0代表数据类型为FP16，2代表FP32
    uint32_t biasDType = TILINGKEY_TPL_FP16;
    if (contextInfo_.args_.geBiasType != contextInfo_.args_.geAType) {
        biasDType = TILINGKEY_TPL_FP32;
    }
    bool x2TransposeFlag = contextInfo_.args_.isBTrans ? true : false;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(contextInfo_.args_.isBias, x2TransposeFlag, TILINGKEY_TPL_NOQUANT,
                                                  biasDType, SOC_ASCEND910_93);
    OP_LOGD(opName_, "hasBias,x2TransposeFlag,biasDtype is: [%d,%d,%lu], and tilingKey is [%lu].",
            contextInfo_.args_.isBias, x2TransposeFlag, biasDType, tilingKey);
    return tilingKey;
}

/**
 * @brief 保存tiling数据到context,并打印相关信息
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllFpMatmulTilingBaseA3::PostTiling()
{
    SetTilingInfo(localTilingData_.alltoAllMatmulTilingInfo);
    AlltoAllMatmulTilingDataA3 *outTilingData = context_->GetTilingData<AlltoAllMatmulTilingDataA3>();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "Failed to get tiling data from context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tilingBufCap < sizeof(localTilingData_)),
                    OP_LOGE(opName_, "TilingBuffer capacity too small, capacity = %zu, need = %zu.", tilingBufCap,
                            sizeof(localTilingData_)),
                    return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(outTilingData, tilingBufCap, &localTilingData_, sizeof(localTilingData_));
    if (ret != EOK) {
        OP_LOGE(opName_, "PostTiling: memcpy_s tiling data failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.", sizeof(AlltoAllMatmulTilingDataA3),
            context_->GetRawTilingData()->GetCapacity());

    context_->GetRawTilingData()->SetDataSize(sizeof(AlltoAllMatmulTilingDataA3));
    context_->SetBlockDim(contextInfo_.args_.aicCoreNum);
    PrintAlltoAllMatmulTilingData(*outTilingData);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 将runInfo的信息拷贝到tilingInfo结构体
 * @param tilingInfo 目标结构体
 */
void AllToAllFpMatmulTilingBaseA3::SetTilingInfo(AlltoAllMatmulTilingInfoA3 &tilingInfo) const
{
    // 基本字段拷贝
    tilingInfo.tileM = inferredInfo_.tileM;
    tilingInfo.tileCnt = inferredInfo_.tileCnt;
    tilingInfo.tailM = inferredInfo_.tailM;
    tilingInfo.tailCnt = inferredInfo_.tailCnt;
    tilingInfo.rankM = contextInfo_.args_.orgMValue;
    tilingInfo.rankN = contextInfo_.args_.nValue;
    tilingInfo.rankK = contextInfo_.args_.orgKValue;
    tilingInfo.commLen = inferredInfo_.commLen;
    tilingInfo.permuteLen = inferredInfo_.permuteLen;
    tilingInfo.biasLen = inferredInfo_.biasLen;
    tilingInfo.rankDim = contextInfo_.args_.rankDim;
    tilingInfo.hcclDataType = (static_cast<uint64_t>(
        mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo_.args_.geAType))); // hccl数据类型
}

/**
 * @brief 打印传递给kernel的tilingData
 *
 * @param outTilingData tilingData参数
 */
void AllToAllFpMatmulTilingBaseA3::PrintAlltoAllMatmulTilingData(AlltoAllMatmulTilingDataA3 &alltoAllMatmulTilingDataA3)
{
    PrintAlltoAllMatmulTilingInfo(opName_, alltoAllMatmulTilingDataA3.alltoAllMatmulTilingInfo);
    PrintMMV3TilingData(opName_, alltoAllMatmulTilingDataA3.mc2MmV3TileTilingData);
    if (alltoAllMatmulTilingDataA3.alltoAllMatmulTilingInfo.tailCnt == 0) {
        return;
    }
    OP_LOGD(opName_, "Matmulalltoall has tail");
    PrintMMV3TilingData(opName_, alltoAllMatmulTilingDataA3.mc2MmV3TailTilingData);
}

/**
 * @brief 打印tilingInfo信息
 *
 * @param opName
 * @param tilingInfo
 */
void AllToAllFpMatmulTilingBaseA3::PrintAlltoAllMatmulTilingInfo(const std::string &opName,
                                                                 AlltoAllMatmulTilingInfoA3 &tilingInfo)
{
    OP_LOGD(opName, "TilingInfo.rankDim: %u", tilingInfo.rankDim);
    OP_LOGD(opName, "TilingInfo.tileM: %u", tilingInfo.tileM);
    OP_LOGD(opName, "TilingInfo.tileCnt: %u", tilingInfo.tileCnt);
    OP_LOGD(opName, "TilingInfo.tailM: %u", tilingInfo.tailM);
    OP_LOGD(opName, "TilingInfo.tailCnt: %u", tilingInfo.tailCnt);
    OP_LOGD(opName, "TilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "TilingInfo.rankM: %u", tilingInfo.rankM);
    OP_LOGD(opName, "TilingInfo.rankN: %u", tilingInfo.rankN);
    OP_LOGD(opName, "TilingInfo.rankK: %u", tilingInfo.rankK);
    OP_LOGD(opName, "TilingInfo.commLen: %u", tilingInfo.commLen);
    OP_LOGD(opName, "TilingInfo.permuteLen: %u", tilingInfo.permuteLen);
    OP_LOGD(opName, "TilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "TilingInfo.hcclDataType: %u", tilingInfo.hcclDataType);
}

/**
 * @brief 打印matmul tiling的信息,注：当前蓝区冒烟找不到mc2_log.h的对应方法，暂时自己实现
 *
 * @param opName
 * @param tiling
 */
void AllToAllFpMatmulTilingBaseA3::PrintMMV3TilingData(const std::string &opName, Mc2MatmulV3TilingData &tiling)
{
    PrintTCubeTilingData(opName, tiling.matmulTiling);
    OP_LOGD(opName, " MMtiling.tileL2cacheTiling.mTileCntL2 %d", tiling.tileL2cacheTiling.mTileCntL2);
    OP_LOGD(opName, " MMtiling.tileL2cacheTiling.nTileCntL2 %d", tiling.tileL2cacheTiling.nTileCntL2);
    OP_LOGD(opName, " MMtiling.tileL2cacheTiling.mTileBlock %d", tiling.tileL2cacheTiling.mTileBlock);
    OP_LOGD(opName, " MMtiling.tileL2cacheTiling.nTileBlock %d", tiling.tileL2cacheTiling.nTileBlock);
    OP_LOGD(opName, " MMtiling.tileL2cacheTiling.calOrder %d", tiling.tileL2cacheTiling.calOrder);
    OP_LOGD(opName, " MMtiling.matmulRunInfo.isHf32 %d", tiling.matmulRunInfo.isHf32);
    OP_LOGD(opName, " MMtiling.matmulRunInfo.isNzA %d", tiling.matmulRunInfo.isNzA);
    OP_LOGD(opName, " MMtiling.matmulRunInfo.isNzB %d", tiling.matmulRunInfo.isNzB);
    OP_LOGD(opName, " MMtiling.matmulRunInfo.nd2nzA %d", tiling.matmulRunInfo.nd2nzA);
    OP_LOGD(opName, " MMtiling.matmulRunInfo.nd2nzB %d", tiling.matmulRunInfo.nd2nzB);
    OP_LOGD(opName, " MMtiling.matmulRunInfo.transA %d", tiling.matmulRunInfo.transA);
    OP_LOGD(opName, " MMtiling.matmulRunInfo.transB %d", tiling.matmulRunInfo.transB);
    OP_LOGD(opName, " MMtiling.l2cacheUseInfo.l2CacheFlag %d", tiling.l2cacheUseInfo.l2CacheFlag);
    OP_LOGD(opName, " MMtiling.baseAN %d", tiling.baseAN);
    OP_LOGD(opName, " MMtiling.baseAD %d", tiling.baseAD);
    OP_LOGD(opName, " MMtiling.baseBN %d", tiling.baseBN);
    OP_LOGD(opName, " MMtiling.baseBD %d", tiling.baseBD);
}

AllToAllFpMatmulTilingBaseA3::AllToAllFpMatmulTilingBaseA3(gert::TilingContext *context)
    : AllToAllMatmulTilingBase(context)
{
}

ge::graphStatus AllToAllFpMatmulHelper::GetShapeAttrsInfo()
{
    auto &&tilingArgs = tilingProcesser_.contextInfo_.args_;
    args_.opName = tilingProcesser_.opName_;
    args_.isATrans = tilingArgs.isATrans;
    args_.isBTrans = tilingArgs.isBTrans;
    args_.hasBias = tilingArgs.isBias;
    args_.aType = tilingArgs.geAType;
    args_.bType = tilingArgs.geBType;
    args_.cType = tilingArgs.geCType;
    args_.biasType = tilingArgs.isBias ? tilingArgs.geBiasType : ge::DT_INT32;

    args_.aFormat = ge::FORMAT_ND;
    args_.bFormat = ge::FORMAT_ND;
    args_.outFormat = ge::FORMAT_ND;

    args_.mValue = tilingArgs.mValue;
    args_.kValue = tilingArgs.kValue;
    args_.nValue = tilingArgs.nValue;

    return ge::GRAPH_SUCCESS;
}

AllToAllFpMatmulHelper::AllToAllFpMatmulHelper(AllToAllFpMatmulTilingBaseA3 &alltoAllMatmulTilingA3,
                                               Mc2MatmulV3TilingData &data)
    : Mc2MatmulV3BaseTiling(alltoAllMatmulTilingA3.context_, &data),
      tilingProcesser_(alltoAllMatmulTilingA3)
{
}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(AlltoAllMatmul, AllToAllFpMatmulTilingBaseA3,
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910B), 1);
} // namespace MC2Tiling
