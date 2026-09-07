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
 * \file fp_matmul_allto_all_tiling_910_93.cpp
 * \brief
 */
#include "fp_matmul_allto_all_tiling_910_93.h"
#include <string>
#include <vector>
#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "platform/platform_infos_def.h"
#include "hccl/hccl_types.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;
using namespace matmul_allto_all_910b_tiling_key;

namespace {
constexpr uint32_t ATTR_GROUP_INDEX = 0;
}

namespace MC2Tiling {

/**
 * @brief 当前非量化过程的准入条件
 *
 * @return true
 */
bool FpMatmulAllToAllTilingBaseA3::IsCapable()
{
    fe::PlatFormInfos *platformInfoPtr = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfoPtr == nullptr, OP_LOGE(opName_, "fail to get platform info"), return ge::GRAPH_FAILED);
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersionStr_);
    OP_LOGD(opName_, "Current SocVersion is : %s", socVersionStr_.c_str());
    QuantMode mode = MatmulAlltoAllTilingUtil::GetQuantMode(context_, opName_);
    if ((mode == QuantMode::NON_QUANT) && (socVersionStr_ == "Ascend910_93")) {
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
ge::graphStatus FpMatmulAllToAllTilingBaseA3::CheckA3NonQuantTensorDataType(const gert::TilingContext *context,
                                                                            const char *opName)
{
    // 获取并校验输入张量描述符
    auto x1TensorDesc = context->GetInputDesc(INPUT_X1_INDEX);
    OP_TILING_CHECK((x1TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1"), return ge::GRAPH_FAILED);
    auto x2TensorDesc = context->GetInputDesc(INPUT_X2_INDEX);
    OP_TILING_CHECK((x2TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2"), return ge::GRAPH_FAILED);
    ge::DataType x1Dtype = x1TensorDesc->GetDataType();
    ge::DataType x2Dtype = x2TensorDesc->GetDataType();
    OP_TILING_CHECK((x1Dtype != x2Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName, "x1,x2", (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype)).c_str(),
                        "The dtype of x1 and x2 must be the same"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(NON_QUANT_X_DTYPE_LIST, x1Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName, "x1,x2", (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype)).c_str(),
                        "The dtype of x1 and x2 must be in the non-quant range (float16/bf16)"),
                    return ge::GRAPH_FAILED);
    // 校验 bias 数据类型（如果存在
    auto biasTensorDesc = context->GetOptionalInputDesc(INPUT_BIAS_INDEX);
    if (biasTensorDesc != nullptr) {
        ge::DataType biasDtype = biasTensorDesc->GetDataType();
        if (x1Dtype == ge::DT_BF16) {
            OP_TILING_CHECK(
                (biasDtype != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "bias", Ops::Base::ToString(biasDtype).c_str(),
                                                      "The dtype of bias must be FLOAT32 when x1 is BF16"),
                return ge::GRAPH_FAILED);
        } else if (x1Dtype == ge::DT_FLOAT16) {
            OP_TILING_CHECK((x1Dtype != biasDtype),
                            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                                opName, "bias", Ops::Base::ToString(biasDtype).c_str(),
                                "The dtype of bias must be the same as that of x1 when x1 is FLOAT16"),
                            return ge::GRAPH_FAILED);
        } else {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "bias", Ops::Base::ToString(biasDtype).c_str(),
                                                  "The dtype of bias must be FLOAT16 or BF16 in non-quantized scene");
            return ge::GRAPH_FAILED;
        }
    }
    // 校验 scale 张量为空（非量化场景
    auto x1ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
    auto x2ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK(
        (x1ScaleTensorDesc != nullptr || x2ScaleTensorDesc != nullptr),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "x1Scale/x2Scale", "non-null",
                                              "The value of x1Scale and x2Scale must be nullptr in non-quant mode"),
        return ge::GRAPH_FAILED);
    // 校验输出张量数据类型
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "y"), return ge::GRAPH_FAILED);
    ge::DataType yDtype = yDesc->GetDataType();
    OP_TILING_CHECK((yDtype != x1Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "y", Ops::Base::ToString(yDtype).c_str(),
                                                          "The dtype of y must be the same as that of x"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验输入信息是否合规:attr,Dtype,shape等，使用通用校验util中的check方法
 *
 * @return ge::graphStatus
 */
ge::graphStatus FpMatmulAllToAllTilingBaseA3::CheckOpInputInfo()
{
    OP_TILING_CHECK(
        MatmulAlltoAllTilingUtil::CheckAttrsInfo(context_, opName_, MATMUL_ALLTOALL_INDEX_SCHEMA) != ge::GRAPH_SUCCESS,
        OP_LOGE(opName_, "Tiling check Attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckTensorFormat(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check format failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckA3NonQuantTensorDataType(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Dtype failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        MatmulAlltoAllTilingUtil::CheckShapeInfo(context_, opName_, MATMUL_ALLTOALL_INDEX_SCHEMA) != ge::GRAPH_SUCCESS,
        OP_LOGE(opName_, "Tiling check shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(Check2DMatrixMulShapes(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckBsRankSizeRange() != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check BS*rankSize range failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FpMatmulAllToAllTilingBaseA3::CheckBsRankSizeRange()
{
    const gert::StorageShape *x1Shape = context_->GetInputShape(INPUT_X1_INDEX);
    uint64_t x1Dim0 = x1Shape->GetStorageShape().GetDim(0);
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    const int64_t *worldSize = attrs->GetAttrPointer<int64_t>(ATTR_WORLD_SIZE_INDEX);
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    int64_t rankSize = 0;
    if (worldSize == nullptr || *worldSize == RANK_DEFAULT_NUM) {
        OP_TILING_CHECK(!mc2tiling::GetRankSize(opName_, group, rankSize), OP_LOGE(opName_, "GetRankSize failed."),
                        return ge::GRAPH_FAILED);
    } else {
        rankSize = *worldSize;
    }
    uint64_t bsRankProduct = x1Dim0 * static_cast<uint64_t>(rankSize);
    OP_TILING_CHECK(bsRankProduct > MAX_INT32_VALUE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x1Dim0", std::to_string(bsRankProduct).c_str(),
                                                          "BS*rankSize must not exceed INT32_MAX"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 根据输入设置tiling参数
 *
 * @return ge::graphStatus
 */
ge::graphStatus FpMatmulAllToAllTilingBaseA3::InitTilingContextParameters()
{
    MC2_CHECK_LOG_RET(
        opName_, MatmulAlltoAllTilingUtil::SetAttrsInfo(context_, opName_, contextInfo, MATMUL_ALLTOALL_INDEX_SCHEMA));
    MC2_CHECK_LOG_RET(opName_, MatmulAlltoAllTilingUtil::SetDataTypeInfo(context_, opName_, contextInfo));
    MC2_CHECK_LOG_RET(opName_, MatmulAlltoAllTilingUtil::SetShapeInfo(context_, contextInfo));
    contextInfo.quantMode = QuantMode::NON_QUANT; // 在isCapable判断过，直接赋值即可
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hccl参数；进行通算切分, 获取mm tiling等
 *
 * @return ge::graphStatus
 */
ge::graphStatus FpMatmulAllToAllTilingBaseA3::DoOpTiling()
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
 * @brief 进行通算切分之后单个块的MM TILING
 *
 * @return ge::graphStatus
 */
ge::graphStatus FpMatmulAllToAllTilingBaseA3::DoMMTiling()
{
    contextInfo.args_.mValue = inferredInfo.tileM;
    FpMatmulAllToAllHelper mmTile(*this, localTilingData_.mc2MmV3TileTilingData);
    MC2_CHECK_LOG_RET(opName_, mmTile.DoTiling());
    if (inferredInfo.tailCnt == 0) {
        return ge::GRAPH_SUCCESS;
    }
    contextInfo.args_.mValue = inferredInfo.tailM;
    FpMatmulAllToAllHelper mmTail(*this, localTilingData_.mc2MmV3TailTilingData);
    auto res = mmTail.DoTiling();
    return res;
}

/**
 * @brief 设置hccl的config,进行hccl对应的通信任务设置
 *
 * @return ge::graphStatus
 */
ge::graphStatus FpMatmulAllToAllTilingBaseA3::SetHcclTiling()
{
    OP_TILING_CHECK(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geCType) ==
                        mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
                    OP_LOGE(opName_, "Cannot find HcclDataType according to ge datatype = %d.",
                            static_cast<int32_t>(contextInfo.args_.geCType)),
                    return ge::GRAPH_FAILED;);
    auto group = context_->GetAttrs()->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(group == nullptr, OP_LOGE(context_->GetNodeName(), "GetAttrPointer for ATTR_GROUP_INDEX failed"),
                    return ge::GRAPH_FAILED);
    auto commMode = context_->GetAttrs()->GetAttrPointer<char>(ATTR_COMM_MODE_INDEX);
    OP_TILING_CHECK(commMode == nullptr,
                    OP_LOGE(context_->GetNodeName(), "GetAttrPointer for ATTR_COMM_MODE_INDEX failed"),
                    return ge::GRAPH_FAILED);
    const size_t maxLength = 7UL;
    OP_TILING_CHECK((strncmp(commMode, "ai_cpu", maxLength) != 0),
                    OP_LOGE(context_->GetNodeName(), "comm_mode only support 'ai_cpu'."), return ge::GRAPH_FAILED);
    uint32_t optype = HcclCMDType::HCCL_CMD_ALLTOALL;
    std::string algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    OP_LOGD(context_->GetNodeName(), "AllToAllFpMatmulTilingBaseA3, SetHcclTiling algConfig is: %s", algConfig.c_str());
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, optype, algConfig);

    OP_TILING_CHECK(mc2CcTilingConfig.SetSkipBufferWindowCopy(
                        static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_DEFAULT)) != 0,
                    OP_LOGE(context_->GetNodeName(), "mc2CcTilingConfig setSkipBufferWindowCopy failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(localTilingData_.mc2InitTiling) != 0,
                    OP_LOGE(context_->GetNodeName(), "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(localTilingData_.mc2CcTiling) != 0,
                    OP_LOGE(context_->GetNodeName(), "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 打印matmul tiling的信息,注：当前蓝区冒烟找不到mc2_log.h的对应方法，暂时自己实现
 *
 * @param opName
 * @param tiling
 */
void FpMatmulAllToAllTilingBaseA3::PrintMMV3TilingData(const std::string &opName, Mc2MatmulV3TilingData &tiling)
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

/**
 * @brief 打印tilingInfo信息
 *
 * @param opName
 * @param tilingInfo
 */
void FpMatmulAllToAllTilingBaseA3::PrintMatmulAlltoAllTilingInfo(const std::string &opName,
                                                                 MatmulAlltoAllTilingInfoA3 &tilingInfo)
{
    OP_LOGD(opName, "tilingInfo.rankDim: %u", tilingInfo.rankDim);
    OP_LOGD(opName, "tilingInfo.tileM: %u", tilingInfo.tileM);
    OP_LOGD(opName, "tilingInfo.tileCnt: %u", tilingInfo.tileCnt);
    OP_LOGD(opName, "tilingInfo.tailM: %u", tilingInfo.tailM);
    OP_LOGD(opName, "tilingInfo.tailCnt: %u", tilingInfo.tailCnt);
    OP_LOGD(opName, "tilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "tilingInfo.rankM: %u", tilingInfo.rankM);
    OP_LOGD(opName, "tilingInfo.rankN: %u", tilingInfo.rankN);
    OP_LOGD(opName, "tilingInfo.rankK: %u", tilingInfo.rankK);
    OP_LOGD(opName, "tilingInfo.mmResultLen: %u", tilingInfo.mmResultLen);
    OP_LOGD(opName, "tilingInfo.permuteLen: %u", tilingInfo.permuteLen);
    OP_LOGD(opName, "tilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "tilingInfo.aicCoreNum: %u", tilingInfo.aicCoreNum);
    OP_LOGD(opName, "tilingInfo.hcclDataType: %u", tilingInfo.hcclDataType);
}

/**
 * @brief 打印传递给kernel的tilingData
 *
 * @param matmulAlltoAllTilingDataA3 tilingData参数
 */
void FpMatmulAllToAllTilingBaseA3::PrintMatmulAlltoAllTilingData(MatmulAlltoAllTilingDataA3 &matmulAlltoAllTilingDataA3)
{
    PrintMatmulAlltoAllTilingInfo(opName_, matmulAlltoAllTilingDataA3.matmulAlltoAllTilingInfo);
    PrintMMV3TilingData(opName_, matmulAlltoAllTilingDataA3.mc2MmV3TileTilingData);
    if (matmulAlltoAllTilingDataA3.matmulAlltoAllTilingInfo.tailCnt == 0) {
        return;
    }
    OP_LOGD(opName_, "Matmulalltoall has tail");
    PrintMMV3TilingData(opName_, matmulAlltoAllTilingDataA3.mc2MmV3TailTilingData);
}

/**
 * @brief 获取对应的tilingKey
 * 使用QUANT_MODE来区分tilingKey,此处的QUANT_MODE指的是x1,x2的QUANT模式组合，以x1为pertoken量化(K)，x2为perchannel量化(C)
 * 为例子，K-C量化就代表一种组合
 *
 * @return uint64_t tilingKey结果
 */
uint64_t FpMatmulAllToAllTilingBaseA3::GetTilingKey() const
{
    // 按照量化组合模式，是否转置，bias数据类型进行展开
    bool x2TransposeFlag = contextInfo.args_.isBTrans ? true : false;
    // 0代表数据类型为FP16，1代表FP32
    uint32_t biasDType = TILINGKEY_TPL_FP16;
    if (contextInfo.args_.geBiasType != contextInfo.args_.geAType) {
        biasDType = TILINGKEY_TPL_FP32;
    }
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(x2TransposeFlag, contextInfo.args_.isBias, biasDType, SOC_ASCEND910_93);
    OP_LOGD(opName_, "x2TransposeFlag,hasBias,biasDtype is: [%d,%d,%d], and tilingKey is [%lu].", x2TransposeFlag,
            contextInfo.args_.isBias, biasDType, tilingKey);
    return tilingKey;
}

/**
 * @brief 设置额外需要的空间，包括计算结果地址，重排地址，偏移地址等
 *
 */
void FpMatmulAllToAllTilingBaseA3::SetUserWorkSpace()
{
    constexpr uint64_t alignAddrLen = 512;
    // MatmulAlltoAll先进行计算，需要有对应的空间先存放结果，假设x1(m,k),
    // x2(k,n),那么计算结果大小为m*n,这里申请的是一块总的空间，通算切分的头尾块偏移由kernel侧自行计算
    inferredInfo.mmResultLen = mc2tiling::AlignUp(
        contextInfo.args_.orgMValue * contextInfo.args_.nValue * contextInfo.args_.outputDtypeSize, alignAddrLen);
    // 重排空间等于mm计算结果空间
    inferredInfo.permuteLen = inferredInfo.mmResultLen;
    if (contextInfo.args_.isBias) {
        inferredInfo.biasLen =
            mc2tiling::AlignUp(contextInfo.args_.nValue, mc2tiling::SHAPE_ALIGN_SIZE) * sizeof(float);
    }
}

/**
 * @brief 获取额外申请的空间
 *
 * @return ge::graphStatus
 */
ge::graphStatus FpMatmulAllToAllTilingBaseA3::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "Get workspace failed"), return ge::GRAPH_FAILED);
    SetUserWorkSpace();
    uint64_t workspaceSize_ =
        libApiWorkSpaceSize_ + inferredInfo.mmResultLen + inferredInfo.permuteLen + inferredInfo.biasLen;
    workspaces[0] = workspaceSize_;
    OP_LOGD(opName_, "Workspaces[0] size=%ld, biasLen=%d, mmResultLen=%d", workspaces[0], inferredInfo.biasLen,
            inferredInfo.mmResultLen);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 保存tiling数据到context
 *
 * @return ge::graphStatus
 */
ge::graphStatus FpMatmulAllToAllTilingBaseA3::PostTiling()
{
    SetTilingInfo(localTilingData_.matmulAlltoAllTilingInfo);
    MatmulAlltoAllTilingDataA3 *outTilingData = context_->GetTilingData<MatmulAlltoAllTilingDataA3>();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "Failed to get tiling data from context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tilingBufCap < sizeof(localTilingData_)),
                    OP_LOGE(opName_, "TilingBuffer capacity too small, capacity = %zu, need = %zu.", tilingBufCap,
                            sizeof(localTilingData_)),
                    return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(outTilingData, tilingBufCap, &localTilingData_, sizeof(localTilingData_));
    if (ret != EOK) {
        OP_LOGE(opName_, "MatmulAlltoAll postTiling: memcpy_s tiling data failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.", sizeof(MatmulAlltoAllTilingDataA3),
            context_->GetRawTilingData()->GetCapacity());

    context_->GetRawTilingData()->SetDataSize(sizeof(MatmulAlltoAllTilingDataA3));
    context_->SetBlockDim(contextInfo.args_.aicCoreNum);
    PrintMatmulAlltoAllTilingData(*outTilingData);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tilingInfo结构体
 *
 * @param tilingInfo 目标结构体
 */
void FpMatmulAllToAllTilingBaseA3::SetTilingInfo(MatmulAlltoAllTilingInfoA3 &tilingInfo) const
{
    // 基本字段拷贝
    tilingInfo.tileM = inferredInfo.tileM;
    tilingInfo.tileCnt = inferredInfo.tileCnt;
    tilingInfo.tailM = inferredInfo.tailM;
    tilingInfo.tailCnt = inferredInfo.tailCnt;
    tilingInfo.rankM = contextInfo.args_.orgMValue;
    tilingInfo.rankN = contextInfo.args_.nValue;
    tilingInfo.rankK = contextInfo.args_.orgKValue;
    tilingInfo.mmResultLen = inferredInfo.mmResultLen;
    tilingInfo.permuteLen = inferredInfo.permuteLen;
    tilingInfo.biasLen = inferredInfo.biasLen;
    tilingInfo.aicCoreNum = contextInfo.args_.aicCoreNum;
    tilingInfo.rankDim = contextInfo.args_.rankDim;
    tilingInfo.hcclDataType =
        (static_cast<uint64_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geAType))); // hccl数据类型
}

/**
 * @brief 构造函数，创建一个FpMatmulAllToAllTilingBase对象
 *
 * @param context
 */
FpMatmulAllToAllTilingBaseA3::FpMatmulAllToAllTilingBaseA3(gert::TilingContext *context)
    : MatmulAllToAllTilingBase(context)
{}

ge::graphStatus FpMatmulAllToAllHelper::GetShapeAttrsInfo()
{
    auto &&tilingArgs = tilingProcesser_.contextInfo.args_;
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

FpMatmulAllToAllHelper::FpMatmulAllToAllHelper(FpMatmulAllToAllTilingBaseA3 &matmulAlltoAllTilingA3,
                                               Mc2MatmulV3TilingData &data)
    : Mc2MatmulV3BaseTiling(matmulAlltoAllTilingA3.context_, &data),
      tilingProcesser_(matmulAlltoAllTilingA3)
{}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAlltoAll, FpMatmulAllToAllTilingBaseA3,
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910B), 1);
} // namespace MC2Tiling
