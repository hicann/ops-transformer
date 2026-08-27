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
 * \file matmul_all_reduce_tiling_950.cc
 * \brief
 */
#include "matmul_all_reduce_tiling_950.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "common/utils/op_mc2.h"
#include "mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_apt_tiling_key.h"
#include "all_reduce_fit_balance_tiling.h"
#include "mc2_comm_utils.h"

using namespace Mc2Tiling;
namespace optiling {
constexpr uint64_t STANDARD_CARD_CGMPAD_WORKSPACE_CNT = 3;
constexpr uint64_t FIVE_ONE_TWO = 512;
bool MatmulAllReduceTilingA5::IsCapable()
{
    OP_LOGI(opName_, "Start with MatmulAllReduceTilingA5 tiling.");
    return true;
}

ge::graphStatus MatmulAllReduceTilingA5::SetMc2HcommAllReduce(const char *groupName, const uint32_t reduceType)
{
    OP_TILING_CHECK(
        mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType) == mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "y", Ops::Base::ToString(args_.geCType).c_str(),
                                  "FLOAT16, BF16, FLOAT or INT8"),
        return ge::GRAPH_FAILED);
    const uint32_t opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
    const uint8_t dataType = static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType));
    OP_TILING_CHECK(context_->GetAttrs() == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "comm_mode"),
                    return ge::GRAPH_FAILED);
    const std::string algConfig = "AllReduce=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType, dataType, dataType);
    uint8_t commMode = 0;
    OP_TILING_CHECK(GetAndConvertCommMode(commMode), OP_LOGE(opName_, "Get commMode failed."), return ge::GRAPH_FAILED);
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        mc2CcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_AICPU);
    } else if (commMode == Mc2Comm::COMM_MODE_CCU) {
        mc2CcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_CCU_SCHED);
    }
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(matmulAllReduce910TilingData_.mc2InitTiling),
                    OP_LOGE(opName_, "Get mc2InitTiling from matmulAllReduce910TilingData failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(matmulAllReduce910TilingData_.mc2CcTiling),
                    OP_LOGE(opName_, "Get mc2CcTiling from matmulAllReduce910TilingData failed."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingA5::SetMc2HcommTwoShot(const char *groupName, const uint32_t reduceType)
{
    uint32_t opType1 = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLTOALL);
    uint32_t opType2 = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    uint8_t dataType = static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType));
    const std::string algConfig1 = "AlltoAll=level0:fullmesh";
    const std::string algConfig2 = "AllGather=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType1, algConfig1, reduceType, dataType, dataType);
    uint8_t commMode = 0;
    OP_TILING_CHECK(GetAndConvertCommMode(commMode), OP_LOGE(opName_, "Get commMode failed."), return ge::GRAPH_FAILED);
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        mc2CcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_AICPU);
    } else if (commMode == Mc2Comm::COMM_MODE_CCU) {
        mc2CcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_CCU_SCHED);
    }
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(matmulAllReduce910TilingData_.mc2InitTiling),
                    OP_LOGE(opName_, "Get mc2InitTiling from MatmulAllReduce910TilingDataA5 failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(matmulAllReduce910TilingData_.mc2CcTiling),
                    OP_LOGE(opName_, "Get mc2CcTiling from MatmulAllReduce910TilingDataA5 failed."),
                    return ge::GRAPH_FAILED);
    mc2CcTilingConfig.SetGroupName(groupName);
    mc2CcTilingConfig.SetOpType(opType2);
    mc2CcTilingConfig.SetAlgConfig(algConfig2);
    mc2CcTilingConfig.SetReduceType(reduceType, dataType, dataType);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(matmulAllReduce910TilingData_.mc2CcTilingComm),
                    OP_LOGE(opName_, "Get mc2CcTilingComm from MatmulAllReduce910TilingDataA5 failed."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingA5::SetMc2Hcomm()
{
    const char *groupName = context_->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    bool isUseA2APath = mc2tiling::IsUseA2APath(args_.rankDim, npuArch_);
    const uint32_t reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
    if (isUseA2APath) {
        OP_TILING_CHECK(SetMc2HcommTwoShot(groupName, reduceType) != ge::GRAPH_SUCCESS,
                        OP_LOGE(opName_, "MatmulAllReduceTilingA5 set Mc2Hcomm config By SetMc2HcommTwoShot failed."),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(SetMc2HcommAllReduce(groupName, reduceType) != ge::GRAPH_SUCCESS,
                        OP_LOGE(opName_, "MatmulAllReduceTilingA5 set Mc2Hcomm config By SetMc2HcommAllReduce failed."),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

void MatmulAllReduceTilingA5::DoEmptyTensorTiling()
{
    MutableTCubeTileTilingData().M = args_.orgMValue;
    MutableTCubeTileTilingData().N = args_.orgNValue;
    MutableTCubeTileTilingData().isBias = args_.isBias;
    MutableTCubeTileTilingData().usedCoreNum = 1;
}

ge::graphStatus MatmulAllReduceTilingA5::DoOpTiling()
{
    MC2_CHECK_LOG_RET(opName_, CheckA16W16());
    MC2_CHECK_LOG_RET(opName_, CheckInput());
    MC2_CHECK_LOG_RET(opName_, SetMc2Hcomm());
    biasOnVec_ = args_.isBias; // arch35 非量化且有 bias 时，bias 移至 vec epilogue
    DoRCSTiling();
    DoSplitMTiling();
    if (!isKZero_) {
        MC2_CHECK_LOG_RET(opName_, Do910Tiling());
    } else {
        DoEmptyTensorTiling();
    }
    DoAllReduceTiling(true);
    // 将 bias 相关字段写入算子 tiling struct（biasUbCnt 在 DoRCSTiling→CalcUbTiling 中计算）
    matmulAllReduce910TilingData_.extTiling.isVecBias = args_.isBias ? 1U : 0U;
    matmulAllReduce910TilingData_.extTiling.biasUbCnt = biasUbCnt_;
    return ge::GRAPH_SUCCESS;
}

uint64_t MatmulAllReduceTilingA5::GetTilingKey() const
{
    if (unlikely(isKZero_)) {
        const uint64_t tilingKey =
            GET_TPL_TILING_KEY(MMTYPE_FP_NULL_TENSOR, false, false, false, SET_NOT_USE_FP_MM_TILING,
                               SET_NOT_USE_QUANT_MM_TILING, SET_NOT_USE_WEIGHT_QUANT_MM_TILING);
        return tilingKey;
    }
    bool matmulWithAdd = true;
    if (!matmulAllReduce910TilingData_.param.isAdd && !args_.isBias) {
        matmulWithAdd = false;
    }
    bool isUseA2APath = mc2tiling::IsUseA2APath(args_.rankDim, npuArch_);
    uint8_t commMode = GetCommMode();
    bool isA2ARSAG = isUseA2APath;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(MMTYPE_FP_MM, false, false, isA2ARSAG, commMode, matmulWithAdd,
                                                  SET_NOT_USE_QUANT_MM_TILING, SET_NOT_USE_WEIGHT_QUANT_MM_TILING);
    OP_LOGD(opName_, "Mc2MatmulAllReduce: matmulWithAdd is:[%d].", matmulWithAdd);
    OP_LOGD(opName_, "Mc2MatmulAllReduce: TilingKey=%lu.", tilingKey);
    return tilingKey;
}

void MatmulAllReduceTilingA5::PrintExtendMatmulTiling(bool isTail)
{
    auto &tiling = matmulAllReduce910TilingData_.mC2Mmv3TailTilingData;
    if (isTail) {
        tiling = matmulAllReduce910TilingData_.mC2Mmv3TailTilingData;
    }

    OP_LOGD(opName_, "Matmul tiling mTailCnt=%u", tiling.mTailCnt);
    OP_LOGD(opName_, "Matmul tiling nTailCnt=%u", tiling.nTailCnt);
    OP_LOGD(opName_, "Matmul tiling kTailCnt=%u", tiling.kTailCnt);
    OP_LOGD(opName_, "Matmul tiling mBaseTailSplitCnt=%u", tiling.mBaseTailSplitCnt);
    OP_LOGD(opName_, "Matmul tiling nBaseTailSplitCnt=%u", tiling.nBaseTailSplitCnt);
    OP_LOGD(opName_, "Matmul tiling mTailMain=%u", tiling.mTailMain);
    OP_LOGD(opName_, "Matmul tiling nTailMain=%u", tiling.nTailMain);
    OP_LOGD(opName_, "Matmul tiling isHf32=%u", tiling.isHf32);
    OP_LOGD(opName_, "Matmul tiling aswWindowLen=%u", tiling.aswWindowLen);
}

ge::graphStatus MatmulAllReduceTilingA5::GetWorkspaceSizeForA2ARSAG()
{
    uint64_t commWorkSpace = 0UL;
    uint64_t cgmPadLen = 0UL;
    uint64_t tempTileSize = MutableTCubeTileTilingData().M * MutableTCubeTileTilingData().N;
    uint64_t tempTailSize = MutableTCubeTailTilingData().M * MutableTCubeTailTilingData().N;

    cgmPadLen = (MutableRCSTilingData().tileCnt + MutableRCSTilingData().tailCnt) * FIVE_ONE_TWO;
    commWorkSpace =
        (tempTileSize * MutableRCSTilingData().tileCnt + tempTailSize * MutableRCSTilingData().tailCnt + cgmPadLen) *
        static_cast<uint64_t>(args_.outputDtypeSize);
    OP_LOGI(opName_, "MatmulAllReduceTilingA5 Set commWorkSpace size=%lu to context.", commWorkSpace);
    // MatMul输出存储+alltoall输出存储+reduceSum输出存储
    uint64_t workspaceSizeCount = STANDARD_CARD_CGMPAD_WORKSPACE_CNT;
    myWorkSpaceSize_ = myWorkSpaceSize_ + commWorkSpace * workspaceSizeCount + commWorkSpace / args_.rankDim;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingA5::GetWorkspaceSize()
{
    MC2_CHECK_LOG_RET(opName_, MatmulAllReduceTilingBase::GetWorkspaceSize());
    OP_LOGI(opName_, "Select max workspace size to context, myWorkSpaceSize_=%lu, workspaceSize_=%lu.",
            myWorkSpaceSize_, workspaceSize_);
    myWorkSpaceSize_ = std::max(myWorkSpaceSize_, workspaceSize_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    if (mc2tiling::IsUseA2APath(args_.rankDim, npuArch_)) {
        GetWorkspaceSizeForA2ARSAG();
    }
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingA5::PostTiling()
{
    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.",
            sizeof(MatmulAllReduce910TilingDataA5), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(sizeof(MatmulAllReduce910TilingDataA5));

    OP_TILING_CHECK(
        (sizeof(MatmulAllReduce910TilingDataA5) % sizeof(uint64_t)) != 0,
        OP_LOGE(opName_, "tiling data size[%zu] is not aligned to 8", sizeof(MatmulAllReduce910TilingDataA5)),
        return ge::GRAPH_FAILED);

    errno_t ret =
        memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                 reinterpret_cast<void *>(&matmulAllReduce910TilingData_), sizeof(MatmulAllReduce910TilingDataA5));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "Memcpy_s failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }

    PrintTilingData();
    context_->SetBlockDim(args_.aicCoreNum);
    // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要设置避免出现网络挂死
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingA5::Do910Tiling()
{
    OP_LOGD(opName_, "Start to execute DoMatmulV3Tiling!");
    // 获取芯片平台信息
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Get platform info failed."), return ge::GRAPH_FAILED);
    // 获取compileInfo
    OP_TILING_CHECK(mc2_matmul_v3_advanced::InitCompileInfo(platformInfo, &compileInfo_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Init compile info failed."), return ge::GRAPH_FAILED);

    // 根据芯片型号获取策略模板
    std::vector<int32_t> priorities;
    OP_TILING_CHECK(mc2tiling::NewGetMatmulV3PriorityPolicy(npuArch_, priorities, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Get mmv3 priority policy failed."), return ge::GRAPH_FAILED);
    Mc2MMRegisterCfg registerCfg{"Mc2MatMulV3", npuArch_, priorities};
    mc2tiling::NewUpdateMatmulV3Args(mmV3Args_, args_, opName_);
    mmV3Args_.hasBias = false; // bias 移至 vec epilogue 累加，关闭 matmul bias（不改第三方代码）

    // 获取tileTiling
    mmV3Args_.mValue = tileMValue_;
    OP_LOGD(opName_, "Do Mc2MatmulV3 tile tiling!");
    Mc2MatmulHelper::Mc2MatmulTilingCfg tileTilingCfg(reinterpret_cast<const void *>(&compileInfo_),
                                                      reinterpret_cast<const void *>(&mmV3Args_));
    MC2_CHECK_LOG_RET(opName_, DoMatmulV3Tiling(tileTilingCfg, registerCfg, MutableMC2MmV3TileTilingData()));
    if (tailMValue_ != 0UL) {
        mmV3Args_.mValue = tailMValue_;
        OP_LOGD(opName_, "Do Mc2MatmulV3 tail tiling!");
        Mc2MatmulHelper::Mc2MatmulTilingCfg tailTilingCfg(reinterpret_cast<const void *>(&compileInfo_),
                                                          reinterpret_cast<const void *>(&mmV3Args_));
        MC2_CHECK_LOG_RET(opName_, DoMatmulV3Tiling(tileTilingCfg, registerCfg, MutableMC2MmV3TailTilingData()));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingA5::DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg &tilingCfg,
                                                          Mc2MMRegisterCfg &registerCfg,
                                                          Mc2MatMulV3TilingData &tilingData)
{
    tilingCfg.SetRankDim(args_.rankDim);
    tilingCfg.SetMatMulV3TilingData(tilingData);
    if (Mc2MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Failed to do MatmulV3Tiling.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

Mc2Tiling::RCSTiling &MatmulAllReduceTilingA5::MutableRCSTilingData()
{
    return matmulAllReduce910TilingData_.param;
}

ge::graphStatus MatmulAllReduceTilingA5::CheckAxisSize()
{
    const uint64_t mValue = MatmulAllReduceTilingBase::GetMValue();
    OP_TILING_CHECK(mValue > static_cast<uint64_t>(INT32_MAX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x1", std::to_string(mValue).c_str(),
                                                          "The value of m of x1 exceeds the upper limit INT32_MAX"),
                    return ge::GRAPH_FAILED);
    const uint64_t kValue = MatmulAllReduceTilingBase::GetKValue();
    OP_TILING_CHECK(kValue > static_cast<uint64_t>(INT32_MAX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x1", std::to_string(kValue).c_str(),
                                                          "The value of k of x1 exceeds the upper limit INT32_MAX"),
                    return ge::GRAPH_FAILED);
    const uint64_t nValue = MatmulAllReduceTilingBase::GetNValue();
    OP_TILING_CHECK(nValue > static_cast<uint64_t>(INT32_MAX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x2", std::to_string(nValue).c_str(),
                                                          "The value of n of x2 exceeds the upper limit INT32_MAX"),
                    return ge::GRAPH_FAILED);

    return CheckEmptyTensor();
}

ge::graphStatus MatmulAllReduceTilingA5::CheckX1X2()
{
    // x2 shape 为 2 维
    size_t x2DimNum = mmrCtxInfo_.x2_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        x2DimNum != DIM_NUM_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "x2", (std::to_string(x2DimNum) + "D").c_str(), "The shape dim of x2 must be 2"),
        return ge::GRAPH_FAILED);
    auto x1Type = mmrCtxInfo_.x1->GetDataType();
    //  x1 为fp16 或者bf16
    OP_TILING_CHECK(!((x1Type == ge::DT_FLOAT16) || (x1Type == ge::DT_BF16)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x1",
                                                          std::to_string(static_cast<int32_t>(x1Type)).c_str(),
                                                          "The value of x1 dtype must be fp16 or bf16"),
                    return ge::GRAPH_FAILED);
    // x1，x2数据类型相同
    auto x2Type = mmrCtxInfo_.x2->GetDataType();
    OP_TILING_CHECK(x1Type != x2Type,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x2",
                                                          std::to_string(static_cast<int32_t>(x2Type)).c_str(),
                                                          "The value of x2 dtype must be the same as that of x1"),
                    return ge::GRAPH_FAILED);
    // x1,bias数据类型相同
    if (mmrCtxInfo_.bias_shape != nullptr) {
        auto biasType = mmrCtxInfo_.bias->GetDataType();
        OP_TILING_CHECK(x1Type != biasType,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "bias",
                                                              std::to_string(static_cast<int32_t>(biasType)).c_str(),
                                                              "The value of bias dtype must be the same as that of x1"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingA5::CheckInput()
{
    MC2_CHECK_LOG_RET(opName_, MatmulAllReduceTilingBase::CheckInput());
    if (CheckX1X2() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto outputDimNum = mmrCtxInfo_.y_shape->GetStorageShape().GetDimNum();
    if (mmrCtxInfo_.x3_shape != nullptr) {
        auto x3DimNum = mmrCtxInfo_.x3_shape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(outputDimNum != x3DimNum,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            context_->GetNodeName(), "x3", (std::to_string(x3DimNum) + "D").c_str(),
                            "The shape dim of x3 must be the same as that of output"),
                        return ge::GRAPH_FAILED);
        for (size_t i = 0U; i < outputDimNum; i++) {
            auto outputDimValue = mmrCtxInfo_.y_shape->GetStorageShape().GetDim(i);
            auto x3DimValue = mmrCtxInfo_.x3_shape->GetStorageShape().GetDim(i);
            OP_TILING_CHECK(
                outputDimValue != x3DimValue,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "x3", Ops::Base::ToString(mmrCtxInfo_.x3_shape->GetStorageShape()).c_str(),
                    "The shape dim of x3 must be the same as that of output"),
                return ge::GRAPH_FAILED);
        }
    }

    return CheckAxisSize();
}

MatmulAllReduceTilingA5::MatmulAllReduceTilingA5(gert::TilingContext *context)
    : MatmulAllReduceTilingBase(context),
      matmulAllReduce910TilingData_(matmulAllReduce910TilingDataSelf_)
{}

MatmulAllReduceTilingA5::MatmulAllReduceTilingA5(gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo,
                                                 MatmulAllReduce910TilingDataA5 *out)
    : MatmulAllReduceTilingBase(context, mmrCtxInfo),
      matmulAllReduce910TilingData_(*out)
{}

CutResult MatmulAllReduceTilingA5::GetTilingResult()
{
    CutResult mCutAllreduce;
    SocVersion inputSocVersion = SocVersion::SOC910_B;
    SetMCutSocVersion(inputSocVersion);
    const gert::StorageShape *commQuantScaleShape1 = mmrCtxInfo_.comm_quant_scale_1_shape;
    const gert::StorageShape *commQuantScaleShape2 = mmrCtxInfo_.comm_quant_scale_2_shape;
    if (mc2tiling::IsStandardCard4P(args_.rankDim, npuArch_)) {
        MMAllReduceFitBalanceTiling allReduceTilingHccl(args_, KernelType::ALL_REDUCE_VIA_TWO_SHOT,
                                                        TopoType::STANDARD_CARD);
        mCutAllreduce = allReduceTilingHccl.GetTiling();
    } else if (mc2tiling::Is8P(args_.rankDim, npuArch_)) {
        MMAllReduceFitBalanceTiling allReduceTilingHccl(args_, KernelType::ALL_REDUCE_VIA_TWO_SHOT, TopoType::EIGHT_P);
        mCutAllreduce = allReduceTilingHccl.GetTiling();
    } else {
        MMPlusAllReduce allReduceTilingHccl(args_, args_.rankDim, KernelType::ALL_REDUCE, inputSocVersion, isPerBlock_);
        allReduceTilingHccl.GetTiling();
        mCutAllreduce = allReduceTilingHccl.tilingM_.cutRes;
    }
    return mCutAllreduce;
}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(MatmulAllReduce, MatmulAllReduceTilingA5, static_cast<int32_t>(NpuArch::DAV_3510),
                                   2);
} // namespace optiling
