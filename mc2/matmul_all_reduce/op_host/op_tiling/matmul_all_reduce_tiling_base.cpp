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
 * \file matmul_all_reduce_tiling_base.cc
 * \brief
 */
#include "matmul_all_reduce_tiling_base.h"
#include <queue>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>

#include "mc2_log.h"
#include "common/utils/op_mc2.h"
#include "all_reduce_formulaic_tiling.h"
#include "util/math_util.h"

#include "op_host/tiling_type.h"
#include "mc2/common/utils/mc2_comm_utils.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Log;
using namespace Mc2Tiling;

namespace optiling {
constexpr char HCCL_BUFFSIZE[] = "HCCL_BUFFSIZE";
// 每卡数据分配几次计算
constexpr uint64_t COMM_TILE = 8;
constexpr int64_t CORE_THRESHOLD = 20;
constexpr uint64_t LENGTH_SUM_THRESHOLD = 8192 * 3;
constexpr uint64_t MAX_SHAPE_DIM = 65535;
constexpr uint64_t MIN_SHAPE_DIM = 1;
constexpr int32_t MAX_M_SHAPE_DIM = 3;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t ADD_X3_FP16_UB_BUF_FACTOR = 6;  // 对应add x3算子中FP16数据的切分数
constexpr uint32_t ADD_X3_BF16_UB_BUF_FACTOR = 10; // 对应add x3算子中BF16数据的切分数
constexpr uint32_t ALIGN_DATA_SIZE = 32;
constexpr uint64_t L2_CACHE_SIZE_910_B4 = 100663296;
constexpr uint32_t COMM_QUANT_MODE_TRUE = 2;
constexpr size_t ATTR_COMM_MODE_INDEX = 9;

struct HcclAicpuOpParam {
    uint8_t res[64];
};

struct KFCMsgBody {
    // Rank* aiv * MsgSize * sizeof(消息)
    HcclAicpuOpParam msgSndArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
    HcclAicpuOpParam msgRcvArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
};

const std::map<ge::DataType, matmul_tiling::DataType> D_TYPE_MAP = {
    {ge::DT_BF16, matmul_tiling::DataType::DT_BFLOAT16},
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
    {ge::DT_INT32, matmul_tiling::DataType::DT_INT32},
    {ge::DT_INT4, matmul_tiling::DataType::DT_INT4},
    {ge::DT_FLOAT8_E4M3FN, matmul_tiling::DataType::DT_FLOAT8_E4M3FN},
    {ge::DT_FLOAT8_E5M2, matmul_tiling::DataType::DT_FLOAT8_E5M2},
    {ge::DT_HIFLOAT8, matmul_tiling::DataType::DT_HIFLOAT8},
    {ge::DT_FLOAT4_E2M1, matmul_tiling::DataType::DT_FLOAT4_E2M1}};

const std::map<ge::DataType, mc2tiling::HcclDataType> HCCL_DATA_TYPE = {
    {ge::DataType::DT_INT8, mc2tiling::HcclDataType::HCCL_DATA_TYPE_INT8},
    {ge::DataType::DT_UINT8, mc2tiling::HcclDataType::HCCL_DATA_TYPE_UINT8},
    {ge::DataType::DT_INT16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_INT16},
    {ge::DataType::DT_UINT16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_UINT16},
    {ge::DataType::DT_INT32, mc2tiling::HcclDataType::HCCL_DATA_TYPE_INT32},
    {ge::DataType::DT_UINT32, mc2tiling::HcclDataType::HCCL_DATA_TYPE_UINT32},
    {ge::DataType::DT_FLOAT16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_FP16},
    {ge::DataType::DT_FLOAT, mc2tiling::HcclDataType::HCCL_DATA_TYPE_FP32},
    {ge::DataType::DT_BF16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_BFP16},
};

ge::graphStatus MatmulAllReduceTilingFunc(gert::TilingContext *context);
ge::graphStatus TilingParseForMatmulAllReduce(gert::TilingParseContext *context);

void MatmulAllReduceTilingBase::Reset()
{
    tileMValue_ = 0;
    tailMValue_ = 0;
    isQuantKey_ = false;
    isPerTensor_ = false;
    antiQuantType_ = AntiQuantType::NONE;
    quantType_ = Mc2QuantType::PER_TENSOR;
    antiGroupSize_ = 0;
    isUbQuant_ = false;
    enableL2Cache_ = false;
    enableBiasConvert_ = false;
    opName_ = nullptr;
    reduceOp_ = nullptr;
    rankSize_ = 8U;
    libApiWorkSpaceSize_ = 0;
    supportL0c2Out_ = false;
    isKZero_ = false;
    isA8W8_ = false;
    isA16W8_ = false;
    isA16W4_ = false;
    isPerBlock_ = false;
    biasOnVec_ = false;
    biasUbCnt_ = 0U;
}

void MatmulAllReduceTilingBase::CalcAllReduceSendRecvParams()
{
    auto &&args = MutableMc2MsgData();
    auto columnNum = args_.orgNValue;
    auto debugMode = mc2tiling::Mc2TilingUtils::GetDebugMode();
    if (debugMode == MC2_DEBUG_ONLY_AICPU && args_.orgKValue < args_.orgNValue) {
        columnNum = args_.orgKValue;
    }
    args.sendOff = MutableTCubeTileTilingData().M * args_.orgNValue * args_.outputDtypeSize;
    args.recvOff = MutableTCubeTileTilingData().M * columnNum * args_.outputDtypeSize;
    args.sendCnt = MutableTCubeTileTilingData().M * args_.orgNValue;
    args.recvCnt = MutableTCubeTileTilingData().M * columnNum;
    args.tailSendOff = MutableTCubeTailTilingData().M * args_.orgNValue * args_.outputDtypeSize;
    args.tailRecvOff = MutableTCubeTailTilingData().M * columnNum * args_.outputDtypeSize;
    args.tailSendCnt = MutableTCubeTailTilingData().M * args_.orgNValue;
    args.tailRecvCnt = MutableTCubeTailTilingData().M * columnNum;
}

void MatmulAllReduceTilingBase::DoAllReduceTiling(bool useHcclApi)
{
    auto &&args = MutableMc2MsgData();
    auto debugMode = mc2tiling::Mc2TilingUtils::GetDebugMode();
    args.debugMode = debugMode;
    args.commType = MutableRCSTilingData().commtype;
    args.reduceOp = MutableRCSTilingData().subtype;

    args.waitPolicy = 1;
    args.rspPolicy = 1;
    args.exitPolicy = 0;
    args.commAlg = 0;
    args.taskType = static_cast<uint8_t>(mc2tiling::KfcTaskType::KFC_TASK_HCC_TASK_DELIVER);

    args.commOrder = 1; // 0先AiCPU后MM;  1为先MM后AICPU
    args.reuseMode = MutableRCSTilingData().tileCnt + MutableRCSTilingData().tailCnt; // 数据空间被使用

    CalcAllReduceSendRecvParams();

    // 总共发送的次数
    args.totalCnt =
        static_cast<uint64_t>(MutableRCSTilingData().rankM) * static_cast<uint64_t>(MutableRCSTilingData().rankN);
    args.turnNum = MutableRCSTilingData().tileCnt + MutableRCSTilingData().tailCnt; // 总轮次
    args.tailNum = MutableRCSTilingData().tailCnt;                                  // 尾块的轮次
    args.stride = 0;                                                                // 跳写间隔

    // workspace 地址
    setUseBufferType();
    args.workspaceOff = libApiWorkSpaceSize_;

    // 消息队列的开始  device notify write/read value偏移
    args.notifyOff = sizeof(KFCMsgBody);
    args.notifyBeginCnt = mc2tiling::NOTIFY_WRITE_CNT; // notify write value的使用个数
    args.notifyEndCnt = 1;                             // notify read value的使用个数

    args.funID = mc2tiling::ALL_REDUCE_FUNC_ID;
    args.dataType = static_cast<uint8_t>(GetDataType(args_.geCType)); // hccl 数据类型
    args.groupNum = 1;
    args.sendArgIndex = 0;
    args.recvArgIndex =
        context_->GetComputeNodeInfo()->GetIrInputsNum() + context_->GetComputeNodeInfo()->GetIrOutputsNum() - 1;
    OP_LOGI(opName_, "IR inputNum: %zu, IR outputNum: %zu.", context_->GetComputeNodeInfo()->GetIrInputsNum(),
            context_->GetComputeNodeInfo()->GetIrOutputsNum());
    if (useHcclApi) {
        args.preparePosition = 1; // 使用HCCLAPI
        args.hasCommOut = 1;
    } else {
        args.preparePosition = 0;
    }
}

void MatmulAllReduceTilingBase::CalcSendOffWithPadM(uint64_t &tileSendOff, uint64_t &tailSendOff)
{
    uint64_t padTileM = MutableTCubeTileTilingData().M;
    uint64_t padTailM = MutableTCubeTailTilingData().M;
    if (padTileM % args_.rankDim != 0) {
        padTileM += args_.rankDim - (padTileM % args_.rankDim); // args_.rankDim :1/2/4/8 不会为0
    }
    tileSendOff = static_cast<uint64_t>(padTileM * MutableTCubeTileTilingData().N * sizeof(uint8_t)) *
                  MutableRCSTilingData().tileCnt;
    if (padTailM % args_.rankDim != 0) {
        padTailM += args_.rankDim - (padTailM % args_.rankDim); // args_.rankDim :1/2/4/8 不会为0
    }
    tailSendOff = static_cast<uint64_t>(padTailM * MutableTCubeTailTilingData().N * sizeof(uint8_t)) *
                  MutableRCSTilingData().tailCnt;
}

uint8_t MatmulAllReduceTilingBase::CalcBufferTypeByWindowSize()
{
    // win区大小为200M
    uint16_t defaultWindowSize = 200;
    if (getenv(HCCL_BUFFSIZE) == nullptr) {
        OP_LOGD(opName_, "Env HCCL_BUFFSIZE don't set.");
    } else {
        try {
            std::string envStr(getenv(HCCL_BUFFSIZE));
            defaultWindowSize = std::stoi(envStr);
        } catch (...) {
            OP_LOGE(opName_, "Unknown Exception encountered when parsing env HCCL_BUFFSIZE.");
        }
    }
    // 1024 * 1024表示1M
    const uint64_t maxWindowSize = static_cast<uint64_t>(defaultWindowSize) * 1024UL * 1024UL;
    uint64_t tileSendOff = static_cast<uint64_t>(MutableMc2MsgData().sendOff) * MutableRCSTilingData().tileCnt;
    uint64_t tailSendOff = static_cast<uint64_t>(MutableMc2MsgData().tailSendOff) * MutableRCSTilingData().tailCnt;
    if (MutableRCSTilingData().isInputCommQuantScale == 1) { // int8低bit通信做alltoall需要pad M使其可以被卡数整除
        CalcSendOffWithPadM(tileSendOff, tailSendOff);
    }
    uint8_t buffer_type;
    if (UINT64_MAX - tileSendOff < tailSendOff || tileSendOff + tailSendOff >= maxWindowSize) {
        buffer_type = static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
    } else {
        buffer_type = static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_WINDOW_IN);
    }
    OP_LOGI(opName_, "Set buffer type to %u, window size %lu/%lu, max %lu.", static_cast<uint32_t>(buffer_type),
            tileSendOff, tailSendOff, maxWindowSize);
    return buffer_type;
}

void MatmulAllReduceTilingBase::setUseBufferType()
{
    uint8_t buffer_type;
    if (npuArch_ == NpuArch::DAV_2002) {
        buffer_type = static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
        OP_LOGI(opName_, "Set buffer type to output for non-910B soc.");
    } else if (MutableMc2MsgData().debugMode == MC2_DEBUG_ONLY_AICPU) {
        buffer_type = static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
        OP_LOGI(opName_, "Set buffer type to output for aicpu debug mode.");
    } else if (MutableMc2MsgData().reuseMode == 0) {
        buffer_type = static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
        OP_LOGI(opName_, "Set buffer type to output for non-reuse mode.");
    } else if (isKZero_) {
        buffer_type = static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
        OP_LOGI(opName_, "Set buffer type to output for empty tensor.");
    } else {
        buffer_type = CalcBufferTypeByWindowSize();
    }
    MutableMc2MsgData().useBufferType = buffer_type;
}

void MatmulAllReduceTilingBase::DoRCSTiling()
{
    MutableRCSTilingData().rankDim = args_.rankDim;
    MutableRCSTilingData().isTransposeA = args_.isATrans;
    MutableRCSTilingData().isTransposeB = args_.isBTrans;
    MutableRCSTilingData().commtype = static_cast<uint32_t>(args_.cmdType);
    if (strncmp(reduceOp_, "sum", 3) == 0) { // 3 is index
        OP_LOGD(opName_, "ReduceOp_ is SUM.");
        MutableRCSTilingData().subtype = static_cast<uint8_t>(mc2tiling::HcclReduceOp::HCCL_REDUCE_SUM);
    } else {
        OP_LOGD(opName_, "ReduceOp_ is RESERVED.");
        MutableRCSTilingData().subtype = static_cast<uint8_t>(mc2tiling::HcclReduceOp::HCCL_REDUCE_RESERVED);
    }
    OP_LOGD(opName_, "MatMulAllReduce DoRCSTiling, args_.orgMValue: %lu, args_.orgNValue: %lu, args_.orgKValue: %lu.",
            args_.orgMValue, args_.orgNValue, args_.orgKValue);
    MutableRCSTilingData().rankM = args_.orgMValue;
    MutableRCSTilingData().rankN = args_.orgNValue;
    MutableRCSTilingData().rankK = args_.orgKValue;
    MutableRCSTilingData().aicCoreNum = args_.aicCoreNum;
    if (MutableRCSTilingData().isAdd || biasOnVec_) {
        CalcUbTiling();
    }
    SetCommQuantScale();
}

void MatmulAllReduceTilingBase::SetMCutSocVersion(SocVersion &inputSocVersion)
{
    if (npuArch_ == NpuArch::DAV_2002) {
        inputSocVersion = SocVersion::SOC310_P;
        OP_LOGD(opName_, "TileCnt enter 310P branch.");
        return;
    }
    // __NPU_ARCH__ == 3510
    if (npuArch_ == NpuArch::DAV_3510) {
        inputSocVersion = SocVersion::SOC950;
        OP_LOGD(opName_, "TileCnt enter 3510 branch.");
        return;
    }
    // end __NPU_ARCH__ == 3510
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t socMemSize = L2_CACHE_SIZE_910_B4;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, socMemSize);
    inputSocVersion = (socMemSize == L2_CACHE_SIZE_910_B4) ? SocVersion::SOC910_B4 : inputSocVersion;
    if (socMemSize == L2_CACHE_SIZE_910_B4) {
        inputSocVersion = SocVersion::SOC910_B4;
        OP_LOGD(opName_, "TileCnt enter 910B4 branch.");
    }
}

CutResult MatmulAllReduceTilingBase::GetTilingResult()
{
    SocVersion inputSocVersion = SocVersion::SOC910_B;
    SetMCutSocVersion(inputSocVersion); // 判断是否是310P或者910B4
    MMPlusAllReduce allReduceTilingHccl(args_, args_.rankDim, KernelType::ALL_REDUCE, inputSocVersion, isPerBlock_);
    allReduceTilingHccl.GetTiling();
    CutResult mCutAllreduce = allReduceTilingHccl.tilingM_.cutRes;
    const gert::StorageShape *commQuantScaleShape1 = mmrCtxInfo_.comm_quant_scale_1_shape;
    const gert::StorageShape *commQuantScaleShape2 = mmrCtxInfo_.comm_quant_scale_2_shape;
    if ((commQuantScaleShape1 != nullptr) && (commQuantScaleShape2 != nullptr)) { // 低bit通信
        OP_LOGD(opName_, "TileCnt enter comm quant.");
        MMPlusQuantAllReduce quantAllReduceTilingHccl(args_, args_.rankDim, KernelType::ALL_REDUCE, inputSocVersion);
        quantAllReduceTilingHccl.GetTiling();
        mCutAllreduce = quantAllReduceTilingHccl.tilingM_.cutRes;
    }

    return mCutAllreduce;
}

void MatmulAllReduceTilingBase::DoSplitMTiling()
{
    auto &&param = MutableRCSTilingData();
    bool isNotBatchOne = (args_.batchValue != 1ULL);
    bool is128Aligned = ((args_.orgMValue / args_.batchValue) & 127ULL) == 0; // 判断原始输入m是否128对齐,batch默认值为1
    if (args_.enableSplitK || isKZero_ || (isPerBlock_ && isNotBatchOne && !is128Aligned) ||
        ((scenario_ == AllReduceScenario::MXFP8) && isNotBatchOne) ||
        ((scenario_ == AllReduceScenario::MXFP4) && isNotBatchOne)) {
        tileMValue_ = args_.orgMValue;
        param.tileCnt = 1;
        param.tailCnt = 0;
        param.tailM = 0;
    } else {
        OP_LOGD(opName_, "Start formulaic tiling.");
        CutResult mCutAllreduce = GetTilingResult();
        if (mCutAllreduce.shortTileAtBack || mCutAllreduce.numShortTile == 0) {
            param.tileCnt = mCutAllreduce.numLongTile;
            param.tailM = mCutAllreduce.shortTileLen;
            tileMValue_ = mCutAllreduce.longTileLen;
            if (mCutAllreduce.numShortTile > 0) { // 有优化空间，不大于零，那就等于零
                tailMValue_ = mCutAllreduce.shortTileLen;
                param.tailCnt = mCutAllreduce.numShortTile;
            } else {
                param.tailCnt = 0;
            }
        } else {
            param.tileCnt = mCutAllreduce.numShortTile;
            param.tailM = mCutAllreduce.longTileLen;
            tileMValue_ = mCutAllreduce.shortTileLen;
            if (mCutAllreduce.numLongTile > 0) {
                tailMValue_ = mCutAllreduce.longTileLen;
                param.tailCnt = mCutAllreduce.numLongTile;
            } else {
                param.tailCnt = 0;
            }
        }
    }
}

void MatmulAllReduceTilingBase::SetCommQuantScale()
{
    bool isInput = false;
    const gert::StorageShape *commQuantScaleShape1 = mmrCtxInfo_.comm_quant_scale_1_shape;
    const gert::StorageShape *commQuantScaleShape2 = mmrCtxInfo_.comm_quant_scale_2_shape;
    if ((commQuantScaleShape1 != nullptr) && (commQuantScaleShape2 != nullptr)) {
        isInput = true;
    }

    MutableRCSTilingData().isInputCommQuantScale = isInput;
    OP_LOGD(opName_, "Is input comm_quant_scale_1_shape and comm_quant_scale_2_shape? %d.", isInput ? 1 : 0);

    const int64_t *commQuantModePtr = mmrCtxInfo_.commQuantModePtr;
    if (commQuantModePtr != nullptr) {
        if (*commQuantModePtr == 1) {
            MutableRCSTilingData().isInputCommQuantScale = COMM_QUANT_MODE_TRUE;
        }
    }
}

ge::graphStatus MatmulAllReduceTilingBase::DoMatmulTiling(matmul_tiling::MultiCoreMatmulTiling &mm1,
                                                          AscendC::tiling::TCubeTiling &cubeTiling)
{
    uint64_t mValue = args_.mValue;
    uint64_t nValue = args_.nValue;
    uint64_t kValue = args_.kValue;
    auto bmmFormat = matmul_tiling::CubeFormat::ND;
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(mmrCtxInfo_.x2->GetStorageFormat())) ==
        ge::Format::FORMAT_FRACTAL_NZ) {
        bmmFormat = matmul_tiling::CubeFormat::NZ;
        isWeightNz_ = true;
    }
    mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args_.aType, args_.isATrans);
    mm1.SetBType(matmul_tiling::TPosition::GM, bmmFormat, args_.bType, args_.isBTrans);
    mm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args_.cType);
    if (args_.isBias) {
        mm1.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args_.biasType);
        mm1.SetBias(true);
    } else {
        mm1.SetBias(false);
    }
    mm1.SetDim(args_.aicCoreNum);

    mm1.SetShape(mValue, nValue, kValue);
    mm1.SetOrgShape(mValue, nValue, kValue);
    mm1.SetBufferSpace(512 * 1024, -1, -1); // 512 * 1024 is buffer size
    int32_t fixCoreM = -1;
    int32_t fixCoreK = -1;
    int32_t fixCoreN = -1;
    mm1.SetSingleShape(fixCoreM, fixCoreN, fixCoreK);
    if (mm1.GetTiling(cubeTiling) == -1) {
        OP_LOGE(opName_, "MValue %lu, nValue %lu, kValue %lu, aicCoreNum %lu.", mValue, nValue, kValue,
                args_.aicCoreNum);
        return ge::GRAPH_FAILED;
    }
    mc2tiling::MatmulFormulaicTiling reduceTiling("MatmulAllReduce");
    reduceTiling.SetSocVersion(socVersion_);
    reduceTiling.SetWeightFormat(bmmFormat);
    reduceTiling.GetCubeTiling(args_, cubeTiling);
    return ge::GRAPH_SUCCESS;
}

void MatmulAllReduceTilingBase::DoL2CacheTiling(Mc2Tiling::Mc2L2cacheTilePara &l2cacheTiling)
{
    L2TilePara tileL2;
    bool enableL2Tile = CalL2TilePara(tileL2, args_.mValue, args_.kValue, args_.nValue, args_.aicCoreNum);
    enableL2Cache_ = enableL2Cache_ && enableL2Tile;
    OP_LOGD(opName_, "EnableL2Tile is %d.", enableL2Tile);
    if (enableL2Tile) {
        l2cacheTiling.mTileCntL2 = tileL2.mTile;
        l2cacheTiling.nTileCntL2 = tileL2.nTile;
        l2cacheTiling.mTileBlock = tileL2.mTileBlock;
        l2cacheTiling.nTileBlock = tileL2.nTileBlock;
        OP_LOGD(opName_, "TileL2.mTile %u, tileL2.nTile %u, tileL2.mTileBlock %u, tileL2.nTileBlock %u.", tileL2.mTile,
                tileL2.nTile, tileL2.mTileBlock, tileL2.nTileBlock);
    }
}

// 根据ctx填充mmrCtxInfo, 这个函数结束之后从context读的操作应该都从mmrCtxInfo读
// 根据ctxinfo做后续处理的AnalyzeShapeAttr操作, 子类重写之后也应该调用一下
ge::graphStatus MatmulAllReduceTilingBase::GetShapeAttrsInfo()
{
    MC2_CHECK_LOG_RET(opName_, ContextTransfer::AssembleMMRCtxInfoFromMMRCtx(context_, mmrCtxInfo_));
    return AnalyzeShapeAttr();
}

ge::graphStatus MatmulAllReduceTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Failed to get platform info."), return ge::GRAPH_FAILED);
    std::string intrinsicName = "Intrinsic_fix_pipe_l0c2out";
    std::string val;
    (void)platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", intrinsicName.c_str(), val);
    supportL0c2Out_ = !val.empty();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    OP_TILING_CHECK(CheckRanksizePlatformSupported() != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Check Ranksize Platform Supported failed."), return ge::GRAPH_FAILED);
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    auto coreNum = ascendcPlatform.GetCoreNumAic();
    args_.aicCoreNum = coreNum;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;

    OP_TILING_CHECK(!CheckPlatformInfo(), OP_LOGE(opName_, "Check Platform Info failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "workspaces"),
                    return ge::GRAPH_FAILED);

    uint32_t biasLen = 0;
    if (args_.isBias && args_.biasType == matmul_tiling::DataType::DT_BFLOAT16) {
        enableBiasConvert_ = true;
        biasLen = mc2tiling::AlignUp(args_.orgNValue, mc2tiling::SHAPE_ALIGN_SIZE) * sizeof(float);
    }
    MutableRCSTilingData().biasLen = biasLen;
    uint64_t gmcFloat = 0;

    // __NPU_ARCH__ == 3510
    // 950需要自己申请一块workSpace存放mm的输出
    if (npuArch_ == NpuArch::DAV_3510) {
        gmcFloat = static_cast<uint64_t>(MutableRCSTilingData().rankM) *
                   static_cast<uint64_t>(MutableRCSTilingData().rankN) * static_cast<uint64_t>(args_.outputDtypeSize);
    }
    // end __NPU_ARCH__ == 3510

    uint64_t mmOutInt32Len = 0;
    if (isUbQuant_) {
        uint32_t maxM = std::max(tilingData_.matmulTiling.M, tilingData_.tailTiling.M);
        mmOutInt32Len =
            static_cast<uint64_t>(maxM) * static_cast<uint64_t>(tilingData_.matmulTiling.N) * sizeof(int32_t);
    }
    uint32_t softSyncSize = mc2tiling::AC_MAX_AIV * 32; // aiv_cnt * 32bytes
    workspaces[0] = libApiWorkSpaceSize_ + biasLen + softSyncSize + mmOutInt32Len + gmcFloat;
    workspaceSize_ = workspaces[0];
    OP_LOGI(opName_,
            "LibApiWorkSpaceSize=%u, biasLen=%d, softSyncSize=%u, mmOutInt32Len=%u,"
            " workspaces[0] size=%ld.",
            libApiWorkSpaceSize_, biasLen, softSyncSize, mmOutInt32Len, workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::PostTiling()
{
    size_t tilingDataSize = sizeof(MatmulAllReduceTilingData);
    OP_LOGD(opName_, "Final tiling data size: %zu.", tilingDataSize);
    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] is not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "Memcpy_s failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    PrintTilingData();

    context_->SetBlockDim(args_.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::CheckRanksizePlatformSupported() const
{
    bool rankSizeSupported = mc2tiling::Mc2TilingUtils::CheckRankSize(npuArch_, rankSize_);
    OP_TILING_CHECK(
        !rankSizeSupported,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "rankSize", std::to_string(rankSize_).c_str(),
                                              "The value of rankSize is not supported by the current SoC version"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool MatmulAllReduceTilingBase::AnalyzeAttrs()
{
    auto group = mmrCtxInfo_.group;
    reduceOp_ = mmrCtxInfo_.reduceOp;
    auto isTransA = mmrCtxInfo_.isTransA;
    auto isTransB = mmrCtxInfo_.isTransB;
    auto commTurn = mmrCtxInfo_.commTurn;
    if (commTurn == 0) {
        commTurn = COMM_TILE;
    }
    const int64_t *antiquantGroupSizePtr = mmrCtxInfo_.antiquantGroupSizePtr;
    if (antiquantGroupSizePtr != nullptr) {
        antiGroupSize_ = static_cast<uint64_t>(*antiquantGroupSizePtr);
    }
    rankSize_ = mc2tiling::MatmulFormulaicTiling::GetRankSize(group);

    OP_LOGD(opName_,
            "Group is %s, rankSize_ is %u, reduceOp_ is %s, isTransA is %d, isTransB is %d,"
            " commTurn is %d antiGroupSize_ is %lu.",
            group, rankSize_, reduceOp_, *isTransA, *isTransB, commTurn, antiGroupSize_);
    args_.isATrans = isTransA ? *isTransA : 0;
    args_.isBTrans = isTransB ? *isTransB : 0;
    args_.cmdType = mc2tiling::AicpuComType::HCCL_CMD_ALLREDUCE;
    args_.rankDim = rankSize_;
    args_.commTurn = commTurn;
    return true;
}

bool MatmulAllReduceTilingBase::IsPerTensorDequantScale(const gert::Shape &dequantShape) const
{
    return dequantShape.GetDimNum() == 1 && dequantShape.GetDim(0) == 1;
}

void MatmulAllReduceTilingBase::SetQuantData()
{
    // 判断是否是量化场景，以及per-tensor还是per-channel
    const gert::StorageShape *matrixDequant = mmrCtxInfo_.dequant_scale_shape;
    const gert::StorageShape *matrixPertoken = mmrCtxInfo_.pertoken_scale_shape;
    if (matrixDequant != nullptr) {
        isQuantKey_ = true;
        const auto &dequantShape = matrixDequant->GetStorageShape();
        isPerTensor_ = IsPerTensorDequantScale(dequantShape);
        quantType_ = isPerTensor_ ? Mc2QuantType::PER_TENSOR : Mc2QuantType::PER_CHANNEL;

        // perblock要求2维scale
        if (matrixPertoken != nullptr) {
            isPerBlock_ = (scenario_ == AllReduceScenario::FP8HIF8) && (dequantShape.GetDimNum() == DIM_NUM_TWO) &&
                          ((matrixPertoken->GetStorageShape().GetDimNum() == DIM_NUM_TWO) ||
                           (matrixPertoken->GetStorageShape().GetDimNum() == DIM_NUM_THREE));
        }
    }

    OP_LOGD(opName_, "Tiling isQuantKey_ is %d, isPerTensor is %d quantType_ is %d.", isQuantKey_ ? 1 : 0,
            isPerTensor_ ? 1 : 0, static_cast<int32_t>(quantType_));
}

void MatmulAllReduceTilingBase::SetAntiQuantData()
{
    antiQuantType_ = GetAntiQuantType();
    OP_LOGD(opName_, "Tiling antiQuantType_is %d.", static_cast<int32_t>(antiQuantType_));
}

void MatmulAllReduceTilingBase::GetAtomicAddData()
{
    // 判断是否需要作Atomic add
    bool isAdd = false;
    const gert::StorageShape *matrixAdd = mmrCtxInfo_.x3_shape;
    if (matrixAdd != nullptr) {
        isAdd = true;
    }
    MutableRCSTilingData().isAdd = isAdd;
}

uint64_t MatmulAllReduceTilingBase::GetNValue() const
{
    const uint16_t maxDimNumForND = 3U;
    const uint16_t nIndexFor3Dim = maxDimNumForND - 1U;
    const uint16_t nIndexFor2Dim = maxDimNumForND - 2U;
    const gert::StorageShape *yShape = mmrCtxInfo_.y_shape;
    return yShape->GetStorageShape().GetDimNum() == maxDimNumForND ? yShape->GetStorageShape().GetDim(nIndexFor3Dim) :
                                                                     yShape->GetStorageShape().GetDim(nIndexFor2Dim);
}

uint64_t MatmulAllReduceTilingBase::GetKValue() const
{
    const gert::StorageShape *inputShape = mmrCtxInfo_.x1_shape;
    size_t inputLen = inputShape->GetStorageShape().GetDimNum();
    return inputShape->GetStorageShape().GetDim(inputLen - 1);
}

uint64_t MatmulAllReduceTilingBase::GetMValue() const
{
    const gert::StorageShape *inputShape = mmrCtxInfo_.x1_shape;
    size_t inputLen = inputShape->GetStorageShape().GetDimNum();
    uint64_t mValue = inputShape->GetStorageShape().GetDim(0);
    if (inputLen == MAX_M_SHAPE_DIM) {
        mValue *= inputShape->GetStorageShape().GetDim(1);
    }
    return mValue;
}

uint64_t MatmulAllReduceTilingBase::GetBatchValue() const
{
    const gert::StorageShape *inputShape = mmrCtxInfo_.x1_shape;
    size_t inputLen = inputShape->GetStorageShape().GetDimNum();
    uint64_t bValue = 1UL;
    if ((inputLen == MAX_M_SHAPE_DIM) && (inputShape->GetStorageShape().GetDim(0) != 0)) {
        bValue = inputShape->GetStorageShape().GetDim(0);
    }
    return bValue;
}

ge::graphStatus MatmulAllReduceTilingBase::CheckInputBias() const
{
    // bias为n值
    if (mmrCtxInfo_.bias_shape != nullptr) {
        size_t biasDimNum = mmrCtxInfo_.bias_shape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(biasDimNum != DIM_NUM_ONE,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "bias",
                                                                 (std::to_string(biasDimNum) + "D").c_str(),
                                                                 "The shape dim of bias must be 1"),
                        return ge::GRAPH_FAILED);
        int64_t biasNValue = mmrCtxInfo_.bias_shape->GetStorageShape().GetDim(0);
        int64_t nValue = static_cast<int64_t>(GetNValue());
        OP_TILING_CHECK(
            biasNValue != nValue,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "bias", Ops::Base::ToString(mmrCtxInfo_.bias_shape->GetStorageShape()).c_str(),
                "The n dimension of bias must be the same as that of output"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::CheckInput()
{
    // 全量化mxfp&fp8hif8支持3种输出
    std::vector<ge::DataType> DTYPE_SUPPORT_LIST_Y;
    if ((scenario_ == AllReduceScenario::MXFP8) || (scenario_ == AllReduceScenario::MXFP4) ||
        (scenario_ == AllReduceScenario::FP8HIF8)) {
        DTYPE_SUPPORT_LIST_Y = {ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16, ge::DataType::DT_FLOAT};
    } else {
        DTYPE_SUPPORT_LIST_Y = {ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16};
    }

    OP_TILING_CHECK(
        std::find(DTYPE_SUPPORT_LIST_Y.begin(), DTYPE_SUPPORT_LIST_Y.end(),
                  static_cast<ge::DataType>(mmrCtxInfo_.y->GetDataType())) == DTYPE_SUPPORT_LIST_Y.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "y", std::to_string(static_cast<int32_t>(mmrCtxInfo_.y->GetDataType())).c_str(),
            "The dtype of y must be fp16, bf16 or float"),
        return ge::GRAPH_FAILED);

    // x1 shape 为2-3维
    size_t x1DimNum = mmrCtxInfo_.x1_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(!((x1DimNum >= DIM_NUM_TWO) && (x1DimNum <= DIM_NUM_THREE)),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x1",
                                                             (std::to_string(x1DimNum) + "D").c_str(),
                                                             "The shape dim of x1 must be 2 or 3"),
                    return ge::GRAPH_FAILED);
    // bias为n值
    if (CheckInputBias() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // reduceOp为sum
    OP_TILING_CHECK(strncmp(mmrCtxInfo_.reduceOp, "sum", 3) != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "reduceOp", mmrCtxInfo_.reduceOp,
                                                          "The value of reduceOp must be sum"),
                    return ge::GRAPH_FAILED);
    // commTurn为0
    OP_TILING_CHECK(mmrCtxInfo_.commTurn != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "commTurn",
                                                          std::to_string(mmrCtxInfo_.commTurn).c_str(),
                                                          "The value of commTurn must be 0"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::CheckA16W16()
{
    uint64_t nValue = GetNValue();
    OP_TILING_CHECK(!CheckBiasShape(nValue), OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "bias"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((mmrCtxInfo_.antiquant_scale_shape != nullptr) ||
                     (mmrCtxInfo_.antiquant_offset_shape != nullptr) || (mmrCtxInfo_.dequant_scale_shape != nullptr)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "antiquantScale/antiquantOffset/dequantScale", "not null",
                        "The value of antiquantScale/antiquantOffset/dequantScale must be nullptr in A16W16 mode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((mmrCtxInfo_.comm_quant_scale_1_shape != nullptr) || (mmrCtxInfo_.comm_quant_scale_2_shape != nullptr)),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "comm_quant_scale", "not null",
                                              "The value of comm_quant_scale must be nullptr in A16W16 mode"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool MatmulAllReduceTilingBase::CheckBiasShape(const uint64_t nValue) const
{
    const auto bias = mmrCtxInfo_.bias_shape;
    if (bias != nullptr) {
        OP_TILING_CHECK(isPerBlock_, OP_LOGE_WITH_INVALID_INPUT(opName_, "bias"), return false);
        const auto biasShapeSize = static_cast<size_t>(bias->GetStorageShape().GetShapeSize());
        uint64_t dimNum = bias->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            (dimNum != 1U) || (biasShapeSize != nValue),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "bias", Ops::Base::ToString(bias->GetStorageShape()).c_str(),
                                                  "The shape of bias must be [n] or [1, n]"),
            return false);
    } else {
        OP_LOGD(context_->GetNodeName(), "No Bias.");
    }

    return true;
}

ge::graphStatus MatmulAllReduceTilingBase::CheckA8W8()
{
    uint64_t mValue = GetMValue();
    uint64_t nValue = GetNValue();
    uint64_t kValue = GetKValue();
    OP_TILING_CHECK(!CheckBiasShape(nValue), OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "bias"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckDequantScaleShape(nValue),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "dequantScale"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckPertokenScaleShape(mValue, kValue),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "pertokenScale"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((mmrCtxInfo_.antiquant_scale_shape != nullptr) || (mmrCtxInfo_.antiquant_offset_shape != nullptr)),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "antiquantScale/antiquantOffset", "not null",
                                              "antiquantScale and antiquantOffset must be nullptr in A8W8 mode"),
        return ge::GRAPH_FAILED);
    if ((socVersion_ == platform_ascendc::SocVersion::ASCEND910B) || (npuArch_ == NpuArch::DAV_3510)) {
        OP_TILING_CHECK(!CheckCommQuantScaleShape(nValue),
                        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "commQuantScale"), return ge::GRAPH_FAILED);
    } else if (npuArch_ == NpuArch::DAV_2002) {
        OP_TILING_CHECK(
            ((mmrCtxInfo_.comm_quant_scale_1_shape != nullptr) || (mmrCtxInfo_.comm_quant_scale_2_shape != nullptr)),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "comm_quant_scale", "not null",
                                                  "The value of comm_quant_scale must be nullptr on 310P"),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "socVersion",
                                              std::to_string(static_cast<int>(npuArch_)).c_str(),
                                              "The value of socVersion must be within the supported range.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::CheckEmptyTensor()
{
    // n为0的时候，框架拦截，走不进tiling逻辑。
    OP_TILING_CHECK(
        (mmrCtxInfo_.bias_shape != nullptr) && (mmrCtxInfo_.bias_shape->GetStorageShape().GetShapeSize() == 0),
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "bias"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((mmrCtxInfo_.x3_shape != nullptr) && (mmrCtxInfo_.x3_shape->GetStorageShape().GetShapeSize() == 0),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "x3"), return ge::GRAPH_FAILED);
    // 校验过input中的batch，M，N和out中的必须相等，所以这里只校验out是否是空tensor
    OP_TILING_CHECK(mmrCtxInfo_.y_shape->GetStorageShape().GetShapeSize() == 0,
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "y"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::CheckQuantEmptyTensor()
{
    OP_TILING_CHECK(
        (mmrCtxInfo_.bias_shape != nullptr) && (mmrCtxInfo_.bias_shape->GetStorageShape().GetShapeSize() == 0),
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "bias"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((mmrCtxInfo_.x3_shape != nullptr) && (mmrCtxInfo_.x3_shape->GetStorageShape().GetShapeSize() == 0),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "x3"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((mmrCtxInfo_.dequant_scale_shape != nullptr) &&
                        (mmrCtxInfo_.dequant_scale_shape->GetStorageShape().GetShapeSize() == 0),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "dequantScale"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((mmrCtxInfo_.pertoken_scale_shape != nullptr) &&
                        (mmrCtxInfo_.pertoken_scale_shape->GetStorageShape().GetShapeSize() == 0),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "pertokenScale"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((mmrCtxInfo_.comm_quant_scale_1_shape != nullptr) &&
                        (mmrCtxInfo_.comm_quant_scale_1_shape->GetStorageShape().GetShapeSize() == 0),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "commQuantScale1"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((mmrCtxInfo_.comm_quant_scale_2_shape != nullptr) &&
                        (mmrCtxInfo_.comm_quant_scale_2_shape->GetStorageShape().GetShapeSize() == 0),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "commQuantScale2"), return ge::GRAPH_FAILED);
    // 校验过input中的batch，M，N和out中的必须相等，所以这里只校验out是否是空tensor
    OP_TILING_CHECK((GetKValue() == 0) || (mmrCtxInfo_.y_shape->GetStorageShape().GetShapeSize() == 0),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "y"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::CheckWeightQuantEmptyTensor()
{
    OP_TILING_CHECK((mmrCtxInfo_.x3_shape != nullptr) && (mmrCtxInfo_.x3_shape->GetStorageShape().GetShapeSize() == 0),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "x3"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (mmrCtxInfo_.bias_shape != nullptr) && (mmrCtxInfo_.bias_shape->GetStorageShape().GetShapeSize() == 0),
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "bias"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((mmrCtxInfo_.antiquant_scale_shape != nullptr) &&
                        (mmrCtxInfo_.antiquant_scale_shape->GetStorageShape().GetShapeSize() == 0),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "antiquantScale"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((mmrCtxInfo_.antiquant_offset_shape != nullptr) &&
                        (mmrCtxInfo_.antiquant_offset_shape->GetStorageShape().GetShapeSize() == 0),
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "antiquantOffset"), return ge::GRAPH_FAILED);
    // 校验过input中的batch，M，N和out中的必须相等，所以这里只校验out是否是空tensor
    OP_TILING_CHECK(mmrCtxInfo_.y_shape->GetStorageShape().GetShapeSize() == 0,
                    OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "y"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTilingBase::CheckA16W8()
{
    uint64_t kValue = GetKValue();
    uint64_t nValue = GetNValue();
    OP_TILING_CHECK(!CheckBiasShape(nValue), OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "bias"),
                    return ge::GRAPH_FAILED);
    if (kValue == 0) {
        OP_LOGD(context_->GetNodeName(),
                "KValue equals zero. "
                "There is no need to check antiquantScale shape in the situation that tensor is empty");
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK(
        (!CheckAntiQuantScaleShape(kValue, nValue)) || (mmrCtxInfo_.dequant_scale_shape != nullptr),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dequantScale", "not null",
                                              "The value of dequantScale must be nullptr in antiquant mode"),
        return ge::GRAPH_FAILED);
    if (!CheckAntiQuantOffsetValid()) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "antiquantOffset", "invalid", "valid antiquantOffset");
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(
        (mmrCtxInfo_.comm_quant_scale_1_shape != nullptr) || (mmrCtxInfo_.comm_quant_scale_2_shape != nullptr),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "comm_quant_scale", "not null",
                                              "The value of comm_quant_scale must be nullptr in A16W8 mode"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool MatmulAllReduceTilingBase::IsA16W8Scenario(const DataType aType, const DataType bType,
                                                const gert::StorageShape *antiQuantScale) const
{
    if ((aType != ge::DT_INT8) && (bType == ge::DT_INT8) && (antiQuantScale != nullptr)) {
        return true;
    } else if (((aType == ge::DT_FLOAT16) || (aType == ge::DT_BF16)) &&
               ((bType == ge::DT_FLOAT8_E4M3FN) || (bType == ge::DT_FLOAT8_E5M2) || (bType == ge::DT_HIFLOAT8)) &&
               (antiQuantScale != nullptr)) {
        return true;
    }

    return false;
}

static AllReduceScenario GetFP8FP4Scenario(const DataType aType, const DataType bType,
                                           const gert::CompileTimeTensorDesc *dequantScaleDesc,
                                           const gert::CompileTimeTensorDesc *pertokeScaleDesc)
{
    // MXFP4: fp4+fp4+fp8
    if ((aType == ge::DT_FLOAT4_E2M1) && (bType == ge::DT_FLOAT4_E2M1) &&
        (dequantScaleDesc->GetDataType() == ge::DT_FLOAT8_E8M0) && (pertokeScaleDesc != nullptr) &&
        (pertokeScaleDesc->GetDataType() == ge::DT_FLOAT8_E8M0)) {
        return AllReduceScenario::MXFP4;
    }
    if (((aType == ge::DT_FLOAT8_E4M3FN) || (aType == ge::DT_FLOAT8_E5M2) || (aType == ge::DT_HIFLOAT8)) &&
        ((bType == ge::DT_FLOAT8_E4M3FN) || (bType == ge::DT_FLOAT8_E5M2) || (bType == ge::DT_HIFLOAT8))) {
        // MXFP8: FP8HIF8+FP8HIF8+FP8. FP8HIF8: FP8HIF8+FP8HIF8+UINT64/FP32.
        if ((dequantScaleDesc->GetDataType() == ge::DT_FLOAT8_E8M0) && (pertokeScaleDesc != nullptr) &&
            (pertokeScaleDesc->GetDataType() == ge::DT_FLOAT8_E8M0)) {
            return AllReduceScenario::MXFP8;
        } else {
            return AllReduceScenario::FP8HIF8;
        }
    }
    return AllReduceScenario::INVALID;
}

AllReduceScenario MatmulAllReduceTilingBase::GetAllReduceScenario(const DataType aType, const DataType bType,
                                                                  const gert::StorageShape *dequantScale,
                                                                  const gert::StorageShape *antiQuantScale) const
{
    if ((aType == ge::DT_INT8) && (bType == ge::DT_INT8) && (dequantScale != nullptr)) {
        return AllReduceScenario::A8W8;
    }

    if (IsA16W8Scenario(aType, bType, antiQuantScale)) {
        return AllReduceScenario::A16W8;
    }

    if ((aType != ge::DT_INT8) && (bType == ge::DT_INT4) && (antiQuantScale != nullptr)) {
        return AllReduceScenario::A16W4;
    }
    if (dequantScale == nullptr) {
        return AllReduceScenario::INVALID;
    }
    auto dequantScaleDesc = mmrCtxInfo_.dequant_scale;
    auto pertokeScaleDesc = mmrCtxInfo_.pertoken_scale;
    return GetFP8FP4Scenario(aType, bType, dequantScaleDesc, pertokeScaleDesc);
}

bool MatmulAllReduceTilingBase::SetArgs(ge::DataType aType, ge::DataType bType, ge::DataType cType,
                                        ge::DataType biasType, bool isBias)
{
    uint64_t batchValue = GetBatchValue();
    uint64_t mValue = GetMValue();
    uint64_t kValue = GetKValue();
    uint64_t nValue = GetNValue();

    uint64_t inputDtypeSize = D_TYPE_SIZE_MAP.at(aType);
    uint64_t outputDtypeSize = D_TYPE_SIZE_MAP.at(cType);

    args_.orgMValue = mValue;
    args_.orgNValue = nValue;
    args_.orgKValue = kValue;
    isKZero_ = args_.orgKValue == 0;
    args_.batchValue = batchValue;
    args_.mValue = mValue;
    args_.nValue = nValue;
    args_.kValue = kValue;
    args_.inputDtypeSize = inputDtypeSize;
    args_.outputDtypeSize = outputDtypeSize;
    args_.enablePad = false;
    args_.enableSplitK = false;
    args_.isBias = isBias;
    args_.geAType = aType;
    args_.geBType = bType;
    args_.geCType = cType;
    args_.geBiasType = biasType;
    try {
        args_.aType = D_TYPE_MAP.at(aType);
        args_.bType = D_TYPE_MAP.at(bType);
        args_.cType = D_TYPE_MAP.at(cType);
        args_.biasType = D_TYPE_MAP.at(biasType);
    } catch (const std::out_of_range &e) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "aType/bType/cType/biasType",
            (std::to_string(static_cast<int32_t>(aType)) + "/" + std::to_string(static_cast<int32_t>(bType)) + "/" +
             std::to_string(static_cast<int32_t>(cType)) + "/" + std::to_string(static_cast<int32_t>(biasType)))
                .c_str(),
            "unsupported dtype combination");
        return false;
    }

    OP_LOGD(opName_, "AType: %d bType: %d cType: %d biasType %d.", static_cast<int32_t>(args_.aType),
            static_cast<int32_t>(args_.bType), static_cast<int32_t>(args_.cType), static_cast<int32_t>(args_.biasType));
    return true;
}

bool MatmulAllReduceTilingBase::AnalyzeInputs()
{
    ge::DataType biasType;
    bool isBias = true;

    auto aType = mmrCtxInfo_.x1->GetDataType();
    auto bType = mmrCtxInfo_.x2->GetDataType();
    auto cType = mmrCtxInfo_.y->GetDataType();
    const gert::StorageShape *antiquantScale = mmrCtxInfo_.antiquant_scale_shape;
    const gert::StorageShape *dequantScale = mmrCtxInfo_.dequant_scale_shape;

    scenario_ = GetAllReduceScenario(aType, bType, dequantScale, antiquantScale);
    isA8W8_ = (scenario_ == AllReduceScenario::A8W8);
    isA16W8_ = (scenario_ == AllReduceScenario::A16W8);
    isA16W4_ = (scenario_ == AllReduceScenario::A16W4);
    OP_LOGD(opName_, "Scenario is %u", static_cast<uint32_t>(scenario_));

    auto dequantDesc = mmrCtxInfo_.dequant_scale;
    isUbQuant_ = ((dequantDesc != nullptr) ? (dequantDesc->GetDataType() == ge::DT_BF16) : isUbQuant_);
    OP_LOGD(opName_, "MatmulAllReduceTilingBase::AnalyzeInput isUbQuant_: %d ", isUbQuant_ ? 1 : 0);
    const gert::StorageShape *matrixBias = mmrCtxInfo_.bias_shape;
    if (matrixBias == nullptr) {
        isBias = false;
        biasType = cType;
    } else {
        biasType = mmrCtxInfo_.bias->GetDataType();
    }

    SetQuantData();
    SetAntiQuantData();
    GetAtomicAddData();

    size_t yDimNum = mmrCtxInfo_.y_shape->GetStorageShape().GetDimNum();
    OP_LOGD(context_->GetNodeName(), "Dim of output is %lu.", yDimNum);
    OP_TILING_CHECK(
        (yDimNum < DIM_NUM_TWO || yDimNum > DIM_NUM_THREE),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "y", (std::to_string(yDimNum) + "D").c_str(),
                                                 "The shape dim of y must be 2 or 3"),
        return false);
    return SetArgs(aType, bType, cType, biasType, isBias);
}

mc2tiling::HcclDataType MatmulAllReduceTilingBase::GetDataType(ge::DataType type)
{
    if (HCCL_DATA_TYPE.find(type) != HCCL_DATA_TYPE.end()) {
        return HCCL_DATA_TYPE.at(type);
    }
    return mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED;
}

void MatmulAllReduceTilingBase::CalcL2TileBlock(L2TilePara &tileL2, uint64_t mValue, uint64_t nValue, uint64_t kValue,
                                                uint64_t blockBaseM, uint64_t blockBaseN, uint32_t tileSize,
                                                uint32_t tileLimit, uint64_t sizeA, uint64_t sizeB,
                                                uint32_t cubeCoreNum) const
{
    const uint32_t ratio = 2;
    const uint32_t kByte = 1024;
    // 仅考虑fp16场景
    tileL2.mTileBlock = (tileSize * kByte * kByte / ratio / kValue + blockBaseM - 1) / blockBaseM;
    tileL2.nTileBlock = (tileSize * kByte * kByte / ratio / kValue + blockBaseN - 1) / blockBaseN;
    // 该处保证L2cache切分的合法性，否则kernel侧切分策略不合法，会导致地址溢出
    tileL2.mTile = (mValue + tileL2.mTileBlock * blockBaseM - 1) / (tileL2.mTileBlock * blockBaseM);
    tileL2.nTile = (nValue + tileL2.nTileBlock * blockBaseN - 1) / (tileL2.nTileBlock * blockBaseN);
    // 如果A或B实际小于tileSize，导致切分的mTileBlock太大，此时可以不用切分
    if (tileL2.mTileBlock >= (mValue / blockBaseM)) {
        tileL2.mTile = 1;
        tileL2.mTileBlock = (mValue + blockBaseM - 1) / blockBaseM;
    }
    if (tileL2.nTileBlock >= (nValue / blockBaseN)) {
        tileL2.nTile = 1;
        tileL2.nTileBlock = (nValue + blockBaseN - 1) / blockBaseN;
    }
    tileL2.mTile = (tileL2.mTile > 0) ? tileL2.mTile : 1;
    tileL2.nTile = (tileL2.nTile > 0) ? tileL2.nTile : 1;
    if (sizeA <= tileLimit) {
        tileL2.mTile = 1;
    }
    if (sizeB <= tileLimit) {
        tileL2.nTile = 1;
    }
    (void)cubeCoreNum;
}

bool MatmulAllReduceTilingBase::CalL2TilePara(L2TilePara &tileL2, uint64_t mValue, uint64_t kValue, uint64_t nValue,
                                              uint32_t cubeCoreNum)
{
    uint64_t blockBaseM = 0;
    uint64_t blockBaseN = 0;
    uint64_t blockBaseK = 0;
    mc2tiling::MatmulFormulaicTiling::GetBaseBlockParm(socVersion_, blockBaseM, blockBaseN, blockBaseK, blockBaseK,
                                                       blockBaseK);
    // size 单位/MB
    const uint32_t kByte = 1024;
    uint64_t sizeA = mValue * kValue * D_MTYPE_SIZE_MAP.at(args_.aType) / kByte / kByte;
    uint64_t sizeB = kValue * nValue * D_MTYPE_SIZE_MAP.at(args_.bType) / kByte / kByte;
    uint64_t sizeC = mValue * nValue * D_MTYPE_SIZE_MAP.at(args_.cType) / kByte / kByte;
    uint64_t totalSize = sizeA + sizeB + sizeC;

    uint64_t l2CacheSize = 0;
    uint64_t singleMatrixSize = 0;
    uint32_t tileSize = 0;
    uint32_t tileLimit = 0;
    bool useNewPara = (cubeCoreNum > CORE_THRESHOLD && (mValue + kValue + nValue) < LENGTH_SUM_THRESHOLD);
    GetL2CacheParm(l2CacheSize, singleMatrixSize, tileSize, tileLimit, useNewPara);

    OP_TILING_CHECK(
        blockBaseM == 0ull || blockBaseN == 0ull,
        OP_LOGE_FOR_INVALID_VALUE(opName_, "blockBaseM, blockBaseN",
                                  (std::string(std::to_string(blockBaseM)) + ", " + std::to_string(blockBaseN)).c_str(),
                                  "positive values"),
        return false);

    if (totalSize >= l2CacheSize || sizeA >= singleMatrixSize || sizeB >= singleMatrixSize ||
        sizeC >= singleMatrixSize) {
        CalcL2TileBlock(tileL2, mValue, nValue, kValue, blockBaseM, blockBaseN, tileSize, tileLimit, sizeA, sizeB,
                        cubeCoreNum);
        if (!isWeightNz_ && (tileL2.mTileBlock * tileL2.nTileBlock < cubeCoreNum)) {
            return false;
        }
        if ((tileL2.mTile > 1 || tileL2.nTile > 1) && mValue >= blockBaseM) {
            return true;
        }
    }
    return false;
}

bool MatmulAllReduceTilingBase::HasAntiQuantOffset() const
{
    return mmrCtxInfo_.antiquant_offset != nullptr;
}
void MatmulAllReduceTilingBase::GetScaleMNAndKIdx(const gert::StorageShape *scaleShape, bool isPertoken,
                                                  uint64_t scaleDimNum, uint64_t &MN, uint64_t &K) const
{
    if (isPertoken) {
        MN = scaleDimNum - 3U;
        K = scaleDimNum - 2U;
        return;
    }
    const auto x2Shape = mmrCtxInfo_.x2_shape;
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);
    bool isTransB = mmrCtxInfo_.isTransB;
    bool nIsOne = isTransB ? (x2Dim0 == 1U) : (x2Dim1 == 1U);
    if (!nIsOne) {
        MN = args_.isBTrans ? scaleDimNum - 3U : scaleDimNum - 2U;
        K = args_.isBTrans ? scaleDimNum - 2U : scaleDimNum - 3U;
    }
}

bool MatmulAllReduceTilingBase::CheckMXScenarioScaleShape(const uint64_t dimZeroValue, const uint64_t kValue,
                                                          const gert::StorageShape *scaleShape, const bool isPertoken,
                                                          const bool isMXfp4) const
{
    uint64_t scaleDimNum = scaleShape->GetStorageShape().GetDimNum();
    const char dimZeroName = isPertoken ? 'm' : 'n';
    uint64_t MN = 0;
    uint64_t K = 0;
    const std::string scaleName = isPertoken ? "pertokenScale(x1Scale)" : "dequantScale(x2Scale)";
    OP_TILING_CHECK((scaleDimNum != DIM_NUM_THREE),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName_, scaleName.c_str(), (std::to_string(scaleDimNum) + "D").c_str(),
                        "When the quantization mode is MXfp8 or MXfp4, the shape dim of scale must be 3"),
                    return false);
    uint64_t scaleDim3 = scaleShape->GetStorageShape().GetDim(scaleDimNum - 1U);
    OP_TILING_CHECK(scaleDim3 != 2,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, scaleName.c_str(), std::to_string(scaleDim3).c_str(),
                                                          "The value of scale dimension 3 (K) must be 2"),
                    return false);
    // 仅支持x1非转置x2转置
    GetScaleMNAndKIdx(scaleShape, isPertoken, scaleDimNum, MN, K);
    uint64_t scaleMN = scaleShape->GetStorageShape().GetDim(MN);
    uint64_t kOverMaxGroupsize = scaleShape->GetStorageShape().GetDim(K);
    (void)dimZeroName;

    // 此时只支持转置，支持非转置后拦截信息需要修改
    OP_TILING_CHECK(scaleMN != dimZeroValue,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, scaleName.c_str(),
                        ("[" + std::to_string(scaleMN) + ", " + std::to_string(kOverMaxGroupsize) + ", 2]").c_str(),
                        "The shape of scale must be [dimZeroValue, ceil(k,64), 2]"),
                    return false);
    if (isMXfp4) {
        OP_TILING_CHECK(
            (Ops::Base::CeilDiv(kValue, MX_GROUP_SIZE_K) % 2) != 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, scaleName.c_str(), ("k=" + std::to_string(kValue)).c_str(),
                                                  "The value of k of scale ceildivided by 32 must be even"),
            return false);
    }
    OP_TILING_CHECK(
        kOverMaxGroupsize != Ops::Base::CeilDiv(kValue, MX_GROUP_SIZE),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, scaleName.c_str(), std::to_string(kOverMaxGroupsize).c_str(),
                                              "The K dimension of scale must match ceil(k, 64)"),
        return false);
    return true;
}

bool MatmulAllReduceTilingBase::CheckDequantScalePerBlock(const gert::StorageShape *dequantScaleShape) const
{
    const auto x2Shape = mmrCtxInfo_.x2_shape;
    const auto dim0Ofscale = dequantScaleShape->GetStorageShape().GetDim(0);
    const auto dim1Ofscale = dequantScaleShape->GetStorageShape().GetDim(1);
    const auto dim0Ofx2 = x2Shape->GetStorageShape().GetDim(0);
    const auto dim1Ofx2 = x2Shape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        (dim0Ofscale != Ops::Base::CeilDiv(dim0Ofx2, SUPPORTED_BLOCK_SIZE)) ||
            (dim1Ofscale != Ops::Base::CeilDiv(dim1Ofx2, SUPPORTED_BLOCK_SIZE)),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "dequantScale(x2Scale)",
            ("(" + std::to_string(dim0Ofscale) + ", " + std::to_string(dim1Ofscale) + ")").c_str(),
            "The shape of dequantScale(x2Scale) must be (ceil(dim0Ofx2, blockSize), ceil(dim1Ofx2, blockSize))"),
        return false);
    return true;
}

bool MatmulAllReduceTilingBase::CheckDequantScaleShape(const uint64_t nValue) const
{
    const auto dequantScaleShape = mmrCtxInfo_.dequant_scale_shape;
    OP_TILING_CHECK(dequantScaleShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "dequantScale(x2Scale)"),
                    return false);

    // perblock 场景
    if (isPerBlock_) {
        return CheckDequantScalePerBlock(dequantScaleShape);
    }

    if (scenario_ == AllReduceScenario::MXFP4) {
        return CheckMXScenarioScaleShape(nValue, GetKValue(), dequantScaleShape, false, true);
    } else if (scenario_ == AllReduceScenario::MXFP8) {
        return CheckMXScenarioScaleShape(nValue, GetKValue(), dequantScaleShape, false, false);
    }

    const uint64_t scaleDimNum = dequantScaleShape->GetStorageShape().GetDimNum();
    OP_LOGD(context_->GetNodeName(), "Dim of dequantScale(x2Scale) is %lu.", scaleDimNum);
    OP_TILING_CHECK(scaleDimNum > DIM_NUM_THREE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName_, "dequantScale(x2Scale)", (std::to_string(scaleDimNum) + "D").c_str(),
                        "The shape dim of dequantScale(x2Scale) must be 1, 2 or 3"),
                    return false);
    const auto scaleShapeSize = static_cast<size_t>(dequantScaleShape->GetStorageShape().GetShapeSize());

    if (scenario_ == AllReduceScenario::MXFP4) {
        return CheckMXScenarioScaleShape(nValue, GetKValue(), dequantScaleShape, false, true);
    } else if (scenario_ == AllReduceScenario::MXFP8) {
        return CheckMXScenarioScaleShape(nValue, GetKValue(), dequantScaleShape, false, false);
    }

    bool perTensorShapeOk = IsPerTensorDequantScale(dequantScaleShape->GetStorageShape());
    OP_TILING_CHECK(
        !((quantType_ == Mc2QuantType::PER_TENSOR && perTensorShapeOk) ||
          (quantType_ == Mc2QuantType::PER_CHANNEL && scaleShapeSize == nValue)),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "dequantScale(x2Scale)",
                                              Ops::Base::ToString(dequantScaleShape->GetStorageShape()).c_str(),
                                              "The shape of dequantScale(x2Scale) must be [1], [n] or [1,n]"),
        return false);
    return true;
}

bool MatmulAllReduceTilingBase::CheckPerblockShape(const uint64_t mValue, const uint64_t kValue) const
{
    const auto pertokenScaleShape = mmrCtxInfo_.pertoken_scale_shape;
    const uint64_t pertokenScaleDimNum = pertokenScaleShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(pertokenScaleDimNum != mmrCtxInfo_.x1_shape->GetStorageShape().GetDimNum(),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName_, "pertokenScale(x1Scale)", (std::to_string(pertokenScaleDimNum) + "D").c_str(),
                        ("The shape dim of pertokenScale(x1Scale) must be " +
                         std::to_string(mmrCtxInfo_.x1_shape->GetStorageShape().GetDimNum()))
                            .c_str()),
                    return false);
    const uint64_t bsOfScale = pertokenScaleShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(
        (pertokenScaleDimNum == DIM_NUM_THREE) && (bsOfScale != GetBatchValue()),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "pertokenScale(x1Scale)", std::to_string(bsOfScale).c_str(),
                                              "The batch of pertokenScale(x1Scale) must be equal to the batch value"),
        return false);
    const auto mOfpertokenScale = pertokenScaleShape->GetStorageShape().GetDim(pertokenScaleDimNum - 2U);
    const auto kOfpertokenScale = pertokenScaleShape->GetStorageShape().GetDim(pertokenScaleDimNum - 1U);
    const int64_t mOfx1 = mValue / GetBatchValue();
    const int64_t kOfx1 = static_cast<int64_t>(kValue);
    OP_TILING_CHECK(
        (mOfpertokenScale != Ops::Base::CeilDiv(mOfx1, SUPPORTED_BLOCK_SIZE)) ||
            (kOfpertokenScale != Ops::Base::CeilDiv(kOfx1, SUPPORTED_BLOCK_SIZE)),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "pertokenScale(x1Scale)",
            ("[" + std::to_string(mOfpertokenScale) + ", " + std::to_string(kOfpertokenScale) + "]").c_str(),
            "The shape of pertokenScale(x1Scale) must be [ceil(m, blockSize), ceil(k, blockSize)]"),
        return false);
    return true;
}

bool MatmulAllReduceTilingBase::CheckPertokenScaleShape(const uint64_t mValue, const uint64_t kValue) const
{
    const auto pertokenScaleShape = mmrCtxInfo_.pertoken_scale_shape;
    if (pertokenScaleShape == nullptr) {
        // pertensor 或 perchannel
        return true;
    }
    const uint64_t pertokenScaleDimNum = pertokenScaleShape->GetStorageShape().GetDimNum();
    // perblock 场景
    if (isPerBlock_) {
        return CheckPerblockShape(mValue, kValue);
    }

    OP_LOGD(opName_, "Dim of pertokenScale(x1Scale) is %lu.", pertokenScaleDimNum);

    // x1为三维时GetMValue获取的mValue是m轴与batch的乘积，需要除以batch得到实际的m轴
    if (scenario_ == AllReduceScenario::MXFP4) {
        uint64_t mOfx1 = mValue / GetBatchValue();
        return CheckMXScenarioScaleShape(mOfx1, kValue, pertokenScaleShape, true, true);
    } else if (scenario_ == AllReduceScenario::MXFP8) {
        uint64_t mOfx1 = mValue / GetBatchValue();
        return CheckMXScenarioScaleShape(mOfx1, kValue, pertokenScaleShape, true, false);
    }

    OP_TILING_CHECK(pertokenScaleDimNum > DIM_NUM_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "pertokenScale(x1Scale)",
                                                             (std::to_string(pertokenScaleDimNum) + "D").c_str(),
                                                             "The shape dim of pertokenScale(x1Scale) must be 1"),
                    return false);

    const auto pertokenScaleShapeSize = static_cast<size_t>(pertokenScaleShape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(
        pertokenScaleShapeSize != mValue,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "pertokenScale(x1Scale)",
                                              Ops::Base::ToString(pertokenScaleShape->GetStorageShape()).c_str(),
                                              "The shape of pertokenScale(x1Scale) must be [m]"),
        return false);
    return true;
}

bool MatmulAllReduceTilingBase::CheckCommQuantScaleShape(const uint64_t nValue) const
{
    const auto commQuantScale1Shape = mmrCtxInfo_.comm_quant_scale_1_shape;
    const auto commQuantScale2Shape = mmrCtxInfo_.comm_quant_scale_2_shape;
    if ((commQuantScale1Shape == nullptr) && (commQuantScale2Shape == nullptr)) {
        return true;
    }
    OP_TILING_CHECK((commQuantScale1Shape == nullptr) || (commQuantScale2Shape == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(opName_, "comm_quant_scale_1/comm_quant_scale_2"), return false);

    uint64_t commQuantScaleOneDimNum = commQuantScale1Shape->GetStorageShape().GetDimNum();
    uint64_t commQuantScaleTwoDimNum = commQuantScale2Shape->GetStorageShape().GetDimNum();
    OP_LOGD(opName_, "Dim of comm_quant_scale_1 and comm_quant_scale_2 is %lu and %lu.", commQuantScaleOneDimNum,
            commQuantScaleTwoDimNum);
    OP_TILING_CHECK(
        (commQuantScaleOneDimNum > DIM_NUM_TWO) || (commQuantScaleTwoDimNum > DIM_NUM_TWO),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            opName_, "comm_quant_scale",
            (std::to_string(commQuantScaleOneDimNum) + "D/" + std::to_string(commQuantScaleTwoDimNum) + "D").c_str(),
            "The shape dim of comm_quant_scale must be 1 or 2"),
        return false);

    const auto commQuantScaleShapeSize1 = static_cast<size_t>(commQuantScale1Shape->GetStorageShape().GetShapeSize());
    const auto commQuantScaleShapeSize2 = static_cast<size_t>(commQuantScale2Shape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(
        (commQuantScaleShapeSize1 != nValue) || (commQuantScaleShapeSize2 != nValue),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "comm_quant_scale",
                                              (Ops::Base::ToString(commQuantScale1Shape->GetStorageShape()) + "/" +
                                               Ops::Base::ToString(commQuantScale2Shape->GetStorageShape()))
                                                  .c_str(),
                                              "The shape of comm_quant_scale must be [n]"),
        return false);
    return true;
}

bool MatmulAllReduceTilingBase::CheckAntiQuantScaleShape(const uint64_t kValue, const uint64_t nValue)
{
    const auto scale = mmrCtxInfo_.antiquant_scale_shape;
    if (scale == nullptr) {
        OP_LOGD(context_->GetNodeName(), "No antiquantScale.");
        return false;
    }
    // 校验scale维度
    int32_t twoDim = 2;
    int32_t scaleShapeDim = scale->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((scaleShapeDim != 1) && (scaleShapeDim != twoDim),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "antiquantScale",
                                                             ("dim=" + std::to_string(scaleShapeDim)).c_str(),
                                                             "The shape dim of antiquantScale must be 1 or 2"),
                    return false);
    const auto scaleShapeSize = static_cast<size_t>(scale->GetStorageShape().GetShapeSize());
    OP_LOGD(context_->GetNodeName(), "ScaleShapeSize %lu, antiGroupSize_ %lu", scaleShapeSize, antiGroupSize_);
    if (scaleShapeSize == 1) {
        OP_TILING_CHECK(antiGroupSize_ != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context_->GetNodeName(), "antiquantScale/antiGroupSize",
                            ("size=1, antiGroupSize=" + std::to_string(antiGroupSize_)).c_str(),
                            "When the shape size of antiquantScale is 1, the value of antiGroupSize must be 0"),
                        return false);
        return true;
    } else if (antiGroupSize_ > 0) {
        OP_TILING_CHECK(kValue < 33,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context_->GetNodeName(), "x1", ("k=" + std::to_string(kValue)).c_str(),
                            "The value of k of x1 must be greater than 33 in per-group mode"),
                        return false);
        return true;
    } else {
        OP_TILING_CHECK(scaleShapeSize != nValue,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "antiquantScale",
                                                              Ops::Base::ToString(scale->GetStorageShape()).c_str(),
                                                              "The shape of antiquantScale must be [1], [n] or [1,n]"),
                        return false);
        return true;
    }
    return false;
}

bool MatmulAllReduceTilingBase::CheckAntiQuantOffsetShape(const gert::StorageShape *scale) const
{
    const gert::StorageShape *antiquantOffset = mmrCtxInfo_.antiquant_offset_shape;
    if (antiquantOffset == nullptr) {
        return true;
    }
    int32_t offsetShapeDim = antiquantOffset->GetStorageShape().GetDimNum();
    int32_t scaleShapeDim = scale->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(offsetShapeDim != scaleShapeDim,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_->GetNodeName(), "antiquantOffset", (std::to_string(offsetShapeDim) + "D").c_str(),
                        "The shape dim of antiquantOffset must be equal to that of antiquantScale"),
                    return false);
    // 校验offset和x2最后一维是否一致
    for (int32_t i = 0; i < offsetShapeDim; ++i) {
        int64_t offsetValue = antiquantOffset->GetStorageShape().GetDim(i);
        int64_t scaleValue = scale->GetStorageShape().GetDim(i);
        OP_TILING_CHECK(offsetValue != scaleValue,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            context_->GetNodeName(), "antiquantOffset", std::to_string(offsetValue).c_str(),
                            "The value of antiquantOffset must be equal to that of antiquantScale"),
                        return false);
    }
    return true;
}

bool MatmulAllReduceTilingBase::CheckAntiQuantOffsetValid() const
{
    const auto scale = mmrCtxInfo_.antiquant_scale_shape;
    if (scale == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "antiquantScale");
        return false;
    }
    // 校验bias dim
    const gert::StorageShape *matrixBias = mmrCtxInfo_.bias_shape;
    int64_t nValue = GetNValue();
    if (matrixBias != nullptr) {
        int32_t biasShapeSize = matrixBias->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(biasShapeSize != 1,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "bias",
                                                                 (std::to_string(biasShapeSize) + "D").c_str(),
                                                                 "The shape dim of bias must be 1"),
                        return false);
        // 校验bias和x2最后一维是否一致
        OP_TILING_CHECK(matrixBias->GetStorageShape().GetDim(biasShapeSize - 1) != nValue,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context_->GetNodeName(), "bias",
                            std::to_string(matrixBias->GetStorageShape().GetDim(biasShapeSize - 1)).c_str(),
                            "The value of bias dimension must be equal to n"),
                        return false);
    }
    // 校验offset维度 与 scale一致
    return CheckAntiQuantOffsetShape(scale);
}

bool MatmulAllReduceTilingBase::CheckA16W4Shape(const uint64_t kValue, const uint64_t nValue)
{
    uint64_t innerN = (MutableRCSTilingData().isTransposeB != 0) ? kValue : nValue;
    OP_TILING_CHECK((innerN & 1) != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName_, "x2", ("k=" + std::to_string(kValue) + ", n=" + std::to_string(nValue)).c_str(),
                        "When the dtype of input is int4, the inner dimension of x2 must be even"),
                    return false);
    return true;
}

bool MatmulAllReduceTilingBase::CheckPlatformInfo() const
{
    if (isA16W8_ || isA16W4_) {
        OP_TILING_CHECK(
            supportL0c2Out_ && ((args_.mValue < MIN_SHAPE_DIM) ||
                                (args_.kValue != 0 && (args_.kValue > MAX_SHAPE_DIM || args_.kValue < MIN_SHAPE_DIM)) ||
                                (args_.nValue > MAX_SHAPE_DIM || args_.nValue < MIN_SHAPE_DIM)),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x1/x2",
                                                  ("m=" + std::to_string(args_.mValue) + ", k=" +
                                                   std::to_string(args_.kValue) + ", n=" + std::to_string(args_.nValue))
                                                      .c_str(),
                                                  "The value of MKN of x1 and x2 must be in the valid range"),
            return false);
    }
    return true;
}

AntiQuantType MatmulAllReduceTilingBase::GetAntiQuantType()
{
    const auto scale = mmrCtxInfo_.antiquant_scale_shape;
    if (scale == nullptr) {
        OP_LOGD(context_->GetNodeName(), "No anti quant scale.");
        return AntiQuantType::NONE;
    }

    args_.antiquantscaleDType = mmrCtxInfo_.antiquant_scale->GetDataType();
    const auto scaleShapeSize = static_cast<size_t>(scale->GetStorageShape().GetShapeSize());
    OP_LOGD(context_->GetNodeName(), "Scale shape size %zu antiGroupSize_ %zu.", scaleShapeSize, antiGroupSize_);
    if (scaleShapeSize == 1) {
        return AntiQuantType::PER_TENSOR;
    } else if (antiGroupSize_ > 0) {
        return AntiQuantType::PER_GROUP;
    } else {
        return AntiQuantType::PER_CHANNEL;
    }
    return AntiQuantType::NONE;
}

void MatmulAllReduceTilingBase::CalcUbTiling()
{
    const int64_t *commQuantModePtr = mmrCtxInfo_.commQuantModePtr;
    bool isPertile = false;
    if (commQuantModePtr != nullptr) {
        isPertile = (*commQuantModePtr == 1);
    }
    uint32_t addX3UbBufFac = ((args_.geCType == ge::DT_BF16) && (npuArch_ != NpuArch::DAV_3510)) || isPertile ?
                                 ADD_X3_BF16_UB_BUF_FACTOR :
                                 ADD_X3_FP16_UB_BUF_FACTOR;
    addX3UbBufFac *= isPertile ? sizeof(float) : D_MTYPE_SIZE_MAP.at(args_.cType);
    uint32_t addX3UbCnt =
        mc2tiling::AlignDown(static_cast<uint32_t>((aicoreParams_.ubSize) / addX3UbBufFac), ALIGN_DATA_SIZE);
    OP_LOGD(context_->GetNodeName(), "The addX3UbCnt=%u, aicoreParams_ubSize=%lu, addX3UbBufFac=%u.", addX3UbCnt,
            aicoreParams_.ubSize, addX3UbBufFac);
    MutableRCSTilingData().addX3UbCnt = addX3UbCnt;

    // biasUbCnt 计算（bias 移至 vec epilogue 累加）
    // bufFactor 取决于实际 UB buffer 分配数量（kernel 侧 InitBuffer）：
    //   仅有 bias: inQueueX(double-buffer=2) + biasBuf(1) + outQueueZ(double-buffer=2) = 5
    //   bias + x3: 上述 + inQueueY(double-buffer=2) = 7
    // bias 走 TBuf（每个 N-chunk 搬入一次，多行复用），arch35 原生 FP16/BF16 Add，不走 fp32 cast
    if (biasOnVec_) {
        constexpr uint32_t DOUBLE_BUFFER_CNT = 2;
        constexpr uint32_t BIAS_BUF_CNT = 1;
        uint32_t biasBufFactor = DOUBLE_BUFFER_CNT + BIAS_BUF_CNT + DOUBLE_BUFFER_CNT; // inX + bias + outZ
        if (MutableRCSTilingData().isAdd) {
            biasBufFactor += DOUBLE_BUFFER_CNT; // inY for x3
        }
        uint32_t biasDtypeSize = D_MTYPE_SIZE_MAP.at(args_.cType);
        biasBufFactor *= biasDtypeSize;
        biasUbCnt_ =
            mc2tiling::AlignDown(static_cast<uint32_t>((aicoreParams_.ubSize) / biasBufFactor), ALIGN_DATA_SIZE);
        // DataCopyParams.blockLen 为 uint16_t，biasUbCnt * sizeof(T) 不可超过 65535
        constexpr uint32_t DATACOPY_BLOCKLEN_MAX = 65535;
        uint32_t maxUbCntByBlockLen = DATACOPY_BLOCKLEN_MAX / biasDtypeSize;
        if (biasUbCnt_ > maxUbCntByBlockLen) {
            biasUbCnt_ = mc2tiling::AlignDown(maxUbCntByBlockLen, ALIGN_DATA_SIZE);
        }
        OP_LOGD(context_->GetNodeName(), "The biasUbCnt=%u, aicoreParams_ubSize=%lu, biasBufFactor=%u.", biasUbCnt_,
                aicoreParams_.ubSize, biasBufFactor);
    }
}
ge::graphStatus MatmulAllReduceTilingBase::AnalyzeShapeAttr()
{
    opName_ = context_->GetNodeName();
    OP_TILING_CHECK(!AnalyzeAttrs() || !AnalyzeInputs(), OP_LOGE(opName_, "Failed to analyze context info."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
void MatmulAllReduceTilingBase::PrintTilingData()
{
    if (MutableRCSTilingData().rankID != 0) {
        return;
    }
    PrintRCSTilingData(context_->GetNodeName(), MutableRCSTilingData());
    PrintExtendMatmulTiling(false);
    PrintTCubeTilingData(context_->GetNodeName(), MutableTCubeTileTilingData());
    PrintMc2MsgData(context_->GetNodeName(), MutableMc2MsgData());
    if (MutableRCSTilingData().tailM <= 0) {
        return;
    }
    OP_LOGD(opName_, "Have tail.");
    PrintExtendMatmulTiling(true);
    PrintTCubeTilingData(context_->GetNodeName(), MutableTCubeTailTilingData());
}
ge::graphStatus MatmulAllReduceTilingBase::GetAndConvertCommMode(uint8_t &commMode) const
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "comm_mode"), return ge::GRAPH_FAILED);
    const char *commModeStr = attrs->GetAttrPointer<char>(ATTR_COMM_MODE_INDEX);
    if (commModeStr == nullptr) {
        commMode = Mc2Comm::COMM_MODE_AICPU;
        OP_LOGI(opName_, "The input attr comm_mode is nullptr, commMode will use default AICPU.");
        return ge::GRAPH_SUCCESS;
    }
    if (std::strcmp(commModeStr, "ai_cpu") == 0) {
        commMode = Mc2Comm::COMM_MODE_AICPU;
        OP_LOGI(opName_, "The input attr comm_mode is ai_cpu, commMode will be set to AICPU.");
    } else if (std::strcmp(commModeStr, "ccu") == 0) {
        commMode = Mc2Comm::COMM_MODE_CCU;
        OP_LOGI(opName_, "The input attr comm_mode is ccu, commMode will be set to CCU.");
    } else {
        OP_LOGE_WITH_INVALID_ATTR(opName_, "comm_mode", commModeStr, "'ai_cpu' or 'ccu'");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
uint8_t MatmulAllReduceTilingBase::GetCommMode() const
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "comm_mode"),
                    return Mc2Comm::COMM_MODE_AICPU);
    const char *commModeStr = attrs->GetAttrPointer<char>(ATTR_COMM_MODE_INDEX);
    if (commModeStr == nullptr) {
        OP_LOGI(opName_, "TilingKey Do AICPU");
        return Mc2Comm::COMM_MODE_AICPU;
    }
    if (std::strcmp(commModeStr, "ccu") == 0) {
        OP_LOGI(opName_, "TilingKey Do CCU");
        return Mc2Comm::COMM_MODE_CCU;
    }
    OP_LOGI(opName_, "TilingKey Do AICPU");
    return Mc2Comm::COMM_MODE_AICPU;
}
ge::graphStatus MatmulAllReduceTilingBase::CheckCommModeA2() const
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "comm_mode"), return ge::GRAPH_FAILED);
    const char *commModeStr = attrs->GetAttrPointer<char>(ATTR_COMM_MODE_INDEX);
    if (commModeStr == nullptr || std::strcmp(commModeStr, "ai_cpu") == 0) {
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE_WITH_INVALID_ATTR(opName_, "comm_mode", commModeStr, "nullptr or 'ai_cpu'");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
