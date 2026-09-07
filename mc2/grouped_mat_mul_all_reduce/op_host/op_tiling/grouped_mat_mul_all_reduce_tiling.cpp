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
 * \file grouped_mat_mul_all_reduce_tiling.cpp
 * \brief
 */
#include "grouped_mat_mul_all_reduce_tiling.h"

#include <climits>

#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr int64_t BEST_L1_PARTA = 256 * 1024;
constexpr int64_t BEST_L1_PARTB = 128 * 1024;
constexpr int64_t BEST_BASEN = 256;
constexpr uint32_t UB_DIVIDE_NUM = 2;
constexpr uint32_t UB_CALSIZE_PER_BLOCK = 16 * 1024;
constexpr uint64_t TILING_KEY = 0;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t MAX_TURN_NUM = 16;
constexpr int32_t MAX_BASE_K = 128;
constexpr uint64_t COMM_TILE = 8;            // 每卡数据分配几次计算
constexpr uint64_t GMM_ALL_REDUCE_FLAG = 99; // 99 grouped_matmul_allreduce hccl flag
constexpr uint64_t Y_INDEX = 4;

static inline uint32_t SixteenAlign(uint32_t a, bool up = false)
{
    if (up) {
        a += 15; // 15: 16 bytes up-align
    }
    return a & ~15; // ~15: 16 bytes down-align
}

static inline uint32_t Ceil(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

struct CoreTilingInfo {
    uint32_t splitM;
    uint32_t splitN;
    uint32_t lambdaM;
    uint32_t lambdaN;
    uint32_t singleM;
    uint32_t singleN;
    uint32_t tileTurnNum;
    uint32_t tailTurnNum;
};

struct HcclAicpuOpParam {
    uint8_t res[64]; // 64:
};

struct KFCMsgBody {
    // Rank* aiv * MsgSize * sizeof(消息)
    HcclAicpuOpParam msgSndArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
    HcclAicpuOpParam msgRcvArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
};

static uint64_t GMMGetSizePlatForm(const platform_ascendc::CoreMemType memType,
                                   platform_ascendc::PlatformAscendC ascendcPlatform)
{
    uint64_t size = 0;
    ascendcPlatform.GetCoreMemSize(memType, size);
    return size;
}

struct PlatFormMemSize {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;

    explicit PlatFormMemSize(platform_ascendc::PlatformAscendC ascendcPlatform)
        : ubSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::UB, ascendcPlatform)),
          l1Size(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L1, ascendcPlatform)),
          l0CSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_C, ascendcPlatform)),
          l0ASize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_A, ascendcPlatform)),
          l0BSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_B, ascendcPlatform))
    {}
};

static void PrintTilingData(optiling::Mc2Msg &msg)
{
    OP_LOGD("GMMAllReduce", " aicpuTiling.sendOff %lu.", msg.get_sendOff());
    OP_LOGD("GMMAllReduce", " aicpuTiling.recvOff %lu.", msg.get_recvOff());
    OP_LOGD("GMMAllReduce", " aicpuTiling.tailSendOff %lu.", msg.get_tailSendOff());
    OP_LOGD("GMMAllReduce", " aicpuTiling.tailRecvOff %lu.", msg.get_tailRecvOff());
    OP_LOGD("GMMAllReduce", " aicpuTiling.sendCnt %lu.", msg.get_sendCnt());
    OP_LOGD("GMMAllReduce", " aicpuTiling.recvCnt %lu.", msg.get_recvCnt());
    OP_LOGD("GMMAllReduce", " aicpuTiling.tailSendCnt %lu.", msg.get_tailSendCnt());
    OP_LOGD("GMMAllReduce", " aicpuTiling.tailRecvCnt %lu.", msg.get_tailRecvCnt());
    OP_LOGD("GMMAllReduce", " aicpuTiling.totalCnt %lu.", msg.get_totalCnt());
    OP_LOGD("GMMAllReduce", " aicpuTiling.turnNum %u.", msg.get_turnNum());
    OP_LOGD("GMMAllReduce", " aicpuTiling.tailNum %u.", msg.get_tailNum());
    OP_LOGD("GMMAllReduce", " aicpuTiling.stride %u.", msg.get_stride());
    OP_LOGD("GMMAllReduce", " aicpuTiling.workspaceOff %u.", msg.get_workspaceOff());
    OP_LOGD("GMMAllReduce", " aicpuTiling.notifyOff %u.", msg.get_notifyOff());

    OP_LOGD("GMMAllReduce", " aicpuTiling.notifyBeginCnt %u.", msg.get_notifyBeginCnt());
    OP_LOGD("GMMAllReduce", " aicpuTiling.notifyEndCnt %u.", msg.get_notifyEndCnt());
    OP_LOGD("GMMAllReduce", " aicpuTiling.useBufferType %u.", msg.get_useBufferType());
    OP_LOGD("GMMAllReduce", " aicpuTiling.funID %u.", msg.get_funID());
    OP_LOGD("GMMAllReduce", " aicpuTiling.dataType %u.", msg.get_dataType());
    OP_LOGD("GMMAllReduce", " aicpuTiling.groupNum %u.", msg.get_groupNum());

    OP_LOGD("GMMAllReduce", " aicpuTiling.reuseMode %u.", msg.get_reuseMode());
    OP_LOGD("GMMAllReduce", " aicpuTiling.commType %u.", msg.get_commType());
    OP_LOGD("GMMAllReduce", " aicpuTiling.reduceOp %u.", msg.get_reduceOp());
    OP_LOGD("GMMAllReduce", " aicpuTiling.commOrder %u.", msg.get_commOrder());
    OP_LOGD("GMMAllReduce", " aicpuTiling.waitPolicy %u.", msg.get_waitPolicy());
    OP_LOGD("GMMAllReduce", " aicpuTiling.rspPolicy %u.", msg.get_rspPolicy());
    OP_LOGD("GMMAllReduce", " aicpuTiling.exitPolicy %u.", msg.get_exitPolicy());
    OP_LOGD("GMMAllReduce", " aicpuTiling.commAlg %u.", msg.get_commAlg());
    OP_LOGD("GMMAllReduce", " aicpuTiling.taskType %u.", msg.get_taskType());
    OP_LOGD("GMMAllReduce", " aicpuTiling.preparePosition %u", msg.get_preparePosition());
}

static void PrintTilingData(optiling::GMMAllReduceCoreTiling &msg)
{
    OP_LOGD("GMMAllReduce", " aicoreTiling.notifyOff %u.", msg.get_notifyOff());
    OP_LOGD("GMMAllReduce", " aicoreTiling.baseParams.groupNum %u.", msg.baseParams.get_groupNum());
    OP_LOGD("GMMAllReduce", " aicoreTiling.baseParams.coreNum %u.", msg.baseParams.get_coreNum());
    OP_LOGD("GMMAllReduce", " aicoreTiling.baseParams.workspaceSize %u.", msg.baseParams.get_workspaceSize());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.M %u.", msg.mmTilingData.get_M());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.N %u.", msg.mmTilingData.get_N());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.Ka %u.", msg.mmTilingData.get_Ka());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.Kb %u.", msg.mmTilingData.get_Kb());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.singleCoreM %u.", msg.mmTilingData.get_singleCoreM());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.singleCoreN %u.", msg.mmTilingData.get_singleCoreN());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.singleCoreK %u.", msg.mmTilingData.get_singleCoreK());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.baseM %u.", msg.mmTilingData.get_baseM());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.baseN %u.", msg.mmTilingData.get_baseN());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.baseK %u.", msg.mmTilingData.get_baseK());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.depthA1 %u.", msg.mmTilingData.get_depthA1());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.depthB1 %u.", msg.mmTilingData.get_depthB1());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.stepM %u.", msg.mmTilingData.get_stepM());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.stepN %u.", msg.mmTilingData.get_stepN());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.isBias %u.", msg.mmTilingData.get_isBias());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.transLength %u.", msg.mmTilingData.get_transLength());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.iterateOrder %u.", msg.mmTilingData.get_iterateOrder());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.shareL1Size %u.", msg.mmTilingData.get_shareL1Size());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.shareL0CSize %u.", msg.mmTilingData.get_shareL0CSize());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.stepKa %u.", msg.mmTilingData.get_stepKa());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.stepKb %u.", msg.mmTilingData.get_stepKb());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.dbL0A %u.", msg.mmTilingData.get_dbL0A());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.dbL0B %u.", msg.mmTilingData.get_dbL0B());
    OP_LOGD("GMMAllReduce", " aicoreTiling.mmTilingData.dbL0C %u.", msg.mmTilingData.get_dbL0C());
}

class GMMAllReduceTiling {
public:
    GMMAllReduceTilingData tilingData;
    mc2tiling::TilingArgs args_;

    ge::graphStatus Init(const gert::TilingContext *context);
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext *context);

protected:
    ge::graphStatus CalMMTiling(const gert::TilingContext *context);
    ge::graphStatus GMMAllReduceSetMMTiling(const gert::TilingContext *context, matmul_tiling::DataType matmulDtype);
    ge::graphStatus DoAllReduceTiling();
    ge::graphStatus DoAiCoreTiling(const gert::TilingContext *context);
    ge::graphStatus SetBaseParams();
    inline void SetArgsValue(uint32_t i);
    ge::graphStatus InitForLoop(const gert::TilingContext *context, uint32_t &groupNum);
    inline void SetMsgValue(Mc2Msg &msg) const;
    void CalculateMMTiling(CoreTilingInfo &coreTilingInfo) const;

private:
    int32_t mList[MAX_TENSOR_CONT] = {0};
    int32_t kList[MAX_TENSOR_CONT] = {0};
    int32_t nList[MAX_TENSOR_CONT] = {0};
    uint32_t ubBaseM;
    uint32_t ubBaseN;
    int32_t maxM;
    int32_t maxN;
    int32_t maxK;
    int32_t baseM_;
    int32_t baseN_;
    int32_t baseK_;
    uint64_t ubSize_;
    uint32_t mmDataTypeSize;
    bool isBias;

    ge::DataType mmDType;
    ge::DataType weightDtype;
    ge::DataType biasDType;
    uint8_t hcclDataType;  // data type in hccl
    int64_t splitItem_;    // attr
    const char *group_;    // attr
    const char *reduceOp_; // attr
    int64_t commTurn_;     // communication turn num
    uint32_t rankSize_;
    uint32_t libApiWorkSpaceSize_;
    const char *opName;
};

ge::graphStatus GMMAllReduceTiling::InitForLoop(const gert::TilingContext *context, uint32_t &groupNum)
{
    // Get shape, dtype information, and the total number of data.
    int64_t maxMKN = 0;
    for (uint32_t i = 0; i < MAX_TENSOR_CONT; ++i) {
        auto xTensor = context->GetDynamicInputTensor(X_INDEX, i);
        auto weightTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, i);
        if (weightTensor == nullptr) {
            break;
        }
        groupNum++;
        if (xTensor == nullptr) {
            continue;
        }
        gert::Shape xShape = xTensor->GetStorageShape();
        gert::Shape wShape = weightTensor->GetStorageShape();
        // calc bs
        int64_t bs = xShape.GetDim(0); // 0: x first dim
        size_t xDimNum = xShape.GetDimNum();
        size_t bsDimNum = xDimNum >= 1 ? xDimNum - 1 : 0; // 1: x last dim k, the other dimensions are bs
        for (size_t j = 1; j < bsDimNum; ++j) {
            bs *= xShape.GetDim(j);
        }
        // Determine whether all data types are consistent.
        if (mmDType == ge::DT_UNDEFINED) { // if is first loop
            auto inputDescPtr0 = context->GetInputDesc(0);
            auto inputDescPtr1 = context->GetInputDesc(1);
            OP_TILING_CHECK(inputDescPtr0 == nullptr || inputDescPtr1 == nullptr,
                            OP_LOGE_WITH_INVALID_INPUT(opName, "inputDescPtr"), return ge::GRAPH_FAILED);
            mmDType = inputDescPtr0->GetDataType();     // save x dtype
            weightDtype = inputDescPtr1->GetDataType(); // save weight dtype
            mmDataTypeSize = GetSizeByDataType(mmDType);
            OP_TILING_CHECK(
                mmDataTypeSize == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, "mmDType", TypeUtils::DataTypeToAscendString(mmDType).GetString(), "get size is 0"),
                return ge::GRAPH_FAILED);
            uint32_t numInOneBlk = std::max<uint32_t>(1, ONE_BLK_SIZE / mmDataTypeSize);
            maxMKN = INT_MAX / numInOneBlk * numInOneBlk;
        }
        OP_TILING_CHECK(bs > maxMKN || xShape.GetDim(xDimNum - 1) > maxMKN || wShape.GetDim(1) > maxMKN,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "shape dimension", std::to_string(maxMKN).c_str(),
                                                              "32B-aligned m, n or k axis is out of int32 range"),
                        return ge::GRAPH_FAILED);
        mList[i] = static_cast<int32_t>(bs);                         // m axis of x
        kList[i] = static_cast<int32_t>(xShape.GetDim(xDimNum - 1)); // x shape is [m, k], k index is xDimNum - 1
        nList[i] = static_cast<int32_t>(wShape.GetDim(1));           // w shape is [k, n], n index is 1
        maxM = std::max(maxM, mList[i]);
        maxK = std::max(maxK, kList[i]);
        maxN = std::max(maxN, nList[i]);
    }
    if (mmDType == ge::DT_BF16) {
        hcclDataType = static_cast<uint8_t>(mc2tiling::HcclDataType::HCCL_DATA_TYPE_BFP16);
    } else {
        hcclDataType = static_cast<uint8_t>(mc2tiling::HcclDataType::HCCL_DATA_TYPE_FP16);
    }
    OP_TILING_CHECK(groupNum > MAX_TENSOR_CONT,
                    OP_LOGE_FOR_INVALID_VALUE(opName, "groupNum", std::to_string(groupNum).c_str(),
                                              std::to_string(MAX_TENSOR_CONT).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMAllReduceTiling::Init(const gert::TilingContext *context)
{
    opName = context->GetNodeName();
    uint32_t groupNum = 0;          // 0: init value
    maxM = 0;                       // init maxM
    maxN = 0;                       // init maxN
    maxK = 0;                       // init maxK
    mmDType = ge::DT_UNDEFINED;     // init mmDType
    weightDtype = ge::DT_UNDEFINED; // init weightDtype
    OP_TILING_CHECK(InitForLoop(context, groupNum) != ge::GRAPH_SUCCESS, OP_LOGE(opName, "InitForLoop failed."),
                    return ge::GRAPH_FAILED);
    auto biasPtr = context->GetDynamicInputTensor(BIAS_INDEX, 0); // 0: 获取tensorList中第一个tensor
    isBias = biasPtr != nullptr && biasPtr->GetStorageShape().GetDimNum() != 0;
    tilingData.aicoreTiling.baseParams.set_mList(mList);
    tilingData.aicoreTiling.baseParams.set_kList(kList);
    tilingData.aicoreTiling.baseParams.set_nList(nList);
    tilingData.aicoreTiling.baseParams.set_groupNum(groupNum);
    OP_LOGD(opName, "groupNum is %u, isBias is %d, maxM is %d, maxK is %d, maxN is %d.", groupNum,
            static_cast<int>(isBias), maxM, maxK, maxN);

    uint32_t index = 0; // 0: start from 0
    splitItem_ = *context->GetAttrs()->GetAttrPointer<int64_t>(index++);
    group_ = context->GetAttrs()->GetAttrPointer<char>(index++);
    reduceOp_ = context->GetAttrs()->GetAttrPointer<char>(index++);
    commTurn_ = *context->GetAttrs()->GetAttrPointer<int>(index++);
    if (commTurn_ == 0) {
        commTurn_ = COMM_TILE;
    } // set default value
    rankSize_ = mc2tiling::MatmulFormulaicTiling::GetRankSize(group_);
    OP_LOGD(opName, "splitItem is %ld, group is %s, rankSize_=%u, reduceOp_ is %s, commTurn is %ld.", splitItem_,
            group_, rankSize_, reduceOp_, commTurn_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMAllReduceTiling::RunFusionKernelTiling(gert::TilingContext *context)
{
    OP_LOGD(opName, "begin RunFusionKernelTiling.");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    static const uint32_t CORE_NUM = ascendcPlatform.GetCoreNumAiv();
    static const uint32_t AIC_NUM = ascendcPlatform.GetCoreNumAic();
    static const uint32_t AIV_NUM = ascendcPlatform.GetCoreNumAiv();
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);
    static const platform_ascendc::SocVersion SOC_VERSION = ascendcPlatform.GetSocVersion();
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    OP_TILING_CHECK(
        (CORE_NUM == 0 || AIV_NUM == 0 || PLATFORM_SIZE.ubSize == 0 || PLATFORM_SIZE.l1Size == 0 ||
         PLATFORM_SIZE.l0CSize == 0 || PLATFORM_SIZE.l0ASize == 0 || PLATFORM_SIZE.l0BSize == 0),
        OP_LOGE(
            opName,
            "platform[%d] info is invalid, coreNum=%u, aivNum=%u, ubSize=%lu, l1Size=%lu, l0CSize=%lu, l0ASize=%lu, "
            "l0BSize=%lu",
            static_cast<int>(SOC_VERSION), CORE_NUM, AIV_NUM, PLATFORM_SIZE.ubSize, PLATFORM_SIZE.l1Size,
            PLATFORM_SIZE.l0CSize, PLATFORM_SIZE.l0ASize, PLATFORM_SIZE.l0BSize),
        return ge::GRAPH_FAILED);
    bool rankSizeSupported = (rankSize_ == 1 || rankSize_ == 2 || rankSize_ == 4 || rankSize_ == 8);
    OP_TILING_CHECK(!rankSizeSupported,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "rankSize", std::to_string(rankSize_).c_str(),
                                                          "The value of rankSize must be 1, 2, 4 or 8"),
                    return ge::GRAPH_FAILED);

    ubSize_ = PLATFORM_SIZE.ubSize;
    args_.cmdType = mc2tiling::AicpuComType::HCCL_CMD_ALLREDUCE; // all reduce
    args_.commTurn = commTurn_;                                  // communication turn num
    args_.enablePad = true;
    args_.enableSplitK = false;
    args_.inputDtypeSize = mmDataTypeSize;
    args_.isATrans = false;
    args_.isBias = isBias;
    args_.isBTrans = false;
    args_.isStorageGather = false;
    args_.outputDtypeSize = mmDataTypeSize;
    args_.aicCoreNum = AIC_NUM;
    args_.rankDim = rankSize_;
    args_.rankTileNum = 1;       // 1: tile num
    args_.usedCoreNum = AIC_NUM; // cube num

    // aicore tiling
    OP_TILING_CHECK(DoAiCoreTiling(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName, "GMM_All_Reduce DoAiCoreTiling failed."), return ge::GRAPH_FAILED);

    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(CORE_NUM, AIC_NUM, AIV_NUM));
    // aicpu tiling
    OP_TILING_CHECK(DoAllReduceTiling() != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName, "GMM_All_Reduce DoAllReduceTiling failed."), return ge::GRAPH_FAILED);

    // set workspsces
    size_t *workspaces = context->GetWorkspaceSizes(1); // 1: fixed value
    workspaces[0] = SYS_WORKSPACE_SIZE;                 // 0: fixed value
    context->SetTilingKey(TILING_KEY);                  // set tilingkey
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    OP_LOGD(opName, "end RunFusionKernelTiling.");
    return ge::GRAPH_SUCCESS;
}

inline void GMMAllReduceTiling::SetArgsValue(uint32_t i)
{
    args_.kValue = kList[i];
    args_.mValue = mList[i];
    args_.nValue = nList[i];
    args_.orgKValue = kList[i];
    args_.orgMValue = mList[i];
    args_.orgNValue = nList[i];
}

inline void GMMAllReduceTiling::SetMsgValue(Mc2Msg &msg) const
{
    msg.set_debugMode(mc2tiling::Mc2TilingUtils::GetDebugMode());
    msg.set_commType(static_cast<uint8_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLREDUCE));
    msg.set_reduceOp(static_cast<uint8_t>(mc2tiling::HcclReduceOp::HCCL_REDUCE_SUM));
    msg.set_stride(0);     // 跳写间隔
    msg.set_waitPolicy(1); // 1: fixed value
    msg.set_rspPolicy(1);  // 1: fixed value
    msg.set_exitPolicy(0); // 0: fixed value
    msg.set_commAlg(0);    // 0: fixed value
    msg.set_taskType(static_cast<uint8_t>(mc2tiling::KfcTaskType::KFC_TASK_HCC_TASK_DELIVER));
    msg.set_commOrder(1); // 0 先AiCPU后MM; 1为先MM后AICPU
    msg.set_funID(mc2tiling::ALL_REDUCE_FUNC_ID);
    msg.set_preparePosition(0);
}

void GMMAllReduceTiling::CalculateMMTiling(CoreTilingInfo &coreTilingInfo) const
{
    coreTilingInfo.lambdaN = Ceil(uint32_t(args_.orgNValue), args_.usedCoreNum * uint32_t(baseN_));
    coreTilingInfo.singleN = coreTilingInfo.lambdaN * baseN_;
    uint32_t numBlocksN = Ceil(uint32_t(args_.orgNValue), coreTilingInfo.singleN);
    if (numBlocksN == 0) {
        numBlocksN = 1;
    }
    coreTilingInfo.splitM = (args_.usedCoreNum % numBlocksN == 0) ? args_.usedCoreNum / numBlocksN : 1;
    coreTilingInfo.lambdaM = Ceil(uint32_t(args_.orgMValue), MAX_TURN_NUM * coreTilingInfo.splitM * uint32_t(baseM_));
    coreTilingInfo.singleM = coreTilingInfo.lambdaM * baseM_;
    coreTilingInfo.tileTurnNum =
        std::max<uint32_t>(1, args_.orgMValue / (coreTilingInfo.singleM * coreTilingInfo.splitM));
    coreTilingInfo.tailTurnNum =
        (coreTilingInfo.tileTurnNum * coreTilingInfo.singleM * coreTilingInfo.splitM < args_.orgMValue);
    OP_LOGI(opName,
            "m = %lu, n = %lu, baseM_ = %u, baseK_ = %u, baseN_ = %u, tileTurnNum = %u, tailTurnNum = %u,"
            "splitM = %u, lambdaM = %u, singleM = %u, numBlocksN = %u, lambdaN = %u, singleN = %u",
            args_.orgMValue, args_.orgNValue, baseM_, baseK_, baseN_, coreTilingInfo.tileTurnNum,
            coreTilingInfo.tailTurnNum, coreTilingInfo.splitM, coreTilingInfo.lambdaM, coreTilingInfo.singleM,
            numBlocksN, coreTilingInfo.lambdaN, coreTilingInfo.singleN);
}

ge::graphStatus GMMAllReduceTiling::DoAllReduceTiling()
{
    OP_LOGD(opName, "begin DoAllReduceTiling.");
    uint32_t groupNum = tilingData.aicoreTiling.baseParams.get_groupNum();
    tilingData.aicpuTiling.set_groupNum(groupNum);
    tilingData.aicpuTiling.set_groupTilingMagicNum(GMM_ALL_REDUCE_FLAG);

    uint8_t *argsU8 = tilingData.aicpuTiling.get_msg();
    Mc2Msg msg(argsU8);
    size_t dataSize = msg.GetDataSize(); // Mc2Msg size
    OP_TILING_CHECK(dataSize != MC2_MSG_SIZE,
                    OP_LOGE_FOR_INVALID_VALUE(opName, "dataSize", std::to_string(dataSize).c_str(),
                                              std::to_string(MC2_MSG_SIZE).c_str()),
                    return ge::GRAPH_FAILED);
    CoreTilingInfo coreTilingInfo;
    for (uint32_t i = 0; i < groupNum; ++i) {
        OP_LOGD(opName, "Start for loop in DoAllReduceTiling. i = %u.", i);
        msg.SetDataPtr(argsU8 + i * dataSize); // move ptr to i msg
        SetMsgValue(msg);                      // set msg some  fixed values
        SetArgsValue(i);                       // set args_ mkn values
        CalculateMMTiling(coreTilingInfo);
        msg.set_reuseMode(coreTilingInfo.tileTurnNum + coreTilingInfo.tailTurnNum); // workspace未复用，为总轮次
        uint32_t totalSingleM = coreTilingInfo.singleM * coreTilingInfo.splitM;
        uint32_t sendCnt = std::min<uint32_t>(args_.orgMValue, totalSingleM) * args_.orgNValue;
        msg.set_sendOff(sendCnt * args_.outputDtypeSize);
        msg.set_recvOff(sendCnt * args_.outputDtypeSize);
        msg.set_sendCnt(sendCnt);
        msg.set_recvCnt(sendCnt);
        // 通信公式化Tiling计算中，可能有尾块
        sendCnt =
            coreTilingInfo.tailTurnNum == 0 ?
                0 :
                (args_.orgMValue - std::min<uint32_t>(args_.orgMValue, totalSingleM * coreTilingInfo.tileTurnNum)) *
                    args_.orgNValue;
        msg.set_tailSendOff(sendCnt * args_.outputDtypeSize);
        msg.set_tailRecvOff(sendCnt * args_.outputDtypeSize);
        msg.set_tailSendCnt(sendCnt);
        msg.set_tailRecvCnt(sendCnt);

        msg.set_totalCnt(args_.orgMValue * args_.orgNValue);
        msg.set_turnNum(coreTilingInfo.tileTurnNum + coreTilingInfo.tailTurnNum); // 总轮次
        msg.set_tailNum(coreTilingInfo.tailTurnNum);                              // 尾块的轮次
        // workspace 地址
        msg.set_useBufferType(static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT));
        msg.set_workspaceOff(libApiWorkSpaceSize_);
        // 消息队列的开始  device notify write/read value偏移
        msg.set_notifyOff(sizeof(KFCMsgBody));
        msg.set_notifyBeginCnt(mc2tiling::NOTIFY_WRITE_CNT); // notify write value的使用个数
        msg.set_notifyEndCnt(1);                             // notify read value的使用个数
        msg.set_dataType(hcclDataType);                      // hccl 数据类型
        msg.set_sendArgIndex(0);
        msg.set_recvArgIndex(Y_INDEX);
        PrintTilingData(msg); // print msg data
    }
    OP_LOGD(opName, "end DoAllReduceTiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMAllReduceTiling::DoAiCoreTiling(const gert::TilingContext *context)
{
    OP_LOGD(opName, "begin DoAiCoreTiling.");

    // GMMAllReduceBaseParams baseParams
    OP_TILING_CHECK(SetBaseParams() != ge::GRAPH_SUCCESS, OP_LOGE(opName, "GMM_All_Reduce SetBaseParams failed."),
                    return ge::GRAPH_FAILED);

    // TCubeTiling mmTilingData
    OP_TILING_CHECK(CalMMTiling(context) != ge::GRAPH_SUCCESS, OP_LOGE(opName, "GMM_All_Reduce CalMMTiling failed."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        GMMAllReduceSetMMTiling(context, static_cast<matmul_tiling::DataType>(mmDType)) != ge::GRAPH_SUCCESS,
        OP_LOGE(opName, "GMM_All_Reduce GMMAllReduceSetMMTiling failed."), return ge::GRAPH_FAILED);

    tilingData.aicoreTiling.set_notifyOff(sizeof(KFCMsgBody)); // used in kernel function
    uint32_t debugMode = mc2tiling::Mc2TilingUtils::GetDebugMode();
    tilingData.aicoreTiling.set_debugMode(debugMode);
    PrintTilingData(tilingData.aicoreTiling);
    OP_LOGD(opName, "end DoAiCoreTiling");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMAllReduceTiling::SetBaseParams()
{
    OP_LOGD(opName, "begin SetBaseParams");
    tilingData.aicoreTiling.baseParams.set_coreNum(args_.aicCoreNum); // Set coreNum
    tilingData.aicoreTiling.baseParams.set_ubBaseM(maxM);             // Set baseM as maxM
    tilingData.aicoreTiling.baseParams.set_ubBaseN(maxN);             // Set baseN as maxN
    tilingData.aicoreTiling.baseParams.set_ubCalSize(UB_CALSIZE_PER_BLOCK);
    tilingData.aicoreTiling.baseParams.set_ubRestSize(UB_CALSIZE_PER_BLOCK);
    tilingData.aicoreTiling.baseParams.set_workspaceSize(SYS_WORKSPACE_SIZE);
    OP_LOGD(opName, "end SetBaseParams.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMAllReduceTiling::CalMMTiling(const gert::TilingContext *context)
{
    OP_LOGD(opName, "begin CalMMTlingData.");

    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);

    uint32_t baseN = BEST_BASEN;
    while (baseN > static_cast<uint32_t>(maxN)) {
        baseN = baseN >> 1;
    }
    if (baseN < static_cast<uint32_t>(maxN)) {
        baseN = baseN << 1;
    }
    baseN_ = std::min<int32_t>(BEST_BASEN, baseN);

    // 基于使能double buffer的L0B内存计算baseK
    baseK_ = (PLATFORM_SIZE.l0BSize / DOUBLE_BUFFER_L0A_L0B) / (baseN_ * mmDataTypeSize);
    baseK_ = SixteenAlign(baseK_);
    if (baseK_ > MAX_BASE_K) {
        baseK_ = MAX_BASE_K;
        int32_t maxBaseN = SixteenAlign(PLATFORM_SIZE.l0BSize / DOUBLE_BUFFER_L0A_L0B / (baseK_ * mmDataTypeSize));
        baseN_ = std::min<int32_t>(baseN_, maxBaseN);
        baseN_ = std::max<int32_t>(16, SixteenAlign(baseN_, true)); // 16: minimum value for baseN
    }
    if (baseK_ > maxK) {
        baseK_ = std::min<int32_t>(baseK_, SixteenAlign(maxK, true));
    }
    OP_TILING_CHECK(baseK_ == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "baseK_", std::to_string(baseK_).c_str(),
                                                          "The value of baseK_ cannot be 0"),
                    return ge::GRAPH_FAILED);
    // 基于使能double buffer的L0A内存和L0B内存计算baseM(cube)
    uint32_t maxBaseM = PLATFORM_SIZE.l0CSize / (baseN_ * sizeof(float));
    baseM_ = std::min<uint32_t>((PLATFORM_SIZE.l0ASize / DOUBLE_BUFFER_L0A_L0B) / (baseK_ * mmDataTypeSize), maxBaseM);
    baseM_ = SixteenAlign(baseM_);
    if (baseM_ > maxM) {
        baseM_ = SixteenAlign(maxM, true);
    }
    OP_TILING_CHECK(baseM_ == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "baseM_", std::to_string(baseM_).c_str(),
                                                          "The value of baseM_ cannot be 0"),
                    return ge::GRAPH_FAILED);
    OP_LOGD(opName, "end CalMMTlingData");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMAllReduceTiling::GMMAllReduceSetMMTiling(const gert::TilingContext *context,
                                                            matmul_tiling::DataType matmulDtype)
{
    OP_LOGD(opName, "Begin GMMAllReduceSetMMTiling.");

    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);

    matmul_tiling::MatmulApiTiling mm;
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, false);
    mm.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN,
                matmul_tiling::DataType::DT_FLOAT16);
    mm.SetBias(isBias);
    if (isBias) {
        mm.SetBiasType(
            matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
            static_cast<matmul_tiling::DataType>(context->GetDynamicInputTensor(BIAS_INDEX, 0)->GetDataType()));
    }
    mm.SetOrgShape(maxM, maxN, maxK);
    mm.SetShape(maxM, baseN_, maxK);
    mm.SetFixSplit(std::min(baseM_, maxM), baseN_);
    mm.SetBufferSpace(PLATFORM_SIZE.l1Size, PLATFORM_SIZE.l0CSize, ubSize_);
    OP_TILING_CHECK(mm.GetTiling(tilingData.aicoreTiling.mmTilingData) == -1,
                    OP_LOGE(context->GetNodeName(), "matmul getTiling failed."), return ge::GRAPH_FAILED);
    // 计算开启dublebuffer之后搬运至L1是所需的参数
    uint32_t mmStepKa = (BEST_L1_PARTB >> 1) / (baseM_ * baseK_ * mmDataTypeSize);
    uint32_t mmStepKb = (BEST_L1_PARTA >> 1) / (baseN_ * baseK_ * mmDataTypeSize);

    if (mmStepKa > mmStepKb) {
        mmStepKa = mmStepKa / mmStepKb * mmStepKb;
    } else if (mmStepKa < mmStepKb) {
        mmStepKb = mmStepKb / mmStepKa * mmStepKa;
    }
    uint32_t stepM = 1; // 1: stepM set fixed value 1
    uint32_t stepN = 1; // 1: stepN set fixed value 1
    uint32_t mmDepthA1 = mmStepKa * DOUBLE_BUFFER_STEPKA_STEPKB * stepM;
    uint32_t mmDepthB1 = mmStepKb * DOUBLE_BUFFER_STEPKA_STEPKB * stepN;

    tilingData.aicoreTiling.mmTilingData.set_usedCoreNum(args_.usedCoreNum);
    tilingData.aicoreTiling.mmTilingData.set_shareMode(0); // 0: no share mode
    tilingData.aicoreTiling.mmTilingData.set_shareL1Size(PLATFORM_SIZE.l1Size);
    tilingData.aicoreTiling.mmTilingData.set_shareL0CSize(PLATFORM_SIZE.l0CSize);
    tilingData.aicoreTiling.mmTilingData.set_dbL0C(1); // 1: no double buffer
    tilingData.aicoreTiling.mmTilingData.set_baseM(baseM_);
    tilingData.aicoreTiling.mmTilingData.set_baseN(baseN_);
    tilingData.aicoreTiling.mmTilingData.set_baseK(baseK_);
    tilingData.aicoreTiling.mmTilingData.set_stepKa(mmStepKa);
    tilingData.aicoreTiling.mmTilingData.set_depthA1(mmDepthA1);
    tilingData.aicoreTiling.mmTilingData.set_stepKb(mmStepKb);
    tilingData.aicoreTiling.mmTilingData.set_depthB1(mmDepthB1);
    tilingData.aicoreTiling.mmTilingData.set_stepM(stepM);
    tilingData.aicoreTiling.mmTilingData.set_stepN(stepN);

    OP_LOGD(opName, "End GMMAllReduceSetMMTiling.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingGMMAllReduce(gert::TilingContext *context)
{
    GMMAllReduceTiling tiling;
    OP_TILING_CHECK(tiling.Init(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "GMM tiling init failed."), return ge::GRAPH_FAILED);
    return tiling.RunFusionKernelTiling(context);
}

struct GMMAllReduceCompileInfo {};

static ge::graphStatus TilingPrepareForGMMAllReduce(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<GMMAllReduceCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatMulAllReduce)
    .Tiling(TilingGMMAllReduce)
    .TilingParse<GMMAllReduceCompileInfo>(TilingPrepareForGMMAllReduce); // 向框架注册入口函数
} // namespace optiling
