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
 * \file mc2_tiling_utils.h
 * \brief
 */

#ifndef __MC2_TILING_UTILS_H__
#define __MC2_TILING_UTILS_H__

#include <cstdint>
#include <map>
#include <string>

#include "exe_graph/runtime/tiling_context.h"
#include "formulaic_tiling_datatype.h"
#include "graph/utils/type_utils.h"
#include "mc2_hcom_topo_info.h"
#include "matmul_formulaic_tiling.h"
#include "platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_type.h"
#include "../../../3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_base_tiling_advanced.h"
#include "op_host/util/op_const_def.h"

namespace mc2tiling {
constexpr uint32_t STANDARD_CARD_4P = 4;
constexpr uint32_t EIGHT_P_8P = 8; // 8P场景（POD8P/Server8P）
constexpr uint32_t COMM_MESH = 0b1U;
constexpr uint32_t COMM_SWITCH = (COMM_MESH << 1U);
constexpr uint32_t COMM_RING = (COMM_MESH << 2U);
constexpr uint32_t COMM_PAIRWISE = (COMM_MESH << 3U);
constexpr uint32_t COMM_UNDEFINED = 0xFFFFFFFFU;
constexpr uint8_t COMM_ALG_FULL_MESH_HOST = 6;
constexpr uint64_t CHECK_VALUE_ODD = 2;
constexpr uint32_t AIC_NUM_950 = 32;
constexpr uint64_t MC2_TILINGKEY_OFFSET = uint64_t(1000000000000000000UL); // 10^18
constexpr size_t RES_LEN = 64;
constexpr size_t MAX_MSG_NUM = 16;
constexpr uint8_t MC2_DEBUG_ONLY_AICPU = 4; // 只通信不计算
constexpr char HCCL_DETERMINISTIC[] = "HCCL_DETERMINISTIC";
/**
当前通信API未提供枚举，后续会提供
0：默认值 1：HOST_TS（A2/3支持 A5不支持）2：AICPU_TS（A2/3支持 A5不支持）
3：AIV 4：AIV_ONLY（A2/3支持 A5不支持） 5：CCU_MS（A2/3支持 A5不支持）
6：CCU_SCHED（A2/3支持 A5不支持） 7：AICPU_UB/ROCE（A5不支持）
**/
constexpr uint8_t A5_AICPU_TS_ENGINE = 2;
constexpr uint8_t AIV_ENGINE = 3;
constexpr uint8_t A5_CCU_ENGINE = 6;
constexpr uint8_t Y_INDEX = 3;
constexpr uint8_t COMM_ALG_DEFAULT = 0;
constexpr uint8_t COMM_ALG_FULL_MESH = 1;
constexpr uint8_t COMM_ALG_DOUBLE_RING = 2;
constexpr uint8_t COMM_ALG_SWITCH_WING = 3;
constexpr uint8_t COMM_VERSION3 = 3;
constexpr double COMM_GROW_RATIO = 1.15;
constexpr uint64_t MTE_STATE_ZONE_SIZE = 1024UL * 1024UL;

constexpr uint64_t LARGE_K = 8192;
constexpr uint64_t LARGE_N = 5120;
constexpr uint64_t SMALL_N_BOUNDARY = 2048;
constexpr uint64_t TINY_M = 512;
constexpr uint64_t SMALL_M = 2048;
constexpr uint64_t MEDIAN_M = 4096;
constexpr double GATHER_LARGERNK_COMM_GROW_RATIO1 = 3;
constexpr double GATHER_LARGERNK_COMM_GROW_RATIO2 = 1.5;

constexpr uint8_t TIME_LOWER_RATIO = 2;
constexpr double TIME_UPPER_RATIO = 3.5;
constexpr double SCATTER_LARGERNK_COMM_GROW_RATIO1 = 1.5;
constexpr double SCATTER_LARGERNK_COMM_GROW_RATIO2 = 1.2;
constexpr double CUBE_UTIL_THRESH = 0.85;
constexpr uint32_t AICPU_NUM_BLOCKS_A2 = 6U;

constexpr uint64_t GROUP_M_OFFSET = 32;
constexpr uint64_t GROUP_N_OFFSET = 16;
constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;

constexpr auto DEFAULT_KEY_FOR_FITTING_MAP = "0_0";

constexpr static uint64_t ALL_GATHER_HCCL_MEM_LIMIT = 256 * 1024 * 1024;
constexpr static uint64_t ALL_GATHER_HCCL_NUM_LIMIT = 16;

enum class Mc2QuantMode {
    DEFAULT = 0,
    PERTENSOR_MODE,
    PERBLOCK_MODE,
    MXFP_MODE,
    INVALID_MODE,
};

struct HcclAicpuOpParam {
    uint8_t res[RES_LEN];
};

struct Mc2MatmulShapeInfo {
    const gert::StorageShape *x1Shape{nullptr};
    const gert::StorageShape *x2Shape{nullptr};
    const gert::StorageShape *x1ScaleShape{nullptr};
    const gert::StorageShape *x2ScaleShape{nullptr};
    bool isMxfp{false};
    bool isBTrans{false};
    const char *opName{nullptr};
};

struct KFCMsgBody {
    // Rank* aiv * MsgSize * sizeof(消息)
    HcclAicpuOpParam msgSndArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
    HcclAicpuOpParam msgRcvArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
};
struct KFCNotify {
    // 消息通信
    HcclAicpuOpParam msgSend[MAX_MSG_NUM]; // 填充16个
    HcclAicpuOpParam msgCnt[MAX_MSG_NUM];
};

constexpr std::initializer_list<ge::DataType> FP8DTYPE_SUPPORT_LIST = {
    ge::DataType::DT_FLOAT8_E4M3FN, ge::DataType::DT_FLOAT8_E5M2, ge::DataType::DT_HIFLOAT8};

constexpr std::initializer_list<ge::DataType> MXFP8DTYPE_SUPPORT_LIST = {ge::DataType::DT_FLOAT8_E4M3FN,
                                                                         ge::DataType::DT_FLOAT8_E5M2};

matmul_tiling::DataType ConvertGeTypeToMmType(const std::string &opName, ge::DataType type);
ge::DataType ConvertMmTypeToGeType(const std::string &opName, matmul_tiling::DataType type);
uint64_t GetDataTypeSize(const std::string &opName, ge::DataType type);
HcclDataType ConvertGeTypeToHcclType(const std::string &opName, ge::DataType type);
bool CheckSuppportedFormat(ge::Format format);
bool IsDeterministic();
bool GetRankSize(const std::string &opName, const char *group, int64_t &rankSize);
bool CheckRankSize(const NpuArch npuArch, const uint32_t rankSize);
uint8_t Mc2GetCommAlgo(int64_t rankDim, uint64_t mValue, const char *group, const gert::TilingContext *context);

bool CheckDataTypeVaild(ge::DataType type, std::initializer_list<ge::DataType> supportDtypeList);

// 解析commAlg属性：未配置(空或"0")时按HCCL环境变量回退选择分层/全互联方案；
// 返回GRAPH_FAILED表示commAlg取值非法，由调用方负责打印错误日志
ge::graphStatus ParseCommAlgWithEnvFallback(const char *entityName, const char *commAlg, bool &isLayered);

// 读取HCCL环境变量判断是否使用分层(HCCL_INTRA_PCIE_ENABLE=1且HCCL_INTRA_ROCE_ENABLE=0)
bool IsHcclPcieLayered(const char *entityName);

void UpdateMatmulV3Args(optiling::mc2_matmul_v3_advanced::Mc2MatMulV3Args &mmV3Args, const mc2tiling::TilingArgs &args,
                        const char *opName);
ge::graphStatus GetMatmulV3PriorityPolicy(const NpuArch npuArch, std::vector<int32_t> &priorities, const char *opName);

inline std::string GetSocVersion(const gert::TilingContext *context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;
    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    return socVersion;
}

inline NpuArch GetNpuArch(const gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    return ascendcPlatform.GetCurNpuArch();
}

class Mc2TilingUtils {
public:
    static uint8_t GetDebugMode();
    static uint8_t GetDebugCommAlg();
    static uint8_t GetDebugStepSize();
    static uint32_t GetCommSets(const char *group);
    static ge::graphStatus CommonParamCheck(const gert::TilingContext *context);
    static mc2tiling::HcclDataType GetDataType(ge::DataType type);
    static uint64_t GetMaxWindowSize();
    static bool CheckRankSize(NpuArch npuArch, uint32_t rankSize);
    static HcclDataType ConvertGeTypeToHcclType(const std::string &opName, ge::DataType type);
    static bool InferGroupSize(Mc2MatmulShapeInfo &mmInfo, uint64_t &groupSizeM, uint64_t &groupSizeN,
                               uint64_t &groupSizeK);
    template <typename T>
    static uint64_t GetTilingKey(T &tilingData, bool isFullMeshHost = false)
    {
        uint8_t commAlg = isFullMeshHost ? COMM_ALG_FULL_MESH_HOST : tilingData.msg.get_commAlg();
        uint64_t castBias = tilingData.param.get_biasLen() == 0 ? 0 : 1;
        // tiling key: commAlg(switch/doublering/fullmesh) nd2nz bias2float
        uint64_t tilingKey = optiling::RecursiveSum(castBias, 1, static_cast<uint64_t>(commAlg));
        return tilingKey;
    };
};

const std::map<ge::DataType, matmul_tiling::DataType> D_TYPE_MAP = {
    {ge::DT_BF16, matmul_tiling::DataType::DT_BFLOAT16},
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
};

const std::map<matmul_tiling::DataType, ge::DataType> D_TYPE_MATMUL_MAP = {
    {matmul_tiling::DataType::DT_BFLOAT16, ge::DT_BF16},
    {matmul_tiling::DataType::DT_FLOAT16, ge::DT_FLOAT16},
    {matmul_tiling::DataType::DT_FLOAT, ge::DT_FLOAT},
};

const std::map<ge::DataType, int64_t> D_TYPE_SIZE_MAP = {
    {ge::DT_BF16, 2},
    {ge::DT_FLOAT16, 2},
    {ge::DT_FLOAT, 4},
};

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
    {ge::DataType::DT_HIFLOAT8, mc2tiling::HcclDataType::HCCL_DATA_TYPE_HIF8}};

const std::map<NpuArch, std::set<uint32_t>> supportedRankSizeSet = {
    {Ops::Base::DAV_2002, {1, 2, 4}},
    {Ops::Base::DAV_2201, {1, 2, 4, 8}},
    {Ops::Base::DAV_3510, {1, 2, 4, 8, 16, 32, 64}},
};

const std::set<ge::Format> SUPPORTED_FORMAT = {ge::FORMAT_NCL,  ge::FORMAT_NCDHW, ge::FORMAT_DHWCN,
                                               ge::FORMAT_NHWC, ge::FORMAT_NCHW,  ge::FORMAT_ND};

inline ge::graphStatus GetCclBufferSize(const char *groupStr, uint64_t *cclBufferSize, const char *nodeName)
{
    HcclComm hcclComm;
    OP_TILING_CHECK(
        Mc2Hcom::MC2HcomTopology::CommGetCclBufferSizeByGroup(groupStr, cclBufferSize, &hcclComm) != HCCL_SUCCESS,
        OP_LOGE(nodeName, "CommGetCclBufferSizeByGroup failed"), return ge::GRAPH_FAILED);
    if (hcclComm == nullptr) {
        OP_TILING_CHECK(Mc2Hcom::MC2HcomTopology::CommGetGroupLocalWindowSize(groupStr, cclBufferSize) != HCCL_SUCCESS,
                        OP_LOGE(nodeName, "GetGroupLocalWindowSize from topoInfo failed"), return ge::GRAPH_FAILED);
        OP_LOGD(nodeName, "Get cclBufferSize by topoInfo");
    } else {
        OP_LOGD(nodeName, "Get cclBufferSize from HCCL");
    }
    OP_TILING_CHECK(*cclBufferSize == 0, OP_LOGE_FOR_INVALID_VALUE(nodeName, "cclBufferSize", "0", "non-zero"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus GetEpWinSize(const gert::TilingContext *context, const char *nodeName,
                                    uint64_t &hcclBufferSizeEp, uint64_t &maxWindowSizeEp, uint32_t attrGroupEpIndex,
                                    bool isLayered)
{
    auto attrs = context->GetAttrs();
    if (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_3510) {
        // A5 暂不支持 Hccl CommGetBufSizeCfg 接口，此处暂作规避
        // A5 实际物理分配为 HCCL_BUFFSIZE 的 2 倍
        hcclBufferSizeEp = mc2tiling::Mc2TilingUtils::GetMaxWindowSize() * 2UL;
        // A5 上前 1MB 作为状态区，剩余空间用作数据区
        maxWindowSizeEp = hcclBufferSizeEp - MTE_STATE_ZONE_SIZE;
    } else {
        if (isLayered) {
            hcclBufferSizeEp = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
        } else {
            auto groupEpHccl = attrs->GetAttrPointer<char>(static_cast<int>(attrGroupEpIndex));
            OP_TILING_CHECK(
                GetCclBufferSize(groupEpHccl, &hcclBufferSizeEp, nodeName) != ge::GRAPH_SUCCESS,
                OP_LOGE(nodeName, "Get Ep HcclBufferSizeEP failed, HcclBufferSizeEP is %lu", maxWindowSizeEp),
                return ge::GRAPH_FAILED);
        }
        maxWindowSizeEp = hcclBufferSizeEp;
    }
    return ge::GRAPH_SUCCESS;
}

// 临时判断是否为标卡4p形态(4卡，950)
inline bool IsStandardCard4P(const uint32_t rankDim, const NpuArch npuArch)
{
    return ((rankDim == STANDARD_CARD_4P) && (npuArch == Ops::Base::DAV_3510));
}

// 判断是否为8P形态(8卡，950)
inline bool Is8P(const uint32_t rankDim, const NpuArch npuArch)
{
    return ((rankDim == EIGHT_P_8P) && (npuArch == Ops::Base::DAV_3510));
}

// 判断是否使用 All2All + Vec Reduce 通路（StandardCard 4P 或 8P）
inline bool IsUseA2APath(const uint32_t rankDim, const NpuArch npuArch)
{
    return ((npuArch == Ops::Base::DAV_3510) && (rankDim == STANDARD_CARD_4P || rankDim == EIGHT_P_8P));
}

// ===== MKN 条件表 tiling 参数公共工具 =====

// 按条件表查值：命中 m/k/n 区间条目返回其键值，否则返回 defaultValue
inline int32_t GetValueFromMKNConditionMap(int32_t m, int32_t k, int32_t n, int32_t defaultValue,
                                           const std::map<int, std::vector<std::vector<int>>> &conditionMap)
{
    // 条件向量列索引：[M_ST, M_END, K_ST, K_END, N_ST, N_END]
    constexpr int32_t CONDITION_M_ST = 0;
    constexpr int32_t CONDITION_M_END = 1;
    constexpr int32_t CONDITION_K_ST = 2;
    constexpr int32_t CONDITION_K_END = 3;
    constexpr int32_t CONDITION_N_ST = 4;
    constexpr int32_t CONDITION_N_END = 5;
    int32_t value = defaultValue;
    for (auto &item : conditionMap) {
        for (auto &condition : item.second) {
            bool inRange = m > condition[CONDITION_M_ST] && m <= condition[CONDITION_M_END] &&
                           k > condition[CONDITION_K_ST] && k <= condition[CONDITION_K_END] &&
                           n > condition[CONDITION_N_ST] && n <= condition[CONDITION_N_END];
            if (inRange) {
                return item.first;
            }
        }
    }
    return value;
}

// 遍历 tiling 参数表：带条件表的项按 MKN 条件查值，否则取 value（-1 表示跳过不赋值）
template <typename TilingValueT>
inline void SetTilingParamsFromConditionMap(int32_t m, int32_t k, int32_t n,
                                            const std::map<int *, TilingValueT> &tilingParamMap)
{
    for (auto &item : tilingParamMap) {
        auto value = item.second.value;
        auto conditionMap = item.second.conditionMap;
        if (!conditionMap.empty()) {
            *item.first = GetValueFromMKNConditionMap(m, k, n, value, conditionMap);
        } else if (value != -1) {
            *item.first = value;
        }
    }
}

// CoCTiling 参数尾处理：ubMoveNum 折算半 KB 单位，并推导 k0/n0 基础块
template <typename CoCTilingT>
inline void FinalizeCoCTilingParam(CoCTilingT &cocTilingData)
{
    constexpr int32_t HALF_KBYTE = 512;  // 半 KB 折算单位
    constexpr int32_t DEFAULT_ROW = 128; // 默认 m0 行数
    constexpr int32_t DEFAULT_COL = 256; // 默认 k0/n0 列数
    cocTilingData.ubMoveNum = cocTilingData.ubMoveNum * HALF_KBYTE;
    if (cocTilingData.m0 >= DEFAULT_ROW) {
        cocTilingData.k0 = DEFAULT_COL;
        cocTilingData.n0 = cocTilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;
    }
}

// 校验 HCCL BUFF 空间大小：获取失败仅告警跳过；成功但小于最小值则报错
inline ge::graphStatus CheckHcclBuffSize(const gert::TilingContext *context, uint32_t attrGroupIndex,
                                         int32_t minBuffBytes, const char *nodeName)
{
    constexpr int32_t MB_SIZE_BYTES = 1024 * 1024; // MB 换算字节数
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(attrGroupIndex));
    uint64_t hcclBuffSize = 0ULL;
    auto cclRet = mc2tiling::GetCclBufferSize(group, &hcclBuffSize, nodeName);
    if (cclRet == ge::GRAPH_SUCCESS) {
        OP_TILING_CHECK(hcclBuffSize < static_cast<uint64_t>(minBuffBytes),
                        OP_LOGE(nodeName, "HCCL_BUFFSIZE (%lu Bytes) too small, min required %lu Bytes (%dMB)",
                                hcclBuffSize, static_cast<uint64_t>(minBuffBytes), minBuffBytes / MB_SIZE_BYTES),
                        return ge::GRAPH_FAILED);
    } else {
        OP_LOGW(nodeName, "Can't get HCCL_BUFFSIZE, skip CCL buffer size validation.");
    }
    return ge::GRAPH_SUCCESS;
}

// 设置 hccl 通信任务配置（MultiPut=level0:fullmesh）并填充 tiling 数据
template <typename TilingDataT>
inline ge::graphStatus SetHcclCcTilingConfig(const gert::TilingContext *context, uint32_t attrGroupIndex,
                                             TilingDataT *tilingData)
{
    constexpr uint32_t MC2_CC_OP_TYPE_BATCH_WRITE = 18; // batch write
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(attrGroupIndex));
    const std::string algConfig = "MultiPut=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, MC2_CC_OP_TYPE_BATCH_WRITE, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}

// Copy common fields without coupling the caller to a particular MatmulV3 argument type.
template <typename MatmulArgsT, typename TilingArgsT>
inline void FillMmV3ArgsFromTilingArgs(MatmulArgsT &args, const TilingArgsT &tilingArgs)
{
    args.isATrans = tilingArgs.isATrans;
    args.isBTrans = tilingArgs.isBTrans;
    args.hasBias = tilingArgs.isBias;
    args.aType = tilingArgs.geAType;
    args.bType = tilingArgs.geBType;
    args.cType = tilingArgs.geCType;
    args.biasType = tilingArgs.isBias ? tilingArgs.geBiasType : ge::DT_INT32;

    args.aFormat = ge::FORMAT_ND;
    args.bFormat = ge::FORMAT_ND;
    args.outFormat = ge::FORMAT_ND;

    args.mValue = tilingArgs.mValue;
    args.kValue = tilingArgs.kValue;
    args.nValue = tilingArgs.nValue;
}

// ===== 非量化 A3 tiling 公共校验 =====

// 工具函数：判断指定 value 是否存在于 list 中
inline bool IsContains(const std::vector<uint32_t> &list, uint32_t value)
{
    return std::count(list.begin(), list.end(), value) > 0;
}

// 非量化场景校验 bias 数据类型：x1 为 BF16 要求 FLOAT；x1 为 FLOAT16 要求与 x1 一致
inline ge::graphStatus CheckNonQuantBiasDataType(const gert::CompileTimeTensorDesc *biasTensorDesc,
                                                 ge::DataType x1Dtype, const char *opName)
{
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
    return ge::GRAPH_SUCCESS;
}

// ===== M 维按通信轮次切分公共工具 =====

// 按通信轮次上限切分 M 维：tileLen 按 128 字节对齐（2 字节 dtype 对齐 64 元素，4 字节对齐 32 元素）；
// commTurn 超过 maxTileCnt 时截断为 maxTileCnt，返回切分后的单份行数
inline uint32_t SplitMValueByCommTurn(uint64_t mValue, uint64_t &commTurn, uint64_t dtypeSize, uint32_t maxTileCnt = 64)
{
    if (commTurn >= maxTileCnt) {
        commTurn = maxTileCnt;
    }

    uint64_t tileLen = 1;
    if (mValue > commTurn) {
        tileLen = mValue / commTurn;
    }

    if (dtypeSize == 2) { // 数据长度为2字节, 则向 2*64 = 128 对齐
        tileLen = AlignUp<uint64_t>(tileLen, 64);
    } else if (dtypeSize == 4) { // 4 is float32 type size, 用于对齐到 128
        tileLen = AlignUp<uint64_t>(tileLen, 32);
    }
    if (mValue > tileLen) {
        return static_cast<uint32_t>(tileLen);
    }
    return static_cast<uint32_t>(mValue);
}
} // namespace mc2tiling

#endif
