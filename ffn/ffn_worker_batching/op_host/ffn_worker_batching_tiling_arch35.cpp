/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ffn_worker_batching_tiling_arch35.cpp
 * \brief FfnWorkerBatching arch35 (Ascend950 / DAV_3510) Regbase tiling（1000 档 + IsRegbaseSocVersion 守卫）。
 *        UB 容量/核数运行时经 GetCoreMemSize/GetCoreNumAiv 取值，禁写死 arch 常量。
 *        切分阈值与 workspace 布局按 A5 四相位（prepare/sort/gather/group_listing）自行推导。
 *        TilingData 采用 host/kernel 共用平铺 struct（GetTilingData<FfnWorkerBatchingArch35TilingData>() 直写）。
 */
#include "ffn_worker_batching_tiling.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "../../../common/include/op_kernel/attention_ffn_schedule.h"
#include <limits>
#include <string>
#include "../op_kernel/arch35/ffn_worker_batching_arch35_tiling_def.h"

namespace optiling {
// arch35 布局契约常量:host 的 UB 预算必须覆盖 kernel 侧固定队列,故与
// op_kernel/arch35 各 a5 头文件中的本地常量按值对齐;不新增 tiling ABI 字段。
namespace Arch35Config {
constexpr int64_t COMPARE_FP32_ELEMENTS = 256 / static_cast<int64_t>(sizeof(float)); // 一个向量 repeat 的 fp32 元素数
constexpr int64_t TOKEN_STRIDE_ALIGNMENT = 512;
constexpr int64_t MX_SCALE_BLOCK_ELEMENTS = 32;
constexpr int64_t FP4_VALUES_PER_BYTE = 2;
constexpr int64_t EXTRACT_MAX_REPEATS = 255; // Extract API 的 repeatTime 上限

// token_dtype 公开属性取值。
constexpr int64_t TOKEN_FP16 = 0;
constexpr int64_t TOKEN_BF16 = 1;
constexpr int64_t TOKEN_INT8 = 2;
constexpr int64_t TOKEN_FP8_E5M2 = 3;
constexpr int64_t TOKEN_FP8_E4M3 = 4;
constexpr int64_t TOKEN_FP4_E2M1 = 5;

// Gather:两个队列,每轮各处理至多 128 个输出行。
constexpr int64_t GATHER_QUEUE_DEPTH = 2;
constexpr int64_t GATHER_ROWS_PER_LOOP = 128;
constexpr int64_t GATHER_OUTPUT_LANES = 5;      // 四路索引 + dynamic scale
constexpr int64_t GATHER_FLOAT_INDEX_LANES = 7; // src, qa, a, remainder, qb, b, k
constexpr int64_t GATHER_INT_INDEX_LANES = 3;   // a, b, k

constexpr int64_t GROUP_VECTOR_BUFFERS = 10; // cur/prev/next/idx/diff/diffF 与四路紧凑数组
constexpr int64_t GROUP_MASK_BUFFERS = 2;    // run 起止
constexpr int64_t GROUP_ROW_FIELDS = 2;      // expert ID 与 token 数(int64)
constexpr int64_t GROUP_MIN_ROWS = 2;
constexpr int64_t BITS_PER_BYTE = 8;
constexpr int64_t RECV_WAITER_FLAG_BUFFERS = 2; // int32 flag 与 FP32 归约暂存

// InitBuffer 在元素存储之外的额外块数。
constexpr int64_t SORT_BUFFER_COUNT = 5;            // 输入、concat、临时、排序结果、掩码
constexpr int64_t MERGE_BUFFER_COUNT = 3;           // 输入、输出、计数缓存
constexpr int64_t EXTRACT_BUFFER_COUNT = 3;         // proposal、expert ID、源下标
constexpr int64_t MERGE_COUNT_CACHE_SEGMENTS = 128; // 有界缓存策略,非硬件上限
} // namespace Arch35Config
namespace {
constexpr uint32_t EXPERT_NUM_ATTR = 0;
constexpr uint32_t MAX_OUT_SHAPE_ATTR = 1;
constexpr uint32_t TOKEN_DTYPE_ATTR = 2;
constexpr uint32_t NEED_SCHEDULE_ATTR = 3;
constexpr uint32_t LAY_NUM_ATTR = 4;
// 追加在 layer_num 后，保持 expert_num 等已有属性的索引稳定。
constexpr uint32_t SYNC_FLAG_ATTR = 5;

constexpr uint32_t INDEX_ZERO = 0;
constexpr uint32_t INDEX_ONE = 1;
constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t INDEX_THREE = 3;

constexpr int64_t BATCH_MODE = 1;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t MAX_RESERVE_WK_NUM = 128;

constexpr int64_t TILING_KEY_NORM = 100;
constexpr int64_t TILING_KEY_RECV = 101;

constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t GL_ROW_BYTES = static_cast<int64_t>(sizeof(int64_t)) * NUM_TWO; // 一行 = [expert_id, tokenNum]
constexpr int64_t GL_UB_FRACTION = 16;                                            // group_list 拼装区取 UB 的 1/16

// 数据块字节数，与 kernel 侧 AscendC::ONE_BLK_SIZE 同值（host 侧无该符号，故此处按同值定义）。
constexpr int64_t ONE_BLK_BYTES = 32;
// MrgSort 单轮归并路数：由 MrgSortSrcList 的 4 个入参与 validBit 的 4 个有效位决定。
constexpr int64_t MRG_LIST_NUM = 4;
// 被 mask 的 token 由上游置为不小于该值的大数，排序前据此压缩剔除。
// 与 kernel 侧判据同源（见 op_kernel/ffn_wb_sort_base.h 的 expertStart_ 及算子文档 mask 约定）。
constexpr int64_t EXPERT_ID_MASK_START = 1000000;

// region proposal 对：fp32 键 + uint32 索引，占 SORT_PAIR_FLOATS 个 float。
constexpr int64_t SORT_PAIR_FLOATS =
    static_cast<int64_t>(sizeof(float) + sizeof(uint32_t)) / static_cast<int64_t>(sizeof(float));
// 段内排序时每元素在 UB 的驻留字节：id + 原下标（各 int32）、比较掩码，
// 以及 Concat/Sort 要求互不重叠的三块 proposal 对区（concat 结果 / 临时区 / 排序结果）。
constexpr int64_t SORT_PAIR_REGIONS = 3;
constexpr int64_t SORT_UB_BYTES_PER_ELEM = static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO +
                                           SORT_PAIR_FLOATS * static_cast<int64_t>(sizeof(float)) * SORT_PAIR_REGIONS +
                                           static_cast<int64_t>(sizeof(uint32_t));
// 拆包时每元素在 UB 的驻留字节：proposal 对 + 拆出的 id 与 idx。
constexpr int64_t EXTRACT_UB_BYTES_PER_ELEM =
    SORT_PAIR_FLOATS * static_cast<int64_t>(sizeof(float)) + static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO;
constexpr int64_t EXPERT_IDX_MAX = 8192;
constexpr int64_t MAX_SESSION_NUM = 1024;
constexpr int64_t MAX_K_NUM = 64;
constexpr int64_t CONTEXT_BYTES = sizeof(aicpu::ScheduleContext);
constexpr int64_t INT32_PER_BLOCK = ONE_BLK_BYTES / sizeof(int32_t);
constexpr int64_t SORT_FIXED_BYTES = Arch35Config::SORT_BUFFER_COUNT * ONE_BLK_BYTES;
constexpr int64_t MERGE_FIXED_BYTES = Arch35Config::MERGE_BUFFER_COUNT * ONE_BLK_BYTES;
constexpr int64_t EXTRACT_FIXED_BYTES = Arch35Config::EXTRACT_BUFFER_COUNT * ONE_BLK_BYTES;

// 本算子的 SIMT(asc_vf_call) 与 VF 计算需要 UB 系统预留区。预留量不自行相减，
// 而是通过平台接口 ReserveLocalMemory 声明——之后 GetCoreMemSize(UB) 返回的即为可用值。
// ReservedSize 是平台定义的枚举（8K/16K/32K），选择依据是本算子用到 SIMT,取最大档。
} // namespace

class FfnWorkerBatchingTilingArch35 : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit FfnWorkerBatchingTilingArch35(gert::TilingContext* context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {}
    ~FfnWorkerBatchingTilingArch35() override = default;

protected:
    bool IsCapable() override
    {
        return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_);
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus CheckInputParam();
    ge::graphStatus GetAttrsInfo();
    ge::graphStatus ParseMaxOutShape(const gert::RuntimeAttrs* attrs);
    ge::graphStatus ParseOptionalAttrs(const gert::RuntimeAttrs* attrs, int64_t expertNum);
    void SplitPrepare();
    void SplitSortAndMerge();
    void SplitGroupList();
    void LayoutWorkspace();

    FfnWorkerBatchingArch35TilingData* tilingDataPtr_ = nullptr;
    int64_t A_ = 0;
    int64_t BS_ = 0;
    int64_t K_ = 0;
    int64_t Y_ = 0;
    int64_t H_ = 0;
    int64_t expertNum_ = 0;
    int64_t tokenDtype_ = 0;
    int64_t needSchedule_ = 0;
    bool syncFlag_ = false;
    int64_t layerNum_ = 0;
    int64_t aivNum_ = 0;
    int64_t coreNum_ = 0;
    int64_t flatElements_ = 0;
    int64_t preparePerLoopRows_ = 0;
    int64_t sortSegNum_ = 0;
    int64_t sortPerSegElements_ = 0;
    int64_t sortLenPerSeg_ = 0;
    int64_t mergeRounds_ = 0;
    int64_t mergeOneLoopElements_ = 0;
    int64_t extractPerLoopElements_ = 0;
    int64_t glRowsPerLoop_ = 0;
    int64_t bskAlign_ = 0; // BS*K 按数据块对齐后的元素数,DoOpTiling 算好后各切分函数共用
    int64_t wsFlatIds_ = 0;
    int64_t wsPairA_ = 0;
    int64_t wsPairB_ = 0;
    int64_t wsSegCnt_ = 0;
    int64_t wsSortedIds_ = 0;
    int64_t wsGatherIdx_ = 0;
    int64_t userWorkspaceWords_ = 0;
    uint64_t ubSize_ = 0;
    uint32_t sysWorkspaceSize_ = 0;
};

ge::graphStatus FfnWorkerBatchingTilingArch35::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "platformInfo"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    aivNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(aivNum_ == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "aivNum", std::to_string(aivNum_),
                                                      "failed to get a positive vector core count"),
                return ge::GRAPH_FAILED);

    // 先声明预留，再取容量：平台在 GetCoreMemSize 中已扣除本次预留，得到 vector core 真实可用 UB。
    ascendcPlatform.ReserveLocalMemory(platform_ascendc::ReservedSize::RESERVED_SIZE_32K);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF(ubSize_ == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubSize", std::to_string(ubSize_),
                                                      "failed to get a positive UB size"),
                return ge::GRAPH_FAILED);

    sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::CheckInputParam()
{
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType xdtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        xdtype != ge::DT_INT8,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "schedule_context", Ops::Base::ToString(xdtype), "int8"),
        return ge::GRAPH_FAILED);

    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto xShape = Ops::Transformer::OpTiling::EnsureNotScalar(inputX->GetStorageShape());
    OP_CHECK_IF(xShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "schedule_context",
                                             std::to_string(xShape.GetDimNum()), "1"),
                return ge::GRAPH_FAILED);
    // 所有公共入口均需保护固定 1024B context 读取；Torch wrapper 的检查不能
    // 覆盖直接 ACLNN/GE 调用。空输入也必须在下发 kernel 前拒绝。
    OP_CHECK_IF(
        xShape.GetDim(0) < CONTEXT_BYTES,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "schedule_context dim[0]", std::to_string(xShape.GetDim(0)),
                                  "at least " + std::to_string(CONTEXT_BYTES)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::GetAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* expertNumPtr = attrs->GetAttrPointer<int64_t>(EXPERT_NUM_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertNumPtr);
    OP_CHECK_IF(*expertNumPtr > EXPERT_IDX_MAX || *expertNumPtr <= 0,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expert_num", std::to_string(*expertNumPtr),
                                          "in (0, " + std::to_string(EXPERT_IDX_MAX) + "]"),
                return ge::GRAPH_FAILED);
    expertNum_ = *expertNumPtr;

    if (ParseMaxOutShape(attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseOptionalAttrs(attrs, *expertNumPtr) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// max_out_shape 是 [A, BS, K, H] 四元组:既定形状上界,也是输出 shape 的静态推导依据。
ge::graphStatus FfnWorkerBatchingTilingArch35::ParseMaxOutShape(const gert::RuntimeAttrs* attrs)
{
    const gert::ContinuousVector* maxOutShapePtr = attrs->GetAttrPointer<gert::ContinuousVector>(MAX_OUT_SHAPE_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maxOutShapePtr);
    OP_CHECK_IF(maxOutShapePtr->GetSize() != static_cast<size_t>(NUM_FOUR),
                OP_LOGE_WITH_INVALID_ATTR_SIZE(context_->GetNodeName(), "max_out_shape",
                                               std::to_string(maxOutShapePtr->GetSize()), std::to_string(NUM_FOUR)),
                return ge::GRAPH_FAILED);
    const int64_t* maxOutShapeArray = reinterpret_cast<const int64_t*>(maxOutShapePtr->GetData());
    A_ = maxOutShapeArray[INDEX_ZERO];
    BS_ = maxOutShapeArray[INDEX_ONE];
    K_ = maxOutShapeArray[INDEX_TWO];
    H_ = maxOutShapeArray[INDEX_THREE];
    OP_CHECK_IF((A_ > MAX_SESSION_NUM || A_ <= 0),
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "max_out_shape[0]", std::to_string(A_),
                                          "in (0, " + std::to_string(MAX_SESSION_NUM) + "]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        BS_ <= 0,
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "max_out_shape[1]", std::to_string(BS_), "greater than 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((K_ > MAX_K_NUM || K_ <= 0),
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "max_out_shape[2]", std::to_string(K_),
                                          "in (0, " + std::to_string(MAX_K_NUM) + "]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        H_ <= 0,
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "max_out_shape[3]", std::to_string(H_), "greater than 0"),
        return ge::GRAPH_FAILED);

    // Check before multiplying: a post-product bound cannot undo signed overflow.
    OP_CHECK_IF(A_ > std::numeric_limits<int64_t>::max() / BS_,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "max_out_shape[0], max_out_shape[1]",
                                                       std::to_string(A_) + ", " + std::to_string(BS_),
                                                       "A*BS must not overflow int64"),
                return ge::GRAPH_FAILED);
    const int64_t aBs = A_ * BS_;
    OP_CHECK_IF(aBs > std::numeric_limits<int64_t>::max() / K_,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "max_out_shape[0], max_out_shape[1], max_out_shape[2]",
                    std::to_string(A_) + ", " + std::to_string(BS_) + ", " + std::to_string(K_),
                    "A*BS*K must not overflow int64"),
                return ge::GRAPH_FAILED);
    Y_ = aBs * K_;
    // Indices/counts are int32. This representation bound does not impose the
    // old 2^22 FP32 limit: large positions still use the integer gather path.
    OP_CHECK_IF(Y_ > std::numeric_limits<int32_t>::max(),
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "A*BS*K", std::to_string(Y_),
                                          "at most " + std::to_string(std::numeric_limits<int32_t>::max())),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 可选属性:缺省时保留成员初值,给出即逐个校验取值域。
ge::graphStatus FfnWorkerBatchingTilingArch35::ParseOptionalAttrs(const gert::RuntimeAttrs* attrs, int64_t expertNum)
{
    const int64_t* tokenDtype = attrs->GetAttrPointer<int64_t>(TOKEN_DTYPE_ATTR);
    if (tokenDtype != nullptr) {
        OP_CHECK_IF((*tokenDtype < Arch35Config::TOKEN_FP16 || *tokenDtype > Arch35Config::TOKEN_FP4_E2M1),
                    OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "token_dtype", std::to_string(*tokenDtype),
                                              "one of [0, 1, 2, 3, 4, 5]"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(*tokenDtype == Arch35Config::TOKEN_FP4_E2M1 && H_ % Arch35Config::FP4_VALUES_PER_BYTE != 0,
                    OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "max_out_shape[3]", std::to_string(H_),
                                              "even when token_dtype is 5 (float4_e2m1)"),
                    return ge::GRAPH_FAILED);
        tokenDtype_ = *tokenDtype;
    }

    const int64_t* needSchedulePtr = attrs->GetAttrPointer<int64_t>(NEED_SCHEDULE_ATTR);
    if (needSchedulePtr != nullptr) {
        OP_CHECK_IF((*needSchedulePtr < 0 || *needSchedulePtr > 1),
                    OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "need_schedule",
                                              std::to_string(*needSchedulePtr), "one of [0, 1]"),
                    return ge::GRAPH_FAILED);
        needSchedule_ = *needSchedulePtr;
    }

    // 未提供时默认 false，兼容旧图的同步接收语义。
    const bool* syncFlagPtr = attrs->GetAttrPointer<bool>(SYNC_FLAG_ATTR);
    if (syncFlagPtr != nullptr) {
        syncFlag_ = *syncFlagPtr;
    }
    const int64_t* layNumPtr = attrs->GetAttrPointer<int64_t>(LAY_NUM_ATTR);
    if (layNumPtr != nullptr) {
        OP_CHECK_IF((*layNumPtr < 0 || *layNumPtr > expertNum),
                    OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "layer_num", std::to_string(*layNumPtr),
                                              "in [0, " + std::to_string(expertNum) + "]"),
                    return ge::GRAPH_FAILED);
        layerNum_ = *layNumPtr;
    }
    // Only asynchronous RECV interprets layer_num. Keep the legacy NORM/synchronous
    // contracts (including layer_num=0) unchanged; never divide by their default 0.
    if (needSchedule_ == 1 && syncFlag_) {
        OP_CHECK_IF(layerNum_ == 0 || expertNum % layerNum_ != 0,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        context_->GetNodeName(), "layer_num, expert_num",
                        std::to_string(layerNum_) + ", " + std::to_string(expertNum),
                        "async RECV requires layer_num > 0 and expert_num divisible by layer_num"),
                    return ge::GRAPH_FAILED);
    }
    // H is stored as uint32 in the runtime context. Check narrowing before
    // deriving payload bytes. FP4 stores two values per byte; MX scale follows
    // the token and uses one byte per 32 logical elements (including tail).
    OP_CHECK_IF(H_ > std::numeric_limits<uint32_t>::max(),
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "max_out_shape[3]", std::to_string(H_),
                                          "at most " + std::to_string(std::numeric_limits<uint32_t>::max())),
                return ge::GRAPH_FAILED);
    const int64_t tokenBytes =
        tokenDtype_ <= Arch35Config::TOKEN_BF16 ?
            H_ * sizeof(uint16_t) :
            (tokenDtype_ == Arch35Config::TOKEN_FP4_E2M1 ? H_ / Arch35Config::FP4_VALUES_PER_BYTE : H_);
    const int64_t scaleBytes =
        tokenDtype_ == Arch35Config::TOKEN_INT8 ?
            sizeof(float) :
            (tokenDtype_ >= Arch35Config::TOKEN_FP8_E5M2 ?
                 (H_ + Arch35Config::MX_SCALE_BLOCK_ELEMENTS - 1) / Arch35Config::MX_SCALE_BLOCK_ELEMENTS :
                 0);
    constexpr int64_t maxAlignedStride =
        static_cast<int64_t>(std::numeric_limits<uint32_t>::max() / Arch35Config::TOKEN_STRIDE_ALIGNMENT) *
        Arch35Config::TOKEN_STRIDE_ALIGNMENT;
    OP_CHECK_IF(tokenBytes + scaleBytes > maxAlignedStride,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "tokenBytes + scaleBytes", std::to_string(tokenBytes + scaleBytes),
                    "must fit a 512B-aligned uint32 stride of at most " + std::to_string(maxAlignedStride) + " bytes"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::GetShapeAttrsInfo()
{
    if (CheckInputParam() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::DoOpTiling()
{
    tilingDataPtr_ = context_->GetTilingData<FfnWorkerBatchingArch35TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingDataPtr_);

    // 用核数由「工作量能否喂饱一个核」决定，不设固定核数阈值：
    // 排序按 ONE_REPEAT_SORT_NUM 为粒度推进，一个核至少要分到一个完整粒度才有意义，
    // 否则多出来的核只是在 SyncAll 上空耗。故上限取平台 aivNum，实际取二者较小值。
    const int64_t coreByWork = (Y_ + ONE_REPEAT_SORT_NUM - 1) / ONE_REPEAT_SORT_NUM;
    coreNum_ = std::max<int64_t>(1, std::min<int64_t>(aivNum_, coreByWork));

    tilingDataPtr_->Y = Y_;
    tilingDataPtr_->H = H_;
    tilingDataPtr_->tokenDtype = tokenDtype_;
    tilingDataPtr_->expertNum = expertNum_;
    tilingDataPtr_->coreNum = coreNum_;
    tilingDataPtr_->ubSize = static_cast<int64_t>(ubSize_);
    tilingDataPtr_->expertStart = EXPERT_ID_MASK_START;

    // 扁平序列长度：RECV 的 expert_id 来自 token_info 的 FfnDataDesc，逐 session 取出后按数据块补齐，
    // 故为 A*align(BS*K)；NORM 的 expert_ids_buf 本就连续，长度即 Y。
    bskAlign_ = (K_ * BS_ * static_cast<int64_t>(sizeof(int32_t)) + ONE_BLK_BYTES - 1) / ONE_BLK_BYTES * ONE_BLK_BYTES /
                static_cast<int64_t>(sizeof(int32_t));
    flatElements_ = (needSchedule_ == 1) ? A_ * bskAlign_ : Y_;
    // RECV padding contributes to the original index space too.
    OP_CHECK_IF(flatElements_ > std::numeric_limits<int32_t>::max(),
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "flatElements", std::to_string(flatElements_),
                                          "at most " + std::to_string(std::numeric_limits<int32_t>::max())),
                return ge::GRAPH_FAILED);
    tilingDataPtr_->flatElements = flatElements_;

    // Reset separates phases, but each phase must be able to allocate its
    // smallest legal block. The gather expression mirrors its fixed queues:
    // two 128-row output/index queues, route IDs, scale scratch and index lanes.
    // Large H is streamed; only one 32B payload block (+INT8 scale block) is required.
    const int64_t sessionAlign = (A_ + INT32_PER_BLOCK - 1) / INT32_PER_BLOCK * INT32_PER_BLOCK;
    const int64_t gatherRows = Arch35Config::GATHER_ROWS_PER_LOOP;
    const int64_t queueDepth = Arch35Config::GATHER_QUEUE_DEPTH;
    const int64_t gatherFixed = queueDepth * gatherRows * sizeof(int32_t) * Arch35Config::GATHER_OUTPUT_LANES +
                                queueDepth * gatherRows * sizeof(int32_t) +
                                sessionAlign * sizeof(int32_t) * queueDepth + gatherRows * ONE_BLK_BYTES * queueDepth +
                                gatherRows * (sizeof(float) * Arch35Config::GATHER_FLOAT_INDEX_LANES +
                                              sizeof(int32_t) * Arch35Config::GATHER_INT_INDEX_LANES);
    const int64_t minPayloadBlock = ONE_BLK_BYTES + (tokenDtype_ == Arch35Config::TOKEN_INT8 ? ONE_BLK_BYTES : 0);
    const int64_t minGather = gatherFixed + queueDepth * minPayloadBlock;
    const int64_t minGroup =
        Arch35Config::GROUP_VECTOR_BUFFERS * (Arch35Config::COMPARE_FP32_ELEMENTS * sizeof(int32_t) + ONE_BLK_BYTES) +
        Arch35Config::GROUP_MASK_BUFFERS *
            (Arch35Config::COMPARE_FP32_ELEMENTS / Arch35Config::BITS_PER_BYTE + ONE_BLK_BYTES) +
        Arch35Config::GROUP_MIN_ROWS * Arch35Config::GROUP_ROW_FIELDS * sizeof(int64_t) + ONE_BLK_BYTES;
    const int64_t minWaiter =
        needSchedule_ == BATCH_MODE ? A_ * ONE_BLK_BYTES * Arch35Config::RECV_WAITER_FLAG_BUFFERS + ONE_BLK_BYTES : 0;
    const int64_t minUb = std::max<int64_t>(CONTEXT_BYTES, std::max(minGather, std::max(minGroup, minWaiter)));
    OP_CHECK_IF(static_cast<int64_t>(ubSize_) < minUb,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "ubSize", std::to_string(ubSize_),
                    "minimum phase buffers require at least " + std::to_string(minUb) + " bytes"),
                return ge::GRAPH_FAILED);

    tilingDataPtr_->syncFlag = syncFlag_;
    tilingDataPtr_->layerNum = layerNum_;
    SplitPrepare();
    SplitSortAndMerge();
    SplitGroupList();
    LayoutWorkspace();
    return ge::GRAPH_SUCCESS;
}

// phase0:单轮块长由运行时 UB 反推。
void FfnWorkerBatchingTilingArch35::SplitPrepare()
{
    // ---------------- phase0(prepare)的切分 ----------------
    // RECV 逐 session 行取 BS*K 个 id,同一轮内 UB 需同时驻留:
    //   · 本轮 id 区        rows * bskAlign * 4B
    //   · 握手回写的清零区   rows * ONE_BLK_BYTES(每行一个 32B 块写 flag)
    // 按运行时 UB 反推每轮行数,不设固定上限;NORM 是整段直搬,按同一公式给出块内元素数即可。
    const int64_t prepBytesPerRow = bskAlign_ * static_cast<int64_t>(sizeof(int32_t)) + ONE_BLK_BYTES;
    // 异步 prepare 另用一个 UB 块读取快照中的单行 flag。
    // 只为异步扣除该空间，保持同步/NORM 的原有批量行数预算。
    const int64_t selectionBytes = (needSchedule_ == 1 && syncFlag_) ? ONE_BLK_BYTES : 0;
    int64_t prepRows = (static_cast<int64_t>(ubSize_) - selectionBytes) / std::max<int64_t>(1, prepBytesPerRow);
    prepRows = std::max<int64_t>(1, std::min<int64_t>(prepRows, A_));
    preparePerLoopRows_ = prepRows;
    tilingDataPtr_->preparePerLoopRows = preparePerLoopRows_;
    // 一行可能大于 UB。保留一次一行的调度粒度，但把 ID 缓冲限制为可用容量，
    // kernel 会在行内分块；不能再把“至少一行”解释为“整行必须驻留”。
    const int64_t idBudget = static_cast<int64_t>(ubSize_) - selectionBytes - prepRows * ONE_BLK_BYTES;
    const int64_t idCapacity = idBudget / ONE_BLK_BYTES * ONE_BLK_BYTES / sizeof(int32_t);
    tilingDataPtr_->preparePerLoopElements = std::min<int64_t>(prepRows * bskAlign_, idCapacity);
}

// phase1~3:段内排序(VBS)、段间归并(VMS)与归并收尾(Extract)的切分,三者共用同一份 UB 预算。
void FfnWorkerBatchingTilingArch35::SplitSortAndMerge()
{
    // ---------------- 段内排序（VBS）的切分 ----------------
    // 一段在 UB 中同时驻留：输入 id + 原下标（各 4B）、proposal 对区与排序临时区（各 8B/元素）、
    // 比较掩码（4B）。故每元素占用 SORT_UB_BYTES_PER_ELEM 字节，据此反推单段元素数上限。
    // ubSize_ 在 GetPlatformInfo 中已扣除系统预留，此处直接使用，勿重复扣减。
    const int64_t ubAvail = static_cast<int64_t>(ubSize_);
    // 五块缓冲各有额外 32B；输入/掩码按 64 元素对齐，proposal 按 32 对齐。
    // 上限取 64 的倍数，使实际 12*align64(n)+24*align32(n)+160 不超过预算。
    constexpr int64_t SORT_COMPARE_ELEMENTS = Arch35Config::COMPARE_FP32_ELEMENTS;
    int64_t segCap =
        (ubAvail - SORT_FIXED_BYTES) / SORT_UB_BYTES_PER_ELEM / SORT_COMPARE_ELEMENTS * SORT_COMPARE_ELEMENTS;
    segCap = std::max<int64_t>(ONE_REPEAT_SORT_NUM, segCap);
    // 先按核数均分；单段超 UB 容量时增加段数（段由各核 grid-stride 认领，段数可多于核数）。
    sortSegNum_ = coreNum_;
    sortPerSegElements_ = (flatElements_ + sortSegNum_ - 1) / sortSegNum_;
    if (sortPerSegElements_ > segCap) {
        sortPerSegElements_ = segCap;
        sortSegNum_ = (flatElements_ + sortPerSegElements_ - 1) / sortPerSegElements_;
    }
    sortPerSegElements_ = std::max<int64_t>(1, sortPerSegElements_);
    sortSegNum_ = std::max<int64_t>(1, sortSegNum_);
    // 每段 proposal 对区：段长按 Sort32 粒度上取整后，每元素占 SORT_PAIR_FLOATS 个 float。
    const int64_t segAlign =
        (sortPerSegElements_ + ONE_REPEAT_SORT_NUM - 1) / ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;
    sortLenPerSeg_ = segAlign * SORT_PAIR_FLOATS;

    tilingDataPtr_->sortSegNum = sortSegNum_;
    tilingDataPtr_->sortPerSegElements = sortPerSegElements_;
    tilingDataPtr_->sortLenPerSeg = sortLenPerSeg_;

    // ---------------- 段间归并（VMS）的切分 ----------------
    // 归并轮数：每轮 MRG_LIST_NUM 路合一，直到剩一路。
    mergeRounds_ = 0;
    for (int64_t lists = sortSegNum_; lists > 1; lists = (lists + MRG_LIST_NUM - 1) / MRG_LIST_NUM) {
        mergeRounds_++;
    }
    // 单次驻留：MRG_LIST_NUM 路输入 + 同宽的输出，均为 proposal 对（每元素 8B）。
    // 预算里必须先扣掉同一 TPipe 上的其它缓冲：各段有效数的读回区，以及每个缓冲按块对齐的余量；
    // 否则输入与输出两块相加恰好等于可用 UB，分配越界后输出会压到输入上（表现为归并结果头部被覆盖）。
    // 大 Y 的段数可能上万，计数不能全部放 UB。只缓存至多 128 个槽，按需换块；
    // 三个 InitBuffer 的额外 32B 也必须扣除，禁止容量不足时回退到完整 UB。
    tilingDataPtr_->mergeCountCacheSegments = std::min<int64_t>(sortSegNum_, Arch35Config::MERGE_COUNT_CACHE_SEGMENTS);
    const int64_t mergeOther = tilingDataPtr_->mergeCountCacheSegments * ONE_BLK_BYTES + MERGE_FIXED_BYTES;
    const int64_t mergeUb = ubAvail - mergeOther;
    int64_t mergeLoop = mergeUb / (MRG_LIST_NUM * NUM_TWO * SORT_PAIR_FLOATS * static_cast<int64_t>(sizeof(float))) /
                        ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;
    mergeOneLoopElements_ = std::max<int64_t>(ONE_REPEAT_SORT_NUM, mergeLoop);
    tilingDataPtr_->mergeRounds = mergeRounds_;
    tilingDataPtr_->mergeOneLoopElements = mergeOneLoopElements_;

    // ---------------- 归并收尾（Extract）的切分 ----------------
    // 单次驻留：proposal 对（8B）+ 拆出的 id 与 idx（各 4B）。
    // Extract 三块缓冲各多 32B，repeatTime 的 API 上限为 255。
    // 按此上限分块也兼容启用范围检查的 CPU 调测，不依赖设备关闭检查。
    int64_t extractLoop =
        (ubAvail - EXTRACT_FIXED_BYTES) / EXTRACT_UB_BYTES_PER_ELEM / ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;
    extractLoop = std::min<int64_t>(extractLoop, Arch35Config::EXTRACT_MAX_REPEATS * ONE_REPEAT_SORT_NUM);
    extractPerLoopElements_ = std::max<int64_t>(ONE_REPEAT_SORT_NUM, std::min<int64_t>(extractLoop, Y_));
    tilingDataPtr_->extractPerLoopElements = extractPerLoopElements_;
}

// phase4:group_list 写出时每块拼多少行。
void FfnWorkerBatchingTilingArch35::SplitGroupList()
{
    // ---------------- group_list 的切分 ----------------
    // 每行 [expert_id, tokenNum] 两个 int64 = GL_ROW_BYTES；拼装区取 UB 的 1/GL_UB_FRACTION，
    // 按 2 行对齐（使块起点落在数据块边界），并以 expertNum 封顶。
    int64_t rows = static_cast<int64_t>(ubSize_) / GL_UB_FRACTION / GL_ROW_BYTES / NUM_TWO * NUM_TWO;
    glRowsPerLoop_ = std::max<int64_t>(Arch35Config::GROUP_MIN_ROWS, std::min<int64_t>(rows, expertNum_));

    tilingDataPtr_->glRowsPerLoop = glRowsPerLoop_;
}

// workspace 段偏移:与 GetWorkspaceSize 的累加顺序严格一致,两处取自同一组成员变量。
void FfnWorkerBatchingTilingArch35::LayoutWorkspace()
{
    // ---------------- workspace 段偏移（以 int32 word 计）----------------
    // 布局与 GetWorkspaceSize 的累加顺序严格一致，两处取自同一组成员变量。
    int64_t off = MAX_RESERVE_WK_NUM;
    wsFlatIds_ = off;
    off += flatElements_;
    wsPairA_ = off;
    off += sortSegNum_ * sortLenPerSeg_;
    wsPairB_ = off;
    off += sortSegNum_ * sortLenPerSeg_;
    wsSegCnt_ = off;
    off += sortSegNum_ * INT32_PER_BLOCK;
    wsSortedIds_ = off;
    off += Y_;
    wsGatherIdx_ = off;
    off += Y_;
    // off 以 int32 word 计；8 words = 32B，先把快照起点对齐到数据块。
    // 异步载荷为 (A+1)*32B：头块低 8B 存选中 micro batch，后 A 块低 4B 各存一个 flag。
    // A=1024 时载荷为 32800B；末尾可能另有最多 28B 的起点对齐空隙。
    // 默认路径只记录段起点，不分配快照载荷；其 kernel 也不会访问该段。
    off = (off + INT32_PER_BLOCK - 1) / INT32_PER_BLOCK * INT32_PER_BLOCK;
    tilingDataPtr_->wsReady = off;
    if (needSchedule_ == 1 && syncFlag_) {
        off += (A_ + 1) * INT32_PER_BLOCK;
    }
    userWorkspaceWords_ = off;

    tilingDataPtr_->wsFlatIds = wsFlatIds_;
    tilingDataPtr_->wsPairA = wsPairA_;
    tilingDataPtr_->wsPairB = wsPairB_;
    tilingDataPtr_->wsSegCnt = wsSegCnt_;
    tilingDataPtr_->wsSortedIds = wsSortedIds_;
    tilingDataPtr_->wsGatherIdx = wsGatherIdx_;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t FfnWorkerBatchingTilingArch35::GetTilingKey() const
{
    // TilingKey 只由 need_schedule 决定（token_dtype 正交，kernel 内处理）。
    return needSchedule_ == 0 ? static_cast<uint64_t>(TILING_KEY_NORM) : static_cast<uint64_t>(TILING_KEY_RECV);
}

ge::graphStatus FfnWorkerBatchingTilingArch35::GetWorkspaceSize()
{
    // 用户区总量由 DoOpTiling 逐段累加得到（userWorkspaceWords_），此处不另算一遍：
    // 段偏移与总量出自同一次累加，避免布局在两处各写一遍而悄悄错位。
    workspaceSize_ = userWorkspaceWords_ * static_cast<int64_t>(sizeof(int32_t)) + sysWorkspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::PostTiling()
{
    context_->SetBlockDim(coreNum_);
    context_->SetScheduleMode(BATCH_MODE);

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(workspaceSize_);

    OP_LOGI(context_->GetNodeName(),
            "arch35 tiling: coreNum:%ld ubSize:%ld Y:%ld H:%ld tokenDtype:%ld expertNum:%ld flatElements:%ld "
            "prepRows:%ld "
            "sortSegNum:%ld sortPerSeg:%ld sortLenPerSeg:%ld mergeRounds:%ld mergeLoop:%ld extractLoop:%ld "
            "glRows:%ld wsWords:%ld tilingKey:%lu",
            tilingDataPtr_->coreNum, tilingDataPtr_->ubSize, tilingDataPtr_->Y, tilingDataPtr_->H,
            tilingDataPtr_->tokenDtype, tilingDataPtr_->expertNum, tilingDataPtr_->flatElements,
            tilingDataPtr_->preparePerLoopRows, tilingDataPtr_->sortSegNum, tilingDataPtr_->sortPerSegElements,
            tilingDataPtr_->sortLenPerSeg, tilingDataPtr_->mergeRounds, tilingDataPtr_->mergeOneLoopElements,
            tilingDataPtr_->extractPerLoopElements, tilingDataPtr_->glRowsPerLoop, userWorkspaceWords_, GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(FfnWorkerBatching, FfnWorkerBatchingTilingArch35, 1000);
} // namespace optiling
