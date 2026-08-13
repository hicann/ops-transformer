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
#include "../op_kernel/arch35/ffn_worker_batching_arch35_tiling_def.h"

namespace optiling {
namespace {
constexpr uint32_t EXPERT_NUM_ATTR = 0;
constexpr uint32_t MAX_OUT_SHAPE_ATTR = 1;
constexpr uint32_t TOKEN_DTYPE_ATTR = 2;
constexpr uint32_t NEED_SCHEDULE_ATTR = 3;
constexpr uint32_t LAY_NUM_ATTR = 4;

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
constexpr int64_t NUM_EIGHT = 8;
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

// 本算子的 SIMT(asc_vf_call) 与 VF 计算需要 UB 系统预留区。预留量不自行相减，
// 而是通过平台接口 ReserveLocalMemory 声明——之后 GetCoreMemSize(UB) 返回的即为可用值。
// ReservedSize 是平台定义的枚举（8K/16K/32K），选择依据是本算子用到 SIMT,取最大档。
} // namespace

class FfnWorkerBatchingTilingArch35 : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit FfnWorkerBatchingTilingArch35(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
    }
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
    ge::graphStatus ParseMaxOutShape(const gert::RuntimeAttrs *attrs);
    ge::graphStatus ParseOptionalAttrs(const gert::RuntimeAttrs *attrs, int64_t expertNum);
    void SplitPrepare();
    void SplitSortAndMerge();
    void SplitGroupList();
    void LayoutWorkspace();

    FfnWorkerBatchingArch35TilingData *tilingDataPtr_ = nullptr;
    int64_t A_ = 0;
    int64_t BS_ = 0;
    int64_t K_ = 0;
    int64_t Y_ = 0;
    int64_t H_ = 0;
    int64_t expertNum_ = 0;
    int64_t tokenDtype_ = 0;
    int64_t needSchedule_ = 0;
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
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_->GetNodeName(), "platformInfo is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    aivNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(aivNum_ == 0, OP_LOGE(context_->GetNodeName(), "Get aivNum failed."), return ge::GRAPH_FAILED);

    // 先声明预留，再取容量：平台在 GetCoreMemSize 中已扣除本次预留，得到 vector core 真实可用 UB。
    ascendcPlatform.ReserveLocalMemory(platform_ascendc::ReservedSize::RESERVED_SIZE_32K);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF(ubSize_ == 0, OP_LOGE(context_->GetNodeName(), "Get ubSize failed: 0."), return ge::GRAPH_FAILED);

    sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::CheckInputParam()
{
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType xdtype = inputDesc->GetDataType();
    OP_CHECK_IF(xdtype != ge::DT_INT8,
                OP_LOGE(context_->GetNodeName(), "Input dtype:%s not int8", Ops::Base::ToString(xdtype).c_str()),
                return ge::GRAPH_FAILED);

    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto xShape = Ops::Transformer::OpTiling::EnsureNotScalar(inputX->GetStorageShape());
    OP_CHECK_IF(xShape.GetDimNum() != 1,
                OP_LOGE(context_->GetNodeName(), "x shape %s dim num not 1", Ops::Base::ToString(xShape).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::GetAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t *expertNumPtr = attrs->GetAttrPointer<int64_t>(EXPERT_NUM_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertNumPtr);
    OP_CHECK_IF(
        *expertNumPtr > EXPERT_IDX_MAX || *expertNumPtr <= 0,
        OP_LOGE(context_->GetNodeName(), "expert_num:%ld should be in range (0, %ld]", *expertNumPtr, EXPERT_IDX_MAX),
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
ge::graphStatus FfnWorkerBatchingTilingArch35::ParseMaxOutShape(const gert::RuntimeAttrs *attrs)
{
    const gert::ContinuousVector *maxOutShapePtr = attrs->GetAttrPointer<gert::ContinuousVector>(MAX_OUT_SHAPE_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maxOutShapePtr);
    OP_CHECK_IF(maxOutShapePtr->GetSize() != static_cast<size_t>(NUM_FOUR),
                OP_LOGE(context_->GetNodeName(), "The max_out_shape size:%lu not equal 4.", maxOutShapePtr->GetSize()),
                return ge::GRAPH_FAILED);
    const int64_t *maxOutShapeArray = reinterpret_cast<const int64_t *>(maxOutShapePtr->GetData());
    A_ = maxOutShapeArray[INDEX_ZERO];
    BS_ = maxOutShapeArray[INDEX_ONE];
    K_ = maxOutShapeArray[INDEX_TWO];
    H_ = maxOutShapeArray[INDEX_THREE];
    OP_CHECK_IF(
        (A_ > MAX_SESSION_NUM || A_ <= 0),
        OP_LOGE(context_->GetNodeName(), "max_out_shape[0]:%ld should be in range of (0, %ld]", A_, MAX_SESSION_NUM),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(BS_ <= 0, OP_LOGE(context_->GetNodeName(), "max_out_shape[1]:%ld should be greater than 0", BS_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((K_ > MAX_K_NUM || K_ <= 0),
                OP_LOGE(context_->GetNodeName(), "max_out_shape[2]:%ld should be in range of (0, %ld]", K_, MAX_K_NUM),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(H_ <= 0, OP_LOGE(context_->GetNodeName(), "max_out_shape[3]:%ld should be greater than 0", H_),
                return ge::GRAPH_FAILED);

    Y_ = A_ * BS_ * K_;

    return ge::GRAPH_SUCCESS;
}

// 可选属性:缺省时保留成员初值,给出即逐个校验取值域。
ge::graphStatus FfnWorkerBatchingTilingArch35::ParseOptionalAttrs(const gert::RuntimeAttrs *attrs, int64_t expertNum)
{
    const int64_t *tokenDtype = attrs->GetAttrPointer<int64_t>(TOKEN_DTYPE_ATTR);
    if (tokenDtype != nullptr) {
        OP_CHECK_IF((*tokenDtype < 0 || *tokenDtype > NUM_TWO),
                    OP_LOGE(context_->GetNodeName(), "token_dtype:%ld must be one of [0, 1, 2]", *tokenDtype),
                    return ge::GRAPH_FAILED);
        tokenDtype_ = *tokenDtype;
    }

    const int64_t *needSchedulePtr = attrs->GetAttrPointer<int64_t>(NEED_SCHEDULE_ATTR);
    if (needSchedulePtr != nullptr) {
        OP_CHECK_IF((*needSchedulePtr < 0 || *needSchedulePtr > 1),
                    OP_LOGE(context_->GetNodeName(), "need_schedule:%ld must be one of [0, 1]", *needSchedulePtr),
                    return ge::GRAPH_FAILED);
        needSchedule_ = *needSchedulePtr;
    }

    const int64_t *layNumPtr = attrs->GetAttrPointer<int64_t>(LAY_NUM_ATTR);
    if (layNumPtr != nullptr) {
        OP_CHECK_IF(
            (*layNumPtr < 0 || *layNumPtr > expertNum),
            OP_LOGE(context_->GetNodeName(), "layer_num:%ld must be in range of [0, %ld]", *layNumPtr, expertNum),
            return ge::GRAPH_FAILED);
        layerNum_ = *layNumPtr;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckInputParam() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckInputParam failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetAttrsInfo failed."),
                return ge::GRAPH_FAILED);
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
    tilingDataPtr_->flatElements = flatElements_;

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
    int64_t prepRows = static_cast<int64_t>(ubSize_) / std::max<int64_t>(1, prepBytesPerRow);
    prepRows = std::max<int64_t>(1, std::min<int64_t>(prepRows, A_));
    preparePerLoopRows_ = prepRows;
    tilingDataPtr_->preparePerLoopRows = preparePerLoopRows_;
}

// phase1~3:段内排序(VBS)、段间归并(VMS)与归并收尾(Extract)的切分,三者共用同一份 UB 预算。
void FfnWorkerBatchingTilingArch35::SplitSortAndMerge()
{
    // ---------------- 段内排序（VBS）的切分 ----------------
    // 一段在 UB 中同时驻留：输入 id + 原下标（各 4B）、proposal 对区与排序临时区（各 8B/元素）、
    // 比较掩码（4B）。故每元素占用 SORT_UB_BYTES_PER_ELEM 字节，据此反推单段元素数上限。
    // ubSize_ 在 GetPlatformInfo 中已扣除系统预留，此处直接使用，勿重复扣减。
    const int64_t ubAvail = static_cast<int64_t>(ubSize_);
    int64_t segCap = ubAvail / SORT_UB_BYTES_PER_ELEM / ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;
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
    const int64_t mergeOther = sortSegNum_ * ONE_BLK_BYTES + (MRG_LIST_NUM + NUM_TWO) * ONE_BLK_BYTES;
    const int64_t mergeUb = (ubAvail > mergeOther) ? (ubAvail - mergeOther) : ubAvail;
    int64_t mergeLoop = mergeUb / (MRG_LIST_NUM * NUM_TWO * SORT_PAIR_FLOATS * static_cast<int64_t>(sizeof(float))) /
                        ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;
    mergeOneLoopElements_ = std::max<int64_t>(ONE_REPEAT_SORT_NUM, mergeLoop);
    tilingDataPtr_->mergeRounds = mergeRounds_;
    tilingDataPtr_->mergeOneLoopElements = mergeOneLoopElements_;

    // ---------------- 归并收尾（Extract）的切分 ----------------
    // 单次驻留：proposal 对（8B）+ 拆出的 id 与 idx（各 4B）。
    int64_t extractLoop = ubAvail / EXTRACT_UB_BYTES_PER_ELEM / ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;
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
    glRowsPerLoop_ = std::max<int64_t>(NUM_TWO, std::min<int64_t>(rows, expertNum_));

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
    off += sortSegNum_ * (ONE_BLK_BYTES / static_cast<int64_t>(sizeof(int32_t)));
    wsSortedIds_ = off;
    off += Y_;
    wsGatherIdx_ = off;
    off += Y_;
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

    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
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
