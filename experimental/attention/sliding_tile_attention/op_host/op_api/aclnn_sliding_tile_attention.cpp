/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sliding_tile_attention.h"
#include "../../../../../attention/block_sparse_attention/op_host/op_api/block_sparse_attention.h"
#include "sliding_tile_attention.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/slice.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace {

constexpr int64_t TILE_T = 6;
constexpr int64_t TILE_H = 8;
constexpr int64_t TILE_W = 8;
constexpr int64_t SPARSE_BLOCK = TILE_T * TILE_H * TILE_W;
// Largest T, H or W of seqShape.
constexpr int64_t MAX_CANVAS_DIM = 200;
constexpr int64_t MAX_HEAD_DIM = 128;
// Head dims the kernels run. This operator's kernel takes 32, 48 and 64 (narrower heads run on the D=32 kernel: the
// D=16 cube loads are several times slower than D=32 ones); the upstream BlockSparseAttention takes 64 and 128. Other
// head dims are zero-padded to the next of these.
constexpr int64_t KERNEL_HEAD_DIM_NARROW = 32;
constexpr int64_t KERNEL_HEAD_DIM_MEDIUM = 48;
constexpr int64_t BSA_HEAD_DIM = 64;
constexpr int64_t BSA_HEAD_DIM_WIDE = 128;
// The text region is one 384-token block, both as KV and as query rows.
constexpr int64_t MAX_TEXT_LENGTH = 384;
constexpr uint64_t WINDOW_DIM_NUM = 3;
// tileGrid attribute of the kernel: T | H << 16 | W << 32.
constexpr int64_t TILE_GRID_SHIFT = 16;
// KV block list bytes of one kernel launch; the heads of larger inputs run in several launches.
constexpr int64_t MAX_LAUNCH_LIST_BYTES = 1LL << 31;
constexpr int64_t LIST_ENTRY_BYTES = sizeof(int32_t);
constexpr int64_t LIST_ENTRIES_PER_C0 = 8;
// Largest block mask that may be attached to one launch as host data. nnopbase keeps launch host data in an
// 800 KB tiling host buffer; exceeding it makes the next execution in the same process fail with 507015.
// 777600 bytes were verified to run repeatedly; the task-book case with 8 distinct windows needs 724808.
constexpr int64_t MAX_HOST_MASK_BYTES = 760 * 1024;

template <typename T>
struct FunctionArity;

template <typename Ret, typename... Args>
struct FunctionArity<Ret (*)(Args...)> : std::integral_constant<size_t, sizeof...(Args)> {};

template <typename BsaFn>
static auto CallBlockSparseAttentionCompat(BsaFn bsa, const aclTensor* query, const aclTensor* key,
                                           const aclTensor* value, const aclTensor* blockMask,
                                           const aclIntArray* blockShape, const aclIntArray* actualSeqLengths,
                                           const aclIntArray* actualSeqLengthsKv, int64_t numHeads, double scaleValue,
                                           int64_t innerPrecise, const aclTensor* output, aclOpExecutor* executor)
{
    constexpr size_t arity = FunctionArity<BsaFn>::value;
    static_assert(arity == 24U || arity == 27U, "unsupported BlockSparseAttention L0 signature");
    if constexpr (arity == 27U) {
        return bsa(query, key, value, blockMask, nullptr, blockShape, actualSeqLengths, actualSeqLengthsKv, nullptr,
                   nullptr, nullptr, nullptr, nullptr, "BNSD", "BNSD", numHeads, 0, scaleValue, innerPrecise, 0,
                   2147483647, 2147483647, 0, 0, 0.0, output, executor);
    } else {
        return bsa(query, key, value, blockMask, nullptr, blockShape, actualSeqLengths, actualSeqLengthsKv, nullptr,
                   nullptr, nullptr, nullptr, "BNSD", "BNSD", numHeads, 0, scaleValue, innerPrecise, 0, 2147483647,
                   2147483647, 0, output, executor);
    }
}

static uint32_t ClampWindowStart(uint32_t coordinate, uint32_t window, uint32_t grid)
{
    window = std::min(window, grid);
    const int64_t centered = static_cast<int64_t>(coordinate) - static_cast<int64_t>(window / 2U);
    const int64_t maximum = static_cast<int64_t>(grid - window);
    return static_cast<uint32_t>(std::max<int64_t>(0, std::min(centered, maximum)));
}

struct TileGrid {
    uint32_t tilesT;
    uint32_t tilesH;
    uint32_t tilesW;
    uint32_t imageTokens;
    uint32_t textLength;
    uint32_t blockCount;
};

// Fills the [queryBlockNum, blockCount] block mask of one head for the query blocks starting at queryBlockBegin.
static void FillHeadBlockMask(const TileGrid& grid, uint32_t windowT, uint32_t windowH, uint32_t windowW,
                              uint32_t queryBlockBegin, uint32_t queryBlockNum, uint8_t* mask)
{
    const uint64_t blockCount = grid.blockCount;
    const uint32_t firstTextBlock = grid.imageTokens / SPARSE_BLOCK;
    // Key blocks of the valid KV length: the text block is one of them only when text keys exist.
    const uint32_t kvBlockCount = firstTextBlock + (grid.textLength != 0 ? 1U : 0U);
    for (uint32_t rowIdx = 0; rowIdx < queryBlockNum; ++rowIdx) {
        const uint32_t queryBlock = queryBlockBegin + rowIdx;
        uint8_t* row = mask + static_cast<uint64_t>(rowIdx) * blockCount;
        std::fill(row, row + blockCount, static_cast<uint8_t>(0));
        const uint32_t firstQuery = queryBlock * SPARSE_BLOCK;
        if (firstQuery >= grid.imageTokens) {
            std::fill(row, row + kvBlockCount, static_cast<uint8_t>(1));
            continue;
        }

        const uint32_t queryTile = firstQuery / SPARSE_BLOCK;
        const uint32_t queryTileW = queryTile % grid.tilesW;
        const uint32_t queryTileH = (queryTile / grid.tilesW) % grid.tilesH;
        const uint32_t queryTileT = queryTile / (grid.tilesH * grid.tilesW);
        const uint32_t startT = ClampWindowStart(queryTileT, windowT, grid.tilesT);
        const uint32_t startH = ClampWindowStart(queryTileH, windowH, grid.tilesH);
        const uint32_t startW = ClampWindowStart(queryTileW, windowW, grid.tilesW);
        for (uint32_t tileTIdx = startT; tileTIdx < startT + windowT; ++tileTIdx) {
            for (uint32_t tileHIdx = startH; tileHIdx < startH + windowH; ++tileHIdx) {
                for (uint32_t tileWIdx = startW; tileWIdx < startW + windowW; ++tileWIdx) {
                    row[(tileTIdx * grid.tilesH + tileHIdx) * grid.tilesW + tileWIdx] = 1;
                }
            }
        }
        if (grid.textLength != 0) {
            std::fill(row + firstTextBlock, row + blockCount, static_cast<uint8_t>(1));
        }
    }
}

// FastVideo centers a window on the query tile with `window / 2` tiles on each side, so a window of an even number
// of tiles covers window + 1 tiles; the result is limited to the grid.
static uint32_t EffectiveWindowDim(int64_t window, uint32_t grid)
{
    const int64_t tiles = 2 * (window / 2) + 1;
    return static_cast<uint32_t>(std::min<int64_t>(tiles, grid));
}

static std::array<uint32_t, WINDOW_DIM_NUM> EffectiveWindow(const TileGrid& grid,
                                                            const std::vector<int64_t>& windowSize, int64_t headIdx)
{
    const size_t base = static_cast<size_t>(headIdx) * WINDOW_DIM_NUM;
    return {EffectiveWindowDim(windowSize[base], grid.tilesT), EffectiveWindowDim(windowSize[base + 1U], grid.tilesH),
            EffectiveWindowDim(windowSize[base + 2U], grid.tilesW)};
}

static bool SharedWindow(const TileGrid& grid, const std::vector<int64_t>& windowSize, int64_t headBegin,
                         int64_t headEnd)
{
    const auto firstWindow = EffectiveWindow(grid, windowSize, headBegin);
    for (int64_t headIdx = headBegin + 1; headIdx < headEnd; ++headIdx) {
        if (EffectiveWindow(grid, windowSize, headIdx) != firstWindow) {
            return false;
        }
    }
    return true;
}

// KV blocks an image query block of the head selects: its window, plus the text block when text keys exist.
static int64_t HeadSelectNum(const TileGrid& grid, const std::vector<int64_t>& windowSize, int64_t headIdx)
{
    const auto window = EffectiveWindow(grid, windowSize, headIdx);
    return static_cast<int64_t>(window[0]) * window[1] * window[2] + (grid.textLength != 0 ? 1 : 0);
}

// Builds the [B, heads, queryBlockNum, blocks] BlockSparseAttention mask of the query blocks starting at
// queryBlockBegin for heads [headBegin, headEnd). Host data is one plane when all heads share a window and one plane
// per head otherwise (callers keep it within MAX_HOST_MASK_BYTES); the batch dimension is always broadcast.
static const aclTensor* BuildDeviceBlockMask(aclOpExecutor* executor, int64_t batch, int64_t headBegin, int64_t headEnd,
                                             const TileGrid& grid, const std::vector<int64_t>& windowSize,
                                             int64_t queryBlockBegin, int64_t queryBlockNum)
{
    const int64_t heads = headEnd - headBegin;
    const int64_t blocks = grid.blockCount;
    const int64_t plane = queryBlockNum * blocks;
    const int64_t hostHeads = SharedWindow(grid, windowSize, headBegin, headEnd) ? 1 : heads;

    std::vector<uint8_t> hostMask(static_cast<size_t>(hostHeads * plane), 0);
    for (int64_t headIdx = 0; headIdx < hostHeads; ++headIdx) {
        const auto window = EffectiveWindow(grid, windowSize, headBegin + headIdx);
        FillHeadBlockMask(grid, window[0], window[1], window[2], static_cast<uint32_t>(queryBlockBegin),
                          static_cast<uint32_t>(queryBlockNum), hostMask.data() + headIdx * plane);
    }
    const aclTensor* planes = executor->ConvertToTensor(hostMask.data(), hostMask.size(), DataType::DT_INT8);
    CHECK_RET(planes != nullptr, nullptr);
    const op::Strides strides({0, hostHeads == 1 ? 0 : plane, blocks, 1});
    const aclTensor* broadcast = executor->CreateView(planes, op::Shape({batch, heads, queryBlockNum, blocks}),
                                                      planes->GetStorageShape(), strides, 0);
    CHECK_RET(broadcast != nullptr, nullptr);
    return l0op::Contiguous(broadcast, executor);
}

// Builds the [1, heads, 1, 3] INT8 window table of this operator's kernel for heads [headBegin, headEnd) and returns
// the largest number of KV blocks an image query block selects.
static const aclTensor* BuildDeviceWindowTable(aclOpExecutor* executor, int64_t headBegin, int64_t headEnd,
                                               const TileGrid& grid, const std::vector<int64_t>& windowSize,
                                               int64_t& maxSelectNum)
{
    const int64_t heads = headEnd - headBegin;
    std::vector<int8_t> table(static_cast<size_t>(heads) * WINDOW_DIM_NUM, 0);
    maxSelectNum = 0;
    for (int64_t headIdx = 0; headIdx < heads; ++headIdx) {
        // Effective windows are at most 33 tiles (T <= 200 tokens) and fit in INT8.
        const auto window = EffectiveWindow(grid, windowSize, headBegin + headIdx);
        for (uint64_t dimIndex = 0; dimIndex < WINDOW_DIM_NUM; ++dimIndex) {
            table[static_cast<size_t>(headIdx) * WINDOW_DIM_NUM + dimIndex] = static_cast<int8_t>(window[dimIndex]);
        }
        maxSelectNum = std::max(maxSelectNum, HeadSelectNum(grid, windowSize, headBegin + headIdx));
    }
    const aclTensor* hostTensor = executor->ConvertToTensor(table.data(), table.size(), DataType::DT_INT8);
    CHECK_RET(hostTensor != nullptr, nullptr);
    const int64_t dimNum = static_cast<int64_t>(WINDOW_DIM_NUM);
    const aclTensor* view =
        executor->CreateView(hostTensor, op::Shape({1, heads, 1, dimNum}), hostTensor->GetStorageShape(),
                             op::Strides({heads * dimNum, dimNum, dimNum, 1}), 0);
    CHECK_RET(view != nullptr, nullptr);
    return l0op::Contiguous(view, executor);
}

// Parses one decimal field of seqShape; returns the character after it, or nullptr when the field is not a number
// in [1, MAX_CANVAS_DIM].
static const char* ParseCanvasDim(const char* text, int64_t& value)
{
    value = 0;
    const char* cursor = text;
    while (*cursor >= '0' && *cursor <= '9') {
        value = value * 10 + (*cursor - '0');
        if (value > MAX_CANVAS_DIM) {
            return nullptr;
        }
        ++cursor;
    }
    if (cursor == text || value == 0) {
        return nullptr;
    }
    return cursor;
}

// Skips the separator of seqShape fields ('x', 'X' or the multiplication sign U+00D7); returns nullptr when there
// is none.
static const char* SkipCanvasSeparator(const char* text)
{
    // UTF-8 encoding of U+00D7.
    constexpr unsigned char MULTIPLY_SIGN_LEAD = 0xC3;
    constexpr unsigned char MULTIPLY_SIGN_TAIL = 0x97;
    if (text == nullptr) {
        return nullptr;
    }
    if (*text == 'x' || *text == 'X') {
        return text + 1;
    }
    if (static_cast<unsigned char>(text[0]) == MULTIPLY_SIGN_LEAD &&
        static_cast<unsigned char>(text[1]) == MULTIPLY_SIGN_TAIL) {
        return text + 2;
    }
    return nullptr;
}

// seqShape is "TxHxW" (image tokens along time, height and width, each in [1, 200] and divisible by the 6x8x8
// tile), or "1", "2", "3" for the three task-book shapes as the ATK case files pass them.
static bool ResolveSeqShape(const char* seqShape, int64_t& canvasT, int64_t& canvasH, int64_t& canvasW)
{
    if (std::strcmp(seqShape, "1") == 0) {
        canvasT = 30;
        canvasH = 48;
        canvasW = 80;
        return true;
    }
    if (std::strcmp(seqShape, "2") == 0) {
        canvasT = 36;
        canvasH = 48;
        canvasW = 48;
        return true;
    }
    if (std::strcmp(seqShape, "3") == 0) {
        canvasT = 18;
        canvasH = 48;
        canvasW = 80;
        return true;
    }
    const char* cursor = SkipCanvasSeparator(ParseCanvasDim(seqShape, canvasT));
    cursor = (cursor == nullptr) ? nullptr : SkipCanvasSeparator(ParseCanvasDim(cursor, canvasH));
    cursor = (cursor == nullptr) ? nullptr : ParseCanvasDim(cursor, canvasW);
    if (cursor == nullptr || *cursor != '\0') {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "seqShape must be \"TxHxW\" with T, H and W in [1, 200] (or 1, 2, 3), but got %s", seqShape);
        return false;
    }
    if (canvasT % TILE_T != 0 || canvasH % TILE_H != 0 || canvasW % TILE_W != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "seqShape %s must have T divisible by 6 and H, W divisible by 8 (6x8x8 tiles)",
                seqShape);
        return false;
    }
    return true;
}

static bool CheckDtype(const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* output)
{
    const auto qDtype = query->GetDataType();
    if (qDtype != DataType::DT_FLOAT16 && qDtype != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query dtype must be fp16 or bf16");
        return false;
    }
    if (key->GetDataType() != qDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "key dtype must equal query dtype");
        return false;
    }
    if (value->GetDataType() != qDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "value dtype must equal query dtype");
        return false;
    }
    if (output->GetDataType() != qDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "q/k/v/output dtype must be consistent");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* output)
{
    const auto& queryShape = query->GetViewShape();
    if (queryShape.GetDimNum() != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query must be 4D BNSD tensor");
        return false;
    }
    if (key->GetViewShape() != queryShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "key shape must equal query shape");
        return false;
    }
    if (value->GetViewShape() != queryShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "value shape must equal query shape");
        return false;
    }
    if (output->GetViewShape() != queryShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "output shape must equal query shape");
        return false;
    }
    for (size_t dimIndex = 0; dimIndex < 4U; ++dimIndex) {
        if (queryShape.GetDim(dimIndex) <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "all BNSD dimensions must be positive");
            return false;
        }
    }
    if (queryShape.GetDim(3) > MAX_HEAD_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "headDim must be no greater than 128");
        return false;
    }
    return true;
}

static aclnnStatus CheckAndFlattenWindowSize(const aclIntArray* const* windowSize, uint64_t windowSizeLen,
                                             int64_t numHeads, std::vector<int64_t>& flattenedWindowSize)
{
    if (windowSizeLen != 1U && windowSizeLen != static_cast<uint64_t>(numHeads)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "windowSizeLen must be 1 or equal to numHeads");
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (uint64_t windowIndex = 0; windowIndex < windowSizeLen; ++windowIndex) {
        if (windowSize[windowIndex] == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "windowSize[%lu] must not be nullptr", windowIndex);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (windowSize[windowIndex]->Size() != WINDOW_DIM_NUM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "each windowSize item must contain exactly three values");
            return ACLNN_ERR_PARAM_INVALID;
        }
        for (uint64_t dimIndex = 0; dimIndex < WINDOW_DIM_NUM; ++dimIndex) {
            const int64_t windowValue = (*windowSize[windowIndex])[dimIndex];
            if (windowValue <= 0 || windowValue > std::numeric_limits<int32_t>::max()) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "windowSize values must be positive integers no greater than INT32_MAX");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }

    flattenedWindowSize.resize(static_cast<size_t>(numHeads) * WINDOW_DIM_NUM);
    for (int64_t headIndex = 0; headIndex < numHeads; ++headIndex) {
        const uint64_t windowIndex = windowSizeLen == 1U ? 0U : static_cast<uint64_t>(headIndex);
        for (uint64_t dimIndex = 0; dimIndex < WINDOW_DIM_NUM; ++dimIndex) {
            flattenedWindowSize[static_cast<size_t>(headIndex) * WINDOW_DIM_NUM + dimIndex] =
                (*windowSize[windowIndex])[dimIndex];
        }
    }
    return ACLNN_SUCCESS;
}

static bool CheckSequence(const aclTensor* query, int64_t textLength, bool hasText, int64_t canvasT, int64_t canvasH,
                          int64_t canvasW)
{
    if (textLength < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "textLength must be non-negative");
        return false;
    }
    if (hasText && textLength > MAX_TEXT_LENGTH) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "textLength must not exceed 384");
        return false;
    }
    const int64_t imgSeqLen = canvasT * canvasH * canvasW;
    const int64_t seqLen = query->GetViewShape().GetDim(2);
    if (!hasText && seqLen != imgSeqLen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "without text, query S must equal the image sequence length");
        return false;
    }
    if (hasText && (seqLen < imgSeqLen + textLength || seqLen > imgSeqLen + MAX_TEXT_LENGTH)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "with text, query S must cover image tokens plus textLength and reserve at most 384 text slots");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* query, const aclTensor* key, const aclTensor* value, aclTensor* output,
                               const aclIntArray* const* windowSize, uint64_t windowSizeLen, int64_t textLength,
                               bool hasText, const char* seqShape, uint64_t* workspaceSize, aclOpExecutor** executor,
                               int64_t& canvasT, int64_t& canvasH, int64_t& canvasW,
                               std::vector<int64_t>& flattenedWindowSize)
{
    if (query == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "query must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (key == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "key must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (value == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "value must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "output must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (windowSize == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "windowSize must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (seqShape == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "seqShape must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (workspaceSize == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    CHECK_RET(CheckDtype(query, key, value, output), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(query, key, value, output), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ResolveSeqShape(seqShape, canvasT, canvasH, canvasW), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSequence(query, textLength, hasText, canvasT, canvasH, canvasW), ACLNN_ERR_PARAM_INVALID);
    const int64_t numHeads = query->GetViewShape().GetDim(1);
    const aclnnStatus windowStatus =
        CheckAndFlattenWindowSize(windowSize, windowSizeLen, numHeads, flattenedWindowSize);
    CHECK_RET(windowStatus == ACLNN_SUCCESS, windowStatus);
    return ACLNN_SUCCESS;
}

// The upstream BlockSparseAttention runs D=64 inputs whose block mask fits one launch's host data, and heads wider
// than 64; this operator's kernel runs the rest, with its own KV block lists instead of a block mask.
enum class SparseBackend {
    UPSTREAM_BSA,
    STA_KERNEL
};

struct SparsePlan {
    SparseBackend backend;
    int64_t runHeadDim; // head dim the chosen kernel runs; narrower inputs are zero-padded to it
};

struct BsaCallInfo {
    int64_t batch;
    int64_t sequence;
    int64_t validSequence;
    double scaleValue;
    TileGrid grid;
};

static SparsePlan ChooseSparsePlan(int64_t headDim, const TileGrid& grid)
{
    if (headDim <= KERNEL_HEAD_DIM_NARROW) {
        return {SparseBackend::STA_KERNEL, KERNEL_HEAD_DIM_NARROW};
    }
    if (headDim <= KERNEL_HEAD_DIM_MEDIUM) {
        return {SparseBackend::STA_KERNEL, KERNEL_HEAD_DIM_MEDIUM};
    }
    if (headDim <= BSA_HEAD_DIM) {
        const int64_t plane = static_cast<int64_t>(grid.blockCount) * grid.blockCount;
        const SparseBackend backend =
            plane <= MAX_HOST_MASK_BYTES ? SparseBackend::UPSTREAM_BSA : SparseBackend::STA_KERNEL;
        return {backend, BSA_HEAD_DIM};
    }
    return {SparseBackend::UPSTREAM_BSA, BSA_HEAD_DIM_WIDE};
}

static const aclTensor* PadHeadDim(aclOpExecutor* executor, const aclTensor* tensor, int64_t padNum)
{
    // Pad takes (before, after) per axis; only the tail of the head dimension is padded, with zeros.
    const int64_t paddings[8] = {0, 0, 0, 0, 0, 0, 0, padNum};
    aclIntArray* padArray = executor->AllocIntArray(paddings, 8);
    CHECK_RET(padArray != nullptr, nullptr);
    const aclTensor* padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);
    CHECK_RET(padTensor != nullptr, nullptr);
    return l0op::Pad(tensor, padTensor, executor);
}

static const aclTensor* SliceHeadDim(aclOpExecutor* executor, const aclTensor* tensor, const op::Shape& shape)
{
    const int64_t offsets[4] = {0, 0, 0, 0};
    const int64_t sizes[4] = {shape.GetDim(0), shape.GetDim(1), shape.GetDim(2), shape.GetDim(3)};
    aclIntArray* offsetArray = executor->AllocIntArray(offsets, 4);
    CHECK_RET(offsetArray != nullptr, nullptr);
    aclIntArray* sizeArray = executor->AllocIntArray(sizes, 4);
    CHECK_RET(sizeArray != nullptr, nullptr);
    return l0op::Slice(tensor, offsetArray, sizeArray, executor);
}

// Zero-padded query/key columns leave every Q*K^T score unchanged (info.scaleValue still uses the real head dim),
// and the padded value columns only reach output columns that are sliced away. The padded tensors are private
// copies, which the kernel may rewrite (see the tiling).
static bool PadInputs(aclOpExecutor* executor, int64_t runHeadDim, const aclTensor*& query, const aclTensor*& key,
                      const aclTensor*& value)
{
    const int64_t padNum = runHeadDim - query->GetViewShape().GetDim(3);
    query = PadHeadDim(executor, query, padNum);
    CHECK_RET(query != nullptr, false);
    key = PadHeadDim(executor, key, padNum);
    CHECK_RET(key != nullptr, false);
    value = PadHeadDim(executor, value, padNum);
    CHECK_RET(value != nullptr, false);
    return true;
}

static const aclTensor* RunStaKernel(aclOpExecutor* executor, const aclTensor* query, const aclTensor* key,
                                     const aclTensor* value, const BsaCallInfo& info,
                                     const std::vector<int64_t>& windowSize, int64_t headBegin, int64_t headEnd,
                                     int64_t runHeadDim)
{
    const op::Shape shape = query->GetViewShape();
    const int64_t headDim = shape.GetDim(3);
    int64_t maxSelectNum = 0;
    const aclTensor* windows =
        BuildDeviceWindowTable(executor, headBegin, headEnd, info.grid, windowSize, maxSelectNum);
    CHECK_RET(windows != nullptr, nullptr);
    const int64_t tileGrid = static_cast<int64_t>(info.grid.tilesT) |
                             (static_cast<int64_t>(info.grid.tilesH) << TILE_GRID_SHIFT) |
                             (static_cast<int64_t>(info.grid.tilesW) << (2 * TILE_GRID_SHIFT));
    if (headDim == runHeadDim) {
        return l0op::SlidingTileAttention(query, key, value, windows, info.validSequence, info.scaleValue, headDim,
                                          tileGrid, maxSelectNum, executor);
    }
    CHECK_RET(PadInputs(executor, runHeadDim, query, key, value), nullptr);
    const aclTensor* paddedOutput = l0op::SlidingTileAttention(
        query, key, value, windows, info.validSequence, info.scaleValue, headDim, tileGrid, maxSelectNum, executor);
    CHECK_RET(paddedOutput != nullptr, nullptr);
    return SliceHeadDim(executor, paddedOutput, shape);
}

static const aclTensor* RunUpstreamBsa(aclOpExecutor* executor, const aclTensor* query, const aclTensor* key,
                                       const aclTensor* value, const BsaCallInfo& info,
                                       const std::vector<int64_t>& windowSize, int64_t headBegin, int64_t headEnd,
                                       int64_t rowBegin, int64_t rowNum, int64_t runHeadDim)
{
    const op::Shape shape = query->GetViewShape();
    const int64_t queryBlockBegin = rowBegin / SPARSE_BLOCK;
    const int64_t queryBlockNum = (rowNum + SPARSE_BLOCK - 1) / SPARSE_BLOCK;
    const aclTensor* blockMask = BuildDeviceBlockMask(executor, info.batch, headBegin, headEnd, info.grid, windowSize,
                                                      queryBlockBegin, queryBlockNum);
    CHECK_RET(blockMask != nullptr, nullptr);
    const bool padded = shape.GetDim(3) != runHeadDim;
    if (padded) {
        CHECK_RET(PadInputs(executor, runHeadDim, query, key, value), nullptr);
    }
    const int64_t blockShapeData[2] = {SPARSE_BLOCK, SPARSE_BLOCK};
    aclIntArray* blockShape = executor->AllocIntArray(blockShapeData, 2);
    CHECK_RET(blockShape != nullptr, nullptr);
    // The sparse mask selects complete 384-token blocks. For a reserved text tail, actualSeqLengthsKv clips
    // the final selected block at textLength, while actualSeqLengths keeps all reserved query rows active.
    const std::vector<int64_t> querySeqLengths(static_cast<size_t>(info.batch), rowNum);
    const std::vector<int64_t> kvSeqLengths(static_cast<size_t>(info.batch), info.validSequence);
    aclIntArray* actualSeqLengths = executor->AllocIntArray(querySeqLengths.data(), querySeqLengths.size());
    CHECK_RET(actualSeqLengths != nullptr, nullptr);
    aclIntArray* actualSeqLengthsKv = executor->AllocIntArray(kvSeqLengths.data(), kvSeqLengths.size());
    CHECK_RET(actualSeqLengthsKv != nullptr, nullptr);
    // FP32 softmax (innerPrecise=0) for fp16 and bf16: the fp16 softmax mode (1) fails the task accuracy
    // threshold on randomized inputs.
    constexpr int64_t innerPrecise = 0;
    // The last tensor argument only supplies the output dtype; the operator allocates its own output.
    auto outputs = CallBlockSparseAttentionCompat(&l0op::BlockSparseAttention, query, key, value, blockMask, blockShape,
                                                  actualSeqLengths, actualSeqLengthsKv, headEnd - headBegin,
                                                  info.scaleValue, innerPrecise, query, executor);
    CHECK_RET(outputs[0] != nullptr, nullptr);
    return padded ? SliceHeadDim(executor, outputs[0], shape) : outputs[0];
}

// Runs the sparse attention of heads [headBegin, headEnd) and query rows [rowBegin, rowBegin + rowNum); query holds
// exactly those heads and rows, key/value those heads and the whole sequence.
static const aclTensor* RunSparseHeads(aclOpExecutor* executor, const aclTensor* query, const aclTensor* key,
                                       const aclTensor* value, const BsaCallInfo& info,
                                       const std::vector<int64_t>& windowSize, int64_t headBegin, int64_t headEnd,
                                       int64_t rowBegin, int64_t rowNum, const SparsePlan& plan)
{
    if (plan.backend == SparseBackend::STA_KERNEL) {
        return RunStaKernel(executor, query, key, value, info, windowSize, headBegin, headEnd, plan.runHeadDim);
    }
    return RunUpstreamBsa(executor, query, key, value, info, windowSize, headBegin, headEnd, rowBegin, rowNum,
                          plan.runHeadDim);
}

static const aclTensor* TensorSlice(aclOpExecutor* executor, const aclTensor* tensor, int64_t headBegin,
                                    int64_t headEnd, int64_t rowBegin, int64_t rowNum)
{
    const auto& shape = tensor->GetViewShape();
    const int64_t offsets[4] = {0, headBegin, rowBegin, 0};
    const int64_t sizes[4] = {shape.GetDim(0), headEnd - headBegin, rowNum, shape.GetDim(3)};
    return l0op::Slice(tensor, executor->AllocIntArray(offsets, 4), executor->AllocIntArray(sizes, 4), executor);
}

// Heads per launch. This operator's kernel keeps the KV block lists of every head of a launch in its workspace, at
// most MAX_LAUNCH_LIST_BYTES of them; the upstream operator splits query rows instead (see RowsPerLaunch).
static int64_t HeadsPerLaunch(const BsaCallInfo& info, const std::vector<int64_t>& windowSize, int64_t numHeads,
                              const SparsePlan& plan)
{
    if (plan.backend == SparseBackend::UPSTREAM_BSA) {
        return numHeads;
    }
    const TileGrid& grid = info.grid;
    int64_t maxSelectNum = 0;
    for (int64_t headIdx = 0; headIdx < numHeads; ++headIdx) {
        maxSelectNum = std::max(maxSelectNum, HeadSelectNum(grid, windowSize, headIdx));
    }
    const int64_t selectCap = (maxSelectNum + LIST_ENTRIES_PER_C0 - 1) / LIST_ENTRIES_PER_C0 * LIST_ENTRIES_PER_C0;
    const int64_t imageBlocks = static_cast<int64_t>(grid.tilesT) * grid.tilesH * grid.tilesW;
    const int64_t headListBytes = imageBlocks * selectCap * LIST_ENTRY_BYTES;
    return std::max<int64_t>(1, MAX_LAUNCH_LIST_BYTES / headListBytes);
}

// Query rows per launch. This operator's kernel always takes the whole sequence; the upstream operator takes the
// query blocks whose mask rows fit into one launch's host data.
static int64_t RowsPerLaunch(const BsaCallInfo& info, const std::vector<int64_t>& windowSize, int64_t headBegin,
                             int64_t headEnd, const SparsePlan& plan)
{
    if (plan.backend == SparseBackend::STA_KERNEL) {
        return info.sequence;
    }
    const int64_t blocks = info.grid.blockCount;
    const int64_t heads = headEnd - headBegin;
    const int64_t hostHeads = SharedWindow(info.grid, windowSize, headBegin, headEnd) ? 1 : heads;
    const int64_t queryBlocks = std::max<int64_t>(1, MAX_HOST_MASK_BYTES / (hostHeads * blocks));
    return std::min(info.sequence, queryBlocks * SPARSE_BLOCK);
}

// Consecutive heads with the same effective window, starting at headBegin. Both kernels hand the tasks of one launch
// to the cores by position, so a launch that mixes window sizes makes the cores that got the small windows wait for
// the core with the largest one: the task book's case 1 (eight different windows) takes 173 ms in one launch and
// 94 ms as one launch per head. Only the head that would extend the run is compared, so grouping every head of a
// call costs numHeads window conversions and not one per pair of heads.
static int64_t SameWindowHeads(const TileGrid& grid, const std::vector<int64_t>& windowSize, int64_t headBegin,
                               int64_t numHeads)
{
    const auto firstWindow = EffectiveWindow(grid, windowSize, headBegin);
    int64_t headEnd = headBegin + 1;
    while (headEnd < numHeads && EffectiveWindow(grid, windowSize, headEnd) == firstWindow) {
        ++headEnd;
    }
    return headEnd - headBegin;
}

// Adds the sparse attention and the copy into output, in one launch unless the heads or the query rows have to be
// split (see HeadsPerLaunch and RowsPerLaunch).
static aclnnStatus BuildSparseGraph(aclOpExecutor* executor, const aclTensor* query, const aclTensor* key,
                                    const aclTensor* value, const aclTensor* output, const BsaCallInfo& info,
                                    const std::vector<int64_t>& windowSize, int64_t numHeads, const SparsePlan& plan)
{
    const int64_t headCap = std::min(numHeads, HeadsPerLaunch(info, windowSize, numHeads, plan));
    const int64_t sharedHeads = std::min(headCap, SameWindowHeads(info.grid, windowSize, 0, numHeads));
    const int64_t groupRows = std::min(info.sequence, RowsPerLaunch(info, windowSize, 0, numHeads, plan));
    if (sharedHeads == numHeads && groupRows == info.sequence) {
        const aclTensor* result =
            RunSparseHeads(executor, query, key, value, info, windowSize, 0, numHeads, 0, info.sequence, plan);
        CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(result, output, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    }

    const auto& outShape = output->GetViewShape();
    const auto& outStrides = output->GetViewStrides();
    for (int64_t headBegin = 0; headBegin < numHeads;) {
        const int64_t groupHeads = std::min(headCap, SameWindowHeads(info.grid, windowSize, headBegin, numHeads));
        const int64_t headEnd = headBegin + groupHeads;
        const int64_t launchRows = std::min(info.sequence, RowsPerLaunch(info, windowSize, headBegin, headEnd, plan));
        // The head range is copied out: a view of it, even the contiguous one that a single batch gives, is
        // rejected by the upstream operator's tiling, which wants its inputs as plain BNSD tensors.
        const aclTensor* groupKey = key;
        const aclTensor* groupValue = value;
        if (groupHeads != numHeads) {
            groupKey = TensorSlice(executor, key, headBegin, headEnd, 0, info.sequence);
            groupValue = TensorSlice(executor, value, headBegin, headEnd, 0, info.sequence);
            CHECK_RET(groupKey != nullptr && groupValue != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        for (int64_t rowBegin = 0; rowBegin < info.sequence; rowBegin += launchRows) {
            const int64_t rowNum = std::min(info.sequence - rowBegin, launchRows);
            const aclTensor* groupQuery = query;
            if (groupHeads != numHeads || rowNum != info.sequence) {
                groupQuery = TensorSlice(executor, query, headBegin, headEnd, rowBegin, rowNum);
                CHECK_RET(groupQuery != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
            const aclTensor* result = RunSparseHeads(executor, groupQuery, groupKey, groupValue, info, windowSize,
                                                     headBegin, headEnd, rowBegin, rowNum, plan);
            CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
            const op::Shape tileShape({outShape.GetDim(0), headEnd - headBegin, rowNum, outShape.GetDim(3)});
            const aclTensor* target =
                executor->CreateView(output, tileShape, output->GetStorageShape(), outStrides,
                                     output->GetViewOffset() + headBegin * outStrides[1] + rowBegin * outStrides[2]);
            CHECK_RET(target != nullptr, ACLNN_ERR_INNER_NULLPTR);
            CHECK_RET(l0op::ViewCopy(result, target, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        headBegin = headEnd;
    }
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" __attribute__((visibility("default"))) aclnnStatus aclnnSlidingTileAttentionGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, aclTensor* output,
    const aclIntArray* const* windowSize, uint64_t windowSizeLen, int64_t textLength, bool hasText,
    const char* seqShape, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSlidingTileAttention, DFX_IN(query, key, value, windowSizeLen, textLength, hasText),
                   DFX_OUT(output));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    int64_t canvasT = 0;
    int64_t canvasH = 0;
    int64_t canvasW = 0;
    std::vector<int64_t> flattenedWindowSize;
    const aclnnStatus paramStatus =
        CheckParams(query, key, value, output, windowSize, windowSizeLen, textLength, hasText, seqShape, workspaceSize,
                    executor, canvasT, canvasH, canvasW, flattenedWindowSize);
    CHECK_RET(paramStatus == ACLNN_SUCCESS, paramStatus);

    aclOpExecutor* l0Executor = uniqueExecutor.get();
    query = l0op::Contiguous(query, l0Executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    key = l0op::Contiguous(key, l0Executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    value = l0op::Contiguous(value, l0Executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const auto& queryShape = query->GetViewShape();
    const uint32_t batch = static_cast<uint32_t>(queryShape.GetDim(0));
    const int64_t numHeads = query->GetViewShape().GetDim(1);
    const uint32_t sequence = static_cast<uint32_t>(queryShape.GetDim(2));
    const int64_t headDim = queryShape.GetDim(3);
    const int64_t effectiveTextLength = hasText ? textLength : 0;
    const double scaleValue = 1.0 / std::sqrt(static_cast<double>(headDim));
    const int64_t validSequence = canvasT * canvasH * canvasW + effectiveTextLength;
    const BsaCallInfo info = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(sequence),
        validSequence,
        scaleValue,
        {static_cast<uint32_t>(canvasT / TILE_T), static_cast<uint32_t>(canvasH / TILE_H),
         static_cast<uint32_t>(canvasW / TILE_W), static_cast<uint32_t>(canvasT * canvasH * canvasW),
         static_cast<uint32_t>(effectiveTextLength),
         static_cast<uint32_t>((sequence + SPARSE_BLOCK - 1U) / SPARSE_BLOCK)}};
    const SparsePlan plan = ChooseSparsePlan(headDim, info.grid);
    const aclnnStatus sparseStatus =
        BuildSparseGraph(l0Executor, query, key, value, output, info, flattenedWindowSize, numHeads, plan);
    CHECK_RET(sparseStatus == ACLNN_SUCCESS, sparseStatus);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" __attribute__((visibility("default"))) aclnnStatus aclnnSlidingTileAttention(void* workspace,
                                                                                        uint64_t workspaceSize,
                                                                                        aclOpExecutor* executor,
                                                                                        aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSlidingTileAttention);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
