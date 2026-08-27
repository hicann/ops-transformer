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
 * \file fused_causal_conv1d_cut_bh_tiling_arch35.h
 * \brief FusedCausalConv1dCutBH tiling implementation
 *
 * BH 模板（CutBH）：沿 Batch 和 Hidden（dim）两个维度做核间切分。
 * 与 BSH 模板的区别在于：BH 模板 seqLen 较小（[1,8]），
 * 核间切分只考虑 batch × dim，核内 UB 循环以 (batch, dim) 为粒度。
 */
#ifndef FUSED_CAUSAL_CONV1D_CUT_BH_TILING_H
#define FUSED_CAUSAL_CONV1D_CUT_BH_TILING_H

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "../op_kernel/arch35/fused_causal_conv1d_cut_bh_struct.h"

namespace optiling {

struct FusedCausalConv1dCutBHCompileInfo {
    uint64_t coreNum = 0; // AIV 核数
    uint64_t ubSize = 0;  // 单核 UB 大小（字节）
};
constexpr uint64_t TILING_KEY_BH_BF16 = 20000; // BF16 数据类型对应的 kernel 分支
constexpr uint64_t TILING_KEY_BH_FP16 = 20001; // FP16 数据类型对应的 kernel 分支

constexpr int32_t X_INDEX = 0;                  // 输入特征 x：[batch, seq_len, dim] 或 [cu_seq_len, dim]
constexpr int32_t WEIGHT_INDEX = 1;             // 卷积权重：[kernel_size, dim]
constexpr int32_t CONV_STATES_INDEX = 2;        // KV 缓存状态：[batch, state_len, dim]
constexpr int32_t QUERY_START_LOC_INDEX = 3;    // 2D 输入时各 batch 的起始位置：[batch+1]（可选）
constexpr int32_t CACHE_INDICES_INDEX = 4;      // cache 索引（可选），非APC为 1D，APC 为 2D
constexpr int32_t INITIAL_STATE_MODE_INDEX = 5; // 初始状态模式（可选）
constexpr int32_t BIAS_INDEX = 6;               // 偏置（可选）
constexpr int32_t NUM_ACCEPTED_TOKEN_INDEX = 7; // 每个 batch 接受的 token 数（可选）
// APC（Attention Page Cache）相关辅助输入
constexpr int32_t NUM_COMPUTED_TOKENS_INDEX = 8;             // 已计算 token 数（可选）
constexpr int32_t BLOCK_IDX_FIRST_SCHEDULED_TOKEN_INDEX = 9; // APC: 首个调度 token 所在 block 索引（可选）
constexpr int32_t BLOCK_IDX_LAST_SCHEDULED_TOKEN_INDEX = 10; // APC: 最后调度 token 所在 block 索引（可选）
constexpr int32_t INITIAL_STATE_IDX_INDEX = 11;              // 初始状态索引（可选）

constexpr int32_t Y_INDEX = 0;                    // 输出特征 y（与 x 形状相同）
constexpr int32_t OUTPUT_CONV_STATES_INDEX = 1;   // 更新后的 conv_states（与输入 conv_states 形状相同）
constexpr int32_t ATTR_ACTIVATION_MODE_INDEX = 0; // 激活函数模式：0=无，1=silu 等
constexpr int32_t ATTR_PAD_SLOT_ID_INDEX = 1;     // padding slot id，用于标识 padding 样本
constexpr int32_t ATTR_RUN_MODE_INDEX = 2;        // 运行模式（预留）
constexpr int32_t ATTR_MAX_QUERY_LEN_INDEX = 3;   // 2D 输入时的最大序列长度
constexpr int32_t ATTR_RESIDUAL_CONNECTION_INDEX = 4; // 是否使用残差连接：0=否，1=是
constexpr int32_t ATTR_BLOCK_SIZE_INDEX = 5;          // APC block 大小（APC 开启时必须非零）
constexpr int32_t ATTR_CONV_MODE_INDEX = 6; // 卷积模式：0=Qwen3，1=Pangu v2（前 width-1 token 置零）
constexpr int32_t ATTR_MAX_DRAFT_TOKENS_INDEX = 7;

constexpr int64_t DIM_ALIGN_ELEMENT = 128; // dim 方向的最小对齐单元（128 个元素，fp16/bf16 = 256 字节）
constexpr int64_t COARSE_GRAIN_THRESHOLD = 1024;                // dim >= 1024 时切换到粗粒度 tiling 模式
constexpr int64_t COARSE_GRAIN_ELEMENT = 1024;                  // 粗粒度模式下每块的元素数
constexpr int64_t COARSE_GRAIN_SIZE = COARSE_GRAIN_ELEMENT * 2; // 粗粒度块字节数（fp16/bf16）
constexpr int64_t DIM_ALIGN_SIZE = 256; // 细粒度对齐块的字节数（128 元素 × 2 字节）
constexpr int64_t ALIGN_BYTES = 32;     // NPU 内存地址基本对齐要求（32 字节）
constexpr int64_t MIN_DIM = 64;         // dim 下限（16 的倍数，最小 64）
constexpr int64_t MAX_DIM = 16384;      // dim 上限
constexpr int64_t MIN_BATCH = 1;        // batch 下限
constexpr int64_t MAX_BATCH = 256;      // batch 上限
constexpr int64_t MIN_M = 0;            // seqLen-1 的最小值（m 值）
constexpr int64_t MAX_M = 16;           // seqLen-1 的最大值（F4 扩展至 [0, 16]）
constexpr int64_t DIM_0 = 0;            // 维度索引常量
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DTYPE_SIZE = 2;                     // bf16/fp16 每个元素占 2 字节
constexpr int64_t BUFFER_NUM = 2;                     // 双 buffer（乒乓缓冲）数量
constexpr int64_t SYSTEM_RESERVED_UB_SIZE = 8 * 1024; // 系统为运行时预留的 UB（8KB），计算可用 UB 时需减去

// x 输入模式常量
constexpr int64_t X_INPUT_3D = 0; // 3D 输入 [batch, seq_len, dim]
constexpr int64_t X_INPUT_2D = 1; // 2D 输入 [cu_seq_len, dim]，需配合 query_start_loc 拆分 batch

class FusedCausalConv1dCutBHTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit FusedCausalConv1dCutBHTiling(gert::TilingContext *context, bool isInplace = false)
        : TilingBaseClass(context),
          isInplace_(isInplace)
    {}

protected:
    // ---- TilingBaseClass 虚函数，由框架按序调用 ----

    bool IsCapable() override;                    // 判断当前平台是否支持此模板（始终返回 true）
    ge::graphStatus GetShapeAttrsInfo() override; // 读取 shape/type/attr/stride 信息并完成合法性校验
    ge::graphStatus GetPlatformInfo() override;   // 读取硬件参数（核数、UB 大小）
    ge::graphStatus DoOpTiling() override;        // 执行核间 + 核内 tiling 计算（主入口）
    ge::graphStatus DoLibApiTiling() override;    // LibAPI tiling（本算子不需要，直接返回成功）
    uint64_t GetTilingKey() const override;       // 返回 TilingKey（按数据类型区分 kernel 分支）
    ge::graphStatus GetWorkspaceSize() override;  // 计算所需 workspace 大小
    ge::graphStatus PostTiling() override;        // 将 tiling 结果写入 RawTilingData 并设置 blockDim
    void DumpTilingInfo() override;               // 打印 tiling 参数（调试用）

    // ---- 各张量形状合法性校验 ----
    ge::graphStatus ValidateXShape();
    ge::graphStatus ValidateWeightShape();
    ge::graphStatus ValidateConvStatesShape();
    ge::graphStatus ValidateQueryStartLocShape();
    ge::graphStatus ValidateCacheIndicesShape();
    ge::graphStatus ValidateNumAcceptedTokenShape();

    // ---- 各张量数据类型合法性校验 ----
    ge::graphStatus ValidateXType();
    ge::graphStatus ValidateWeightType();
    ge::graphStatus ValidateConvStatesType();
    ge::graphStatus ValidateQueryStartLocType();
    ge::graphStatus ValidateCacheIndicesType();
    ge::graphStatus ValidateNumAcceptedTokenType();

    // ---- 属性合法性校验 ----
    ge::graphStatus ValidateAttrs();

    // ---- 总校验入口（依次调用上面的 Validate* 函数）----
    ge::graphStatus CheckInputParams();

private:
    // ---- 核数上限计算 ----
    int64_t CalculateLimitedCoreNum(); // 细粒度模式：根据数据量限制最大核数（每 256 字节分配 1 核）
    int64_t CalculateLimitedCoreNumCoarse(); // 粗粒度模式：根据数据量限制最大核数（每 2048 字节分配 1 核）

    // ---- GetShapeAttrsInfo 的子函数 ----
    ge::graphStatus GetShapeInfo();  // 读取 x/weight/conv_states 形状，判断 APC/cacheIndices 是否有效
    ge::graphStatus GetTypeInfo();   // 读取各输入的数据类型，判断 hasAcceptTokenNum
    ge::graphStatus GetAttrInfo();   // 读取算子属性（activation_mode / pad_slot_id / run_mode 等）
    ge::graphStatus GetStrideInfo(); // 读取 x 和 conv_states 的 stride（支持非连续张量/view）

    // ---- 核间切分（细粒度：dim < 1024）----
    ge::graphStatus ComputeInterCoreSplit();
    // ---- 核内 UB 切分（细粒度）----
    ge::graphStatus ComputeIntraCoreUbTiling();
    // ---- UB 切分核心计算（细粒度，被 ComputeIntraCoreUbTiling 调用）----
    void ComputeUbFor(int64_t coreDimElems, int64_t coreBS, int64_t availableUbSize, int64_t &outUbDim,
                      int64_t &outUbBS, int64_t &outLoopDim, int64_t &outLoopBS, int64_t &outUbTailDim,
                      int64_t &outUbTailBS);

    // ---- 核间切分（粗粒度：dim >= 1024）----
    ge::graphStatus ComputeInterCoreSplitCoarse();
    // ---- 核内 UB 切分（粗粒度）----
    ge::graphStatus ComputeIntraCoreUbTilingCoarse();
    // ---- UB 切分核心计算（粗粒度，被 ComputeIntraCoreUbTilingCoarse 调用）----
    void ComputeUbForCoarse(int64_t coreDimElems, int64_t coreBS, int64_t availableUbSize, int64_t &outUbDim,
                            int64_t &outUbBS, int64_t &outLoopDim, int64_t &outLoopBS, int64_t &outUbTailDim,
                            int64_t &outUbTailBS);

    uint64_t ubSize_ = 0;       // 单核 UB 总大小（字节）
    uint64_t totalCoreNum_ = 0; // 总 AIV 核数
    uint64_t ubBlockSize_ = 0;  // UB 块大小（用于辅助计算，来自 GetUbBlockSize）

    // ---- 运行时 UB 可用量（由 ComputeIntraCoreUbTiling 计算）----
    int64_t fixedUBSize = 0; // 固定占用的 UB（acceptTokenBuf / indicesBuf / queryStartLocBuf 等）

    // ---- 输入张量形状（由 GetShapeInfo 填充）----
    int64_t batchSize_ = 0;  // batch 大小
    int64_t seqLen_ = 0;     // 序列长度（3D 输入时有效；2D 时在 UB 计算阶段赋值为 maxQueryLen_）
    int64_t cuSeqLen_ = 0;   // 2D 输入时 x 的第一维（= 所有 batch 的 token 总数）
    int64_t dim_ = 0;        // 隐层维度大小
    int64_t kernelSize_ = 0; // 卷积核大小（来自 weight 的第 0 维）
    int64_t stateLen_ = 0;   // conv_states 第 1 维大小（= kernel_size - 1 + max_seq_len）

    // ---- 数据类型（由 GetTypeInfo 填充）----
    ge::DataType xDtype_;
    ge::DataType weightDtype_;
    ge::DataType convStatesDtype_;
    ge::DataType queryStartLocDtype_;
    ge::DataType cacheIndicesDtype_;
    ge::DataType numAcceptedTokenDtype_;
    size_t xDtypeSize_ = 0; // x 数据类型的字节大小

    // ---- 算子属性（由 GetAttrInfo 填充）----
    int64_t activationMode_ = 0;     // 激活函数模式
    int64_t xStride_ = 0;            // x 在 seq_len 维度上的 stride（view 时非 dim_）
    int64_t cacheStride0_ = 0;       // conv_states 在 batch 维度的 stride
    int64_t cacheStride1_ = 0;       // conv_states 在 state_len 维度的 stride
    int64_t padSlotId_ = -1;         // padding sample 的 slot id（-1 表示无效）
    int64_t runMode_ = 0;            // 运行模式（预留扩展）
    int64_t xInputMode_ = 0;         // x 输入模式：0=3D，1=2D
    int64_t hasAcceptTokenNum_ = 0;  // 是否提供 num_accepted_token 输入：0=否，1=是
    int64_t residualConnection_ = 0; // 是否使用残差连接：0=否，1=是

    int64_t apcEnable_ = 0; // APC 开关：block_idx_first_scheduled_token 存在即初判开启，缺其余必要条件立即报错
    int64_t blockSize_ = 0;       // APC block 大小（由属性读取）
    int64_t maxNumBlocks_ = 0;    // APC 开启时 cacheIndices 第二维大小
    int64_t hasCacheIndices_ = 0; // cacheIndices 是否有效：0=无，1=有
    int64_t inplace_ = 0;         // conv_states 是否原地更新：0=否，1=是（由 isInplace_ 决定）
    bool isInplace_ = false;      // 构造时传入，替代 attr 读取
    int64_t convMode_ = 0;        // 卷积模式：0=Qwen3，1=Pangu v2（前 width-1 token 置零）
    int64_t maxDraftTokens_ = 0;
    int64_t hasNumComputedTokens_ = 0; // numComputedTokens 是否有效：0=无，1=有
    int64_t maxQueryLen_ = MAX_M + 1;  // 2D 输入时最大序列长度（来自 max_query_len 属性）

    int64_t limitedCoreNum_ = 0; // 按数据量限制的最大可用核数（参考值）
    int64_t usedCoreNum_ = 0;    // 实际使用的总核数（= dimCoreCnt_ × batchCoreCnt_）
    int64_t dimCoreCnt_ = 0;     // dim 方向分配的核数
    int64_t batchCoreCnt_ = 0;   // batch 方向分配的核数
    // dim 方向非均匀切分
    int64_t dimMainCoreCnt_ = 0; // dim 方向"大核"数量（每核处理 mainCoredimLen_ 个元素）
    int64_t dimTailCoreCnt_ = 0; // dim 方向"小核"数量（每核处理 tailCoredimLen_ 个元素）
    int64_t mainCoredimLen_ = 0; // 大核负责的 dim 元素数（大 = base+1 块）
    int64_t tailCoredimLen_ = 0; // 小核负责的 dim 元素数（小 = base 块）
    // batch 方向非均匀切分
    int64_t batchMainCoreCnt_ = 0; // batch 方向"大核"数量
    int64_t batchTailCoreCnt_ = 0; // batch 方向"小核"数量
    int64_t mainCoreBatchNum_ = 0; // 大核负责的 batch 数
    int64_t tailCoreBatchNum_ = 0; // 小核负责的 batch 数

    // ---- 大核（main core）UB 参数 ----
    int64_t loopNumBS_ = 0;       // 大核 batch 方向循环次数
    int64_t loopNumDim_ = 0;      // 大核 dim 方向循环次数
    int64_t ubMainFactorBS_ = 0;  // 大核每次 UB 循环处理的 batch 数（主块大小）
    int64_t ubTailFactorBS_ = 0;  // 大核 batch 方向最后一次循环的 batch 数（尾块大小）
    int64_t ubMainFactorDim_ = 0; // 大核每次 UB 循环处理的 dim 元素数（主块大小）
    int64_t ubTailFactorDim_ = 0; // 大核 dim 方向最后一次循环的 dim 元素数（尾块大小）
    // ---- 小核（tail core）UB 参数 ----
    int64_t tailBlockloopNumBS_ = 0;       // 小核 batch 方向循环次数
    int64_t tailBlockloopNumDim_ = 0;      // 小核 dim 方向循环次数
    int64_t tailBlockubFactorBS_ = 0;      // 小核每次 UB 循环处理的 batch 数（主块大小）
    int64_t tailBlockubTailFactorBS_ = 0;  // 小核 batch 方向尾块大小
    int64_t tailBlockubFactorDim_ = 0;     // 小核每次 UB 循环处理的 dim 元素数（主块大小）
    int64_t tailBlockubTailFactorDim_ = 0; // 小核 dim 方向尾块大小
    // ---- 最后一个 dim 核（last dim core）UB 参数 ----
    // 最后一个 dim 核在普通 tail core 参数基础上额外处理 dimRemainderElems_ 个余数元素
    int64_t dimRemainderElems_ = 0;       // dim % 128，最后一个 dim 核额外处理的余数元素数
    int64_t lastCoreloopNumDim_ = 0;      // 最后一个 dim 核的 dim 方向循环次数
    int64_t lastCoreubMainFactorDim_ = 0; // 最后一个 dim 核的 UB 主块 dim 大小
    int64_t lastCoreubTailFactorDim_ = 0; // 最后一个 dim 核的 UB 尾块 dim 大小

    FusedCausalConv1dCutBHTilingData tilingData_;
};

} // namespace optiling

#endif // FUSED_CAUSAL_CONV1D_CUT_BH_TILING_H
