/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ffn_worker_batching_arch35_tiling_def.h
 * \brief arch35 (Ascend950 / DAV_3510) kernel 侧平铺 TilingData struct，host/kernel 共用 ABI。
 *        host 经 context_->GetTilingData<FfnWorkerBatchingArch35TilingData>() 直写；
 *        kernel 经 GET_TILING_DATA_WITH_STRUCT(FfnWorkerBatchingArch35TilingData, ...) 读取。
 *
 *        原则：**kernel 侧不自算任何切分与地址**。核间/UB 切分、各阶段单次搬运量、
 *        workspace 各段偏移全部在此下发；kernel 只按下发值取用。
 */
#ifndef FFN_WB_ARCH35_TILING_DEF_H
#define FFN_WB_ARCH35_TILING_DEF_H

struct FfnWorkerBatchingArch35TilingData {
    // ---- 形状与平台 ----
    int64_t Y{0};          // A*BS*K
    int64_t H{0};          // hidden size（max_out_shape[3]）
    int64_t tokenDtype{0}; // 0:FP16 1:BF16 2:dynamic_quant_int8
    int64_t expertNum{0};
    int64_t coreNum{0}; // 运行时 GetCoreNumAiv 取值
    int64_t ubSize{0};  // 运行时 GetCoreMemSize(UB) 取值

    // ---- 段内排序（VBS）----
    // 扁平 expert_id 的元素数：NORM 为 Y；RECV 每个 session 的 BS*K 需按数据块补齐，
    // 故为 A*align(BS*K)，与 gather 解码下标时用的 bskProduct 口径一致。
    int64_t flatElements{0};
    int64_t preparePerLoopRows{0}; // phase0 每轮搬多少个 session 行(由运行时 UB 推导)
    int64_t sortSegNum{0};         // 序列切成多少段
    int64_t sortPerSegElements{0}; // 每段元素数
    int64_t sortLenPerSeg{0};      // 每段 proposal 对区的 float 个数
    int64_t expertStart{0};        // >= 此值的 expert_id 视为被 mask（上游置大值）

    // ---- 段间归并（VMS）----
    int64_t mergeRounds{0};          // 归并轮数 = ceil(log4(sortSegNum))
    int64_t mergeOneLoopElements{0}; // 单路单次载入 UB 的元素数

    // ---- 归并收尾（Extract）----
    int64_t extractPerLoopElements{0}; // 单次拆包元素数（Sort32 粒度整数倍）

    // ---- group_list ----
    int64_t glRowsPerLoop{0}; // 写出时每块拼多少行（每行 16B）

    // ---- workspace 段偏移（以 int32 word 计，相对 userWorkspace 起始）----
    int64_t wsFlatIds{0};   // 归一后的扁平 expert_id：Y 个 int32
    int64_t wsPairA{0};     // 排序/归并工作区 A：sortSegNum*sortLenPerSeg 个 float
    int64_t wsPairB{0};     // 排序/归并工作区 B（与 A 乒乓）
    int64_t wsSegCnt{0};    // 各段有效元素数：每段一个 32B 槽
    int64_t wsSortedIds{0}; // 有序 expert_id：Y 个 int32
    int64_t wsGatherIdx{0}; // gather_idx：Y 个 int32
};

#endif // FFN_WB_ARCH35_TILING_DEF_H
