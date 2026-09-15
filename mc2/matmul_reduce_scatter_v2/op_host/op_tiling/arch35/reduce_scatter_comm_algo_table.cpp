/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_comm_algo_table.h"
#include "mc2_tiling_utils.h"
#include "hccl/hccl_rank_graph.h"

namespace Mc2Tiling {

constexpr uint32_t MIN_LAYERS = 1;
constexpr uint32_t MAX_LAYERS_UBX = 1;
constexpr uint32_t MAX_LAYERS_SERVER = 2;
constexpr uint32_t MIN_DATA_BYTES_ZERO = 0;
constexpr uint32_t DATA_BYTES_16M = 16ULL * 1024ULL * 1024ULL;
constexpr uint32_t DATA_BYTES_256M = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t MAX_DATA_BYTES_UNLIMITED = ~0ULL;
constexpr uint32_t MIN_RANK_SIZE = 0;
constexpr uint32_t RANK_SIZE_2 = 2;
constexpr uint32_t RANK_SIZE_4 = 4;
constexpr uint32_t RANK_SIZE_8 = 8;
constexpr uint32_t MAX_RANK_SIZE = ~0U;

enum CommAlgoPriority : int32_t {
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3,
};

static constexpr Mc2Hcom::CommAlgoEntry REDUCE_SCATTER_COMM_ALGO_TABLE[] = {
    // AICPU通信算法
    // Server/pod 小/中数据量 → AicpuReduceScatterSoleMesh
    {mc2tiling::A5_AICPU_TS_ENGINE, COMM_TOPO_1DMESH, MIN_LAYERS, MAX_LAYERS_SERVER, MIN_DATA_BYTES_ZERO,
     DATA_BYTES_256M, MIN_RANK_SIZE, MAX_RANK_SIZE, MEDIUM, "AicpuReduceScatterSoleMesh"},
    // Server/pod 大数据量 → AicpuReduceScatterSoleMeshChunk
    {mc2tiling::A5_AICPU_TS_ENGINE, COMM_TOPO_1DMESH, MIN_LAYERS, MAX_LAYERS_SERVER, DATA_BYTES_256M,
     MAX_DATA_BYTES_UNLIMITED, MIN_RANK_SIZE, MAX_RANK_SIZE, MEDIUM, "AicpuReduceScatterSoleMeshChunk"},
    // UBX/8p 小/中数据量 → AicpuReduceScatterSoleNHR
    {mc2tiling::A5_AICPU_TS_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, MIN_DATA_BYTES_ZERO, DATA_BYTES_256M,
     RANK_SIZE_8, MAX_RANK_SIZE, MEDIUM, "AicpuReduceScatterSoleNHR"},
    // UBX/8p 大数据量 → AicpuReduceScatterParallelMeshNHRUBX
    {mc2tiling::A5_AICPU_TS_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, DATA_BYTES_256M,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_8, MAX_RANK_SIZE, MEDIUM, "AicpuReduceScatterParallelMeshNHRUBX"},
    // UBX/4p 小/中数据量 → AicpuReduceScatterSoleMesh
    {mc2tiling::A5_AICPU_TS_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, MIN_DATA_BYTES_ZERO, DATA_BYTES_256M,
     MIN_RANK_SIZE, RANK_SIZE_4, MEDIUM, "AicpuReduceScatterSoleMesh"},
    // UBX/4p 大数据量 → AicpuReduceScatterConcurMeshNHR
    {mc2tiling::A5_AICPU_TS_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, DATA_BYTES_256M,
     MAX_DATA_BYTES_UNLIMITED, MIN_RANK_SIZE, RANK_SIZE_4, MEDIUM, "AicpuReduceScatterConcurMeshNHR"},
    // CCU通信算法
    // Server/pod 2p → CcuSchedReduceScatterSoleMesh
    {mc2tiling::A5_CCU_ENGINE, COMM_TOPO_1DMESH, MIN_LAYERS, MAX_LAYERS_SERVER, MIN_DATA_BYTES_ZERO,
     MAX_DATA_BYTES_UNLIMITED, MIN_RANK_SIZE, RANK_SIZE_2, MEDIUM, "CcuSchedReduceScatterSoleMesh"},
    // Server/pod 4p/8p → CcuSchedAllToAllSoleMesh
    {mc2tiling::A5_CCU_ENGINE, COMM_TOPO_1DMESH, MIN_LAYERS, MAX_LAYERS_SERVER, MIN_DATA_BYTES_ZERO,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_4, RANK_SIZE_8, MEDIUM, "CcuSchedAllToAllSoleMesh"},
    // UBX/8p → CcuSchedAllToAllSoleMeshMultiJetty
    {mc2tiling::A5_CCU_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, MIN_DATA_BYTES_ZERO,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_8, MAX_RANK_SIZE, MEDIUM, "CcuSchedAllToAllSoleMeshMultiJetty"},
    // UBX/4p → CcuSchedAllToAllSoleMeshConcurrent
    {mc2tiling::A5_CCU_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, MIN_DATA_BYTES_ZERO,
     MAX_DATA_BYTES_UNLIMITED, MIN_RANK_SIZE, RANK_SIZE_4, MEDIUM, "CcuSchedAllToAllSoleMeshConcurrent"},
};

const Mc2Hcom::CommAlgoEntry *GetReduceScatterCommAlgoTable(uint32_t &count)
{
    count = sizeof(REDUCE_SCATTER_COMM_ALGO_TABLE) / sizeof(REDUCE_SCATTER_COMM_ALGO_TABLE[0]);
    return REDUCE_SCATTER_COMM_ALGO_TABLE;
}

} // namespace Mc2Tiling
