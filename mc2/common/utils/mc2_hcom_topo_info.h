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
 * \file mc2_hcom_topo_info.h
 * \brief
 */

#ifndef MC2_HCOM_TOPOLOGY_H
#define MC2_HCOM_TOPOLOGY_H

#include <memory>
#include <vector>
#include "hccl/hcom.h"

#ifdef BUILD_OPEN_PROJECT
#include "hccl/hccl_rank_graph.h"
#endif

static constexpr uint32_t COMM_ALG_DEFAULT = 0U;
static constexpr uint32_t COMM_MESH = 0b1U;
static constexpr uint32_t COMM_SWITCH = (COMM_MESH << 1U);
static constexpr uint32_t COMM_RING = (COMM_MESH << 2U);
static constexpr uint32_t COMM_PAIRWISE = (COMM_MESH << 3U);
static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
namespace Mc2Hcom {
class MC2HcomTopology {
public:
    static HcclResult CommGetInstSizeByGroup(const char *group, uint32_t *rankNum);
    static HcclResult TryGetGroupTopoType(const char *group, uint32_t *topoType);
    static HcclResult CommGetCclBufferSizeByGroup(const char *group, uint64_t *cclBufferSize, HcclComm *hcclComm);
    static HcclResult CommGetGroupLocalWindowSize(const char *group, uint64_t *cclBufferSize);
    static HcclResult CommGetHcclBufferByGroup(const char *group, void **buffer, uint64_t *size);
    static HcclResult CommGetNetLayersByGroup(const char *group, std::vector<uint32_t> &layers);
    static HcclResult CommGetTopoTypeByLayer(const char *group, uint32_t layer, uint32_t *topoType);

private:
    static MC2HcomTopology &GetInstance();
    explicit MC2HcomTopology(const char *libPath);
    HcclResult CallHcomGetCommHandleByGroup(const char *group, HcclComm *commHandle) const;
    HcclResult CallCommGetNetLayers(HcclComm comm, uint32_t **netLayer, uint32_t *netLayerNum) const;
    HcclResult CallCommGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *topoType) const;
#ifdef BUILD_OPEN_PROJECT
    HcclResult CallHcomGetRankSizeEx(const char *group, uint32_t *ranksize, uint32_t flag) const;
    HcclResult CallHcomGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t flag) const;
#else
    HcclResult CallCommGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum) const;
#endif
    HcclResult CallCommGetCCLBufSizeCfg(HcclComm comm, uint64_t *cclBufferSize) const;
    HcclResult CallCommGetHcclBuffer(HcclComm comm, void **buffer, uint64_t *size) const;

    void *handle_ = nullptr;

    using FuncGetHandle = HcclResult (*)(const char *, HcclComm *);
    using FuncGetNetLayers = HcclResult (*)(HcclComm, uint32_t **, uint32_t *);
    using FuncGetTopoTypeByLayer = HcclResult (*)(HcclComm, uint32_t, CommTopo *);
#ifdef BUILD_OPEN_PROJECT
    using FuncGetRankSize = HcclResult (*)(HcclComm, uint32_t *);
#else
    using FuncGetInstSize = HcclResult (*)(HcclComm, uint32_t, uint32_t *);
#endif
    using FuncGetHcclBuffer = HcclResult (*)(HcclComm, void **, uint64_t *);
    void *hcclHandle_ = nullptr;
    FuncGetHandle getCommHandle_ = nullptr;
    FuncGetNetLayers getNetLayers_ = nullptr;
    FuncGetTopoTypeByLayer getTopoTypeByLayer_ = nullptr;
#ifdef BUILD_OPEN_PROJECT
    FuncGetRankSize getRankSize_ = nullptr;
#else
    FuncGetInstSize getInstSize_ = nullptr;
#endif
    FuncGetHcclBuffer getHcclBuffer_ = nullptr;
};
} // namespace Mc2Hcom
#endif
