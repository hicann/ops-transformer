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
 * \file elastic_buffer.cpp
 * \brief
 */

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <chrono>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <cstring>
#include <cstdio>
#include <atomic>
#include <cstdint>
#include <algorithm>
#include <array>
#include <mutex>
#include <unordered_map>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>

// CANN ACL Runtime API
#include "acl/acl.h"

// HCCL types
#include "hccl/hccl_types.h"

// HCCL common utilities
#include "hccl_common.h"
#include "moe_shared_jetty_adapter.h"

// ACLNN common utilities
#include "aclnn_common.h"

// torch_npu stream utilities
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace Mc2Api {

// Constants
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;
constexpr uint32_t HCCL_MIN_RANK_SIZE = 2;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr int64_t NETWORK_DIRECT = 0;
constexpr int64_t NETWORK_HYBRID = 1;
constexpr int64_t BUFFER_ALIGNMENT = 2 * 1024 * 1024;
constexpr int64_t MB_SIZE = 1024LL * 1024LL;
constexpr int64_t HUGE1G_SIZE = 1024ULL * 1024ULL * 1024ULL;
constexpr int DIM_ONE = 1;
constexpr int DIM_TWO = 2;
constexpr uint32_t MOE_CHANNEL_HANDLE_NUM = 64U;
constexpr uint32_t MOE_CHANNEL_NOTIFY_NUM = 3U;
constexpr uint32_t MEM_HANDLE_NUM = 1U;
constexpr int64_t SEND_COUNTS_ALIGN_FACTOR = 8;
constexpr int64_t MOE_EP_METADATA_FIELDS = 4;
constexpr int64_t MOE_EP_METADATA_ALIGN_BYTES = 512;
constexpr uint32_t MIX_LAYERED = 2U;
constexpr uint32_t MIX_CLOS_SLOT = 0U;
constexpr uint32_t MIX_MESH_SLOT = 1U;
constexpr uint32_t MIX_LAYERED_RANK_SIZE = 8U;
constexpr uint64_t VMM_RESERVE_MEM_FLAG = 1U;

// RAII guard for multi-step host buffer allocation
struct HostBufferGuard {
    void* hostPtr = nullptr;
    bool registered = false;

    ~HostBufferGuard()
    {
        if (registered && hostPtr) {
            aclrtHostUnregister(hostPtr);
        }
        if (hostPtr) {
            aclrtFreeHost(hostPtr);
        }
    }

    void Release()
    {
        hostPtr = nullptr;
        registered = false;
    }
};

struct FdGuard {
    int fd = -1;

    explicit FdGuard(int f)
        : fd(f)
    {}

    ~FdGuard()
    {
        if (fd >= 0) {
            (void)close(fd);
        }
    }

    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
};

// Helper functions
static inline int64_t CeilDiv(int64_t x, int64_t y)
{
    TORCH_CHECK(y > 0, "CeilDiv divisor must be positive, got ", y);
    TORCH_CHECK(x <= INT64_MAX - y + 1, "CeilDiv overflow: x=", x, " y=", y);
    return (x + y - 1) / y;
}

static inline int64_t AlignTo(int64_t x, int64_t y)
{
    TORCH_CHECK(y > 0, "AlignTo divisor must be positive, got ", y);
    TORCH_CHECK(x <= INT64_MAX - y + 1, "AlignTo overflow: x=", x, " y=", y);
    return CeilDiv(x, y) * y;
}

static inline uint32_t GetAivCoreNum()
{
    static const uint32_t aivCoreNum = [] {
        int32_t deviceId = 0;
        aclError derr = c10_npu::GetDevice(&deviceId);
        if (derr != ACL_SUCCESS) {
            deviceId = static_cast<int32_t>(c10_npu::current_device());
        }

        int64_t aiv = 0;
        aclError ret = aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId), ACL_DEV_ATTR_VECTOR_CORE_NUM, &aiv);
        if (ret != ACL_SUCCESS || aiv <= 0) {
            ret = aclGetDeviceCapability(static_cast<uint32_t>(deviceId), ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv);
        }
        TORCH_CHECK(ret == ACL_SUCCESS && aiv > 0, "Failed to query AIV core num, device=", deviceId,
                    ", ret=", static_cast<int>(ret), ", aiv=", aiv);
        return static_cast<uint32_t>(aiv);
    }();
    return aivCoreNum;
}

static inline void CheckMoeEpMetadataTensor(const at::Tensor& metadata, const char* name, int64_t capacity,
                                            int64_t epWorldSize, int64_t routeSlots, const at::Device& device)
{
    TORCH_CHECK(capacity >= 0 && capacity <= INT32_MAX, "metadata capacity must be in [0, INT32_MAX]");
    TORCH_CHECK(epWorldSize >= 2 && epWorldSize <= 1024, "ep_world_size must be in [2, 1024]");
    TORCH_CHECK(routeSlots >= 0 && routeSlots <= INT32_MAX, "metadata route_slots must be in [0, INT32_MAX]");
    // Packed ABI: four-column rows, rank offsets, local reverse index, slot ends, recv-X IDs and top-k IDs.
    const int64_t elementBytes = sizeof(int32_t);
    const int64_t rankOffsetsOffset =
        AlignTo(capacity * MOE_EP_METADATA_FIELDS * elementBytes, MOE_EP_METADATA_ALIGN_BYTES);
    const int64_t localIndexOffset =
        rankOffsetsOffset + AlignTo((epWorldSize + 1) * elementBytes, MOE_EP_METADATA_ALIGN_BYTES);
    const int64_t elements = (localIndexOffset + AlignTo(routeSlots * elementBytes, MOE_EP_METADATA_ALIGN_BYTES) +
                              3 * AlignTo(capacity * elementBytes, MOE_EP_METADATA_ALIGN_BYTES)) /
                             elementBytes;
    TORCH_CHECK(metadata.scalar_type() == at::kInt && metadata.dim() == DIM_ONE, name,
                " must be a 1D int32 packed tensor");
    TORCH_CHECK(metadata.numel() == static_cast<int64_t>(elements), name, " packed length must be ", elements);
    TORCH_CHECK(metadata.is_contiguous() && metadata.storage_offset() == 0, name,
                " must describe the full contiguous packed allocation, not a metadata-only view");
    TORCH_CHECK(metadata.device() == device, name, " must be on device ", device);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(metadata.data_ptr()) % MOE_EP_METADATA_ALIGN_BYTES == 0, name,
                " base address must be 512-byte aligned");
}

static inline void NpuStreamWait(aclrtStream waitStream, aclrtStream recordStream)
{
    if (waitStream == recordStream) {
        return;
    }
    aclrtEvent event = nullptr;
    aclError ret = aclrtCreateEvent(&event);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtCreateEvent failed, ret: ", ret);
    ret = aclrtRecordEvent(event, recordStream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtRecordEvent failed, ret: ", ret);
    ret = aclrtStreamWaitEvent(waitStream, event);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtStreamWaitEvent failed, ret: ", ret);
    ret = aclrtDestroyEvent(event);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtDestroyEvent failed, ret: ", ret);
}

// CommContext structure for HCCL communication
struct EngramCommContext {
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    // rail布局：超外按machine粒度，超内按rank粒度依次排布
    uint64_t virtualAddrList[HCCL_MAX_RANK_SIZE] = {};
    uint64_t hcommHandle[HCCL_MAX_RANK_SIZE] = {};
    uint32_t channelsPerRank = 1;
    uint32_t ranksPerMachine = 0;
    uint32_t ranksPerSupernode = 0;
    uint64_t rankSlotStride = 0;
};

// 布局：[0,16) 头部标量 | nodeWinBase[] | nodeChannel[]（定长容量，建链只写有效前缀）
struct EngramRailCtxHead {
    uint32_t localId = 0;           // L = rank mod C
    uint32_t supernodeId = 0;       // sn = rank / C
    uint32_t supernodeCnt = 0;      // 超节点数 = world / C
    uint32_t ranksPerSupernode = 0; // C = 每超节点卡数
};
static_assert(sizeof(EngramRailCtxHead) == 16, "rail ctx head must stay 16B");
constexpr uint32_t RAIL_SN_DOMAIN_MAX = 2048U;
constexpr uint32_t RAIL_NODE_MAX = HCCL_MAX_RANK_SIZE + RAIL_SN_DOMAIN_MAX; // node 表容量 = C 上限 + 超节点数上限
constexpr uint64_t RAIL_CTX_WIN_OFF = sizeof(EngramRailCtxHead);            // nodeWinBase 偏移
constexpr uint64_t RAIL_CTX_CH_OFF = RAIL_CTX_WIN_OFF + 8ULL * RAIL_NODE_MAX;     // nodeChannel 偏移
constexpr uint64_t RAIL_CTX_TOTAL_BYTES = RAIL_CTX_CH_OFF + 8ULL * RAIL_NODE_MAX; // 49168B
constexpr int64_t RAIL_CTX_ELEMS =
    static_cast<int64_t>(RAIL_CTX_TOTAL_BYTES / sizeof(int32_t)); // 12292 = tiling CTX_RAIL_ELEMS
static_assert(RAIL_CTX_TOTAL_BYTES % sizeof(int32_t) == 0, "rail ctx must be int32-flat");
constexpr uint64_t RAIL_WIN_SLOT_BYTES = 32ULL; // 窗头写者槽/epoch 槽步长（与 kernel 侧 STATE_OFFSET 一致）

struct MoeCommContext {
    uint32_t epRankId = 0;
    uint32_t rankSizePerServer = 0;
    uint64_t epHcclBuffer[HCCL_MAX_RANK_SIZE] = {};
    ChannelHandle hcommHandle[HCCL_MAX_RANK_SIZE] = {};
    uint32_t channelsPerRank = 1;
};

struct RankLinkInfo {
    CommProtocol protocol;
    uint32_t layer;
};

// 分轨拓扑感知结果：只感知一数 ranksPerSupernode，超节点数/轨道对端集全推导
struct RailTopoInfo {
    uint32_t world{0};
    uint32_t ranksPerSupernode{0};        // 每超节点（直连域）卡数
    uint32_t ranksPerMachine{0};          // 成对 env 声明时填充；训练不读
    std::vector<uint32_t> supernodePeers; // 同超节点成员（全局 rank，不含我）
    std::vector<uint32_t> railPeers;      // 轨道对端（各超节点的同 L 卡）
};

// 分轨建链规格：逐对端一条（cpr=1——每通道单生产者由 kernel 侧 S 核统一发送保证）
struct RailPeerSpec {
    uint32_t peerRank;
    uint32_t tableIdx; // 平铺 node 表落表下标
    bool inSupernode;  // true→期望 UB_CTP；false→期望 UB_RTP（探测内含二选一兜底）
};
struct LayerRanks {
    uint32_t layer;
    std::vector<uint32_t> ranks;
};
struct MixLayeredInfo {
    // Channel 0 uses Clos; channel 1 uses the direct mesh.
    std::array<uint32_t, MIX_LAYERED> layerIds = {};
    std::array<std::vector<CommLink>, MIX_LAYERED> linksByRank;
    bool isUseMixLayered = false;
};

struct MoeContextResources {
    at::Tensor contextTensor;
    int64_t cclBufferSize = 0;
    uint32_t rankSizePerServer = 0;
    void* deviceBufPtr = nullptr;
    aclrtDrvMemHandle physicalMemHandle = nullptr;
    HcclMemHandle memHandle = nullptr;
    std::string contextTag;
};

// 进程级 MoE 通信 buffer 共享池：HCCL 引擎 ctx 以 tag 为单例缓存在通信域内，注册内存
// (HcclCommMemReg)与 channel 均无反注册/销毁接口，其内容(epHcclBuffer[])指向本类自建的
// 物理内存。因此物理内存生命周期与 tag 绑定并保留至进程级：同 tag 多实例共享，Destroy 仅
// 解除本实例引用、不释放内存，同 group destroy 后重建 ElasticBuffer 时按既有容量直接复用
// (重建声明的 cclBufferSize 不得超过首建值)。
struct MoeSharedBufferEntry {
    void* deviceBufPtr = nullptr;
    aclrtDrvMemHandle physicalMemHandle = nullptr;
    HcclMemHandle memHandle = nullptr;
    int64_t cclBufferSize = 0; // 实际已申请的物理内存字节数
};
static std::mutex gMoeSharedBufferMutex;
static std::unordered_map<std::string, MoeSharedBufferEntry> gMoeSharedBuffers;

struct EngramCtxResBase {
    HcclMemHandle memHandle = nullptr;
    void* hostBufPtr = nullptr;
    void* deviceBufPtr = nullptr;
    bool externalRegistered = false;
    int64_t commBufferSize = 0;
    EngramCommContext context;
    void* sfHostBufPtr = nullptr;
    void* sfDeviceBufPtr = nullptr;
    bool sfExternalRegistered = false;

    void CopyBase(const EngramCtxResBase& o)
    {
        static_cast<EngramCtxResBase&>(*this) = o;
    }
};

struct EngramContextResources : EngramCtxResBase {
    HcclComm hcclComm = nullptr;
    at::Tensor contextTensor;
};

// 进程级 Engram 通信 buffer 共享池：注册内存与 channel 均无反注册接口，生命周期与 tag 绑定
// 保留至进程级——Destroy 仅解除引用，同 group 重建直接复用（容量/地址须与首建一致）。
struct EngramSharedBufferEntry : EngramCtxResBase {
    bool external = false;         // buffer 是否来自调用方零拷贝存储
    int64_t registeredBytes = 0;   // 实际已注册的字节数
    bool sfExternal = false;       // SF 表是否来自调用方零拷贝存储
    int64_t sfRegisteredBytes = 0; // SF 表实际已注册的字节数
    std::string netLayer = "auto"; // 首建生效的 netlayer（"auto" 或数字串），重建需一致
};
static std::mutex gEngramSharedBufferMutex;
static std::unordered_map<std::string, EngramSharedBufferEntry> gEngramSharedBuffers;

struct LocalEngramMapping {
    int64_t bytes = 0;
    void* devicePtr = nullptr;
    uint64_t refcount = 0;
};
static std::mutex gLocalEngramMappingsMutex;
static std::unordered_map<void*, LocalEngramMapping> gLocalEngramMappings;

static void* AcquireLocalHostMapping(void* hostPtr, int64_t bytes)
{
    TORCH_CHECK(hostPtr != nullptr, "local zero-copy storage pointer is null");
    TORCH_CHECK(bytes > 0, "local zero-copy storage size must be positive, got ", bytes);

    std::lock_guard<std::mutex> lock(gLocalEngramMappingsMutex);
    auto iter = gLocalEngramMappings.find(hostPtr);
    if (iter != gLocalEngramMappings.end()) {
        TORCH_CHECK(iter->second.bytes == bytes,
                    "local zero-copy storage address is already registered with a different size, registered ",
                    iter->second.bytes, " bytes, got ", bytes, " bytes");
        ++iter->second.refcount;
        return iter->second.devicePtr;
    }

    aclError ret = aclrtHostRegisterV2(hostPtr, static_cast<uint64_t>(bytes), ACL_HOST_REG_MAPPED);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtHostRegisterV2(", bytes, " B) failed, ret=", ret);
    void* devicePtr = nullptr;
    ret = aclrtHostGetDevicePointer(hostPtr, &devicePtr, 0);
    if (ret != ACL_SUCCESS) {
        (void)aclrtHostUnregister(hostPtr);
        TORCH_CHECK(false, "aclrtHostGetDevicePointer failed, ret=", ret, ", host memory cannot be mapped to device");
    }
    auto [insertedIter, inserted] = gLocalEngramMappings.emplace(hostPtr, LocalEngramMapping{bytes, devicePtr, 1});
    TORCH_CHECK(inserted, "local host mapping insert unexpectedly found an existing entry");
    return insertedIter->second.devicePtr;
}

static void ReleaseLocalHostMapping(void* hostPtr, int64_t bytes)
{
    if (hostPtr == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(gLocalEngramMappingsMutex);
    auto iter = gLocalEngramMappings.find(hostPtr);
    TORCH_CHECK(iter != gLocalEngramMappings.end(), "local host mapping is missing during destroy");
    TORCH_CHECK(iter->second.bytes == bytes, "local host mapping size changed during destroy, registered ",
                iter->second.bytes, " bytes, got ", bytes, " bytes");
    TORCH_CHECK(iter->second.refcount > 0, "local host mapping refcount underflow");
    if (iter->second.refcount > 1) {
        --iter->second.refcount;
        return;
    }

    aclError ret = aclrtHostUnregister(hostPtr);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtHostUnregister failed, ret: ", ret);
    gLocalEngramMappings.erase(iter);
}

static void RollbackExternalHostRegistration(EngramContextResources& resources, void* externalHostPtr, void* sfHostPtr)
{
    if (externalHostPtr != nullptr && resources.externalRegistered) {
        (void)aclrtHostUnregister(externalHostPtr);
        resources.externalRegistered = false;
    }
    if (sfHostPtr != nullptr && resources.sfExternalRegistered) {
        (void)aclrtHostUnregister(sfHostPtr);
        resources.sfExternalRegistered = false;
    }
}

struct ExternalHostRegGuard {
    EngramContextResources* resources = nullptr;
    void* externalHostPtr = nullptr;
    void* sfHostPtr = nullptr;
    bool released = false;

    ExternalHostRegGuard(EngramContextResources& res, void* extHostPtr, void* sfTblHostPtr)
        : resources(&res),
          externalHostPtr(extHostPtr),
          sfHostPtr(sfTblHostPtr)
    {}

    ~ExternalHostRegGuard()
    {
        if (!released && resources != nullptr) {
            RollbackExternalHostRegistration(*resources, externalHostPtr, sfHostPtr);
        }
    }

    void Release()
    {
        released = true;
    }
};

template <typename ContextT>
static at::Tensor CreateCommContextTensor(const ContextT& context)
{
    int64_t numElements = (sizeof(ContextT) + sizeof(int32_t) - 1) / sizeof(int32_t);
    at::Tensor tensor = at::empty({numElements}, at::TensorOptions()
                                                     .dtype(at::kInt)
                                                     .device(c10::DeviceType::PrivateUse1)
                                                     .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor hostContext = at::empty({numElements}, at::TensorOptions().dtype(at::kInt));
    errno_t memRet = memcpy_s(hostContext.data_ptr<int32_t>(), hostContext.nbytes(), &context, sizeof(ContextT));
    TORCH_CHECK(memRet == EOK, "memcpy_s failed, ret=", memRet);
    tensor.copy_(hostContext);
    return tensor;
}

static void DumpRankPairLinks(const HcclComm& commHandle, uint32_t srcRankId, uint32_t rankSize,
                              const uint32_t* layerList, uint32_t layerNum)
{
    ASCEND_LOGI("[DumpTopo] src=%u rankSize=%u layerNum=%u", srcRankId, rankSize, layerNum);

    for (uint32_t li = 0; li < layerNum; ++li) {
        uint32_t layerId = layerList[li];

        CommTopo topo = COMM_TOPO_RESERVED;
        HcclRankGraphGetTopoTypeByLayerFunc(commHandle, layerId, &topo);
        ASCEND_LOGI("[DumpTopo] layer %u: topo=%d", layerId, static_cast<int>(topo));

        uint32_t* rankList = nullptr;
        uint32_t rankCnt = 0;
        if (HcclRankGraphGetRanksByLayerFunc(commHandle, layerId, &rankList, &rankCnt) == HCCL_SUCCESS) {
            std::string members;
            for (uint32_t i = 0; i < rankCnt; ++i) {
                members += std::to_string(rankList[i]);
                if (i + 1 < rankCnt) {
                    members += ",";
                }
            }
            ASCEND_LOGI("[DumpTopo] layer %u members: %s", layerId, members.c_str());
        }

        for (uint32_t dstRankId = 0; dstRankId < rankSize; ++dstRankId) {
            if (dstRankId == srcRankId) {
                continue;
            }
            CommLink* links = nullptr;
            uint32_t linkNum = 0;
            auto linkRet = HcclRankGraphGetLinksFunc(commHandle, layerId, srcRankId, dstRankId, &links, &linkNum);
            if (linkRet != HCCL_SUCCESS) {
                ASCEND_LOGI("[DumpTopo] layer %u src %u dst %u: GetLinks failed ret=%d", layerId, srcRankId, dstRankId,
                            static_cast<int>(linkRet));
                continue;
            }
            std::string protocols;
            for (uint32_t i = 0; i < linkNum; ++i) {
                protocols += std::to_string(static_cast<int>(links[i].linkAttr.linkProtocol));
                if (i + 1 < linkNum) {
                    protocols += ",";
                }
            }
            ASCEND_LOGI("[DumpTopo] layer %u src %u dst %u: linkNum=%u protocols=[%s]", layerId, srcRankId, dstRankId,
                        linkNum, protocols.c_str());
        }
    }
    ASCEND_LOGI("[DumpTopo] done");
}

// env 值防御解析：空串/非数字/负号/溢出/零值走 err 失败路径，不得以 stoul 异常逃出探测契约
static bool ParseEnvUint32(const char* name, const char* raw, uint32_t& value, std::string& err)
{
    const std::string rangeHint = "env " + std::string(name) + "='" + raw + "' must be a decimal integer in [1, " +
                                  std::to_string(UINT32_MAX) + "]";
    if (raw[0] == '\0' || raw[0] == '-') {
        err = rangeHint;
        return false;
    }
    errno = 0;
    char* endPtr = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &endPtr, 10);
    if (endPtr == raw || *endPtr != '\0' || errno == ERANGE || parsed == 0ULL || parsed > UINT32_MAX) {
        err = rangeHint;
        return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

class HcclContextBuilderBase {
public:
    // 按通信组名包装的拓扑探测入口（EnsureEngramContext 预探测用）
    static bool ProbeRailTopoByGroup(const std::string& groupName, RailTopoInfo& topo, std::string& reason)
    {
        HcclComm comm = nullptr;
        if (HcomGetCommHandleByGroupFunc(groupName.c_str(), &comm) != HCCL_SUCCESS || comm == nullptr) {
            reason = "get comm handle failed";
            return false;
        }
        uint32_t myRank = 0;
        uint32_t world = 0;
        if (HcclGetRankIdFunc(comm, &myRank) != HCCL_SUCCESS || HcclGetRankSizeFunc(comm, &world) != HCCL_SUCCESS ||
            world == 0U) {
            reason = "get rank info failed";
            return false;
        }
        return ProbeRailTopo(comm, myRank, world, topo, reason);
    }

protected:
    static void AcquireHcclHandle(const std::string& groupName, HcclComm& hcclComm)
    {
        auto hcclRet = HcomGetCommHandleByGroupFunc(groupName.c_str(), &hcclComm);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", groupName.c_str(), ", ret: ", hcclRet);
    }

    static void CheckContextTag(const std::string& contextTag)
    {
        TORCH_CHECK(contextTag.size() <= 255, "Mc2ContextTag is too long, max size is 255, got ", contextTag.size());
    }

    static void CreateEngineContext(const HcclComm& commHandle, const std::string& contextTag, const CommEngine& engine,
                                    uint64_t contextSize, void*& ctx)
    {
        auto hcclRet = HcclEngineCtxCreateFunc(commHandle, contextTag.c_str(), engine, contextSize, &ctx);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Create HCCL context memory failed, ret: ", hcclRet);
    }

    static void CopyContextToDevice(const HcclComm& commHandle, const std::string& contextTag, const CommEngine& engine,
                                    const void* context, uint64_t contextSize)
    {
        auto hcclRet =
            HcclEngineCtxCopyFunc(commHandle, engine, contextTag.c_str(), const_cast<void*>(context), contextSize, 0);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Copy context from host to device failed, ret: ", hcclRet);
    }

    static void GetRankInfo(const HcclComm& commHandle, uint32_t& rankId, uint32_t& rankSize)
    {
        auto hcclRet = HcclGetRankIdFunc(commHandle, &rankId);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank ID failed, ret: ", hcclRet);

        hcclRet = HcclGetRankSizeFunc(commHandle, &rankSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank size failed, ret: ", hcclRet);
    }

    static void AcquireChannels(const HcclComm& commHandle, const CommEngine& engine,
                                std::vector<HcclChannelDesc>& descs, ChannelHandle* channels)
    {
        auto hcclRet =
            HcclChannelAcquireFunc(commHandle, engine, descs.data(), static_cast<uint32_t>(descs.size()), channels);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Acquire HCCL channel failed, ret: ", hcclRet);
    }

    static void GetNetLayers(const HcclComm& commHandle, uint32_t*& netLayerList, uint32_t& netLayerNum)
    {
        auto hcclRet = HcclRankGraphGetLayersFunc(commHandle, &netLayerList, &netLayerNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL layers failed, ret: ", hcclRet);
    }

    // 查询 src→dst 在 rank-graph 中是否有任意链路（不区分协议）
    static bool HasAnyLink(const HcclComm& commHandle, const uint32_t* layerList, uint32_t layerNum, uint32_t srcRank,
                           uint32_t dstRank)
    {
        for (uint32_t li = 0; li < layerNum; ++li) {
            CommLink* links = nullptr;
            uint32_t linkNum = 0;
            if (HcclRankGraphGetLinksFunc(commHandle, layerList[li], srcRank, dstRank, &links, &linkNum) ==
                    HCCL_SUCCESS &&
                linkNum > 0) {
                return true;
            }
        }
        return false;
    }

    // 候选 B 实测校验：同号卡轨道对端逐个（超限时等距抽样，上限 64）实测有链路；按 B 降序，首个通过即最大有效 B
    static bool RailCandidateValid(const HcclComm& commHandle, const uint32_t* layerList, uint32_t layerNum,
                                   uint32_t myRank, uint32_t world, uint32_t candB)
    {
        constexpr uint32_t kRailProbeCap = 64;
        const uint32_t snCnt = world / candB;
        const uint32_t mySn = myRank / candB;
        const uint32_t myL = myRank % candB;
        const uint32_t total = snCnt - 1U;
        const uint32_t step = (total > kRailProbeCap) ? (total / kRailProbeCap) : 1U;
        uint32_t peerIdx = 0;
        uint32_t probed = 0;
        for (uint32_t sn = 0U; sn < snCnt && probed < kRailProbeCap; ++sn) {
            if (sn == mySn) {
                continue;
            }
            if (peerIdx % step == 0U) {
                const uint32_t peer = sn * candB + myL;
                ++probed;
                if (!HasAnyLink(commHandle, layerList, layerNum, myRank, peer)) {
                    return false;
                }
            }
            ++peerIdx;
        }
        return probed > 0U;
    }

    // 共享拓扑探测：env（单设 RANKS_PER_SUPERNODE / 成对 RANK_SIZE_PER_*）→ 层形状候选 + 链路实测
    static bool TryProbeHybridTopo(const HcclComm& shareComm, uint32_t rankId, uint32_t& ranksPerSupernode,
                                   uint32_t* ranksPerMachine, std::string& err)
    {
        const char* envPerSupernode = std::getenv("RANKS_PER_SUPERNODE");
        if (envPerSupernode != nullptr) {
            if (!ParseEnvUint32("RANKS_PER_SUPERNODE", envPerSupernode, ranksPerSupernode, err)) {
                return false;
            }
            // B 容量上界（ctx node 表超节点段容量）：超限 fail-fast 防 node 表越段写
            if (ranksPerSupernode > HCCL_MAX_RANK_SIZE) {
                err = "env RANKS_PER_SUPERNODE='" + std::string(envPerSupernode) +
                      "' exceeds ctx grp table capacity (max " + std::to_string(HCCL_MAX_RANK_SIZE) + ")";
                return false;
            }
            return true;
        }
        const char* envPerMachine = std::getenv("RANK_SIZE_PER_MACHINE");
        const char* envSnSize = std::getenv("RANK_SIZE_PER_SUPERNODE");
        if (envPerMachine != nullptr || envSnSize != nullptr) {
            if (envPerMachine == nullptr || envSnSize == nullptr) {
                err = "RANK_SIZE_PER_MACHINE and RANK_SIZE_PER_SUPERNODE must be set together";
                return false;
            }
            uint32_t perMachine = 0;
            uint32_t perSupernode = 0;
            if (!ParseEnvUint32("RANK_SIZE_PER_MACHINE", envPerMachine, perMachine, err) ||
                !ParseEnvUint32("RANK_SIZE_PER_SUPERNODE", envSnSize, perSupernode, err)) {
                return false;
            }
            if (perSupernode > HCCL_MAX_RANK_SIZE) {
                err = "env RANK_SIZE_PER_SUPERNODE='" + std::string(envSnSize) +
                      "' exceeds ctx grp table capacity (max " + std::to_string(HCCL_MAX_RANK_SIZE) + ")";
                return false;
            }
            uint32_t world = 0;
            if (HcclGetRankSizeFunc(shareComm, &world) != HCCL_SUCCESS || world == 0U) {
                err = "get rank size failed";
                return false;
            }
            if (perMachine == 0U || perSupernode == 0U || perSupernode % perMachine != 0U ||
                world % perSupernode != 0U) {
                err = "env RANK_SIZE_PER_MACHINE=" + std::to_string(perMachine) +
                      " / RANK_SIZE_PER_SUPERNODE=" + std::to_string(perSupernode) +
                      " inconsistent with world=" + std::to_string(world);
                return false;
            }
            ranksPerSupernode = perSupernode;
            if (ranksPerMachine != nullptr) {
                *ranksPerMachine = perMachine;
            }
            return true;
        }
        uint32_t world = 0;
        if (HcclGetRankSizeFunc(shareComm, &world) != HCCL_SUCCESS || world == 0U) {
            err = "get rank size failed";
            return false;
        }
        uint32_t* layerList = nullptr;
        uint32_t layerNum = 0;
        GetNetLayers(shareComm, layerList, layerNum);
        // 符号缺失 fail-fast：仅分轨探测消费，老版 libhccl_fwk 缺符号时经 err 上报；存量路径不经此
        if (HcclRankGraphGetInstSizeListByLayerFunc == nullptr) {
            err = "rail-aligned topo probe requires HCCL symbol HcclRankGraphGetInstSizeListByLayer "
                  "(libhccl_fwk too old for this CANN pairing; update CANN/HCCL)";
            return false;
        }
        // 每层形状（instCnt × instSize；instSize 全等且乘积覆盖全集群才参与候选）
        struct LayerShape {
            uint32_t layer;
            uint32_t instCnt;
            uint32_t instSize;
        };
        std::vector<LayerShape> shapes;
        for (uint32_t li = 0; li < layerNum; ++li) {
            uint32_t* instSizeList = nullptr;
            uint32_t instListSize = 0;
            (void)HcclRankGraphGetInstSizeListByLayerFunc(shareComm, layerList[li], &instSizeList, &instListSize);
            if (instListSize == 0 || instSizeList == nullptr) {
                continue;
            }
            bool uniform = true;
            for (uint32_t i = 1; i < instListSize; ++i) {
                if (instSizeList[i] != instSizeList[0]) {
                    uniform = false;
                    break;
                }
            }
            if (!uniform || static_cast<uint64_t>(instListSize) * instSizeList[0] != world) {
                continue;
            }
            shapes.push_back({layerList[li], instListSize, instSizeList[0]});
        }
        for (const auto& sh : shapes) {
            if (sh.instCnt == 1U && sh.instSize == world) {
                ranksPerSupernode = world;
                return true;
            }
        }
        std::vector<uint32_t> candidates;
        for (const auto& sh : shapes) {
            for (uint32_t v : {sh.instCnt, sh.instSize}) {
                if (v >= 2U && v <= HCCL_MAX_RANK_SIZE && world % v == 0U && world / v >= 2U &&
                    world / v <= RAIL_SN_DOMAIN_MAX) {
                    candidates.push_back(v);
                }
            }
        }
        std::sort(candidates.begin(), candidates.end(), std::greater<uint32_t>());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        if (candidates.empty()) {
            err = "structured layer selection failed: no valid B candidate from layer shapes (world=" +
                  std::to_string(world) + ", shapes=" + std::to_string(shapes.size()) + ")";
            return false;
        }
        for (uint32_t candB : candidates) {
            if (RailCandidateValid(shareComm, layerList, layerNum, rankId, world, candB)) {
                ranksPerSupernode = candB;
                return true;
            }
        }
        err = "structured layer selection failed: no candidate B passed rail-peer link validation (world=" +
              std::to_string(world) + ")";
        return false;
    }

    // 兜底拓扑探测：world 因子分解枚举候选 B + 轨道对端链路实测
    static bool TryDiscoverRailTopology(const HcclComm& comm, uint32_t myRank, uint32_t world, RailTopoInfo& topo,
                                        std::string& err)
    {
        uint32_t* layerList = nullptr;
        uint32_t layerNum = 0;
        if (HcclRankGraphGetLayersFunc(comm, &layerList, &layerNum) != HCCL_SUCCESS || layerNum == 0) {
            err = "no net layers visible on comm (world=" + std::to_string(world) + ")";
            return false;
        }
        const uint32_t bMax = std::min(HCCL_MAX_RANK_SIZE, world / 2U);
        for (uint32_t candB = bMax; candB >= 2U; --candB) { // 降序：首个通过即最大有效 B
            const uint32_t snCnt = world / candB;
            if (world % candB != 0U || snCnt < 2U || snCnt > RAIL_SN_DOMAIN_MAX) {
                continue;
            }
            if (!RailCandidateValid(comm, layerList, layerNum, myRank, world, candB)) {
                continue;
            }
            const uint32_t mySn = myRank / candB;
            const uint32_t myL = myRank % candB;
            topo.ranksPerSupernode = candB;
            for (uint32_t r = mySn * candB; r < mySn * candB + candB; ++r) {
                if (r != myRank) {
                    topo.supernodePeers.push_back(r);
                }
            }
            for (uint32_t sn = 0; sn < snCnt; ++sn) { // 轨道集 = 纯推导（可达性由建链实测兜底）
                if (sn != mySn) {
                    topo.railPeers.push_back(sn * candB + myL);
                }
            }
            return true;
        }
        err = "no divisor B of world passed rail-peer link validation (world=" + std::to_string(world) + ")";
        return false;
    }

    // 拓扑探测核心（comm 级）：优先共享探测，失败回退因子分解兜底；成功后按轨道公式展开 peer 集
    static bool ProbeRailTopo(const HcclComm& comm, uint32_t myRank, uint32_t world, RailTopoInfo& topo,
                              std::string& reason)
    {
        RailTopoInfo probe;
        uint32_t sharedRanksPerSupernode = 0;
        uint32_t sharedRanksPerMachine = 0;
        std::string sharedErr;
        if (TryProbeHybridTopo(comm, myRank, sharedRanksPerSupernode, &sharedRanksPerMachine, sharedErr)) {
            const uint32_t ranksPerSupernode = sharedRanksPerSupernode;
            if (ranksPerSupernode == 0U || ranksPerSupernode > HCCL_MAX_RANK_SIZE) { // rail ctx node 表超节点段容量上限
                reason = "ranksPerSupernode=" + std::to_string(ranksPerSupernode) +
                         " exceeds rail ctx node table supernode-segment limit 1024";
                return false;
            }
            if (world < ranksPerSupernode || world % ranksPerSupernode != 0U) {
                reason = "world(" + std::to_string(world) +
                         ") is not a multiple of ranksPerSupernode=" + std::to_string(ranksPerSupernode) +
                         " — rail-aligned requires uniform supernodes";
                return false;
            }
            const uint32_t snCnt = world / ranksPerSupernode;
            if (snCnt > RAIL_SN_DOMAIN_MAX) { // rail ctx node 表轨道段容量上限
                reason = "supernode count(world/ranksPerSupernode=" + std::to_string(snCnt) +
                         ") exceeds rail ctx node table rail-segment limit 2048";
                return false;
            }
            probe.ranksPerSupernode = ranksPerSupernode;
            probe.ranksPerMachine = sharedRanksPerMachine;
            // peer 集按轨道公式展开
            const uint32_t mySn = myRank / ranksPerSupernode;
            const uint32_t myL = myRank % ranksPerSupernode;
            for (uint32_t l = 0U; l < ranksPerSupernode; ++l) {
                const uint32_t r = mySn * ranksPerSupernode + l;
                if (r != myRank) {
                    probe.supernodePeers.push_back(r);
                }
            }
            for (uint32_t sn = 0U; sn < snCnt; ++sn) {
                if (sn != mySn) {
                    probe.railPeers.push_back(sn * ranksPerSupernode + myL);
                }
            }
            ASCEND_LOGI("[RailTopo] shared topo: ranksPerSupernode=%u snCnt=%u (world=%u)", ranksPerSupernode, snCnt,
                        world);
        } else {
            // 因子分解兜底
            std::string err;
            if (!TryDiscoverRailTopology(comm, myRank, world, probe, err)) {
                reason = "not a rail topology (world=" + std::to_string(world) + "): shared probe [" + sharedErr +
                         "]; divisor probe [" + err + "]";
                return false;
            }
            ASCEND_LOGI("[RailTopo] divisor topo fallback: ranksPerSupernode=%u", probe.ranksPerSupernode);
        }
        probe.world = world;
        topo = probe;
        return true;
    }

    // 逐层探测（不抛异常版）
    static bool ProbeLink(const HcclComm& commHandle, const uint32_t* layerList, uint32_t layerNum, uint32_t srcRank,
                          uint32_t dstRank, CommProtocol protocol, RankLinkInfo& out)
    {
        for (uint32_t li = 0; li < layerNum; ++li) {
            CommLink* links = nullptr;
            uint32_t linkNum = 0;
            auto hcclRet = HcclRankGraphGetLinksFunc(commHandle, layerList[li], srcRank, dstRank, &links, &linkNum);
            for (uint32_t i = 0; hcclRet == HCCL_SUCCESS && links != nullptr && i < linkNum; ++i) {
                if (links[i].linkAttr.linkProtocol == protocol) {
                    out = {links[i].linkAttr.linkProtocol, layerList[li]};
                    return true;
                }
            }
        }
        return false;
    }

    // 按对端规格批量建链（探测 + desc + Acquire），返回与 peers 等长等序的句柄集
    static std::vector<ChannelHandle> AcquireRailChannels(const HcclComm& commHandle, const uint32_t* layerList,
                                                          uint32_t layerNum, uint32_t myRank,
                                                          const std::vector<RailPeerSpec>& peers,
                                                          HcclMemHandle& memHandle)
    {
        std::vector<ChannelHandle> handles(peers.size(), 0);
        if (peers.empty()) {
            return handles;
        }
        std::vector<HcclChannelDesc> descs;
        descs.reserve(peers.size());
        for (const RailPeerSpec& spec : peers) {
            // 协议期望：超节点内 UB_CTP 直连；出超节点（轨道腿）UB_RTP
            CommProtocol expected =
                spec.inSupernode ? CommProtocol::COMM_PROTOCOL_UB_CTP : CommProtocol::COMM_PROTOCOL_UB_RTP;
            RankLinkInfo linkInfo{};
            bool linkFound = ProbeLink(commHandle, layerList, layerNum, myRank, spec.peerRank, expected, linkInfo);
            if (!linkFound) {
                // 协议二选一兜底：env 覆盖形态与物理链路错位时换另一协议再试
                expected = (expected == CommProtocol::COMM_PROTOCOL_UB_CTP) ? CommProtocol::COMM_PROTOCOL_UB_RTP :
                                                                              CommProtocol::COMM_PROTOCOL_UB_CTP;
                linkFound = ProbeLink(commHandle, layerList, layerNum, myRank, spec.peerRank, expected, linkInfo);
            }
            if (!linkFound) {
                // 诊断：dump 该对端在全部层的全部链路协议
                std::string linksDesc;
                for (uint32_t li = 0; li < layerNum; ++li) {
                    CommLink* dbgLinks = nullptr;
                    uint32_t dbgLinkNum = 0;
                    if (HcclRankGraphGetLinksFunc(commHandle, layerList[li], myRank, spec.peerRank, &dbgLinks,
                                                  &dbgLinkNum) == HCCL_SUCCESS &&
                        dbgLinks != nullptr) {
                        for (uint32_t i = 0; i < dbgLinkNum; ++i) {
                            linksDesc += " [layer=" + std::to_string(layerList[li]) + " proto=" +
                                         std::to_string(static_cast<uint32_t>(dbgLinks[i].linkAttr.linkProtocol)) + "]";
                        }
                    }
                }
                TORCH_CHECK(false, "rail topo probe: no ",
                            (expected == CommProtocol::COMM_PROTOCOL_UB_CTP ? "UB_CTP" : "UB_RTP"), " link to peer ",
                            spec.peerRank, " [", (spec.inSupernode ? "supernode-peer" : "rail-peer"),
                            ", myRank=", myRank,
                            "] (probe result inconsistent with layer members, check cluster wiring); visible links:",
                            linksDesc.empty() ? " <none in any layer>" : linksDesc);
            }
            CommLink* links = nullptr;
            GetHcclCommLink(commHandle, linkInfo.layer, myRank, spec.peerRank, expected, links);
            HcclChannelDesc desc;
            HcclResult initRet = HcclChannelDescInit(&desc, 1);
            TORCH_CHECK(initRet == HCCL_SUCCESS, "HcclChannelDescInit failed, ret=", initRet);
            desc.remoteRank = spec.peerRank;
            desc.channelProtocol = linkInfo.protocol;
            desc.localEndpoint = links->srcEndpointDesc;
            desc.remoteEndpoint = links->dstEndpointDesc;
            desc.notifyNum = MOE_CHANNEL_NOTIFY_NUM;
            desc.memHandles = &memHandle; // V1 同款：channel 绑定存储注册内存
            desc.memHandleNum = 1;
            descs.push_back(desc);
        }
        AcquireChannels(commHandle, CommEngine::COMM_ENGINE_AIV, descs, handles.data());
        return handles;
    }

    bool SupportsProtocol(const HcclComm& commHandle, uint32_t layerId, uint32_t srcRankId, uint32_t dstRankId,
                          const CommProtocol& protocol) const
    {
        CommLink* linksList = nullptr;
        uint32_t netLinkNum = 0;
        auto hcclRet = HcclRankGraphGetLinksFunc(commHandle, layerId, srcRankId, dstRankId, &linksList, &netLinkNum);
        if (hcclRet != HCCL_SUCCESS || netLinkNum == 0) {
            ASCEND_LOGW("Get HCCL links failed when checking protocol support, srcRankId: %u, dstRankId: %u, "
                        "layerId: %u, protocol: %u, ret: %d, netLinkNum: %u",
                        srcRankId, dstRankId, layerId, protocol, hcclRet, netLinkNum);
            return false;
        }
        return HasLinkWithProtocol(linksList, netLinkNum, protocol);
    }

    void CheckProtocolSupport(const HcclComm& commHandle, const uint32_t* layerList, uint32_t layerNum,
                              const CommProtocol& protocol)
    {
        ASCEND_LOGI("start CheckProtocolSupport");
        uint32_t srcRankId = 0;
        uint32_t rankSize = 0;
        GetRankInfo(commHandle, srcRankId, rankSize);
        rankLinkMap_.clear();
        rankNumPerUbDomain_ = 0;

        LayerRanks ubDomain;
        if (!FindUbDomain(commHandle, layerList, layerNum, protocol, srcRankId, ubDomain)) {
            TORCH_CHECK(false, "Failed to determine UB domain for rank ", srcRankId,
                        ", rank has no peer supporting protocol ", static_cast<int>(protocol));
        }
        rankNumPerUbDomain_ = static_cast<uint32_t>(ubDomain.ranks.size());
        ASCEND_LOGI("Layer %u is current rank's UB domain, rankNumPerUbDomain_: %u", ubDomain.layer,
                    rankNumPerUbDomain_);

        ASCEND_LOGI("complete FindUbDomain");
        if (IsSingleUbDomain(ubDomain, rankSize)) {
            return;
        }

        TORCH_CHECK(rankSize >= rankNumPerUbDomain_ && rankSize % rankNumPerUbDomain_ == 0,
                    "rankNumPerUbDomain_ must be less than rankSize and divisible, rankNumPerUbDomain_: ",
                    rankNumPerUbDomain_, ", rankSize: ", rankSize);

        TORCH_CHECK(CheckIntraUbDomainProtocol(commHandle, srcRankId), "Rank ", srcRankId,
                    " does not support UB_CTP with peers inside its UB domain");
        TORCH_CHECK(CheckCrossUbDomainProtocols(commHandle, layerList, layerNum, srcRankId), "Rank ", srcRankId,
                    " does not support UBG with peers outside its UB domain");
        TORCH_CHECK(rankLinkMap_.size() == rankSize - 1, "Incomplete topology info for rank ", srcRankId, ", recorded ",
                    rankLinkMap_.size(), " of ", rankSize - 1);
        ASCEND_LOGI("Cross-server confirmed, use UB_CTP inside UB domain and UBG across UB domains");
    }

    bool FindUbDomain(const HcclComm& commHandle, const uint32_t* layerList, uint32_t layerNum,
                      const CommProtocol& domainProtocol, uint32_t srcRankId, LayerRanks& ubDomain)
    {
        bool hasDomainLayer = false;
        for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
            LayerRanks layer = GetLayerRanks(commHandle, layerList[layerIndex]);
            if (!SupportsDomainProtocolWithAllRanks(commHandle, layer, domainProtocol, srcRankId)) {
                break;
            }
            RecordDomainLayerRanks(domainProtocol, layer, srcRankId);
            ubDomain = std::move(layer);
            hasDomainLayer = true;
        }
        return hasDomainLayer;
    }

    bool CheckIntraUbDomainProtocol(const HcclComm& commHandle, uint32_t srcRankId)
    {
        for (auto& linkEntry : rankLinkMap_) {
            uint32_t dstRank = linkEntry.first;
            uint32_t layer = linkEntry.second.layer;
            if (!SupportsProtocol(commHandle, layer, srcRankId, dstRank, CommProtocol::COMM_PROTOCOL_UB_CTP)) {
                ASCEND_LOGW("Rank %u does not support UB_CTP with rank %u in UB domain layer %u", srcRankId, dstRank,
                            layer);
                return false;
            }
            linkEntry.second.protocol = CommProtocol::COMM_PROTOCOL_UB_CTP;
        }
        return true;
    }

    bool CheckCrossUbDomainProtocols(const HcclComm& commHandle, const uint32_t* layerList, uint32_t layerNum,
                                     uint32_t srcRankId)
    {
        bool isSupportUBG = false;
        for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
            LayerRanks layer = GetLayerRanks(commHandle, layerList[layerIndex]);
            for (uint32_t dstRank : layer.ranks) {
                if (dstRank == srcRankId || rankLinkMap_.count(dstRank) > 0) {
                    continue;
                }
                if (SupportsProtocol(commHandle, layer.layer, srcRankId, dstRank, CommProtocol::COMM_PROTOCOL_UB_RTP)) {
                    ASCEND_LOGI("Rank %u does support UBG with cross-domain rank %u in layer %u", srcRankId, dstRank,
                                layer.layer);
                    isSupportUBG = true;
                    rankLinkMap_[dstRank] = {CommProtocol::COMM_PROTOCOL_UB_RTP, layer.layer};
                }
            }
        }
        return isSupportUBG;
    }

    LayerRanks GetLayerRanks(const HcclComm& commHandle, uint32_t layerId) const
    {
        uint32_t rankNum = 0;
        uint32_t* rankList = nullptr;
        auto hcclRet = HcclRankGraphGetRanksByLayerFunc(commHandle, layerId, &rankList, &rankNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank IDs by layer failed, ret: ", hcclRet);
        return {layerId, std::vector<uint32_t>(rankList, rankList + rankNum)};
    }

    bool SupportsDomainProtocolWithAllRanks(const HcclComm& commHandle, const LayerRanks& layer,
                                            const CommProtocol& domainProtocol, uint32_t srcRankId) const
    {
        for (uint32_t dstRank : layer.ranks) {
            if (dstRank == srcRankId || rankLinkMap_.count(dstRank) > 0) {
                continue;
            }
            if (!SupportsProtocol(commHandle, layer.layer, srcRankId, dstRank, domainProtocol)) {
                return false;
            }
        }
        return true;
    }

    bool IsSingleUbDomain(const LayerRanks& ubDomain, uint32_t rankSize) const
    {
        return ubDomain.ranks.size() == rankSize;
    }

    void RecordDomainLayerRanks(const CommProtocol& protocol, const LayerRanks& layer, uint32_t srcRankId)
    {
        for (uint32_t dstRank : layer.ranks) {
            if (dstRank != srcRankId && rankLinkMap_.count(dstRank) == 0) {
                rankLinkMap_[dstRank] = {protocol, layer.layer};
            }
        }
    }

    static bool HasLinkWithProtocol(const CommLink* linksList, uint32_t netLinkNum, const CommProtocol& protocol)
    {
        for (uint32_t linkIdx = 0; linkIdx < netLinkNum; ++linkIdx) {
            if (linksList[linkIdx].linkAttr.linkProtocol == protocol) {
                return true;
            }
        }
        return false;
    }

    static void GetHcclCommLink(const HcclComm& commHandle, uint32_t netLayerId, uint32_t srcRankId, uint32_t dstRankId,
                                const CommProtocol& protocol, CommLink*& links)
    {
        CommLink* linksList = nullptr;
        uint32_t netLinkNum = 0;
        auto hcclRet = HcclRankGraphGetLinksFunc(commHandle, netLayerId, srcRankId, dstRankId, &linksList, &netLinkNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL Communication link failed, ret: ", hcclRet);
        TORCH_CHECK(netLinkNum > 0, "The Net Link Is nullptr. srcRankId is ", srcRankId, ", dstRankId is ", dstRankId,
                    ", layerId is ", netLayerId);
        uint32_t index = 0;
        for (; index < netLinkNum; ++index) {
            if (linksList[index].linkAttr.linkProtocol == protocol) {
                links = &linksList[index];
                break;
            }
        }
        TORCH_CHECK(index < netLinkNum, "No matching communication protocol found in HCCL links, protocol is ",
                    static_cast<int>(protocol));
        ASCEND_LOGI("Get HCCL Communication link Success, srcRankId %u, dstRankId %u, layerId %u, protocol is: %u",
                    srcRankId, dstRankId, netLayerId, static_cast<int>(protocol));
    }

    RankLinkInfo ResolveLinkInfo(uint32_t dstRank, const CommProtocol& protocol, const uint32_t* netLayerList) const
    {
        auto linkIter = rankLinkMap_.find(dstRank);
        if (linkIter != rankLinkMap_.end()) {
            return linkIter->second;
        }
        uint32_t fallbackLayer = netLayerList[HCCL_COMM_LAYERS_UB_MEM];
        ASCEND_LOGW("Topology info not found for dstRank: %u, fallback to layer %u with protocol %d", dstRank,
                    fallbackLayer, static_cast<int>(protocol));
        return {protocol, fallbackLayer};
    }

    std::unordered_map<uint32_t, RankLinkInfo> rankLinkMap_;
    uint32_t rankNumPerUbDomain_ = 0;
};

struct EngramRailStorage {
    uint32_t ranksPerMachine = 0;
    uint32_t ranksPerSupernode = 0;
    uint32_t machineCount = 0;
    uint32_t machinesPerSupernode = 0;
    uint32_t supernodeStartRank = 0;
    uint32_t superStartMachine = 0;
    void* shareVa = nullptr;
    void* shareOwnSlotVa = nullptr;
    uint64_t slotBytes = 0;
};

class EngramContextBuilder : public HcclContextBuilderBase {
public:
    EngramContextBuilder(int64_t hybridMode = 0, aclrtStream commStream = nullptr)
        : hybridMode_(hybridMode),
          commStream_(commStream)
    {}

    static EngramContextResources BuildLocal(int64_t numCpuBytes, void* externalHostPtr, int64_t externalBytes)
    {
        TORCH_CHECK(externalHostPtr == nullptr || externalBytes > 0,
                    "local zero-copy storage size must be positive, got ", externalBytes);

        EngramContextResources resources;
        if (externalHostPtr != nullptr) {
            resources.hostBufPtr = externalHostPtr;
            resources.deviceBufPtr = AcquireLocalHostMapping(externalHostPtr, externalBytes);
            resources.externalRegistered = true;
        } else {
            TORCH_CHECK(numCpuBytes > 0, "local copy window size must be positive, got ", numCpuBytes);
            aclError ret = aclrtMallocHost(&resources.hostBufPtr, static_cast<uint64_t>(numCpuBytes));
            TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMallocHost(", numCpuBytes, " B) failed, ret=", ret);
            try {
                resources.deviceBufPtr = AcquireLocalHostMapping(resources.hostBufPtr, numCpuBytes);
            } catch (...) {
                (void)aclrtFreeHost(resources.hostBufPtr);
                resources.hostBufPtr = nullptr;
                throw;
            }
        }

        try {
            resources.context.rankId = 0;
            resources.context.rankSize = 1;
            resources.context.channelsPerRank = 1;
            resources.context.virtualAddrList[0] = reinterpret_cast<uint64_t>(resources.deviceBufPtr);
            resources.contextTensor = CreateCommContextTensor(resources.context);
        } catch (...) {
            ReleaseLocalHostMapping(resources.hostBufPtr, externalHostPtr != nullptr ? externalBytes : numCpuBytes);
            if (externalHostPtr == nullptr) {
                (void)aclrtFreeHost(resources.hostBufPtr);
            }
            resources.hostBufPtr = nullptr;
            resources.deviceBufPtr = nullptr;
            resources.externalRegistered = false;
            throw;
        }
        return resources;
    }

    // rail_aligned推理 交换共享内存句柄
    static std::vector<aclrtMemFabricHandle> ExchangeRailHandles(const std::string& groupName, HcclComm shareComm,
                                                                 aclrtStream commStream, uint32_t rankId,
                                                                 uint32_t rankSize, uint32_t ranksPerMachine,
                                                                 uint32_t machineStartRank,
                                                                 aclrtMemFabricHandle& shareBlob)
    {
        int32_t bareTgid = 0;
        aclError ret = aclrtDeviceGetBareTgid(&bareTgid);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtDeviceGetBareTgid failed, ret=", ret);
        constexpr uint32_t handleBlobU64 = sizeof(aclrtMemFabricHandle) / sizeof(uint64_t);
        constexpr uint32_t exchangeElems = handleBlobU64 + 1U;
        constexpr uint64_t exchangeBytes = static_cast<uint64_t>(exchangeElems) * sizeof(uint64_t);
        std::vector<uint64_t> sendVals(exchangeElems, 0);
        memcpy(sendVals.data(), shareBlob.data, sizeof(shareBlob.data));
        sendVals[handleBlobU64] = static_cast<uint64_t>(bareTgid);

        std::string shmName;
        for (char c : groupName) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-') {
                shmName += c;
            }
        }
        const std::string shmPath = "/dev/shm/engram_rail_" + shmName + ".bin";
        const int fd = open(shmPath.c_str(), O_RDWR | O_CREAT, 0666);
        TORCH_CHECK(fd >= 0, "open(", shmPath, ") failed, errno=", errno);
        FdGuard fdGuard(fd);
        const int ftRet = ftruncate(fd, static_cast<off_t>(exchangeBytes * rankSize));
        TORCH_CHECK(ftRet == 0, "ftruncate(", shmPath, ", ", exchangeBytes * rankSize, ") failed, errno=", errno);
        const ssize_t wRet = pwrite(fd, sendVals.data(), exchangeBytes, static_cast<off_t>(exchangeBytes * rankId));
        TORCH_CHECK(wRet == static_cast<ssize_t>(exchangeBytes), "pwrite(own slot) failed, errno=", errno);

        auto barrierRet = HcclBarrierFunc(shareComm, commStream);
        TORCH_CHECK(barrierRet == HCCL_SUCCESS,
                    "HcclBarrier(handle exchange sync) failed, ret=", static_cast<int>(barrierRet));
        ret = aclrtSynchronizeStream(commStream);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtSynchronizeStream(handle exchange sync) failed, ret=", ret);

        std::vector<uint64_t> recvVals(static_cast<size_t>(exchangeElems) * rankSize, 0);
        for (uint32_t g = machineStartRank; g < machineStartRank + ranksPerMachine; ++g) {
            const ssize_t rRet = pread(fd, &recvVals[static_cast<size_t>(g) * exchangeElems], exchangeBytes,
                                       static_cast<off_t>(exchangeBytes * g));
            TORCH_CHECK(rRet == static_cast<ssize_t>(exchangeBytes), "pread(slot=", g, ") failed, errno=", errno);
        }

        std::vector<aclrtMemFabricHandle> sharePeerBlobs(rankSize, aclrtMemFabricHandle{});
        std::vector<int32_t> peerPids(rankSize, 0);
        for (uint32_t rank = 0; rank < rankSize; ++rank) {
            memcpy(sharePeerBlobs[rank].data, &recvVals[static_cast<size_t>(rank) * exchangeElems],
                   sizeof(sharePeerBlobs[rank].data));
            peerPids[rank] = static_cast<int32_t>(recvVals[static_cast<size_t>(rank) * exchangeElems + handleBlobU64]);
        }

        if (rankId == machineStartRank) {
            (void)unlink(shmPath.c_str());
        }

        std::vector<int32_t> peerPidList;
        peerPidList.reserve(ranksPerMachine - 1U);
        for (uint32_t rank = machineStartRank; rank < machineStartRank + ranksPerMachine; ++rank) {
            if (rank != rankId) {
                peerPidList.push_back(peerPids[rank]);
            }
        }
        std::string pidStr;
        for (int32_t p : peerPidList) {
            pidStr += (pidStr.empty() ? "" : ",") + std::to_string(p);
        }
        ASCEND_LOGI("rail_aligned collect handle: rank=%u machineStartRank=%u rpm=%u pidNum=%zu pids=[%s]", rankId,
                    machineStartRank, ranksPerMachine, peerPidList.size(), pidStr.c_str());
        ret = aclrtMemSetPidToShareableHandleV2(shareBlob.data, ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT, peerPidList.data(),
                                                peerPidList.size());
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemSetPidToShareableHandleV2(peers=", peerPidList.size(),
                    ") failed, ret=", ret);
        barrierRet = HcclBarrierFunc(shareComm, commStream);
        TORCH_CHECK(barrierRet == HCCL_SUCCESS,
                    "HcclBarrier(handle whitelist sync) failed, ret=", static_cast<int>(barrierRet));
        ret = aclrtSynchronizeStream(commStream);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtSynchronizeStream(whitelist sync) failed, ret=", ret);
        return sharePeerBlobs;
    }

    // rail_aligned推理 内存映射
    static void MapRailMemory(uint32_t rankId, uint32_t ranksPerMachine, uint32_t machineStartRank, int32_t deviceId,
                              uint64_t totalBytes, uint64_t slotBytes, aclrtDrvMemHandle sharePhysHandle,
                              std::vector<aclrtMemFabricHandle>& sharePeerBlobs, void*& shareVa, void*& shareOwnSlotVa,
                              std::vector<aclrtDrvMemHandle>& shareImportedHandles)
    {
        aclError ret = ACL_SUCCESS;
        int32_t logicDeviceId = deviceId;
        {
            int32_t converted = -1;
            aclError convRet = aclrtGetLogicDevIdByUserDevId(deviceId, &converted);
            if (convRet == ACL_SUCCESS && converted >= 0) {
                logicDeviceId = converted;
            } else {
                ASCEND_LOGW("aclrtGetLogicDevIdByUserDevId(dev=%d) failed, ret=%d, fallback to user devid", deviceId,
                            static_cast<int>(convRet));
            }
        }
        aclrtMemAccessDesc accessDesc = {};
        accessDesc.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
        accessDesc.location.id = static_cast<uint32_t>(logicDeviceId);
        accessDesc.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;

        shareVa = nullptr;
        ret = aclrtReserveMemAddress(&shareVa, totalBytes, 0, nullptr, VMM_RESERVE_MEM_FLAG);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtReserveMemAddress(", totalBytes, " B) failed, ret=", ret);

        auto slotVaOf = [&](uint32_t g) {
            return static_cast<uint8_t*>(shareVa) + static_cast<uint64_t>(g - machineStartRank) * slotBytes;
        };

        for (uint32_t g = machineStartRank; g < machineStartRank + ranksPerMachine; ++g) {
            aclrtDrvMemHandle slotHandle = sharePhysHandle;
            if (g != rankId) {
                ret = aclrtMemImportFromShareableHandleV2(sharePeerBlobs[g].data, ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT, 0,
                                                          &slotHandle);
                TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemImportFromShareableHandleV2(rank=", g, ") failed, ret=", ret);
                shareImportedHandles.push_back(slotHandle);
            }
            void* slotVa = slotVaOf(g);
            ret = aclrtMapMem(slotVa, slotBytes, 0, slotHandle, 0);
            TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMapMem(slot=", g, ") failed, ret=", ret);
        }

        for (uint32_t g = machineStartRank; g < machineStartRank + ranksPerMachine; ++g) {
            void* slotVa = slotVaOf(g);
            ret = aclrtMemSetAccess(slotVa, slotBytes, &accessDesc, 1);
            TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemSetAccess(slot=", g, ") failed: va=", slotVa, ", ret=", ret);
        }
        shareOwnSlotVa = static_cast<uint8_t*>(shareVa) + static_cast<uint64_t>(rankId - machineStartRank) * slotBytes;

        ret = aclrtMemset(shareOwnSlotVa, slotBytes, 0, slotBytes);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemset(local slot) failed, ret=", ret);
    }

    // rail_aligned推理 申请内存
    void AllocRailMemory(const std::string& memBufferTag, int64_t numCpuBytes, EngramContextResources& resources)
    {
        const uint32_t rankId = resources.context.rankId;
        const uint32_t rankSize = resources.context.rankSize;
        const uint32_t ranksPerMachine = railStorage_.ranksPerMachine;
        const uint32_t machineStartRank = (rankId / ranksPerMachine) * ranksPerMachine;

        int32_t deviceId = 0;
        aclError ret = aclrtGetDevice(&deviceId);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtGetDevice failed, ret=", ret);

        aclrtPhysicalMemProp prop = {};
        prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
        prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        prop.memAttr = ACL_MEM_P2P_HUGE;
        prop.location.id = 0;
        prop.location.type = ACL_MEM_LOCATION_TYPE_HOST;
        prop.reserve = 0;

        size_t granularity = 0;
        ret = aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity);
        if (ret != ACL_SUCCESS || granularity == 0) {
            ASCEND_LOGW("aclrtMemGetAllocationGranularity failed, ret=%d, fallback to 2MB", static_cast<int>(ret));
            granularity = 2UL * 1024UL * 1024UL;
        }
        const uint64_t slotBytes = (static_cast<uint64_t>(numCpuBytes) + granularity - 1U) / granularity * granularity;
        const uint64_t totalBytes = slotBytes * ranksPerMachine;
        aclrtDrvMemHandle sharePhysHandle = nullptr;
        ret = aclrtMallocPhysical(&sharePhysHandle, slotBytes, &prop, 0);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMallocPhysical(ACL_MEM_P2P_HUGE@HOST, ", slotBytes,
                    " B) failed, ret=", ret);

        aclrtMemFabricHandle shareBlob = {};
        ret = aclrtMemExportToShareableHandleV2(sharePhysHandle, 0, ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT, shareBlob.data);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemExportToShareableHandleV2(DEFAULT) failed, ret=", ret);

        std::vector<aclrtMemFabricHandle> sharePeerBlobs =
            ExchangeRailHandles(groupName_, resources.hcclComm, commStream_, rankId, rankSize, ranksPerMachine,
                                machineStartRank, shareBlob);

        void* shareVa = nullptr;        // 虚拟映射首地址
        void* shareOwnSlotVa = nullptr; // 虚拟映射当前分段地址
        std::vector<aclrtDrvMemHandle> shareImportedHandles;
        MapRailMemory(rankId, ranksPerMachine, machineStartRank, deviceId, totalBytes, slotBytes, sharePhysHandle,
                      sharePeerBlobs, shareVa, shareOwnSlotVa, shareImportedHandles);
        ASCEND_LOGI("rail_aligned alloc: rank=%u va=%p ownSlotVa=%p slotBytes=%lu totalBytes=%lu", rankId, shareVa,
                    shareOwnSlotVa, slotBytes, totalBytes);

        CommMem mem = {};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = shareVa;
        mem.size = slotBytes * ranksPerMachine;
        auto memRegRet = HcclCommMemRegFunc(resources.hcclComm, memBufferTag.c_str(), &mem, &resources.memHandle);
        TORCH_CHECK(memRegRet == HCCL_SUCCESS, "HcclCommMemReg(tag='", memBufferTag, "', size=", mem.size,
                    ") failed, ret=", memRegRet);

        auto barrierRet = HcclBarrierFunc(resources.hcclComm, commStream_);
        TORCH_CHECK(barrierRet == HCCL_SUCCESS,
                    "HcclBarrier(rail build reg sync) failed, ret=", static_cast<int>(barrierRet));
        ret = aclrtSynchronizeStream(commStream_);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtSynchronizeStream(rail build reg sync) failed, ret=", ret);

        railStorage_.shareVa = shareVa;
        railStorage_.shareOwnSlotVa = shareOwnSlotVa;
        railStorage_.slotBytes = slotBytes;
        resources.hostBufPtr = shareOwnSlotVa;
        resources.deviceBufPtr = shareOwnSlotVa;
    }

    // rail_aligned推理 通信建链
    void GetRailCommResource(const HcclComm& commHandle, EngramContextResources& resources,
                             const std::string& targetTag)
    {
        const uint32_t rankId = resources.context.rankId;
        const uint32_t ranksPerMachine = railStorage_.ranksPerMachine;
        const uint32_t ranksPerSupernode = railStorage_.ranksPerSupernode;
        const uint32_t machineCount = railStorage_.machineCount;
        const uint32_t machinesPerSupernode = railStorage_.machinesPerSupernode;
        const uint32_t superStartMachine = railStorage_.superStartMachine;
        const uint64_t slotBytes = railStorage_.slotBytes;
        EngramCommContext& ctx = resources.context;
        ctx.channelsPerRank = 1;
        std::vector<std::pair<uint32_t, bool>> linkedPeers;
        auto buildOneChannel = [&](uint32_t peer, bool inSuper) -> ChannelHandle {
            auto linkIter = rankLinkMap_.find(peer);
            TORCH_CHECK(linkIter != rankLinkMap_.end(), "rail_aligned: no routing info for peer rank ", peer);
            CommLink* link = nullptr;
            GetHcclCommLink(commHandle, linkIter->second.layer, rankId, peer, linkIter->second.protocol, link);
            HcclChannelDesc desc;
            auto chRet = HcclChannelDescInit(&desc, 1);
            TORCH_CHECK(chRet == HCCL_SUCCESS, "HcclChannelDescInit(peer=", peer,
                        ") failed, ret=", static_cast<int>(chRet));
            desc.remoteRank = peer;
            desc.channelProtocol = linkIter->second.protocol;
            desc.localEndpoint = link->srcEndpointDesc;
            desc.remoteEndpoint = link->dstEndpointDesc;
            desc.notifyNum = MOE_CHANNEL_NOTIFY_NUM;
            desc.memHandles = &resources.memHandle;
            desc.memHandleNum = 1;
            std::vector<HcclChannelDesc> descs{desc};
            ChannelHandle channel = 0;
            AcquireChannels(commHandle, CommEngine::COMM_ENGINE_AIV, descs, &channel);
            linkedPeers.push_back({peer, inSuper});
            return channel;
        };
        auto queryPeerBase = [&](uint32_t peer, ChannelHandle channel) -> uint64_t {
            uint32_t memNum = 0;
            CommMem* remoteMems = nullptr;
            char** memTags = nullptr;
            auto memRet = HcclChannelGetRemoteMemsFunc(commHandle, channel, &memNum, &remoteMems, &memTags);
            TORCH_CHECK(memRet == HCCL_SUCCESS, "HcclChannelGetRemoteMems(peer=", peer, ") failed, ret=", memRet);
            uint64_t peerBase = 0;
            for (uint32_t j = 0; j < memNum; ++j) {
                if (memTags != nullptr && remoteMems != nullptr && memTags[j] != nullptr && targetTag == memTags[j]) {
                    peerBase = reinterpret_cast<uint64_t>(remoteMems[j].addr);
                }
            }
            TORCH_CHECK(peerBase != 0, "remote rail buffer tag '", targetTag, "' not found for peer ", peer);
            return peerBase;
        };
        const uint32_t myMachine = rankId / ranksPerMachine;
        const uint32_t cardIdx = rankId % ranksPerMachine;
        auto superSlot = [&](uint32_t ownerRank) {
            return superStartMachine + ownerRank - railStorage_.supernodeStartRank;
        };
        auto outMachineSlot = [&](uint32_t m) {
            return (m < superStartMachine) ? m : m + ranksPerSupernode - machinesPerSupernode;
        };
        for (uint32_t m = 0; m < machineCount; ++m) {
            if (m == myMachine) {
                for (uint32_t slot = 0; slot < ranksPerMachine; ++slot) {
                    const uint32_t ownerRank = m * ranksPerMachine + slot;
                    ctx.virtualAddrList[superSlot(ownerRank)] =
                        reinterpret_cast<uint64_t>(railStorage_.shareVa) + static_cast<uint64_t>(slot) * slotBytes;
                }
                continue;
            }
            const bool inSuper = (m / machinesPerSupernode) == myMachine / machinesPerSupernode;
            if (inSuper) {
                for (uint32_t slot = 0; slot < ranksPerMachine; ++slot) {
                    const uint32_t ownerRank = m * ranksPerMachine + slot;
                    const ChannelHandle channel = buildOneChannel(ownerRank, true);
                    const uint32_t slotIdx = superSlot(ownerRank);
                    ctx.virtualAddrList[slotIdx] = queryPeerBase(ownerRank, channel);
                    ctx.hcommHandle[slotIdx] = static_cast<uint64_t>(channel);
                }
            } else {
                const uint32_t peer = m * ranksPerMachine + cardIdx;
                const ChannelHandle channel = buildOneChannel(peer, false);
                ctx.virtualAddrList[outMachineSlot(m)] = queryPeerBase(peer, channel);
                ctx.hcommHandle[outMachineSlot(m)] = static_cast<uint64_t>(channel);
            }
        }
        ctx.ranksPerMachine = ranksPerMachine;
        ctx.ranksPerSupernode = ranksPerSupernode;
        ctx.rankSlotStride = slotBytes;
        ASCEND_LOGI("rail_aligned build: rank=%u machine=%u cardIdx=%u machineCount=%u vaBase=%p ownSlotVa=%lld "
                    "ranksPerMachine=%u",
                    rankId, myMachine, cardIdx, machineCount, railStorage_.shareVa,
                    reinterpret_cast<int64_t>(railStorage_.shareOwnSlotVa), ranksPerMachine);
        {
            std::string localPeers;
            for (uint32_t owner = 0; owner < ranksPerSupernode; ++owner) {
                if (ctx.virtualAddrList[superStartMachine + owner] != 0U) {
                    localPeers += (localPeers.empty() ? "" : ",") + std::to_string(owner);
                }
            }
            std::string inSuperPeers;
            std::string outSuperPeers;
            uint32_t channelCount = 0U;
            for (const auto& [peer, inSuper] : linkedPeers) {
                auto& list = inSuper ? inSuperPeers : outSuperPeers;
                list += (list.empty() ? "" : ",") + std::to_string(peer);
                ++channelCount;
            }
            ASCEND_LOGI("rail_aligned links: rank=%u local=[%s] inSuper=[%s] outSuper=[%s] channels=%u", rankId,
                        localPeers.c_str(), inSuperPeers.c_str(), outSuperPeers.c_str(), channelCount);
        }
    }

    EngramContextResources Build(const std::string& groupName, int64_t numCpuBytes, bool withGrad,
                                 void* externalHostPtr = nullptr, int64_t externalBytes = 0, void* sfHostPtr = nullptr,
                                 int64_t sfBytes = 0, const std::string& netLayer = "auto")
    {
        withGrad_ = withGrad;
        netLayer_ = netLayer;
        groupName_ = groupName;
        EngramContextResources resources;
        AcquireHcclHandle(groupName, resources.hcclComm);
        std::string contextTag =
            groupName + (hybridMode_ == NETWORK_HYBRID ? "engram_embedding_rail_aligned" : "engram_embedding");
        CheckContextTag(contextTag);

        std::lock_guard<std::mutex> lock(gEngramSharedBufferMutex);

        bool ctxCreated = HasEngramCommCtx(resources.hcclComm, contextTag);
        if (!ctxCreated) {
            // 首次建链：创建 ctx -> 注册存储 -> 登记共享池
            CreateEngramContext(resources, contextTag, numCpuBytes, externalHostPtr, externalBytes, sfHostPtr, sfBytes);
            return resources;
        }
        // ctx 已存在(同 group 曾创建过，含 destroy 后重建)：注册内存与 channel 均无反注册接口，
        // 由进程级共享池按 tag 复用，Destroy 不释放。
        EngramSharedBufferEntry entry;
        auto iter = gEngramSharedBuffers.find(contextTag);
        TORCH_CHECK(iter != gEngramSharedBuffers.end(), "Engram comm context of group '", contextTag,
                    "' exists but its buffer is missing from the shared pool; an earlier build on this "
                    "group may have failed midway, please use a new comm group");
        entry = iter->second;
        if (entry.hostBufPtr == nullptr) {
            // 场景1：ctx 已存在但存储尚未注册(先前仅 barrier 初始化)：补注册一次并回填共享池
            return CreateEngramContext(resources, contextTag, numCpuBytes, externalHostPtr, externalBytes, sfHostPtr,
                                       sfBytes, false);
        }
        // 场景2：重建复用——校验本次参数与首建注册的存储形态/地址/大小一致
        TORCH_CHECK(entry.external == (externalHostPtr != nullptr) &&
                        (externalHostPtr == nullptr ||
                         (externalHostPtr == entry.hostBufPtr && externalBytes == entry.registeredBytes)),
                    "Engram comm context of group '", contextTag, "' has registered external storage at address ",
                    entry.hostBufPtr, ", size ", entry.registeredBytes,
                    "; zero-copy rebuild requires the same pinned storage address and size, self-allocated buffer "
                    "rebuild is not supported, please use a new comm group");
        TORCH_CHECK(externalHostPtr != nullptr || numCpuBytes <= entry.registeredBytes,
                    "engram buffer size exceeds the existing buffer of this group, requested ", numCpuBytes,
                    ", allocated ", entry.registeredBytes,
                    "; the shared engram storage is sized by the first ElasticBuffer created on this group, "
                    "destroy and recreate with a larger size is not supported, please use a new comm group");
        TORCH_CHECK(
            entry.sfExternal == (sfHostPtr != nullptr) &&
                (sfHostPtr == nullptr || (sfHostPtr == entry.sfHostBufPtr && sfBytes == entry.sfRegisteredBytes)),
            "Engram comm context of group '", contextTag, "' has registered external SF table at address ",
            entry.sfHostBufPtr, ", size ", entry.sfRegisteredBytes,
            "; zero-copy rebuild requires the same pinned SF table address and size, rebuild without SF is "
            "not supported, please use a new comm group");
        TORCH_CHECK(entry.netLayer == netLayer_, "Engram comm context of group '", contextTag,
                    "' was built with netLayer '", entry.netLayer,
                    "'; rebuild with a different netlayer is not supported, please use a new comm group");
        resources.CopyBase(entry);
        resources.contextTensor = CreateCommContextTensor(resources.context);
        return resources;
    }

    // 训练分轨建链：感知拓扑、取协议 WIN、注册存储、建 channel、组装平铺 rail ctx。
    // ctx tag 与推理 rail（engram_embedding_rail_aligned）/老路径（engram_embedding）隔离——
    // 平铺 rail ctx 布局，kernel/tiling 按 CTX_RAIL_ELEMS 识别。
    EngramContextResources BuildRailContext(const std::string& groupName, int64_t numCpuBytes, void* externalHostPtr,
                                            int64_t externalBytes, void* sfHostPtr, int64_t sfBytes)
    {
        constexpr uint32_t kRailChannelsPerRank = 1U; // 每通道单生产者：kernel 侧 S 核统一发送，S/F/barrier 共用
        EngramContextResources resources;
        AcquireHcclHandle(groupName, resources.hcclComm);
        uint32_t myRank = 0;
        uint32_t world = 0;
        GetRankInfo(resources.hcclComm, myRank, world);
        const std::string contextTag = groupName + "engram_embedding_rail";
        CheckContextTag(contextTag);

        std::lock_guard<std::mutex> lock(gEngramSharedBufferMutex);
        if (HasEngramCommCtx(resources.hcclComm, contextTag)) {
            // 复用：channel 与内存注册均无反注册接口，按 tag 复用首建资源
            auto iter = gEngramSharedBuffers.find(contextTag);
            TORCH_CHECK(iter != gEngramSharedBuffers.end(), "rail comm context of group '", contextTag,
                        "' exists but its buffer is missing from the shared pool; an earlier build on this "
                        "group may have failed midway, please use a new comm group");
            const EngramSharedBufferEntry& entry = iter->second;
            TORCH_CHECK(entry.context.ranksPerSupernode > 0U, "comm context of group '", contextTag,
                        "' was built by the legacy path; "
                        "rail and legacy builders cannot share one comm group, please use a new comm group");
            TORCH_CHECK(entry.external == (externalHostPtr != nullptr) &&
                            (externalHostPtr == nullptr ||
                             (externalHostPtr == entry.hostBufPtr && externalBytes == entry.registeredBytes)),
                        "rail rebuild requires the same engram storage presence/address/size as the first "
                        "build (zero-copy storage must be the same pinned buffer; got ptr=",
                        externalHostPtr, ", bytes=", externalBytes, ", registered ptr=", entry.hostBufPtr,
                        ", bytes=", entry.registeredBytes, ")");
            TORCH_CHECK(externalHostPtr != nullptr || numCpuBytes <= entry.registeredBytes, "rail storage budget (",
                        numCpuBytes, ") exceeds the first-built buffer (", entry.registeredBytes,
                        "); destroy and recreate with a larger size is not supported, "
                        "please use a new comm group");
            TORCH_CHECK(
                entry.sfExternal == (sfHostPtr != nullptr) &&
                    (sfHostPtr == nullptr || (sfHostPtr == entry.sfHostBufPtr && sfBytes == entry.sfRegisteredBytes)),
                "rail rebuild requires the same SF table presence/address/size as the first build "
                "(zero-copy SF must be the same pinned buffer)");
            resources.CopyBase(entry);
            resources.contextTensor = GetRailCtxTensor(resources.hcclComm, contextTag);
            return resources;
        }

        RailTopoInfo topo;
        {
            std::string probeErr;
            TORCH_CHECK(ProbeRailTopo(resources.hcclComm, myRank, world, topo, probeErr),
                        "rail-aligned topo probe failed in Build: ", probeErr);
        }

        // 协议 WIN = HCCL 默认 buffer；集合通信共享 scratch 可能覆写窗头，ReZeroRailWindowHead 兜底
        void* localWinBuf = nullptr;
        uint64_t localWinSize = 0;
        auto winRet = HcclGetHcclBufferFunc(resources.hcclComm, &localWinBuf, &localWinSize);
        TORCH_CHECK(winRet == HCCL_SUCCESS, "rail HcclGetHcclBuffer failed, ret=", winRet);
        TORCH_CHECK(localWinBuf != nullptr && localWinSize > 0, "rail got empty HCCL default buffer");
        const int64_t winBytes = static_cast<int64_t>(localWinSize);

        // engine ctx = 平铺 rail ctx（头部标量 + node 表两张，定长 RAIL_CTX_TOTAL_BYTES）
        void* railEngineCtx = nullptr;
        CreateEngineContext(resources.hcclComm, contextTag, CommEngine::COMM_ENGINE_AIV, RAIL_CTX_TOTAL_BYTES,
                            railEngineCtx);
        HostBufferGuard guard;
        const std::string memTag = contextTag + "_buffer";
        TORCH_CHECK(externalHostPtr != nullptr || numCpuBytes > 0,
                    "rail build requires external storage or a positive num_cpu_bytes budget");
        // 自建表拷贝 buffer 按 2MB 对齐开（HcclCommMemReg 注册尺寸对齐惯例）
        const int64_t storageAllocBytes =
            (externalHostPtr != nullptr) ? externalBytes :
                                           (numCpuBytes + BUFFER_ALIGNMENT - 1) / BUFFER_ALIGNMENT * BUFFER_ALIGNMENT;
        ExternalHostRegGuard extRegGuard(resources, externalHostPtr, sfHostPtr);
        AllocateAndRegisterBuffer(resources.hcclComm, memTag, storageAllocBytes, resources, guard, externalHostPtr,
                                  externalBytes);
        if (sfHostPtr != nullptr && sfBytes > 0) {
            MapSfTableBuffer(resources, sfHostPtr, sfBytes);
        }
        resources.commBufferSize = winBytes; // WIN 大小即 commBufferSize（tiling 按 N 分满）

        // host 侧元信息：本结构不拷贝落设备（设备侧 ctx = 平铺 rail ctx），供 getter/共享池校验/窗头复位用
        const uint32_t ranksPerSupernode = topo.ranksPerSupernode;
        const uint32_t snCnt = world / ranksPerSupernode;
        resources.context.rankId = myRank % ranksPerSupernode;
        resources.context.rankSize = ranksPerSupernode;
        resources.context.channelsPerRank = kRailChannelsPerRank;
        resources.context.virtualAddrList[resources.context.rankId] = reinterpret_cast<uint64_t>(localWinBuf);
        resources.context.ranksPerSupernode = topo.ranksPerSupernode;

        // 平铺 rail ctx 组装（布局与 kernel route.h / tiling_data.h 一致）
        std::vector<uint8_t> railCtx(RAIL_CTX_TOTAL_BYTES, 0);
        auto railHead = reinterpret_cast<EngramRailCtxHead*>(railCtx.data());
        railHead->localId = myRank % ranksPerSupernode;
        railHead->supernodeId = myRank / ranksPerSupernode;
        railHead->supernodeCnt = snCnt;
        railHead->ranksPerSupernode = ranksPerSupernode;
        auto nodeWinBaseAt = [&](uint32_t p) -> uint64_t& {
            return *reinterpret_cast<uint64_t*>(railCtx.data() + RAIL_CTX_WIN_OFF + 8ULL * p);
        };
        auto nodeChannelAt = [&](uint32_t p) -> uint64_t& {
            return *reinterpret_cast<uint64_t*>(railCtx.data() + RAIL_CTX_CH_OFF + 8ULL * p);
        };
        nodeWinBaseAt(railHead->localId) = reinterpret_cast<uint64_t>(localWinBuf); // 自槽无通道

        uint32_t* layerList = nullptr;
        uint32_t layerNum = 0;
        GetNetLayers(resources.hcclComm, layerList, layerNum);
        TORCH_CHECK(layerNum > 0, "rail link setup: no net layers visible (world=", world, ")");
        std::vector<RailPeerSpec> supernodeSpecs; // 超节点同伴：期望 UB_CTP 直连
        supernodeSpecs.reserve(topo.supernodePeers.size());
        for (uint32_t r : topo.supernodePeers) {
            supernodeSpecs.push_back({r, r % ranksPerSupernode, true});
        }
        std::vector<RailPeerSpec> railSpecs; // 轨道对端：期望 UB_RTP，tableIdx 落 node 表轨道段
        railSpecs.reserve(topo.railPeers.size());
        for (uint32_t r : topo.railPeers) {
            railSpecs.push_back({r, ranksPerSupernode + r / ranksPerSupernode, false});
        }
        std::vector<ChannelHandle> supernodeHandles =
            AcquireRailChannels(resources.hcclComm, layerList, layerNum, myRank, supernodeSpecs, resources.memHandle);
        std::vector<ChannelHandle> railHandles =
            AcquireRailChannels(resources.hcclComm, layerList, layerNum, myRank, railSpecs, resources.memHandle);
        // spec 顺序 = 句柄顺序落表；filledIdx 显式记录已落槽下标（句柄值不做 0 哨兵假设）
        std::vector<uint32_t> supernodeIdx;
        supernodeIdx.reserve(supernodeSpecs.size());
        for (size_t i = 0; i < supernodeSpecs.size(); ++i) {
            nodeChannelAt(supernodeSpecs[i].tableIdx) = supernodeHandles[i];
            supernodeIdx.push_back(supernodeSpecs[i].tableIdx);
        }
        std::vector<uint32_t> railIdx;
        railIdx.reserve(railSpecs.size());
        for (size_t i = 0; i < railSpecs.size(); ++i) {
            TORCH_CHECK(railSpecs[i].tableIdx < RAIL_NODE_MAX, "rail node idx ", railSpecs[i].tableIdx,
                        " exceeds node table capacity ", RAIL_NODE_MAX);
            nodeChannelAt(railSpecs[i].tableIdx) = railHandles[i];
            railIdx.push_back(railSpecs[i].tableIdx);
        }
        PullRailPeerWins(resources.hcclComm, railCtx, supernodeIdx, winBytes);
        PullRailPeerWins(resources.hcclComm, railCtx, railIdx, winBytes);

        // 窗头清零：单调编码协议要求首调 epoch=0 直通，建链时一次性清零并同步全设备
        InitWindowHeadOnDevice(localWinBuf, winBytes, ranksPerSupernode, world);
        CopyContextToDevice(resources.hcclComm, contextTag, CommEngine::COMM_ENGINE_AIV, railCtx.data(),
                            railCtx.size());
        resources.contextTensor = GetRailCtxTensor(resources.hcclComm, contextTag);
        RegisterBufferPoolEntry(contextTag, resources, externalHostPtr, externalBytes, sfHostPtr, sfBytes,
                                storageAllocBytes);
        extRegGuard.Release();
        guard.Release();
        return resources;
    }

    // 窗头字节数 = (W+1)×32，W = ranksPerSupernode + 超节点数（kernel RAIL_WIN_SLOT_BYTES 同口径）
    static uint64_t WinHeadBytes(uint32_t ranksPerSupernode, uint32_t world)
    {
        const uint64_t snCnt = (ranksPerSupernode > 0U) ? world / ranksPerSupernode : 0U;
        return (static_cast<uint64_t>(ranksPerSupernode) + snCnt + 1ULL) * RAIL_WIN_SLOT_BYTES;
    }

private:
    int64_t hybridMode_ = 0;
    aclrtStream commStream_ = nullptr;
    EngramRailStorage railStorage_;
    std::string groupName_;
    bool withGrad_ = false;
    std::string netLayer_ = "auto";

    // 功能：将本次建链结果登记进共享池
    void RegisterBufferPoolEntry(const std::string& contextTag, const EngramContextResources& resources,
                                 void* externalHostPtr, int64_t externalBytes, void* sfHostPtr, int64_t sfBytes,
                                 int64_t numCpuBytes)
    {
        try {
            EngramSharedBufferEntry& entry = gEngramSharedBuffers[contextTag];
            entry.CopyBase(resources);
            entry.netLayer = netLayer_;
            entry.external = (externalHostPtr != nullptr);
            entry.registeredBytes =
                resources.hostBufPtr != nullptr ? ((externalHostPtr != nullptr) ? externalBytes : numCpuBytes) : 0;
            entry.sfExternal = (sfHostPtr != nullptr);
            entry.sfRegisteredBytes = resources.sfHostBufPtr != nullptr ? sfBytes : 0;
        } catch (...) {
            gEngramSharedBuffers.erase(contextTag);
            ASCEND_LOGW("failed to record engram shared buffer for tag %s, buffer will be rolled back",
                        contextTag.c_str());
            throw;
        }
    }

    // 功能：查询同 group 同 tag 的引擎 ctx 是否已创建过
    bool HasEngramCommCtx(const HcclComm& commHandle, const std::string& contextTag)
    {
        uint64_t ctxSize = 0;
        void* ctx = nullptr;
        return HcclEngineCtxGetFunc(commHandle, contextTag.c_str(), CommEngine::COMM_ENGINE_AIV, &ctx, &ctxSize) ==
               HCCL_SUCCESS;
    }

    // 功能：首次建链——创建 ctx、注册存储并登记共享池（未创建过分支）
    EngramContextResources CreateEngramContext(EngramContextResources& resources, const std::string& contextTag,
                                               int64_t numCpuBytes, void* externalHostPtr, int64_t externalBytes,
                                               void* sfHostPtr, int64_t sfBytes, bool createEngineCtx = true)
    {
        HostBufferGuard guard;
        HostBufferGuard sfGuard;
        try {
            CreateContext(resources, contextTag, numCpuBytes, guard, externalHostPtr, externalBytes, sfHostPtr, sfBytes,
                          sfGuard, createEngineCtx);
        } catch (...) {
            RollbackExternalHostRegistration(resources, externalHostPtr, sfHostPtr);
            throw;
        }
        resources.contextTensor = CreateCommContextTensor(resources.context);
        // 首次创建: 注册内存入共享池(与 tag 绑定的不可销毁资源同生命周期)，Destroy 不释放。
        // 入池成功前不解除 guard 所有权，入池抛异常时由 guard 析构兜底释放自建 buffer，防资源失联。
        RegisterBufferPoolEntry(contextTag, resources, externalHostPtr, externalBytes, sfHostPtr, sfBytes, numCpuBytes);
        guard.Release();
        sfGuard.Release();
        return resources;
    }

    static int32_t ParseNetLayerId(const std::string& netLayer)
    {
        TORCH_CHECK(!netLayer.empty() && netLayer.find_first_not_of("0123456789") == std::string::npos,
                    "netlayer must be 'auto' or a non-negative integer string (e.g. '1', '2'), got '", netLayer, "'");
        TORCH_CHECK(netLayer.size() <= 9U, "netlayer value is invalid: '", netLayer, "'");
        return std::stoi(netLayer);
    }

    void ProbeRailTopology(const HcclComm& commHandle, uint32_t srcRankId, uint32_t rankSize, const uint32_t* layerList,
                           uint32_t layerNum)
    {
        uint32_t ranksPerMachine = 0;
        uint32_t ranksPerSupernode = 0;
        const char* envPerMachine = std::getenv("RANK_SIZE_PER_MACHINE");
        const char* envPerSupernode = std::getenv("RANK_SIZE_PER_SUPERNODE");
        TORCH_CHECK((envPerMachine == nullptr) == (envPerSupernode == nullptr),
                    "RANK_SIZE_PER_MACHINE and RANK_SIZE_PER_SUPERNODE must be set together");
        TORCH_CHECK(HcclRankGraphGetInstSizeListByLayerFunc != nullptr,
                    "rail_aligned inference topo probe requires HcclRankGraphGetInstSizeListByLayer, "
                    "which is missing from the loaded libhccl; upgrade CANN/HCCL or set "
                    "RANK_SIZE_PER_MACHINE + RANK_SIZE_PER_SUPERNODE explicitly");
        if (envPerMachine != nullptr && envPerSupernode != nullptr) {
            std::string envErr;
            TORCH_CHECK(ParseEnvUint32("RANK_SIZE_PER_MACHINE", envPerMachine, ranksPerMachine, envErr) &&
                            ParseEnvUint32("RANK_SIZE_PER_SUPERNODE", envPerSupernode, ranksPerSupernode, envErr),
                        "RANK_SIZE_PER_MACHINE/RANK_SIZE_PER_SUPERNODE invalid: ", envErr);
            ASCEND_LOGI("rail_aligned topo: rank=%u env set ranksPerMachine=%u ranksPerSupernode=%u", srcRankId,
                        ranksPerMachine, ranksPerSupernode);
        } else {
            for (uint32_t li = 0; li < layerNum; ++li) {
                uint32_t sizeByLayer = 0;
                HcclResult szRet = HcclRankGraphGetRankSizeByLayerFunc(commHandle, layerList[li], &sizeByLayer);
                TORCH_CHECK(szRet == HCCL_SUCCESS, "HcclRankGraphGetRankSizeByLayer(layer=", layerList[li],
                            ") failed, ret=", static_cast<int>(szRet));
                uint32_t* instSizeList = nullptr;
                uint32_t instListSize = 0;
                HcclResult listRet =
                    HcclRankGraphGetInstSizeListByLayerFunc(commHandle, layerList[li], &instSizeList, &instListSize);
                TORCH_CHECK(listRet == HCCL_SUCCESS, "HcclRankGraphGetInstSizeListByLayer(layer=", layerList[li],
                            ") failed, ret=", static_cast<int>(listRet));
                uint32_t instCnt = (instSizeList != nullptr) ? instListSize : 0U;
                std::string instSizeStr;
                for (uint32_t i = 0; i < instCnt; ++i) {
                    instSizeStr += (i == 0 ? "" : ",") + std::to_string(instSizeList[i]);
                }
                ASCEND_LOGI("rail_aligned topo: rank=%u layer=%u size=%u instListSize=%u instSizeList=[%s]", srcRankId,
                            layerList[li], sizeByLayer, instListSize, instSizeStr.c_str());
                if (li == 1U) {
                    TORCH_CHECK(instCnt > 0, "rail_aligned: layer1 instSizeList is empty");
                    ranksPerSupernode = instSizeList[0];
                }
                if (li == 2U) {
                    TORCH_CHECK(instCnt > 0, "rail_aligned: layer2 instSizeList is empty");
                    ranksPerMachine = instListSize;
                }
            }
        }
        railStorage_.ranksPerMachine = ranksPerMachine;
        railStorage_.ranksPerSupernode = ranksPerSupernode;
        TORCH_CHECK(ranksPerMachine != 0 && ranksPerSupernode != 0 && ranksPerSupernode <= rankSize &&
                        rankSize % ranksPerMachine == 0 && ranksPerSupernode % ranksPerMachine == 0 &&
                        (rankSize / ranksPerMachine) % (ranksPerSupernode / ranksPerMachine) == 0,
                    "rail_aligned topology invalid: rankSize=", rankSize, " ranksPerMachine=", ranksPerMachine,
                    " ranksPerSupernode=", ranksPerSupernode);

        railStorage_.machineCount = rankSize / ranksPerMachine;
        railStorage_.machinesPerSupernode = ranksPerSupernode / ranksPerMachine;
        railStorage_.supernodeStartRank = (srcRankId / ranksPerSupernode) * ranksPerSupernode;
        railStorage_.superStartMachine = railStorage_.supernodeStartRank / ranksPerMachine;
        ASCEND_LOGI("rail_aligned topo: rank=%u ranksPerMachine=%u ranksPerSupernode=%u machineCount=%u "
                    "machinesPerSupernode=%u",
                    srcRankId, ranksPerMachine, ranksPerSupernode, railStorage_.machineCount,
                    railStorage_.machinesPerSupernode);
    }

    // netlayer 选层："auto" 自首个网络分层起逐层探测；数字串（如 "1"）指定分层编号，仅在该分层内
    // 建链（该层所有对端都必须可达），指定分层不存在或该层内建链失败直接报错
    void ResolveRankLinks(const HcclComm& commHandle, uint32_t srcRankId, uint32_t rankSize, const uint32_t* layerList,
                          uint32_t layerNum)
    {
        uint32_t preferIdx = 0;
        if (netLayer_ != "auto") {
            int32_t netLayerId = ParseNetLayerId(netLayer_);
            bool layerExists = false;
            for (uint32_t li = 0; li < layerNum; ++li) {
                if (static_cast<int32_t>(layerList[li]) == netLayerId) {
                    preferIdx = li;
                    layerExists = true;
                    break;
                }
            }
            TORCH_CHECK(layerExists, "netLayer ", netLayer_, " not exist in HCCL net layers");
        }

        auto recordLink = [&](uint32_t peer, uint32_t startLayer, uint32_t endLayer) {
            bool found = false;
            for (uint32_t li = startLayer; li < endLayer && !found; ++li) {
                CommLink* links = nullptr;
                uint32_t linkNum = 0;
                auto hcclRet = HcclRankGraphGetLinksFunc(commHandle, layerList[li], srcRankId, peer, &links, &linkNum);
                for (uint32_t i = 0; hcclRet == HCCL_SUCCESS && links != nullptr && i < linkNum; ++i) {
                    CommProtocol protocol = links[i].linkAttr.linkProtocol;
                    if (protocol == CommProtocol::COMM_PROTOCOL_UB_CTP ||
                        protocol == CommProtocol::COMM_PROTOCOL_UB_RTP) {
                        rankLinkMap_[peer] = {protocol, layerList[li]};
                        found = true;
                        break;
                    }
                }
            }
            return found;
        };
        if (hybridMode_ == NETWORK_HYBRID) {
            ProbeRailTopology(commHandle, srcRankId, rankSize, layerList, layerNum);
            const uint32_t ranksPerMachine = railStorage_.ranksPerMachine;
            const uint32_t machineCount = railStorage_.machineCount;
            const uint32_t machinesPerSupernode = railStorage_.machinesPerSupernode;
            const uint32_t myMachine = srcRankId / ranksPerMachine;
            const uint32_t cardIdx = srcRankId % ranksPerMachine;
            for (uint32_t m = 0; m < machineCount; ++m) {
                if (m == myMachine) {
                    continue;
                }
                if ((m / machinesPerSupernode) == myMachine / machinesPerSupernode) {
                    for (uint32_t slot = 0; slot < ranksPerMachine; ++slot) {
                        const uint32_t peer = m * ranksPerMachine + slot;
                        TORCH_CHECK(recordLink(peer, preferIdx, layerNum),
                                    "rail_aligned: no UBC/UBRTP link between rank ", srcRankId, " and peer ", peer);
                    }
                } else {
                    const uint32_t peer = m * ranksPerMachine + cardIdx;
                    TORCH_CHECK(recordLink(peer, preferIdx, layerNum), "rail_aligned: no UBC/UBRTP link between rank ",
                                srcRankId, " and peer ", peer);
                }
            }
            return;
        }
        uint32_t endLayer = netLayer_ != "auto" ? (preferIdx + 1) : layerNum;
        for (uint32_t peer = 0; peer < rankSize; ++peer) {
            if (peer == srcRankId) {
                continue;
            }
            TORCH_CHECK(recordLink(peer, preferIdx, endLayer), "No UBC/UBG link found for peer rank ", peer,
                        " in netLayer ", netLayer_);
        }
        ASCEND_LOGI("netLayer %s applied", netLayer_.c_str());
    }

    static void ValidateRankSize(uint32_t rankSize)
    {
        TORCH_CHECK(rankSize >= HCCL_MIN_RANK_SIZE, "rankSize must be at least HCCL_MIN_RANK_SIZE, got ", rankSize,
                    ", min ", HCCL_MIN_RANK_SIZE);
        TORCH_CHECK(rankSize <= HCCL_MAX_RANK_SIZE, "rankSize exceeds HCCL_MAX_RANK_SIZE, got ", rankSize, ", max ",
                    HCCL_MAX_RANK_SIZE);
    }

    static void AllocateAndRegisterBuffer(const HcclComm& commHandle, const std::string& memBufferTag,
                                          int64_t numCpuBytes, EngramContextResources& resources,
                                          HostBufferGuard& guard, void* externalHostPtr = nullptr,
                                          int64_t externalBytes = 0)
    {
        if (externalHostPtr == nullptr) {
            aclError ar = aclrtMallocHost(&guard.hostPtr, static_cast<uint64_t>(numCpuBytes));
            TORCH_CHECK(ar == ACL_SUCCESS, "aclrtMallocHost(", numCpuBytes, " B) failed, ret=", ar);

            ar = aclrtHostRegisterV2(guard.hostPtr, static_cast<uint64_t>(numCpuBytes), ACL_HOST_REG_MAPPED);
            TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostRegisterV2(", numCpuBytes, " B) failed, ret=", ar);
            guard.registered = true;
        } else {
            // Explicit zero-copy (or training direct mapping): registration failure is a hard error.
            aclError ar =
                aclrtHostRegisterV2(externalHostPtr, static_cast<uint64_t>(externalBytes), ACL_HOST_REG_MAPPED);
            TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostRegisterV2(", externalBytes, " B) failed, ret=", ar);
            resources.externalRegistered = (ar == ACL_SUCCESS);
        }

        void* hostPtr = (externalHostPtr != nullptr) ? externalHostPtr : guard.hostPtr;
        void* devPtr = nullptr;
        aclError ar = aclrtHostGetDevicePointer(hostPtr, &devPtr, 0);
        TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostGetDevicePointer failed, ret=", ar,
                    ", host memory cannot be mapped to device");

        CommMem mem;
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = devPtr;
        mem.size = static_cast<uint64_t>((externalHostPtr != nullptr) ? externalBytes : numCpuBytes);
        auto hcclRet = HcclCommMemRegFunc(commHandle, memBufferTag.c_str(), &mem, &resources.memHandle);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclCommMemReg(tag='", memBufferTag, "', size=", mem.size,
                    ") failed, ret=", hcclRet);

        resources.hostBufPtr = hostPtr;
        resources.deviceBufPtr = devPtr;
    }

    // SF 表仅本核经映射地址直读 (参数路由): 只做 host->device 映射, 不自建内存、不入 HCCL 通信域。
    static void MapSfTableBuffer(EngramContextResources& resources, void* sfHostPtr, int64_t sfBytes)
    {
        aclError ar = aclrtHostRegisterV2(sfHostPtr, static_cast<uint64_t>(sfBytes), ACL_HOST_REG_MAPPED);
        if (ar != ACL_SUCCESS) {
            ASCEND_LOGW("aclrtHostRegisterV2(%lld B) failed, ret=%d, fallback to existing registration", sfBytes,
                        static_cast<int>(ar));
        }
        resources.sfExternalRegistered = (ar == ACL_SUCCESS);

        void* devPtr = nullptr;
        ar = aclrtHostGetDevicePointer(sfHostPtr, &devPtr, 0);
        TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostGetDevicePointer failed, ret=", ar,
                    ", storage host memory cannot be mapped to device");

        resources.sfHostBufPtr = sfHostPtr;
        resources.sfDeviceBufPtr = devPtr;
    }

    // 逐节点查其 HCCL 默认 buffer 基址，按节点索引落平铺 node 表（filledIdx = 已落 channel 的节点索引集）。
    // remoteBase == localBase 为预期形态（对称 VA），非"返回了本端 buffer"：各卡默认 buffer 映射在同一
    // VA，通道远程写/信用读目的地址按对端设备 VA 空间解析。
    static void PullRailPeerWins(const HcclComm& comm, std::vector<uint8_t>& railCtx,
                                 const std::vector<uint32_t>& filledIdx, int64_t localWinBytes)
    {
        for (uint32_t idx : filledIdx) {
            void* remoteBuffer = nullptr;
            uint64_t remoteBufSize = 0;
            const uint64_t chHandle = *reinterpret_cast<const uint64_t*>(railCtx.data() + RAIL_CTX_CH_OFF + 8ULL * idx);
            auto hcclRet = HcclChannelGetHcclBufferFunc(comm, chHandle, &remoteBuffer, &remoteBufSize);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "rail HcclChannelGetHcclBuffer(idx=", idx, ") failed, ret=", hcclRet);
            TORCH_CHECK(remoteBuffer != nullptr, "rail remote HCCL default buffer is null for node idx ", idx);
            if (localWinBytes > 0) {
                TORCH_CHECK(static_cast<int64_t>(remoteBufSize) >= localWinBytes, "rail remote win size(",
                            remoteBufSize, ") < local(", localWinBytes, ") for node idx ", idx,
                            " — cross-rank default buffer size mismatch");
            }
            *reinterpret_cast<uint64_t*>(railCtx.data() + RAIL_CTX_WIN_OFF + 8ULL * idx) =
                reinterpret_cast<uint64_t>(remoteBuffer);
        }
    }

    // 取平铺 rail ctx tensor：from_blob 非持有视图，内存归 HCCL 所有（生命周期与 comm 一致）
    static at::Tensor GetRailCtxTensor(const HcclComm& commHandle, const std::string& contextTag)
    {
        void* ctx = nullptr;
        uint64_t ctxSize = 0;
        auto hcclRet =
            HcclEngineCtxGetFunc(commHandle, contextTag.c_str(), CommEngine::COMM_ENGINE_AIV, &ctx, &ctxSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS && ctx != nullptr && ctxSize > 0, "get flat rail ctx of tag '", contextTag,
                    "' failed, ret=", hcclRet);
        TORCH_CHECK(ctxSize >= RAIL_CTX_TOTAL_BYTES, "flat rail ctx size ", ctxSize, " < expected ",
                    RAIL_CTX_TOTAL_BYTES);
        auto options = at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1);
        return at_npu::native::from_blob(ctx, {RAIL_CTX_ELEMS}, options);
    }

    // 建链时把协议 WIN 窗头一次性 device 清零并同步全设备（默认 buffer 初始内容非零，
    // 单调编码协议要求首调 epoch=0 直通）
    static void InitWindowHeadOnDevice(void* winBase, int64_t winBytes, uint32_t ranksPerSupernode, uint32_t world)
    {
        uint64_t headBytes = WinHeadBytes(ranksPerSupernode, world);
        TORCH_CHECK(winBase != nullptr && winBytes >= static_cast<int64_t>(headBytes), "rail win too small: ", winBytes,
                    " < head ", headBytes);
        aclError ar = aclrtMemset(winBase, static_cast<size_t>(winBytes), 0, static_cast<size_t>(headBytes));
        TORCH_CHECK(ar == ACL_SUCCESS, "aclrtMemset(win head ", headBytes, "B) failed, ret=", ar);
        ar = aclrtSynchronizeDevice();
        TORCH_CHECK(ar == ACL_SUCCESS, "aclrtSynchronizeDevice after win head memset failed, ret=", ar);
    }

    void BuildChannelDescs(const HcclComm& commHandle, uint32_t srcRankId, uint32_t rankDim, uint32_t channelsPerRank,
                           HcclMemHandle& memHandle, std::vector<HcclChannelDesc>& channelDesc)
    {
        channelDesc.clear();
        uint32_t totalChannels = (rankDim - 1) * channelsPerRank;
        channelDesc.reserve(totalChannels);

        for (uint32_t peer = 0; peer < rankDim; ++peer) {
            if (peer == srcRankId) {
                continue;
            }
            auto linkIter = rankLinkMap_.find(peer);
            TORCH_CHECK(linkIter != rankLinkMap_.end(), "No routing info for peer rank ", peer,
                        " after protocol negotiation");
            RankLinkInfo linkInfo = linkIter->second;
            CommLink* links = nullptr;
            GetHcclCommLink(commHandle, linkInfo.layer, srcRankId, peer, linkInfo.protocol, links);
            for (uint32_t ch = 0; ch < channelsPerRank; ++ch) {
                HcclChannelDesc desc;
                HcclResult initRet = HcclChannelDescInit(&desc, 1);
                TORCH_CHECK(initRet == HCCL_SUCCESS, "HcclChannelDescInit failed, ret=", initRet);
                desc.remoteRank = peer;
                desc.channelProtocol = linkInfo.protocol;
                desc.localEndpoint = links->srcEndpointDesc;
                desc.remoteEndpoint = links->dstEndpointDesc;
                desc.notifyNum = MOE_CHANNEL_NOTIFY_NUM;
                desc.memHandles = &memHandle;
                desc.memHandleNum = 1;
                channelDesc.push_back(desc);
            }
        }
    }

    void GetHcclCommChannel(const HcclComm& commHandle, uint32_t rankDim, uint32_t srcRankId, uint32_t channelsPerRank,
                            HcclMemHandle& memHandle, ChannelHandle* channels)
    {
        std::vector<HcclChannelDesc> descs;
        ChannelHandle channelBuf[HCCL_MAX_RANK_SIZE] = {};
        BuildChannelDescs(commHandle, srcRankId, rankDim, channelsPerRank, memHandle, descs);
        AcquireChannels(commHandle, CommEngine::COMM_ENGINE_AIV, descs, channelBuf);
        uint32_t descIdx = 0;
        for (uint32_t peer = 0; peer < rankDim; ++peer) {
            if (peer == srcRankId)
                continue;
            for (uint32_t ch = 0; ch < channelsPerRank; ++ch) {
                channels[peer * channelsPerRank + ch] = channelBuf[descIdx];
                descIdx++;
            }
        }
    }

    void GetHcclCommResource(const HcclComm& commHandle, EngramContextResources& resources,
                             const std::string& targetTag)
    {
        uint32_t rankId = resources.context.rankId;
        bool hasUbGPeer = false;
        for (auto& entry : rankLinkMap_) {
            if (entry.second.protocol == CommProtocol::COMM_PROTOCOL_UB_RTP) {
                hasUbGPeer = true;
                break;
            }
        }
        if (hasUbGPeer) {
            resources.context.channelsPerRank = 1U;
        } else {
            resources.context.channelsPerRank =
                static_cast<uint32_t>(CeilDiv(GetAivCoreNum(), resources.context.rankSize));
            if (resources.context.channelsPerRank == 0U) {
                resources.context.channelsPerRank = 1U;
            }
        }
        uint32_t channelsPerRank = resources.context.channelsPerRank;

        ChannelHandle handlesByRank[HCCL_MAX_RANK_SIZE] = {};
        GetHcclCommChannel(commHandle, resources.context.rankSize, rankId, channelsPerRank, resources.memHandle,
                           handlesByRank);

        for (uint32_t peer = 0; peer < resources.context.rankSize; ++peer) {
            if (peer == rankId)
                continue;
            for (uint32_t ch = 0; ch < channelsPerRank; ++ch) {
                resources.context.hcommHandle[peer * channelsPerRank + ch] = handlesByRank[peer * channelsPerRank + ch];
            }
        }

        resources.context.virtualAddrList[rankId] = reinterpret_cast<uint64_t>(resources.deviceBufPtr);

        for (uint32_t i = 0; i < resources.context.rankSize; ++i) {
            if (i == rankId)
                continue;
            uint32_t memNum = 0;
            CommMem* remoteMems = nullptr;
            char** memTags = nullptr;
            auto hcclRet = HcclChannelGetRemoteMemsFunc(commHandle, resources.context.hcommHandle[i * channelsPerRank],
                                                        &memNum, &remoteMems, &memTags);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclChannelGetRemoteMems(peer=", i, ") failed, ret=", hcclRet);
            bool hasTargetMem = false;
            for (uint32_t j = 0; j < memNum; j++) {
                if (memTags == nullptr || remoteMems == nullptr) {
                    break;
                }
                if (memTags[j] != nullptr && targetTag == memTags[j]) {
                    uint64_t targetMemAddr = reinterpret_cast<uint64_t>(remoteMems[j].addr);
                    resources.context.virtualAddrList[i] = targetMemAddr;
                    ASCEND_LOGI("Get Target Mem(%s) Success, Mem id is %d, Addr is %lu", targetTag.c_str(), j,
                                targetMemAddr);
                    hasTargetMem = true;
                }
            }
            TORCH_CHECK(hasTargetMem, "Target Mem : ", targetTag, " is not found.");
        }
    }

    void QueryHcclBufferResource(const HcclComm& commHandle, EngramContextResources& resources)
    {
        uint32_t rankId = resources.context.rankId;
        uint32_t rankSize = resources.context.rankSize;
        TORCH_CHECK(rankSize > 0, "rankSize must be positive, got ", rankSize);
        bool hasUbGPeer = false;
        for (auto& entry : rankLinkMap_) {
            if (entry.second.protocol == CommProtocol::COMM_PROTOCOL_UB_RTP) {
                hasUbGPeer = true;
                break;
            }
        }

        uint32_t channelsPerPeer = 1U;
        if (!hasUbGPeer) {
            uint32_t numSendCores = GetAivCoreNum() / 2U;
            if (numSendCores == 0U) {
                numSendCores = 1U;
            }
            uint32_t groupSize = numSendCores / rankSize;
            if (groupSize == 0U) {
                groupSize = 1U;
            }
            channelsPerPeer = groupSize;
        }
        resources.context.channelsPerRank = channelsPerPeer;
        TORCH_CHECK(channelsPerPeer <= HCCL_MAX_RANK_SIZE / rankSize,
                    "HCCL channel handles exceed capacity, rank size ", rankSize, ", channels per rank ",
                    channelsPerPeer);

        std::vector<ChannelHandle> handlesByRank(rankSize * channelsPerPeer);
        GetHcclCommChannel(commHandle, rankSize, rankId, channelsPerPeer, resources.memHandle, handlesByRank.data());
        for (uint32_t peer = 0; peer < rankSize; ++peer) {
            if (peer == rankId)
                continue;
            for (uint32_t ch = 0; ch < channelsPerPeer; ++ch) {
                resources.context.hcommHandle[peer * channelsPerPeer + ch] = handlesByRank[peer * channelsPerPeer + ch];
            }
        }

        void* localBuffer = nullptr;
        uint64_t localBufferSize = 0;
        auto hcclRet = HcclGetHcclBufferFunc(commHandle, &localBuffer, &localBufferSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclGetHcclBuffer failed, ret=", hcclRet);
        TORCH_CHECK(localBuffer != nullptr && localBufferSize > 0,
                    "HCCL default buffer is null or empty, size=", localBufferSize);

        resources.commBufferSize = static_cast<int64_t>(localBufferSize);
        resources.context.virtualAddrList[rankId] = reinterpret_cast<uint64_t>(localBuffer);

        for (uint32_t i = 0; i < resources.context.rankSize; ++i) {
            if (i == rankId) {
                continue;
            }
            void* remoteBuffer = nullptr;
            uint64_t remoteBufSize = 0;
            hcclRet = HcclChannelGetHcclBufferFunc(commHandle, resources.context.hcommHandle[i * channelsPerPeer],
                                                   &remoteBuffer, &remoteBufSize);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclChannelGetHcclBuffer(peer=", i, ") failed, ret=", hcclRet);
            TORCH_CHECK(remoteBuffer != nullptr, "HCCL remote buffer is null for peer=", i);
            resources.context.virtualAddrList[i] = reinterpret_cast<uint64_t>(remoteBuffer);
        }
    }

    void CreateContext(EngramContextResources& resources, const std::string& contextTag, int64_t numCpuBytes,
                       HostBufferGuard& guard, void* externalHostPtr, int64_t externalBytes, void* sfHostPtr,
                       int64_t sfBytes, HostBufferGuard& sfGuard, bool createEngineCtx = true)
    {
        if (createEngineCtx) {
            uint64_t contextSize = sizeof(EngramCommContext);
            void* ctx = nullptr;
            CreateEngineContext(resources.hcclComm, contextTag, CommEngine::COMM_ENGINE_AIV, contextSize, ctx);
        }

        GetRankInfo(resources.hcclComm, resources.context.rankId, resources.context.rankSize);
        ValidateRankSize(resources.context.rankSize);

        SetupEngramBuffer(resources, contextTag, numCpuBytes, guard, externalHostPtr, externalBytes, sfHostPtr, sfBytes,
                          sfGuard);
    }

    // 注册存储并构建 channel/远端地址信息，最后把完整 context 拷贝到设备端引擎 ctx。
    // 复用路径(ctx 已存在但共享池尚无注册内存)重建时也会走到这里。
    void SetupEngramBuffer(EngramContextResources& resources, const std::string& contextTag, int64_t numCpuBytes,
                           HostBufferGuard& guard, void* externalHostPtr, int64_t externalBytes, void* sfHostPtr,
                           int64_t sfBytes, HostBufferGuard& sfGuard)
    {
        if (numCpuBytes == 0 && externalHostPtr == nullptr) {
            return;
        }

        std::string memBufferTag = contextTag + "_buffer";

        uint32_t* netLayerList = nullptr;
        uint32_t netLayerNum = 0;
        GetNetLayers(resources.hcclComm, netLayerList, netLayerNum);
        DumpRankPairLinks(resources.hcclComm, resources.context.rankId, resources.context.rankSize, netLayerList,
                          netLayerNum);
        ResolveRankLinks(resources.hcclComm, resources.context.rankId, resources.context.rankSize, netLayerList,
                         netLayerNum);

        if (hybridMode_ == NETWORK_HYBRID) {
            TORCH_CHECK(externalHostPtr == nullptr,
                        "rail_aligned does not support zero-copy external storage, use self-allocated buffer");
            AllocRailMemory(memBufferTag, numCpuBytes, resources);
        } else {
            AllocateAndRegisterBuffer(resources.hcclComm, memBufferTag, numCpuBytes, resources, guard, externalHostPtr,
                                      externalBytes);
        }

        if (sfHostPtr != nullptr && sfBytes > 0) {
            MapSfTableBuffer(resources, sfHostPtr, sfBytes);
        }

        if (hybridMode_ == NETWORK_HYBRID) {
            GetRailCommResource(resources.hcclComm, resources, memBufferTag);
        } else if (withGrad_) {
            QueryHcclBufferResource(resources.hcclComm, resources);
        } else {
            GetHcclCommResource(resources.hcclComm, resources, memBufferTag);
        }

        CopyContextToDevice(resources.hcclComm, contextTag, CommEngine::COMM_ENGINE_AIV, &resources.context,
                            sizeof(EngramCommContext));
    }
};

class MoeContextBuilder : public HcclContextBuilderBase {
public:
    MoeContextBuilder() = default;
    ~MoeContextBuilder()
    {
        // RAII: Build 完整成功(所有权已转移)之外的任意异常路径，回滚已申请的物理内存，防泄露
        if (!ownershipReleased_) {
            ReleaseLocalDeviceBuffer();
        }
    }
    MoeContextBuilder(const MoeContextBuilder&) = delete;
    MoeContextBuilder& operator=(const MoeContextBuilder&) = delete;

    MoeContextResources Build(const std::string& groupName, int64_t customCclBufferSize)
    {
        ASCEND_LOGI("start build");
        InitHcclEngineCtxFunctions();

        HcclComm hcclComm = nullptr;
        AcquireHcclHandle(groupName, hcclComm);

        CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UB_CTP;
        GetCommProtocol(hcclComm, protocol);
        rankNumPerServer_ = rankNumPerUbDomain_;
        CheckIsMixLayered(hcclComm);

        // 所需实际总字节数需要大于0，且2MB对齐
        TORCH_CHECK(customCclBufferSize > 0, "ccl buffer size must be greater than 0, got ", customCclBufferSize);
        TORCH_CHECK(customCclBufferSize % (2 * MB_SIZE) == 0, "ccl buffer size must be divisible by 2MB, got ",
                    customCclBufferSize);
        cclBufferSize_ = customCclBufferSize;

        MoeCommContext context;
        TORCH_CHECK(rankNumPerServer_ > 0, "rank_num_per_server must be positive after resolving MoE topology");
        context.rankSizePerServer = rankNumPerServer_;
        const std::string opName = "moe_dispatch_combine_multi_channel";

        void* ctx = nullptr;
        BuildContext(hcclComm, groupName, opName, protocol, context, ctx);
        TORCH_CHECK(ctx != nullptr, "Create MoE context tensor failed: ctx is nullptr");
        int64_t numElements = (sizeof(MoeCommContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
        auto options = at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1);
        // HCCL owns ctx; the tensor only provides a non-owning view of the cached device context.
        MoeContextResources resources;
        resources.contextTensor = at_npu::native::from_blob(ctx, {numElements}, options);
        resources.cclBufferSize = cclBufferSize_;
        resources.rankSizePerServer = rankNumPerServer_;
        resources.deviceBufPtr = deviceBufPtr_;
        resources.physicalMemHandle = physicalMemHandle_;
        resources.memHandle = memHandle_;
        resources.contextTag = groupName + opName;
        if (createdNew_) {
            // 首次创建: 物理内存入共享池(与 tag 绑定的不可销毁资源同生命周期)，Destroy 不释放，同 group 重建时复用
            std::lock_guard<std::mutex> lock(gMoeSharedBufferMutex);
            MoeSharedBufferEntry& entry = gMoeSharedBuffers[resources.contextTag];
            entry.deviceBufPtr = deviceBufPtr_;
            entry.physicalMemHandle = physicalMemHandle_;
            entry.memHandle = memHandle_;
            entry.cclBufferSize = cclBufferSize_;
        }
        ownershipReleased_ = true; // 所有权已转移(共享池/既有持有者)，builder 析构不释放
        return resources;
    }

private:
    // 回滚本 builder 持有的物理内存(异常路径)，顺序与申请严格互逆
    void ReleaseLocalDeviceBuffer()
    {
        if (deviceBufPtr_ != nullptr) {
            if (physicalMemHandle_ != nullptr) {
                (void)aclrtUnmapMem(deviceBufPtr_);
                (void)aclrtFreePhysical(physicalMemHandle_);
                physicalMemHandle_ = nullptr;
            }
            (void)aclrtReleaseMemAddress(deviceBufPtr_);
            deviceBufPtr_ = nullptr;
        }
        memHandle_ = nullptr;
    }

    static bool CoversAllRanks(const std::vector<uint32_t>& ranks, uint32_t rankSize)
    {
        for (uint32_t peer = 0; peer < rankSize; ++peer) {
            if (std::find(ranks.begin(), ranks.end(), peer) == ranks.end()) {
                return false;
            }
        }
        return true;
    }

    static bool MatchesInstanceEndpoint(const EndpointDesc& linkEndpoint, const EndpointDesc& instanceEndpoint)
    {
        if (linkEndpoint.protocol != instanceEndpoint.protocol ||
            linkEndpoint.loc.locType != instanceEndpoint.loc.locType ||
            linkEndpoint.commAddr.type != instanceEndpoint.commAddr.type) {
            return false;
        }
        // Instance queries do not populate device location IDs; compare the active address field only.
        const auto& lhs = linkEndpoint.commAddr;
        const auto& rhs = instanceEndpoint.commAddr;
        switch (lhs.type) {
            case COMM_ADDR_TYPE_EID:
                return std::memcmp(lhs.eid, rhs.eid, sizeof(lhs.eid)) == 0;
            case COMM_ADDR_TYPE_IP_V4:
                return std::memcmp(&lhs.addr, &rhs.addr, sizeof(lhs.addr)) == 0;
            case COMM_ADDR_TYPE_IP_V6:
                return std::memcmp(&lhs.addr6, &rhs.addr6, sizeof(lhs.addr6)) == 0;
            case COMM_ADDR_TYPE_ID:
                return lhs.id == rhs.id;
            default:
                return false;
        }
    }

    bool CollectMixLinks(const HcclComm& commHandle, uint32_t layerId, uint32_t srcRankId, uint32_t rankSize,
                         const std::vector<EndpointDesc>* instanceEndpoints, std::vector<CommLink>& selectedLinks)
    {
        selectedLinks.resize(rankSize);
        for (uint32_t peer = 0; peer < rankSize; ++peer) {
            if (peer == srcRankId) {
                continue;
            }
            CommLink* links = nullptr;
            uint32_t linkNum = 0;
            auto ret = HcclRankGraphGetLinksFunc(commHandle, layerId, srcRankId, peer, &links, &linkNum);
            TORCH_CHECK(ret == HCCL_SUCCESS, "Get mixed-layer links failed, layer: ", layerId, ", srcRank: ", srcRankId,
                        ", peer: ", peer, ", ret: ", ret);
            TORCH_CHECK(linkNum == 0 || links != nullptr, "Null mixed-layer links, layer: ", layerId);
            bool found = false;
            for (uint32_t i = 0; i < linkNum; ++i) {
                if (links[i].linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UB_CTP) {
                    continue;
                }
                if (instanceEndpoints != nullptr &&
                    std::none_of(instanceEndpoints->begin(), instanceEndpoints->end(), [&](const EndpointDesc& ep) {
                        return MatchesInstanceEndpoint(links[i].srcEndpointDesc, ep);
                    })) {
                    continue;
                }
                // HCCL owns the returned list. Keep the exact validated link for channel construction.
                selectedLinks[peer] = links[i];
                found = true;
                break;
            }
            if (!found) {
                ASCEND_LOGI("[MixLayer] rank %u layer %u: no matching UB_CTP link to peer %u", srcRankId, layerId,
                            peer);
                return false;
            }
        }
        return true;
    }

    std::vector<uint32_t> GetTopologyInstanceIds(const HcclComm& commHandle, uint32_t layerId)
    {
        uint32_t* instanceList = nullptr;
        uint32_t instanceCount = 0;
        auto ret = HcclRankGraphGetTopoInstsByLayerFunc(commHandle, layerId, &instanceList, &instanceCount);
        TORCH_CHECK(ret == HCCL_SUCCESS, "Get topology instances failed, layer: ", layerId, ", ret: ", ret);
        TORCH_CHECK(instanceCount == 0 || instanceList != nullptr, "Null topology instance list, layer: ", layerId);
        if (instanceCount == 0) {
            return {};
        }
        return {instanceList, instanceList + instanceCount};
    }

    bool InstanceCoversAllRanks(const HcclComm& commHandle, uint32_t layerId, uint32_t instanceId, uint32_t rankSize)
    {
        uint32_t* rankList = nullptr;
        uint32_t rankCount = 0;
        auto ret = HcclRankGraphGetRanksByTopoInstFunc(commHandle, layerId, instanceId, &rankList, &rankCount);
        TORCH_CHECK(ret == HCCL_SUCCESS, "Get instance ranks failed, layer: ", layerId, ", instance: ", instanceId,
                    ", ret: ", ret);
        TORCH_CHECK(rankCount == 0 || rankList != nullptr, "Null instance ranks, layer: ", layerId);
        return rankCount != 0 && CoversAllRanks(std::vector<uint32_t>(rankList, rankList + rankCount), rankSize);
    }

    std::vector<EndpointDesc> GetInstanceEndpoints(const HcclComm& commHandle, uint32_t layerId, uint32_t instanceId)
    {
        uint32_t endpointCapacity = 0;
        auto ret = HcclRankGraphGetEndpointNumFunc(commHandle, layerId, instanceId, &endpointCapacity);
        TORCH_CHECK(ret == HCCL_SUCCESS, "Get instance endpoint count failed, layer: ", layerId,
                    ", instance: ", instanceId, ", ret: ", ret);
        if (endpointCapacity == 0) {
            return {};
        }
        std::vector<EndpointDesc> endpoints(endpointCapacity);
        uint32_t actualEndpointCount = endpointCapacity;
        ret = HcclRankGraphGetEndpointDescFunc(commHandle, layerId, instanceId, &actualEndpointCount, endpoints.data());
        TORCH_CHECK(ret == HCCL_SUCCESS && actualEndpointCount <= endpointCapacity,
                    "Get instance endpoints failed, layer: ", layerId, ", instance: ", instanceId, ", ret: ", ret);
        endpoints.resize(actualEndpointCount);
        return endpoints;
    }

    bool CollectInstanceMixLinks(const HcclComm& commHandle, uint32_t layerId, uint32_t instanceId, uint32_t srcRankId,
                                 uint32_t rankSize, std::vector<CommLink>& selectedLinks)
    {
        if (!InstanceCoversAllRanks(commHandle, layerId, instanceId, rankSize)) {
            ASCEND_LOGI("[MixLayer] rank %u skips layer %u instance %u: incomplete rank coverage", srcRankId, layerId,
                        instanceId);
            return false;
        }
        const auto endpoints = GetInstanceEndpoints(commHandle, layerId, instanceId);
        if (endpoints.empty()) {
            ASCEND_LOGI("[MixLayer] rank %u skips layer %u instance %u: no local endpoints", srcRankId, layerId,
                        instanceId);
            return false;
        }
        if (!CollectMixLinks(commHandle, layerId, srcRankId, rankSize, &endpoints, selectedLinks)) {
            ASCEND_LOGI("[MixLayer] rank %u skips layer %u instance %u: incomplete instance CTP links", srcRankId,
                        layerId, instanceId);
            return false;
        }
        return true;
    }

    bool FindCustomMixInstance(const HcclComm& commHandle, uint32_t layerId, CommTopo targetType, uint32_t srcRankId,
                               uint32_t rankSize, std::vector<CommLink>& selectedLinks)
    {
        const auto instanceIds = GetTopologyInstanceIds(commHandle, layerId);
        for (uint32_t instanceId : instanceIds) {
            CommTopo instanceType = COMM_TOPO_RESERVED;
            auto ret = HcclRankGraphGetTopoTypeFunc(commHandle, layerId, instanceId, &instanceType);
            TORCH_CHECK(ret == HCCL_SUCCESS, "Get instance type failed, layer: ", layerId, ", instance:  ", instanceId,
                        ", ret: ", ret);
            if (instanceType != targetType) {
                ASCEND_LOGD("[MixLayer] rank %u layer %u instance %u: type %d, looking for %d", srcRankId, layerId,
                            instanceId, static_cast<int>(instanceType), static_cast<int>(targetType));
                continue;
            }
            if (!CollectInstanceMixLinks(commHandle, layerId, instanceId, srcRankId, rankSize, selectedLinks)) {
                continue;
            }
            ASCEND_LOGI("[MixLayer] rank %u selects CUSTOM layer %u instance %u as %s", srcRankId, layerId, instanceId,
                        targetType == COMM_TOPO_CLOS ? "Clos" : "fullmesh");
            return true;
        }
        ASCEND_LOGI("[MixLayer] rank %u CUSTOM layer %u: no qualified %s instance among %zu instances", srcRankId,
                    layerId, targetType == COMM_TOPO_CLOS ? "Clos" : "fullmesh", instanceIds.size());
        return false;
    }

    void SelectExplicitMixLayers(const HcclComm& commHandle, const std::vector<uint32_t>& layerIds, uint32_t srcRankId,
                                 uint32_t rankSize, std::vector<uint32_t>& customLayers,
                                 std::array<bool, MIX_LAYERED>& found)
    {
        for (uint32_t layerId : layerIds) {
            CommTopo topoType = COMM_TOPO_RESERVED;
            auto ret = HcclRankGraphGetTopoTypeByLayerFunc(commHandle, layerId, &topoType);
            TORCH_CHECK(ret == HCCL_SUCCESS, "Get HCCL topology type failed, layer: ", layerId, ", ret: ", ret);
            ASCEND_LOGI("[MixLayer] rank %u layer %u: topology %d", srcRankId, layerId, static_cast<int>(topoType));
            if (topoType == COMM_TOPO_CUSTOM) {
                customLayers.push_back(layerId);
                continue;
            }
            if (topoType != COMM_TOPO_CLOS && topoType != COMM_TOPO_1DMESH) {
                continue;
            }
            const uint32_t slot = topoType == COMM_TOPO_CLOS ? MIX_CLOS_SLOT : MIX_MESH_SLOT;
            if (found[slot]) {
                continue;
            }
            if (!CoversAllRanks(GetLayerRanks(commHandle, layerId).ranks, rankSize)) {
                ASCEND_LOGI("[MixLayer] rank %u skips layer %u: incomplete rank coverage", srcRankId, layerId);
                continue;
            }
            if (CollectMixLinks(commHandle, layerId, srcRankId, rankSize, nullptr, mixLayeredInfo_.linksByRank[slot])) {
                mixLayeredInfo_.layerIds[slot] = layerId;
                found[slot] = true;
                if (found[MIX_CLOS_SLOT] && found[MIX_MESH_SLOT]) {
                    break;
                }
            }
        }
    }

    void SelectCustomMixLayers(const HcclComm& commHandle, const std::vector<uint32_t>& customLayers,
                               uint32_t srcRankId, uint32_t rankSize, std::array<bool, MIX_LAYERED>& found)
    {
        // When both roles are missing, prefer fullmesh first, then Clos on a different layer.
        for (uint32_t slot : {MIX_MESH_SLOT, MIX_CLOS_SLOT}) {
            if (found[slot]) {
                continue;
            }
            const uint32_t otherSlot = slot == MIX_MESH_SLOT ? MIX_CLOS_SLOT : MIX_MESH_SLOT;
            const CommTopo targetType = slot == MIX_MESH_SLOT ? COMM_TOPO_1DMESH : COMM_TOPO_CLOS;
            for (uint32_t layerId : customLayers) {
                if (found[otherSlot] && layerId == mixLayeredInfo_.layerIds[otherSlot]) {
                    continue;
                }
                if (FindCustomMixInstance(commHandle, layerId, targetType, srcRankId, rankSize,
                                          mixLayeredInfo_.linksByRank[slot])) {
                    mixLayeredInfo_.layerIds[slot] = layerId;
                    found[slot] = true;
                    break;
                }
            }
        }
    }

    void CheckIsMixLayered(const HcclComm& commHandle)
    {
        mixLayeredInfo_ = {};
        uint32_t srcRankId = 0;
        uint32_t rankSize = 0;
        GetRankInfo(commHandle, srcRankId, rankSize);
        if (rankNumPerServer_ != MIX_LAYERED_RANK_SIZE || rankSize != rankNumPerServer_) {
            return;
        }
        uint32_t layerNum = 0;
        uint32_t* layerList = nullptr;
        GetNetLayers(commHandle, layerList, layerNum);
        if (layerNum < MIX_LAYERED) {
            return;
        }
        TORCH_CHECK(layerList != nullptr, "HCCL returned a null layer list");
        std::vector<uint32_t> layerIds(layerList, layerList + layerNum);
        std::vector<uint32_t> customLayers;
        std::array<bool, MIX_LAYERED> found = {};
        // Preserve HCCL order and prefer explicit types before filling missing roles from CUSTOM layers.
        SelectExplicitMixLayers(commHandle, layerIds, srcRankId, rankSize, customLayers, found);
        SelectCustomMixLayers(commHandle, customLayers, srcRankId, rankSize, found);
        mixLayeredInfo_.isUseMixLayered = found[MIX_CLOS_SLOT] && found[MIX_MESH_SLOT];
        if (mixLayeredInfo_.isUseMixLayered) {
            ASCEND_LOGI("[MixLayer] rank %u enables mixed CTP channels: Clos layer %u, fullmesh layer %u", srcRankId,
                        mixLayeredInfo_.layerIds[MIX_CLOS_SLOT], mixLayeredInfo_.layerIds[MIX_MESH_SLOT]);
        } else {
            ASCEND_LOGI("[MixLayer] rank %u uses default topology: Clos found %u, fullmesh found %u", srcRankId,
                        static_cast<uint32_t>(found[MIX_CLOS_SLOT]), static_cast<uint32_t>(found[MIX_MESH_SLOT]));
        }
    }

    void BuildContext(const HcclComm& commHandle, const std::string& groupName, const std::string& opName,
                      const CommProtocol& protocol, MoeCommContext& context, void*& ctx)
    {
        std::string contextTag = groupName + opName;
        CheckContextTag(contextTag);
        CommEngine engine = CommEngine::COMM_ENGINE_AIV;

        GetOrCreateContext(commHandle, contextTag, engine, protocol, ctx, context);
    }

    void CreateContext(const HcclComm& commHandle, const std::string& contextTag, const CommEngine& engine,
                       const CommProtocol& protocol, void*& ctx, MoeCommContext* context)
    {
        uint64_t contextSize = sizeof(MoeCommContext);
        CreateEngineContext(commHandle, contextTag, engine, contextSize, ctx);

        uint32_t rankSize = 0;
        GetRankInfo(commHandle, context->epRankId, rankSize);

        std::string memBufferTag = contextTag + "_buffer";
        CheckContextTag(memBufferTag);
        AllocateAndRegisterDeviceBuffer(commHandle, memBufferTag);

        GetHcclCommResource(commHandle, engine, protocol, *context, rankSize, memBufferTag);

        CopyContextToDevice(commHandle, contextTag, engine, context, contextSize);
    }

    void GetOrCreateContext(const HcclComm& commHandle, const std::string& contextTag, const CommEngine& engine,
                            const CommProtocol& protocol, void*& ctx, MoeCommContext& context)
    {
        uint64_t ctxSize = 0;
        auto hcclRet = HcclEngineCtxGetFunc(commHandle, contextTag.c_str(), engine, &ctx, &ctxSize);
        if (hcclRet != HCCL_SUCCESS) {
            CreateContext(commHandle, contextTag, engine, protocol, ctx, &context);
            createdNew_ = true;
            return;
        }
        // ctx 已存在(同 group 曾创建过，含 destroy 后重建)：物理内存由共享池按 tag 保留，复用既有 buffer。
        // 重建声明的容量不得超过首建容量，否则 tiling 按声明值校验通过、 kernel 经共享 ctx 写入会超出实际物理内存。
        MoeSharedBufferEntry entry;
        {
            std::lock_guard<std::mutex> lock(gMoeSharedBufferMutex);
            auto iter = gMoeSharedBuffers.find(contextTag);
            TORCH_CHECK(iter != gMoeSharedBuffers.end(), "MoE comm context of group '", contextTag,
                        "' exists but its buffer is missing from the shared pool; an earlier build on this "
                        "group may have failed midway, please use a new comm group");
            TORCH_CHECK(cclBufferSize_ <= iter->second.cclBufferSize,
                        "ccl buffer size exceeds the existing buffer of this group, requested ", cclBufferSize_,
                        ", allocated ", iter->second.cclBufferSize,
                        "; the shared comm buffer is sized by the first ElasticBuffer created on this group, "
                        "destroy and recreate with a larger size is not supported, please use a new comm group");
            entry = iter->second;
        }
        deviceBufPtr_ = entry.deviceBufPtr;
        physicalMemHandle_ = entry.physicalMemHandle;
        memHandle_ = entry.memHandle;
        ownershipReleased_ = true; // 复用资源不归本 builder 所有，析构不得释放
    }

    void GetCommProtocol(const HcclComm& commHandle, CommProtocol& protocol)
    {
        ASCEND_LOGI("start GetCommProtocol");
        uint32_t layerNum = 0;
        uint32_t* layerList = nullptr;
        GetNetLayers(commHandle, layerList, layerNum);

        if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
            GetRankSizePerServer(commHandle, rankNumPerUbDomain_);
            return;
        }

        CheckProtocolSupport(commHandle, layerList, layerNum, protocol);
    }

    bool HasUbGPeer() const
    {
        for (const auto& entry : rankLinkMap_) {
            if (entry.second.protocol == CommProtocol::COMM_PROTOCOL_UB_RTP) {
                return true;
            }
        }
        return false;
    }

    void InitHcclChannel(const HcclComm& commHandle, uint32_t rankDim, uint32_t srcRankId, uint32_t channelsPerRank,
                         const CommProtocol& protocol, std::vector<HcclChannelDesc>& channelDesc)
    {
        uint32_t channelNum = static_cast<uint32_t>(channelDesc.size());
        auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HCCL channel init failed, ret: ", hcclRet);

        uint32_t netLayerNum = 0;
        uint32_t* netLayerList = nullptr;
        GetNetLayers(commHandle, netLayerList, netLayerNum);
        TORCH_CHECK(netLayerNum > 0, "Get HCCL net layers failed, netLayerNum is ", netLayerNum);

        for (uint32_t peer = 0; peer < rankDim; ++peer) {
            if (peer == srcRankId) {
                continue;
            }
            uint32_t peerIndex = (peer > srcRankId) ? (peer - 1) : peer;
            RankLinkInfo linkInfo = ResolveLinkInfo(peer, protocol, netLayerList);
            CommLink* links = nullptr;
            if (!mixLayeredInfo_.isUseMixLayered) {
                GetHcclCommLink(commHandle, linkInfo.layer, srcRankId, peer, linkInfo.protocol, links);
            }
            for (uint32_t channel = 0; channel < channelsPerRank; ++channel) {
                // Handles are compact by remote rank because the local rank does not need an HCOMM channel.
                if (mixLayeredInfo_.isUseMixLayered) {
                    links = &mixLayeredInfo_.linksByRank[channel % MIX_LAYERED][peer];
                }
                uint32_t channelId = peerIndex * channelsPerRank + channel;
                channelDesc[channelId].channelProtocol =
                    mixLayeredInfo_.isUseMixLayered ? CommProtocol::COMM_PROTOCOL_UB_CTP : linkInfo.protocol;
                channelDesc[channelId].remoteRank = peer;
                channelDesc[channelId].notifyNum = MOE_CHANNEL_NOTIFY_NUM;
                channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
                channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
                channelDesc[channelId].memHandles = &memHandle_;
                channelDesc[channelId].memHandleNum = MEM_HANDLE_NUM;
            }
        }
    }

    void GetHcclCommChannel(const HcclComm& commHandle, const CommEngine& engine, uint32_t rankDim, uint32_t srcRankId,
                            const CommProtocol& protocol, MoeCommContext& context)
    {
        TORCH_CHECK(rankDim >= HCCL_MIN_RANK_SIZE && rankDim <= HCCL_MAX_RANK_SIZE, "Invalid HCCL rank size ", rankDim);
        uint32_t remoteRankNum = rankDim - 1;
        bool hasUbGPeer = HasUbGPeer();
        context.channelsPerRank = 1U;
        if (!hasUbGPeer && rankDim < MOE_CHANNEL_HANDLE_NUM) {
            context.channelsPerRank = MOE_CHANNEL_HANDLE_NUM / rankDim;
        }
        TORCH_CHECK(context.channelsPerRank > 0, "No HCCL channel capacity for rank size ", rankDim);
        TORCH_CHECK(context.channelsPerRank <= HCCL_MAX_RANK_SIZE / rankDim,
                    "HCCL channel handles exceed capacity, rank size ", rankDim, ", channels per rank ",
                    context.channelsPerRank);
        uint32_t channelNum = remoteRankNum * context.channelsPerRank;
        std::vector<HcclChannelDesc> channelDesc(channelNum);
        ChannelHandle channelBuf[HCCL_MAX_RANK_SIZE] = {};

        InitHcclChannel(commHandle, rankDim, srcRankId, context.channelsPerRank, protocol, channelDesc);
        AcquireChannels(commHandle, engine, channelDesc, channelBuf);

        uint32_t channelIndex = 0;
        for (uint32_t peer = 0; peer < rankDim; ++peer) {
            if (peer == srcRankId) {
                continue;
            }
            for (uint32_t channel = 0; channel < context.channelsPerRank; ++channel) {
                context.hcommHandle[peer * context.channelsPerRank + channel] = channelBuf[channelIndex++];
            }
        }
    }

    void GetHcclCommResource(const HcclComm& commHandle, const CommEngine& engine, const CommProtocol& protocol,
                             MoeCommContext& context, uint32_t rankSize, const std::string& targetTag)
    {
        uint32_t rankId = context.epRankId;
        GetHcclCommChannel(commHandle, engine, rankSize, rankId, protocol, context);

        GetRegisteredCommResource(commHandle, context, rankSize, targetTag);
    }

    // 单次物理内存申请(预留虚拟地址→分配物理→映射→设权限→清零)，失败时回滚已完成的步骤。
    // prop由调用方预先构造，内部仅按memAttr变更大页规格后使用。
    bool AllocateHugePageBuffer(uint64_t allocSizeBytes, aclrtPhysicalMemProp& prop, aclrtMemAttr memAttr)
    {
        void* virPtr = nullptr;
        aclError ret = aclrtReserveMemAddress(&virPtr, allocSizeBytes, 0, nullptr, 0);
        if (ret != ACL_SUCCESS) {
            ASCEND_LOGI("aclrtReserveMemAddress(%lu) failed, ret=%d", allocSizeBytes, ret);
            return false;
        }

        prop.memAttr = memAttr;
        aclrtDrvMemHandle physicalHandle = nullptr;
        ret = aclrtMallocPhysical(&physicalHandle, allocSizeBytes, &prop, 0);
        if (ret != ACL_SUCCESS) {
            ASCEND_LOGI("aclrtMallocPhysical(%lu) failed, ret=%d", allocSizeBytes, ret);
            (void)aclrtReleaseMemAddress(virPtr);
            return false;
        }

        ret = aclrtMapMem(virPtr, allocSizeBytes, 0, physicalHandle, 0);
        if (ret != ACL_SUCCESS) {
            ASCEND_LOGI("aclrtMapMem(%lu) failed, ret=%d", allocSizeBytes, ret);
            (void)aclrtFreePhysical(physicalHandle);
            (void)aclrtReleaseMemAddress(virPtr);
            return false;
        }

        int32_t deviceId = 0;
        ret = aclrtGetDevice(&deviceId);
        if (ret != ACL_SUCCESS) {
            // 设备句柄获取失败非内存不足，直接报错；报错前回滚已完成的步骤防泄露
            (void)aclrtUnmapMem(virPtr);
            (void)aclrtFreePhysical(physicalHandle);
            (void)aclrtReleaseMemAddress(virPtr);
            TORCH_CHECK(false, "aclrtGetDevice failed, ret=", ret);
        }
        aclrtMemAccessDesc accessDesc = {};
        accessDesc.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
        accessDesc.location.id = static_cast<uint32_t>(deviceId);
        accessDesc.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        ret = aclrtMemSetAccess(virPtr, allocSizeBytes, &accessDesc, 1);
        if (ret != ACL_SUCCESS) {
            ASCEND_LOGI("aclrtMemSetAccess failed, ret=%d", ret);
            (void)aclrtUnmapMem(virPtr);
            (void)aclrtFreePhysical(physicalHandle);
            (void)aclrtReleaseMemAddress(virPtr);
            return false;
        }

        deviceBufPtr_ = virPtr;
        physicalMemHandle_ = physicalHandle;
        return true;
    }

    void AllocateAndRegisterDeviceBuffer(const HcclComm& commHandle, const std::string& memBufferTag)
    {
        TORCH_CHECK(cclBufferSize_ > 0, "ccl buffer size must be greater than 0, got ", cclBufferSize_);
        uint64_t bufferSizeBytes = static_cast<uint64_t>(cclBufferSize_);
        if (deviceBufPtr_ == nullptr) {
            // 物理内存属性: PINNED(物理地址连续) + 当前设备定位，大页规格按实际情况调整
            int32_t deviceId = 0;
            aclError ret = aclrtGetDevice(&deviceId);
            TORCH_CHECK(ret == ACL_SUCCESS, "aclrtGetDevice failed, ret=", ret);
            aclrtPhysicalMemProp prop = {};
            prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
            prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.id = static_cast<uint32_t>(deviceId);
            prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;

            // 优先1G大页(申请量按1G向上对齐)，不足时降级2M大页(申请量=原始需求，天然2M对齐)
            uint64_t allocBufferBytesAlign = CeilDiv(cclBufferSize_, HUGE1G_SIZE) * HUGE1G_SIZE;
            if (!AllocateHugePageBuffer(allocBufferBytesAlign, prop, ACL_MEM_HUGE1G)) {
                ASCEND_LOGI("1G hugepage allocation failed, fallback to 2M hugepage");
                TORCH_CHECK(AllocateHugePageBuffer(bufferSizeBytes, prop, ACL_MEM_HUGE),
                            "Allocate ccl buffer failed for both 1G and 2M hugepage, size=", bufferSizeBytes);
            }

            // 按实际通信窗口大小清零（1G大页对齐时申请量大于需求，尾部无需清零）
            ret = aclrtMemset(deviceBufPtr_, bufferSizeBytes, 0, bufferSizeBytes);
            TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemset(deviceBufPtr_) failed, ret=", ret);
        }
        if (memHandle_ == nullptr) {
            CommMem mem;
            mem.type = COMM_MEM_TYPE_DEVICE;
            mem.addr = deviceBufPtr_;
            mem.size = bufferSizeBytes;
            auto hcclRet = HcclCommMemRegFunc(commHandle, memBufferTag.c_str(), &mem, &memHandle_);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclCommMemReg(tag='", memBufferTag, "', size=", bufferSizeBytes,
                        ") failed, ret=", hcclRet);
        }
    }

    void GetRegisteredCommResource(const HcclComm& commHandle, MoeCommContext& context, uint32_t rankSize,
                                   const std::string& targetTag)
    {
        uint32_t rankId = context.epRankId;
        context.epHcclBuffer[rankId] = reinterpret_cast<uint64_t>(deviceBufPtr_);
        for (uint32_t peer = 0; peer < rankSize; ++peer) {
            if (peer == rankId) {
                continue;
            }
            uint32_t memNum = 0;
            CommMem* remoteMems = nullptr;
            char** memTags = nullptr;
            auto hcclRet = HcclChannelGetRemoteMemsFunc(commHandle, context.hcommHandle[peer * context.channelsPerRank],
                                                        &memNum, &remoteMems, &memTags);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclChannelGetRemoteMems(peer=", peer, ") failed, ret=", hcclRet);
            bool hasTargetMem = false;
            for (uint32_t j = 0; j < memNum; ++j) {
                if (memTags == nullptr || remoteMems == nullptr) {
                    break;
                }
                if (memTags[j] != nullptr && targetTag == memTags[j]) {
                    context.epHcclBuffer[peer] = reinterpret_cast<uint64_t>(remoteMems[j].addr);
                    hasTargetMem = true;
                    break;
                }
            }
            TORCH_CHECK(hasTargetMem, "Target Mem : ", targetTag, " is not found.");
        }
    }

    static void GetRankSizePerServer(const HcclComm& commHandle, uint32_t& rankSizePerServer)
    {
        uint32_t* netLayerList = nullptr;
        uint32_t netLayerNum = 0;
        GetNetLayers(commHandle, netLayerList, netLayerNum);

        uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER];
        auto hcclRet = HcclRankGraphGetRankSizeByLayerFunc(commHandle, netLayers, &rankSizePerServer);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL rank size per server failed, ret: ", hcclRet);
    }

    MixLayeredInfo mixLayeredInfo_;
    uint32_t rankNumPerServer_ = 2;
    int64_t cclBufferSize_ = 0;
    void* deviceBufPtr_ = nullptr;
    aclrtDrvMemHandle physicalMemHandle_ = nullptr;
    HcclMemHandle memHandle_ = nullptr;
    bool createdNew_ = false;        // 本次 Build 是否首次创建(ctx 不存在)
    bool ownershipReleased_ = false; // 物理内存所有权是否已转移(共享池/复用)，析构不再释放
};

// ElasticBuffer class - unified interface for Engram storage and MoE EP kernels
class ElasticBuffer {
public:
    // 同轨道通信开关（对标 DeepEP allow_hybrid_mode）：0=关闭；1=分轨——训练走分轨模型，
    // 拓扑自动探测（env 覆盖 → 链路实测 → 因子分解兜底）。worldSize 由 python 传入（dist world）。
    ElasticBuffer(const std::optional<std::string>& groupName, int64_t numCpuBytes, int64_t numMaxTokensPerRank = 0,
                  bool withGrad = false, bool explicitlyDestroy = false, const std::string& netLayer = "auto",
                  const std::string& moeCommAlg = MOE_COMM_ALG_DEFAULT, int64_t allowHybridMode = NETWORK_DIRECT,
                  int64_t worldSize = 0);
    ~ElasticBuffer();

    void EngramWrite(const at::Tensor& storage, const c10::optional<at::Tensor>& sf, bool zeroCopy);
    void EngramBarrier(bool useCommStream = true, bool withCpuSync = false);
    void Destroy();

    const at::Tensor& GetContextTensor() const
    {
        return engramContextTensor_;
    }

    // ctx 是否已建链（python attr getter 防御：未建链时不缓存缺省拓扑）
    bool IsEngramContextInitialized() const
    {
        return engramContextInitialized_;
    }

    // 读取感知数每超节点卡数（python getter → attr 透传给 tiling；未建链/老路径返回 0）
    uint32_t GetRanksPerSupernode() const
    {
        return engramCommContext_.ranksPerSupernode;
    }

    const at::Tensor& GetLocalStorageAddrTensor() const
    {
        return localStorageAddrTensor_;
    }
    const at::Tensor& GetLocalSfTableAddrTensor() const
    {
        return localSfTableTensor_;
    }
    int64_t GetCommBufferSize() const
    {
        return commBufferSize_;
    }
    uint32_t GetRankSize() const
    {
        // 分轨（探测命中轨道即建链，含 ≤1024 缩比床）：ctx group 实例的 rankSize 只是编号空间
        // 大小，不是通信域尺寸——python 校验（send_counts=world×8）与 maxR 需要 world；
        // 老路径/本地全词表维持原语义（通信域 rank 数）。
        if (engramRailContext_) {
            return static_cast<uint32_t>(engramWorldSize_);
        }
        return engramCommContext_.rankSize;
    }

    using EngramFetchOutput = std::tuple<at::Tensor, at::Tensor>;
    static EngramFetchOutput EngramFetch(const at::Tensor& context, const at::Tensor& indices, int64_t hiddenSize,
                                         int64_t numEntries, int64_t dtypeEnum, const at::Tensor& sfTable,
                                         int64_t hybridMode);
    using EngramFetchTrainOutput =
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
    static EngramFetchTrainOutput EngramFetchTrain(const at::Tensor& context, const at::Tensor& indices,
                                                   int64_t hiddenSize, int64_t numEntries, int64_t dtypeEnum,
                                                   const at::Tensor& localStorageAddr, at::ScalarType sfDtype,
                                                   int64_t sfPacks, const at::Tensor& sfTableAddr,
                                                   int64_t numMaxTokensPerRank, int64_t commBufferSize,
                                                   int64_t rankSize, int64_t ranksPerSupernode = 0);
    static at::Tensor EngramFetchWait(const at::Tensor& context, const at::Tensor& fetched);

    // Stateless static method for training backward (graph-mode compatible).
    // Outputs: gradUnique [maxR, H], uniqueLocalEntry [maxR], numUnique [1] (NOT narrowed).
    // Caller is responsible for narrowing by numUnique.item() outside the graph.
    using EngramFetchGradOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
    // 感知数（HCCL 感知 → python get_ranks_per_supernode → 透传）：缺省 0 = 未感知
    static EngramFetchGradOutput EngramFetchGrad(const at::Tensor& context, const at::Tensor& gradFetched,
                                                 const at::Tensor& perm, const at::Tensor& sendCounts,
                                                 const at::Tensor& recvCounts, const at::Tensor& recvLocalEntry,
                                                 const at::Tensor& numRecv, int64_t numEntries, int64_t commBufferSize,
                                                 int64_t numMaxTokensPerRank, int64_t rankSize,
                                                 int64_t ranksPerSupernode = 0);

    static int64_t GetEngramStorageSizeHint(int64_t numEntries, int64_t hiddenSize,
                                            at::ScalarType dtype = at::kBFloat16);

    using DispatchTensorList = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
    using DispatchEpilogueTensorList =
        std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>>;
    using CombineEpilogueTensorList = std::tuple<at::Tensor, c10::optional<at::Tensor>>;

    DispatchTensorList MoeEpDispatch(
        const at::Tensor& x, const at::Tensor& topkIdx, const c10::optional<at::Tensor>& topkWeights,
        const c10::optional<at::Tensor>& scales, const c10::optional<at::Tensor>& cachedDstSlotIdx,
        const c10::optional<at::Tensor>& cachedRouteCount, const c10::optional<at::Tensor>& cachedRouteDstScaleout,
        const c10::optional<at::Tensor>& cachedRouteScaleoutSlot, int64_t epWorldSize, int64_t epRankId,
        int64_t numExperts, int64_t numMaxTokensPerRank, int64_t expertAlignment, bool doCpuSync,
        int64_t hostPinnedCounterAddr, int64_t cclBufferSize);
    DispatchEpilogueTensorList MoeEpDispatchEpilogue(
        const at::Tensor& x, const at::Tensor& topkIdx, const at::Tensor& numRecvPerRank,
        const at::Tensor& numRecvPerExpert, const c10::optional<at::Tensor>& cachedRecvSrcMetadata, int64_t epWorldSize,
        int64_t epRankId, int64_t numExperts, int64_t numMaxTokensPerRank, int64_t cclBufferSize, at::Tensor& recvX,
        at::Tensor& recvSrcMetadata, const c10::optional<at::Tensor>& recvTopkWeightsOpt,
        const c10::optional<at::Tensor>& recvScalesOpt);
    void MoeEpCombine(const at::Tensor& x, const at::Tensor& topkIdx, const at::Tensor& recvSrcMetadata,
                      const at::Tensor& numRecvTokensPerExpert, const c10::optional<at::Tensor>& topkWeights,
                      int64_t epWorldSize, int64_t epRankId, int64_t numExperts, int64_t numMaxTokensPerRank,
                      int64_t cclBufferSize);
    CombineEpilogueTensorList MoeEpCombineEpilogue(const at::Tensor& x, const at::Tensor& topkIdx,
                                                   const at::Tensor& recvSrcMetadata,
                                                   const c10::optional<at::Tensor>& topkWeights, int64_t epWorldSize,
                                                   int64_t epRankId, int64_t numExperts, int64_t numMaxTokensPerRank,
                                                   int64_t cclBufferSize, at::Tensor& combinedX,
                                                   const c10::optional<at::Tensor>& combinedTopkWeightsOpt);

private:
    void EnsureEngramContext(void* externalHostPtr = nullptr, int64_t externalBytes = 0, void* sfHostPtr = nullptr,
                             int64_t sfBytes = 0);
    void EnsureMoeContext(int64_t cclBufferSize, int64_t epWorldSize);
    int64_t ResolveRankNumPerServer(int64_t epWorldSize) const;
    int64_t ResolveTopoType(int64_t epWorldSize, int64_t rankNumPerServer) const;
    void CheckExternalStorageUnchanged(const at::Tensor& storage, bool zeroCopy, bool localMode) const;
    void ReZeroRailWindowHead(aclrtStream stream);

    std::optional<std::string> groupName_;
    bool localFullVocab_ = false;
    int64_t engramNumCpuBytes_;
    int64_t numMaxTokensPerRank_ = 0;
    bool explicitlyDestroy_ = false;
    bool withGrad_ = false;
    std::string netLayer_ = "auto";
    std::string moeCommAlg_ = MOE_COMM_ALG_DEFAULT;
    int64_t hybridMode_ = NETWORK_DIRECT;
    uint32_t rankId_ = 0;
    uint32_t rankSize_ = 0;
    int64_t engramWorldSize_ = 0;
    bool engramRailContext_ = false;

    void* engramHostBufPtr_ = nullptr;
    void* engramDeviceBufPtr_ = nullptr;
    HcclMemHandle engramMemHandle_ = nullptr;
    int64_t commBufferSize_ = 0; // HCCL 默认 buffer 大小（从 HcclGetHcclBufferFunc 查询，训练 a2a 收发缓冲）
    HcclComm engramHcclComm_ = nullptr;
    EngramCommContext engramCommContext_;
    at::Tensor engramContextTensor_;    // Cached Engram context tensor
    at::Tensor localStorageAddrTensor_; // int64 scalar tensor, stores deviceBufPtr_ address
    at::Tensor localSfTableTensor_;     // FP8 device tensor view of HCCL-registered sf table
    void* sfDeviceBufPtr_ = nullptr;
    bool sfExternalRegistered_ = false;
    bool engramContextInitialized_ = false;
    bool engramStorageExternal_ = false;
    bool engramExternalRegisteredByUs_ = false;
    int64_t engramExternalBytes_ = 0;
    bool engramBufferPooled_ = false; // host buffer 归进程级共享池所有，Destroy 不反注册/释放
    // local 模式下，本实例是否已经释放 mapping 引用；free host 失败后的重试只需继续释放内存。
    bool localMappingReleased_ = false;

    at::Tensor moeContextTensor_;
    int64_t moeCclBufferSize_ = 0; // MoE 通信 buffer 大小（首次调用时按算子参数计算并内部申请注册）
    uint32_t moeRankSizePerServer_ = 2;
    void* moeDeviceBufPtr_ = nullptr;
    aclrtDrvMemHandle moePhysicalMemHandle_ = nullptr;
    HcclMemHandle moeMemHandle_ = nullptr;
    std::string moeContextTag_; // 共享池 key（group + opName），复用/入池时使用
    bool moeContextInitialized_ = false;

    int64_t engramHiddenSize_ = 0;
    int64_t engramNumEntries_ = 0;
    at::ScalarType engramDtype_ = at::kBFloat16;

    bool destroyed_ = false;
    bool engramWriteCalled_ = false;
    aclrtStream commStream_ = nullptr;
};

// Constructor

ElasticBuffer::ElasticBuffer(const std::optional<std::string>& groupName, int64_t numCpuBytes,
                             int64_t numMaxTokensPerRank, bool withGrad, bool explicitlyDestroy,
                             const std::string& netLayer, const std::string& moeCommAlg, int64_t allowHybridMode,
                             int64_t worldSize)
    : groupName_(groupName),
      localFullVocab_(!groupName.has_value()),
      engramNumCpuBytes_(numCpuBytes),
      destroyed_(false),
      engramWriteCalled_(false),
      numMaxTokensPerRank_(numMaxTokensPerRank),
      explicitlyDestroy_(explicitlyDestroy),
      withGrad_(withGrad),
      netLayer_(netLayer),
      moeCommAlg_(moeCommAlg),
      hybridMode_(allowHybridMode),
      engramWorldSize_(worldSize)
{
    TORCH_CHECK(moeCommAlg_ == MOE_COMM_ALG_AUTO || moeCommAlg_ == MOE_COMM_ALG_CLOS_ONLY,
                "MoE comm_alg must be 'auto' or 'clos_only', but got '", moeCommAlg_, "'.");
    TORCH_CHECK(hybridMode_ == NETWORK_DIRECT || hybridMode_ == NETWORK_HYBRID,
                "allow_hybrid_mode must be 0 (off) or 1 (rail-aligned), got ", hybridMode_, ".");
    if (!localFullVocab_) {
        InitHcclEngineCtxFunctions();
        InitHcclFunctions();
    }
    if (hybridMode_ == NETWORK_HYBRID) {
        TORCH_CHECK(!localFullVocab_, "allow_hybrid_mode only supports grouped mode (group != None)");
        if (!withGrad_) {
            // 推理分轨依赖 host 锁页共享槽直读，须显式 2MB 对齐 num_cpu_bytes；训练分轨自建对齐缓冲
            TORCH_CHECK(engramNumCpuBytes_ > 0 && engramNumCpuBytes_ % BUFFER_ALIGNMENT == 0,
                        "rail_aligned inference requires explicit 2MB-aligned num_cpu_bytes, got ", engramNumCpuBytes_);
        }
    }
}

// Destructor - automatic resource cleanup only when explicitlyDestroy is false
ElasticBuffer::~ElasticBuffer()
{
    if (explicitlyDestroy_) {
        if (!destroyed_) {
            ASCEND_LOGI("ElasticBuffer is destroyed without explicit destroy() call, "
                        "resource leak may occur when explicitly_destroy is set to true.");
        }
        return;
    }
    try {
        Destroy();
    } catch (const std::exception& e) {
        ASCEND_LOGE("ElasticBuffer destructor cleanup failed: %s", e.what());
    }
}

void ElasticBuffer::EnsureEngramContext(void* externalHostPtr, int64_t externalBytes, void* sfHostPtr, int64_t sfBytes)
{
    TORCH_CHECK(!destroyed_, "ElasticBuffer cannot be used after destroy, please create a new ElasticBuffer instance");
    if (engramContextInitialized_) {
        return;
    }
    EngramContextResources resources;
    if (localFullVocab_) {
        resources = EngramContextBuilder::BuildLocal(engramNumCpuBytes_, externalHostPtr, externalBytes);
    } else {
        commStream_ = c10_npu::getNPUStreamFromPool().stream(false);
        TORCH_CHECK(commStream_ != nullptr, "Failed to get NPU stream from pool for comm stream");
        if (hybridMode_ == NETWORK_HYBRID && withGrad_) {
            RailTopoInfo probeTopo;
            std::string probeReason;
            TORCH_CHECK(EngramContextBuilder::ProbeRailTopoByGroup(*groupName_, probeTopo, probeReason),
                        "allow_hybrid_mode (training rail-aligned) topo probe failed: ", probeReason,
                        "; set RANKS_PER_SUPERNODE for test clusters (ranks per supernode), "
                        "or check supernode wiring and HCCL topo");
            if (engramWorldSize_ > probeTopo.ranksPerSupernode) {
                EngramContextBuilder builder(hybridMode_, commStream_);
                resources = builder.BuildRailContext(*groupName_, withGrad_ ? 0 : engramNumCpuBytes_, externalHostPtr,
                                                     externalBytes, sfHostPtr, sfBytes);
                engramRailContext_ = true; // GetRankSize 语义开关：分轨下返回 world
            } else {
                TORCH_CHECK(engramWorldSize_ <= HCCL_MAX_RANK_SIZE, "single supernode world=", engramWorldSize_,
                            " exceeds legacy builder rank limit 1024 (all-direct domain too large); "
                            "rail-aligned requires a multi-supernode rail topology");
                EngramContextBuilder builder;
                resources = builder.Build(*groupName_, withGrad_ ? 0 : engramNumCpuBytes_, withGrad_, externalHostPtr,
                                          externalBytes, sfHostPtr, sfBytes, netLayer_);
            }
        } else if (hybridMode_ == NETWORK_HYBRID) {
            EngramContextBuilder builder(hybridMode_, commStream_);
            resources = builder.Build(*groupName_, withGrad_ ? 0 : engramNumCpuBytes_, withGrad_, externalHostPtr,
                                      externalBytes, sfHostPtr, sfBytes, netLayer_);
        } else {
            TORCH_CHECK(engramWorldSize_ <= HCCL_MAX_RANK_SIZE, "world=", engramWorldSize_,
                        " > 1024 requires rail-aligned link setup: pass "
                        "allow_hybrid_mode=1 to ElasticBuffer (training rail-aligned); "
                        "legacy builder cannot host world > 1024");
            try {
                EngramContextBuilder builder;
                resources = builder.Build(*groupName_, withGrad_ ? 0 : engramNumCpuBytes_, withGrad_, externalHostPtr,
                                          externalBytes, sfHostPtr, sfBytes, netLayer_);
            } catch (const std::exception& e) {
                const std::string errMsg = e.what();
                if (errMsg.find("No UBC/UBG link found") == std::string::npos) {
                    throw;
                }
                RailTopoInfo probeTopo;
                std::string probeReason;
                if (EngramContextBuilder::ProbeRailTopoByGroup(*groupName_, probeTopo, probeReason) &&
                    probeTopo.world > probeTopo.ranksPerSupernode && probeTopo.ranksPerSupernode > 0U) {
                    TORCH_CHECK(false, "direct link setup failed on a rail topology: world=", probeTopo.world,
                                ", ranksPerSupernode=", probeTopo.ranksPerSupernode,
                                " (cross-supernode peers are "
                                "only reachable via same-L rail links); original error: ",
                                errMsg, "; pass allow_hybrid_mode=1 to ElasticBuffer for rail-aligned link setup");
                }
                throw;
            }
        }
    }
    rankId_ = resources.context.rankId;
    rankSize_ = resources.context.rankSize;
    engramHcclComm_ = resources.hcclComm;
    engramMemHandle_ = resources.memHandle;
    engramHostBufPtr_ = resources.hostBufPtr;
    engramDeviceBufPtr_ = resources.deviceBufPtr;
    engramCommContext_ = resources.context;
    engramContextTensor_ = resources.contextTensor;
    commBufferSize_ = resources.commBufferSize;
    engramStorageExternal_ = (externalHostPtr != nullptr);
    engramExternalRegisteredByUs_ = resources.externalRegistered;
    sfDeviceBufPtr_ = resources.sfDeviceBufPtr;
    sfExternalRegistered_ = resources.sfExternalRegistered;
    if (sfDeviceBufPtr_ != nullptr) {
        int64_t sfAddrValue = reinterpret_cast<int64_t>(sfDeviceBufPtr_);
        auto hostSfAddrTensor = at::full({1}, sfAddrValue, at::TensorOptions().dtype(at::kLong));
        localSfTableTensor_ = hostSfAddrTensor.to(c10::DeviceType::PrivateUse1);
    }
    engramExternalBytes_ = externalBytes;
    engramBufferPooled_ = (!localFullVocab_ && engramHostBufPtr_ != nullptr);
    int64_t addrValue = reinterpret_cast<int64_t>(engramDeviceBufPtr_);
    auto hostAddrTensor = at::full({1}, addrValue, at::TensorOptions().dtype(at::kLong));
    localStorageAddrTensor_ = hostAddrTensor.to(c10::DeviceType::PrivateUse1);
    engramContextInitialized_ = true;
}

// Lazy-initialize the MoE comm context on the first MoE kernel invocation (four entry points
// keep equal triggering rights, same as the original pattern). The CCL buffer size is computed
// by Python from the actual operator arguments and passed in; it is internally allocated as
// 2M-hugepage pinned physical memory and registered to the comm domain.
void ElasticBuffer::EnsureMoeContext(int64_t cclBufferSize, int64_t epWorldSize)
{
    TORCH_CHECK(!destroyed_, "ElasticBuffer cannot be used after destroy, please create a new ElasticBuffer instance");
    TORCH_CHECK(!localFullVocab_, "MoE operations are not supported when group is None");
    if (moeContextInitialized_) {
        TORCH_CHECK(cclBufferSize <= moeCclBufferSize_,
                    "MoE communication buffer cannot grow after context initialization, requested ", cclBufferSize,
                    ", allocated ", moeCclBufferSize_, "; use a new communication group");
        return;
    }
    TORCH_CHECK(epWorldSize >= HCCL_MIN_RANK_SIZE, "MoE requires at least ", HCCL_MIN_RANK_SIZE, " ranks, but got ",
                epWorldSize);
    if (moeCommAlg_ == MOE_COMM_ALG_CLOS_ONLY) {
        moeContextTensor_ = BuildMoeSharedJettyContext(*groupName_, static_cast<uint64_t>(cclBufferSize));
        moeCclBufferSize_ = cclBufferSize;
    } else {
        MoeContextBuilder builder;
        MoeContextResources resources = builder.Build(*groupName_, cclBufferSize);
        moeContextTensor_ = resources.contextTensor;
        moeCclBufferSize_ = resources.cclBufferSize;
        moeRankSizePerServer_ = resources.rankSizePerServer;
        moeDeviceBufPtr_ = resources.deviceBufPtr;
        moePhysicalMemHandle_ = resources.physicalMemHandle;
        moeMemHandle_ = resources.memHandle;
        moeContextTag_ = resources.contextTag;
    }
    moeContextInitialized_ = true;
}

int64_t ElasticBuffer::ResolveRankNumPerServer(int64_t epWorldSize) const
{
    if (moeCommAlg_ == MOE_COMM_ALG_CLOS_ONLY) {
        return epWorldSize;
    }
    int64_t rankNumPerServer = static_cast<int64_t>(moeRankSizePerServer_);
    TORCH_CHECK(rankNumPerServer > 0, "rank_num_per_server must be positive, got ", rankNumPerServer);
    TORCH_CHECK(epWorldSize % rankNumPerServer == 0, "ep_world_size must be divisible by rank_num_per_server, got ",
                epWorldSize, " and ", rankNumPerServer);
    return rankNumPerServer;
}

int64_t ElasticBuffer::ResolveTopoType(int64_t epWorldSize, int64_t rankNumPerServer) const
{
    if (moeCommAlg_ == MOE_COMM_ALG_CLOS_ONLY) {
        return NETWORK_DIRECT;
    }
    return (epWorldSize / rankNumPerServer > 1) ? NETWORK_HYBRID : NETWORK_DIRECT;
}

void ElasticBuffer::CheckExternalStorageUnchanged(const at::Tensor& storage, bool zeroCopy, bool localMode) const
{
    const int64_t currentBytes = zeroCopy && storage.storage_offset() == 0 ?
                                     static_cast<int64_t>(storage.storage().nbytes()) :
                                     static_cast<int64_t>(storage.nbytes());
    if (withGrad_) {
        TORCH_CHECK(engramHostBufPtr_ == storage.data_ptr(),
                    "engram_write: engram storage is already initialized from a different address, expected ptr=",
                    engramHostBufPtr_, ", got ptr=", storage.data_ptr(),
                    "; note: in with_grad mode the first engram API call initializes the context, an earlier "
                    "engram_barrier without storage would initialize it with no storage");
        TORCH_CHECK(engramExternalBytes_ == static_cast<int64_t>(storage.nbytes()),
                    "engram_write: storage size changed after the engram storage was registered from this address, "
                    "registered ",
                    engramExternalBytes_, " bytes, got ", storage.nbytes(), " bytes");
        return;
    }

    TORCH_CHECK(zeroCopy == engramStorageExternal_,
                "engram_write zero_copy mode cannot change after the storage window is initialized");
    if (!zeroCopy) {
        return;
    }

    const char* mode = localMode ? "local zero-copy" : "zero-copy";
    TORCH_CHECK(engramHostBufPtr_ == storage.data_ptr(), "engram_write: ", mode, " storage address changed, expected ",
                engramHostBufPtr_, ", got ", storage.data_ptr());
    TORCH_CHECK(engramExternalBytes_ == currentBytes, "engram_write: ", mode, " storage size changed, registered ",
                engramExternalBytes_, " bytes, got ", currentBytes, " bytes");
}

// EngramWrite - write data with automatic barrier.
void ElasticBuffer::EngramWrite(const at::Tensor& storage, const c10::optional<at::Tensor>& sf, bool zeroCopy)
{
    TORCH_CHECK(!destroyed_, "engram_write cannot be called after destroy, "
                             "please create a new ElasticBuffer instance");
    // Zero-copy inference may be backed by a padded mmap.  Register the full
    // storage when the tensor starts at offset 0 so the driver sees a 2 MiB
    // aligned length; otherwise preserve the tensor's logical byte count.
    const int64_t registrationBytes = storage.storage_offset() == 0 ? static_cast<int64_t>(storage.storage().nbytes()) :
                                                                      static_cast<int64_t>(storage.nbytes());
    void* externalHostPtr = nullptr;
    int64_t externalBytes = 0;
    void* sfHostPtr = nullptr;
    int64_t sfBytes = 0;
    if (localFullVocab_) {
        TORCH_CHECK(!withGrad_, "local full-vocab mode supports inference only");
        if (engramContextInitialized_) {
            CheckExternalStorageUnchanged(storage, zeroCopy, true);
        } else {
            if (zeroCopy) {
                externalHostPtr = storage.data_ptr();
                externalBytes = registrationBytes;
            } else if (engramNumCpuBytes_ == 0) {
                engramNumCpuBytes_ = AlignTo(static_cast<int64_t>(storage.nbytes()), BUFFER_ALIGNMENT);
            }
        }
    } else if (withGrad_) {
        TORCH_CHECK(storage.nbytes() > 0, "engram_write in with_grad mode requires non-empty storage, got ",
                    storage.nbytes(), " bytes");
        if (engramContextInitialized_) {
            CheckExternalStorageUnchanged(storage, true, false);
        } else {
            externalHostPtr = storage.data_ptr();
            externalBytes = static_cast<int64_t>(storage.nbytes());
            // 空 SF tensor 归一化为无 SF：避免首建登记 sfExternal=true 但未注册映射，导致重建一致性校验必失败
            if (sf.has_value() && sf.value().nbytes() > 0) {
                sfHostPtr = sf.value().data_ptr();
                sfBytes = static_cast<int64_t>(sf.value().nbytes());
            }
        }
    } else {
        if (engramContextInitialized_) {
            CheckExternalStorageUnchanged(storage, zeroCopy, false);
        } else if (zeroCopy) {
            externalHostPtr = storage.data_ptr();
            externalBytes = registrationBytes;
        }
    }
    EnsureEngramContext(externalHostPtr, externalBytes, sfHostPtr, sfBytes);

    if (!withGrad_ && !engramStorageExternal_) {
        TORCH_CHECK(storage.nbytes() <= static_cast<size_t>(engramNumCpuBytes_), "storage size ", storage.nbytes(),
                    " exceeds buffer capacity ", engramNumCpuBytes_);
    }

    constexpr int64_t int32Max = static_cast<int64_t>(INT32_MAX);
    if (localFullVocab_) {
        TORCH_CHECK(storage.size(0) <= int32Max,
                    "num_entries must not exceed INT32_MAX in full-vocab mode, got num_entries=", storage.size(0));
    } else {
        const int64_t rankSizeForBound = static_cast<int64_t>(GetRankSize());
        TORCH_CHECK(storage.size(0) * rankSizeForBound <= int32Max,
                    "num_entries * rank_size must not exceed INT32_MAX, got num_entries=", storage.size(0),
                    ", rank_size=", rankSizeForBound, ", product=", storage.size(0) * rankSizeForBound);
    }

    if (!localFullVocab_) {
        EngramBarrier(false, true);
    }

    engramHiddenSize_ = storage.size(1);
    engramNumEntries_ = storage.size(0);
    engramDtype_ = storage.scalar_type();

    if (!withGrad_ && !engramStorageExternal_ && engramNumEntries_ > 0) {
        constexpr size_t MEMCPY_MAX_BYTES = 0x7fffffff;
        size_t totalBytes = storage.nbytes();
        size_t remaining = totalBytes;
        uint8_t* dst = static_cast<uint8_t*>(engramHostBufPtr_);
        const uint8_t* src = static_cast<const uint8_t*>(storage.data_ptr());
        while (remaining > 0) {
            size_t chunkSize = std::min(remaining, MEMCPY_MAX_BYTES);
            if (hybridMode_ == NETWORK_HYBRID) {
                aclError copyRet = aclrtMemcpy(dst, chunkSize, src, chunkSize, ACL_MEMCPY_HOST_TO_HOST);
                TORCH_CHECK(copyRet == ACL_SUCCESS, "aclrtMemcpy(rail write) failed, ret=", static_cast<int>(copyRet),
                            ", chunkSize=", chunkSize);
            } else {
                errno_t memRet = memcpy_s(dst, chunkSize, src, chunkSize);
                TORCH_CHECK(memRet == EOK, "memcpy_s failed, ret=", memRet, ", offset=", totalBytes - remaining,
                            ", chunkSize=", chunkSize);
            }
            dst += chunkSize;
            src += chunkSize;
            remaining -= chunkSize;
        }
    }

    if (!localFullVocab_) {
        EngramBarrier(false, true);
    }

    engramWriteCalled_ = true;
}

// EngramFetch - stateless static method for torch CustomOp registration (graph-mode compatible).
ElasticBuffer::EngramFetchOutput ElasticBuffer::EngramFetch(const at::Tensor& context, const at::Tensor& indices,
                                                            int64_t hiddenSize, int64_t numEntries, int64_t dtypeEnum,
                                                            const at::Tensor& sfTable, int64_t hybridMode)
{
    auto dtype = static_cast<at::ScalarType>(dtypeEnum);
    int64_t numTokens = indices.size(0);
    auto fetched = at::empty({numTokens, hiddenSize}, at::TensorOptions().dtype(dtype).device(indices.device()));
    int64_t sfPacks = (sfTable.dim() == 2 && sfTable.size(1) > 0) ? sfTable.size(1) : 0;
    auto fetchedSf =
        sfPacks > 0 ? at::empty({numTokens, sfPacks}, sfTable.options()) : at::empty({0}, sfTable.options());
    if (numTokens == 0) {
        return std::make_tuple(fetched, fetchedSf);
    }
    aclTensor* nullTensor = nullptr;
    int64_t zero = 0;
    ACLNN_CMD(aclnnEngramFetch, context, indices, nullTensor, sfTable, fetched, nullTensor, nullTensor, nullTensor,
              nullTensor, nullTensor, fetchedSf, hiddenSize, numEntries, zero, zero, zero, hybridMode, zero);
    return {fetched, fetchedSf};
}

// EngramFetchTrain - stateless static method for training forward (graph-mode compatible).
// Outputs: fetched, fetchedSf, and save-for-backward ctx tensors (perm, sendCounts, recvCounts, recvLocalEntry,
// numRecv). The scalar SF metadata provides the output shape/dtype; sfTableAddr
// remains the training kernel's mapped host-table address.
ElasticBuffer::EngramFetchTrainOutput ElasticBuffer::EngramFetchTrain(
    const at::Tensor& context, const at::Tensor& indices, int64_t hiddenSize, int64_t numEntries, int64_t dtypeEnum,
    const at::Tensor& localStorageAddr, at::ScalarType sfDtype, int64_t sfPacks, const at::Tensor& sfTableAddr,
    int64_t numMaxTokensPerRank, int64_t commBufferSize, int64_t rankSize, int64_t ranksPerSupernode)
{
    auto dtype = static_cast<at::ScalarType>(dtypeEnum);
    TORCH_CHECK(rankSize > 0 && numMaxTokensPerRank <= INT64_MAX / rankSize,
                "numMaxTokensPerRank * rankSize overflow, got numMaxTokensPerRank=", numMaxTokensPerRank,
                ", rankSize=", rankSize);
    TORCH_CHECK(sfPacks >= 0, "sf_packs must be non-negative, got ", sfPacks);
    TORCH_CHECK(sfDtype == at::kFloat || sfDtype == at::kFloat8_e8m0fnu,
                "sf_dtype must be float32 or float8_e8m0fnu, got ", sfDtype);
    int64_t numTokens = indices.size(0);
    auto fetched = at::empty({numTokens, hiddenSize}, at::TensorOptions().dtype(dtype).device(indices.device()));
    auto intOpts = at::TensorOptions().dtype(at::kInt).device(indices.device());
    int64_t maxR = numMaxTokensPerRank * rankSize;
    at::Tensor perm = at::zeros({numTokens}, intOpts);
    at::Tensor sendCounts = at::zeros({rankSize * SEND_COUNTS_ALIGN_FACTOR}, intOpts); // 32b对齐
    at::Tensor recvCounts = at::zeros({rankSize}, intOpts);
    at::Tensor recvLocalEntry = at::zeros({maxR}, intOpts);
    at::Tensor numRecv = at::zeros({1}, intOpts);
    auto sfOpts = at::TensorOptions().dtype(sfDtype).device(indices.device());
    at::Tensor fetchedSf = sfPacks > 0 ? at::empty({numTokens, sfPacks}, sfOpts) : at::empty({0}, sfOpts);

    if (numTokens > 0) {
        constexpr int64_t withGrad = 1;
        int64_t zeroHybridMode = 0; // 训练不走推理分轨（hybrid_mode=0）
        ACLNN_CMD(aclnnEngramFetch, context, indices, localStorageAddr, sfTableAddr, fetched, perm, sendCounts,
                  recvCounts, recvLocalEntry, numRecv, fetchedSf, hiddenSize, numEntries, numMaxTokensPerRank,
                  commBufferSize, withGrad, zeroHybridMode, ranksPerSupernode);
    }
    return std::make_tuple(fetched, fetchedSf, perm, sendCounts, recvCounts, recvLocalEntry, numRecv);
}

// EngramFetchWait - stateless static method for torch CustomOp registration.
at::Tensor ElasticBuffer::EngramFetchWait(const at::Tensor& context, const at::Tensor& fetched)
{
    if (fetched.size(0) == 0) {
        return fetched;
    }
    ACLNN_CMD(aclnnEngramFetchWait, context, fetched);
    return fetched;
}

// EngramFetchGrad - stateless static method for training backward (graph-mode compatible).
ElasticBuffer::EngramFetchGradOutput ElasticBuffer::EngramFetchGrad(
    const at::Tensor& context, const at::Tensor& gradFetched, const at::Tensor& perm, const at::Tensor& sendCounts,
    const at::Tensor& recvCounts, const at::Tensor& recvLocalEntry, const at::Tensor& numRecv, int64_t numEntries,
    int64_t commBufferSize, int64_t numMaxTokensPerRank, int64_t rankSize, int64_t ranksPerSupernode)
{
    int64_t hidden = gradFetched.size(1);
    TORCH_CHECK(rankSize > 0 && numMaxTokensPerRank <= INT64_MAX / rankSize,
                "numMaxTokensPerRank * rankSize overflow, got numMaxTokensPerRank=", numMaxTokensPerRank,
                ", rankSize=", rankSize);
    int64_t maxR = numMaxTokensPerRank * rankSize;

    auto gradUnique =
        at::empty({maxR, hidden}, at::TensorOptions().dtype(gradFetched.dtype()).device(gradFetched.device()));
    auto uniqueLocalEntry = at::empty({maxR}, at::TensorOptions().dtype(at::kInt).device(gradFetched.device()));
    auto numUnique = at::empty({1}, at::TensorOptions().dtype(at::kInt).device(gradFetched.device()));

    ACLNN_CMD(aclnnEngramFetchGrad, context, gradFetched, perm, sendCounts, recvCounts, recvLocalEntry, numRecv,
              gradUnique, uniqueLocalEntry, numUnique, numEntries, commBufferSize, ranksPerSupernode);

    return std::make_tuple(gradUnique, uniqueLocalEntry, numUnique);
}

// EngramBarrier - cross-rank synchronization
void ElasticBuffer::EngramBarrier(bool useCommStream, bool withCpuSync)
{
    TORCH_CHECK(!destroyed_,
                "engram_barrier cannot be called after destroy, please create a new ElasticBuffer instance");
    TORCH_CHECK(!localFullVocab_, "engram_barrier is not supported in local full-vocab mode");
    EnsureEngramContext();
    TORCH_CHECK(engramHcclComm_ != nullptr, "HCCL comm not initialized");

    aclrtStream computeStream = c10_npu::getCurrentNPUStream().stream(false);
    aclrtStream stream = useCommStream ? commStream_ : computeStream;

    if (useCommStream) {
        NpuStreamWait(commStream_, computeStream);
    }

    if (withCpuSync) {
        aclError aclRet = aclrtSynchronizeDevice();
        TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed, ret: ", aclRet);
    }

    HcclResult ret = HcclBarrierFunc(engramHcclComm_, stream);
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclBarrier failed, ret: ", ret);

    if (engramRailContext_) {
        ReZeroRailWindowHead(stream);
    }

    if (withCpuSync) {
        aclError aclRet = aclrtSynchronizeDevice();
        TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed, ret: ", aclRet);
    }

    if (useCommStream) {
        NpuStreamWait(computeStream, commStream_);
    }
}

void ElasticBuffer::ReZeroRailWindowHead(aclrtStream stream)
{
    if (!engramRailContext_) {
        return;
    }
    void* winBase = reinterpret_cast<void*>(engramCommContext_.virtualAddrList[engramCommContext_.rankId]);
    TORCH_CHECK(winBase != nullptr, "rail window head reset: null protocol win base");
    const uint32_t ranksPerSupernode = GetRanksPerSupernode();
    TORCH_CHECK(ranksPerSupernode > 0U && static_cast<uint64_t>(engramWorldSize_) % ranksPerSupernode == 0,
                "rail window head reset: invalid topology ranksPerSupernode=", ranksPerSupernode,
                ", world=", engramWorldSize_);
    const uint64_t headBytes =
        EngramContextBuilder::WinHeadBytes(ranksPerSupernode, static_cast<uint32_t>(engramWorldSize_));
    TORCH_CHECK(commBufferSize_ >= static_cast<int64_t>(headBytes), "rail window head reset: commBufferSize(",
                commBufferSize_, ") < win head(", headBytes, ")");
    aclError ar =
        aclrtMemsetAsync(winBase, static_cast<size_t>(commBufferSize_), 0, static_cast<size_t>(headBytes), stream);
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtMemsetAsync(rail win head ", headBytes, "B) failed, ret=", ar);
}

// Destroy - explicit resource cleanup
void ElasticBuffer::Destroy()
{
    if (destroyed_) {
        return;
    }

    if (localFullVocab_) {
        aclError syncRet = aclrtSynchronizeDevice();
        TORCH_CHECK(syncRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed during local destroy, ret: ", syncRet);
    } else if (hybridMode_ == NETWORK_HYBRID) {
        // rail_aligned 模式：注册内存/物理内存/导入句柄均无与正常路径一致的释放通道，按约定不释放
        ASCEND_LOGI("rail_aligned destroy: leak rail resources by design");
    } else {
        try {
            EngramBarrier(true, true);
        } catch (const std::exception& e) {
            ASCEND_LOGE("EngramBarrier in Destroy failed: %s", e.what());
        }
    }

    // HCCL 默认 buffer 由框架管理，无需手动释放
    commBufferSize_ = 0;

    // MoE 通信 buffer 已按 tag 注册进 HCCL 通信域(HcclCommMemReg 无反注册接口)，其地址被
    // 引擎 ctx(含其他 rank 缓存的 epHcclBuffer[])引用且无法销毁：Destroy 仅解除本实例引用，
    // 物理内存由共享池按 tag 保留，同 group 重建 ElasticBuffer 时复用(容量以首建为准)。
    moeDeviceBufPtr_ = nullptr;
    moePhysicalMemHandle_ = nullptr;
    moeMemHandle_ = nullptr;
    moeContextTag_.clear();
    moeCclBufferSize_ = 0;

    // 注册内存与 host buffer 已按 tag 入进程级共享池(引擎 ctx 的 virtualAddrList[] 引用且无法
    // 反注册)：Destroy 仅解除本实例引用、不释放内存，同 group 重建 ElasticBuffer 时复用。
    if (localFullVocab_ && engramHostBufPtr_ != nullptr) {
        if (!localMappingReleased_) {
            ReleaseLocalHostMapping(engramHostBufPtr_,
                                    engramStorageExternal_ ? engramExternalBytes_ : engramNumCpuBytes_);
            localMappingReleased_ = true;
        }
        if (!engramStorageExternal_) {
            aclError ret = aclrtFreeHost(engramHostBufPtr_);
            TORCH_CHECK(ret == ACL_SUCCESS, "aclrtFreeHost failed, ret: ", ret);
        }
        engramHostBufPtr_ = nullptr;
        engramDeviceBufPtr_ = nullptr;
    } else if (engramHostBufPtr_ != nullptr) {
        if (!engramBufferPooled_) {
            if (!engramStorageExternal_ || engramExternalRegisteredByUs_) {
                aclError ret = aclrtHostUnregister(engramHostBufPtr_);
                TORCH_CHECK(ret == ACL_SUCCESS, "aclrtHostUnregister failed, ret: ", ret);
            }
            if (!engramStorageExternal_) {
                aclError ret = aclrtFreeHost(engramHostBufPtr_);
                TORCH_CHECK(ret == ACL_SUCCESS, "aclrtFreeHost failed, ret: ", ret);
            }
        }
        engramHostBufPtr_ = nullptr;
        engramDeviceBufPtr_ = nullptr;
        engramStorageExternal_ = false;
        engramExternalRegisteredByUs_ = false;
        engramExternalBytes_ = 0;
        engramBufferPooled_ = false;
    }
    engramContextInitialized_ = false;
    moeContextInitialized_ = false;
    engramContextTensor_ = at::Tensor();
    localStorageAddrTensor_ = at::Tensor();
    localSfTableTensor_ = at::Tensor();
    moeContextTensor_ = at::Tensor();

    // SF 注册内存同样按 tag 入进程级共享池，Destroy 仅解除本实例引用、不释放内存。
    sfDeviceBufPtr_ = nullptr;
    sfExternalRegistered_ = false;

    destroyed_ = true;
}

// GetEngramStorageSizeHint - calculate recommended CPU buffer size (static method)
int64_t ElasticBuffer::GetEngramStorageSizeHint(int64_t numEntries, int64_t hiddenSize, at::ScalarType dtype)
{
    int64_t dtypeSize = at::elementSize(dtype);
    TORCH_CHECK(hiddenSize <= INT64_MAX / dtypeSize, "hiddenSize * dtypeSize overflow");
    int64_t hiddenSizeBytes = hiddenSize * dtypeSize;
    int64_t numSfPacks = (dtypeSize <= 1) ? CeilDiv(hiddenSize, 32) : 0;
    TORCH_CHECK(hiddenSizeBytes <= INT64_MAX - numSfPacks * 4, "numBytesPerEntry addition overflow");
    int64_t numBytesPerEntry = AlignTo(hiddenSizeBytes + numSfPacks * 4, 32);
    TORCH_CHECK(numBytesPerEntry > 0 && numEntries <= INT64_MAX / numBytesPerEntry,
                "numBytesPerEntry * numEntries overflow");
    int64_t numCpuBytes = AlignTo(numBytesPerEntry * numEntries, BUFFER_ALIGNMENT);
    return numCpuBytes;
}

} // namespace Mc2Api

namespace OpApi {
namespace {

#define ACL_CHECK(expr) \
    do { \
        aclError _s = (expr); \
        if (_s != ACL_SUCCESS) { \
            throw std::runtime_error("ACL error: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                                     " code=" + std::to_string(_s)); \
        } \
    } while (0)

} // namespace

class HostPinnedCounter {
public:
    HostPinnedCounter()
    {
        ACL_CHECK(aclrtMallocHost(&hostPtr_, 4 * sizeof(int64_t)));
        ACL_CHECK(aclrtHostRegisterV2(hostPtr_, 4 * sizeof(int64_t), ACL_HOST_REG_MAPPED));
        ACL_CHECK(aclrtHostGetDevicePointer(hostPtr_, &devPtr_, 0));
        Reset();
    }

    ~HostPinnedCounter()
    {
        if (hostPtr_ != nullptr) {
            aclrtHostUnregister(hostPtr_);
            aclrtFreeHost(hostPtr_);
            hostPtr_ = nullptr;
            devPtr_ = nullptr;
        }
    }

    void Reset()
    {
        *reinterpret_cast<volatile int64_t*>(hostPtr_) = -1;
    }

    int64_t SpinWait()
    {
        while (true) {
            int64_t v = *reinterpret_cast<volatile int64_t*>(hostPtr_);
            if (v >= 0) {
                return v;
            }
        }
    }

    uintptr_t DevicePtr() const
    {
        return reinterpret_cast<uintptr_t>(devPtr_);
    }

    uintptr_t HostPtr() const
    {
        return reinterpret_cast<uintptr_t>(hostPtr_);
    }

private:
    void* hostPtr_ = nullptr;
    void* devPtr_ = nullptr;
};

} // namespace OpApi

Mc2Api::ElasticBuffer::DispatchTensorList Mc2Api::ElasticBuffer::MoeEpDispatch(
    const at::Tensor& x, const at::Tensor& topkIdx, const c10::optional<at::Tensor>& topkWeights,
    const c10::optional<at::Tensor>& scales, const c10::optional<at::Tensor>& cachedDstSlotIdx,
    const c10::optional<at::Tensor>& cachedRouteCount, const c10::optional<at::Tensor>& cachedRouteDstScaleout,
    const c10::optional<at::Tensor>& cachedRouteScaleoutSlot, int64_t epWorldSize, int64_t epRankId, int64_t numExperts,
    int64_t numMaxTokensPerRank, int64_t expertAlignment, bool doCpuSync, int64_t hostPinnedCounterAddr,
    int64_t cclBufferSize)
{
    TORCH_CHECK(x.dim() == DIM_TWO, "x dims must be 2, but got ", x.dim());
    TORCH_CHECK(topkIdx.dim() == DIM_TWO, "topk_idx dims must be 2, but got ", topkIdx.dim());
    TORCH_CHECK(epWorldSize > 0, "ep_world_size must be positive, got ", epWorldSize);
    EnsureMoeContext(cclBufferSize, epWorldSize);
    int64_t rankNumPerServer = ResolveRankNumPerServer(epWorldSize);
    int64_t topoType = ResolveTopoType(epWorldSize, rankNumPerServer);

    bool anyCached = cachedDstSlotIdx.has_value();
    TORCH_CHECK(!(anyCached && doCpuSync), "cached mode is incompatible with do_cpu_sync=True");
    bool anyCachedRoute =
        cachedRouteCount.has_value() || cachedRouteDstScaleout.has_value() || cachedRouteScaleoutSlot.has_value();
    bool allCachedRoute =
        cachedRouteCount.has_value() && cachedRouteDstScaleout.has_value() && cachedRouteScaleoutSlot.has_value();
    TORCH_CHECK(!anyCachedRoute || allCachedRoute, "cached route tensors must be all present or all absent");
    bool isDirect = (topoType == NETWORK_DIRECT);
    bool hybridCached = anyCached && topoType == NETWORK_HYBRID;
    TORCH_CHECK(!hybridCached || allCachedRoute, "hybrid cached dispatch requires all cached route tensors");

    auto xSize = x.sizes();
    int64_t numTokens = xSize[0];
    int64_t topK = topkIdx.size(1);
    int64_t numLocalExperts = numExperts / epWorldSize;
    int64_t routeCapacity = topK;

    at::Tensor numRecvPerRank = at::empty({epWorldSize}, x.options().dtype(at::kInt));
    at::Tensor numRecvPerExpert = at::empty({numLocalExperts}, x.options().dtype(at::kLong));
    at::Tensor dstSlot = isDirect ? at::full({epWorldSize, numTokens + 1}, -1, x.options().dtype(at::kInt)) :
                                    at::full({numTokens, topK}, -1, x.options().dtype(at::kInt));
    at::Tensor routeCount = at::zeros({numTokens}, x.options().dtype(at::kInt));
    at::Tensor routeDstScaleout = at::full({numTokens, routeCapacity}, -1, x.options().dtype(at::kInt));
    at::Tensor routeScaleoutSlot = at::full({numTokens, routeCapacity}, -1, x.options().dtype(at::kInt));

    at::Tensor topkWeightsTensor = topkWeights.has_value() ? *topkWeights : at::Tensor();
    at::Tensor cachedSlotTensor = cachedDstSlotIdx.has_value() ? *cachedDstSlotIdx : at::Tensor();
    at::Tensor cachedRouteCountTensor = cachedRouteCount.has_value() ? *cachedRouteCount : at::Tensor();
    at::Tensor cachedRouteDstScaleoutTensor =
        cachedRouteDstScaleout.has_value() ? *cachedRouteDstScaleout : at::Tensor();
    at::Tensor cachedRouteScaleoutSlotTensor =
        cachedRouteScaleoutSlot.has_value() ? *cachedRouteScaleoutSlot : at::Tensor();

    at::Tensor scalesTensor = scales.has_value() ? *scales : at::Tensor();
    aclDataType scalesDtype = (scales.has_value() && scalesTensor.scalar_type() == at::kByte) ?
                                  aclDataType::ACL_FLOAT8_E8M0 :
                                  ConvertToAclDataType(scales.has_value() ? scalesTensor.scalar_type() : at::kFloat);
    TensorWrapper scalesWrapper = TensorWrapper{scalesTensor, scalesDtype};
    ACLNN_CMD(aclnnMoeEpDispatch, moeContextTensor_, x, topkIdx, topkWeightsTensor, scalesWrapper, cachedSlotTensor,
              cachedRouteCountTensor, cachedRouteDstScaleoutTensor, cachedRouteScaleoutSlotTensor, epWorldSize,
              epRankId, numExperts, numMaxTokensPerRank, moeCclBufferSize_, expertAlignment, doCpuSync,
              hostPinnedCounterAddr, topoType, rankNumPerServer, moeCommAlg_, numRecvPerRank, numRecvPerExpert, dstSlot,
              routeCount, routeDstScaleout, routeScaleoutSlot);

    return std::make_tuple(numRecvPerRank, numRecvPerExpert, dstSlot, routeCount, routeDstScaleout, routeScaleoutSlot);
}

Mc2Api::ElasticBuffer::DispatchEpilogueTensorList Mc2Api::ElasticBuffer::MoeEpDispatchEpilogue(
    const at::Tensor& x, const at::Tensor& topkIdx, const at::Tensor& numRecvPerRank,
    const at::Tensor& numRecvPerExpert, const c10::optional<at::Tensor>& cachedRecvSrcMetadata, int64_t epWorldSize,
    int64_t epRankId, int64_t numExperts, int64_t numMaxTokensPerRank, int64_t cclBufferSize, at::Tensor& recvX,
    at::Tensor& recvSrcMetadata, const c10::optional<at::Tensor>& recvTopkWeightsOpt,
    const c10::optional<at::Tensor>& recvScalesOpt)
{
    TORCH_CHECK(x.dim() == DIM_TWO, "x dims must be 2, but got ", x.dim());
    TORCH_CHECK(recvX.dim() == DIM_TWO, "recv_x dims must be 2, but got ", recvX.dim());
    CheckMoeEpMetadataTensor(recvSrcMetadata, "recv_src_metadata", recvX.size(0), epWorldSize, topkIdx.numel(),
                             recvX.device());
    if (cachedRecvSrcMetadata.has_value()) {
        CheckMoeEpMetadataTensor(*cachedRecvSrcMetadata, "cached_recv_src_metadata", recvX.size(0), epWorldSize,
                                 topkIdx.numel(), recvX.device());
    }

    EnsureMoeContext(cclBufferSize, epWorldSize);
    int64_t rankNumPerServer = ResolveRankNumPerServer(epWorldSize);
    int64_t topoType = ResolveTopoType(epWorldSize, rankNumPerServer);

    at::Tensor cachedRecvSrcMetadataTensor = cachedRecvSrcMetadata.has_value() ? *cachedRecvSrcMetadata : at::Tensor();

    aclDataType recvScalesDtype = aclDataType::ACL_FLOAT;
    at::Tensor recvScalesTensor =
        recvScalesOpt.has_value() ? *recvScalesOpt : at::empty({1}, recvX.options().dtype(at::kFloat));
    if (recvScalesOpt.has_value() && recvScalesTensor.scalar_type() == at::kByte) {
        recvScalesDtype = aclDataType::ACL_FLOAT8_E8M0;
    }

    TensorWrapper recvScalesWrapper = TensorWrapper{recvScalesTensor, recvScalesDtype};

    at::Tensor recvTopkWeightsTensor =
        recvTopkWeightsOpt.has_value() ? *recvTopkWeightsOpt : at::empty({1}, recvX.options().dtype(at::kFloat));
    bool hasTopkWeights = recvTopkWeightsOpt.has_value();
    ACLNN_CMD(aclnnMoeEpDispatchEpilogue, moeContextTensor_, x, topkIdx, numRecvPerRank, numRecvPerExpert,
              cachedRecvSrcMetadataTensor, epWorldSize, epRankId, numExperts, numMaxTokensPerRank, moeCclBufferSize_,
              hasTopkWeights, topoType, rankNumPerServer, moeCommAlg_, recvX, recvSrcMetadata, recvTopkWeightsTensor,
              recvScalesWrapper);

    c10::optional<at::Tensor> recvTopkWeightsOutput;
    if (recvTopkWeightsOpt.has_value()) {
        recvTopkWeightsOutput = *recvTopkWeightsOpt;
    }
    c10::optional<at::Tensor> recvScalesOutput;
    if (recvScalesOpt.has_value()) {
        recvScalesOutput = *recvScalesOpt;
    }
    return std::make_tuple(recvX, recvSrcMetadata, recvTopkWeightsOutput, recvScalesOutput);
}

void Mc2Api::ElasticBuffer::MoeEpCombine(const at::Tensor& x, const at::Tensor& topkIdx,
                                         const at::Tensor& recvSrcMetadata, const at::Tensor& numRecvTokensPerExpert,
                                         const c10::optional<at::Tensor>& topkWeights, int64_t epWorldSize,
                                         int64_t epRankId, int64_t numExperts, int64_t numMaxTokensPerRank,
                                         int64_t cclBufferSize)
{
    TORCH_CHECK(x.dim() == DIM_TWO, "x dims must be 2, but got ", x.dim());
    TORCH_CHECK(topkIdx.dim() == DIM_TWO, "topk_idx dims must be 2, but got ", topkIdx.dim());
    CheckMoeEpMetadataTensor(recvSrcMetadata, "recv_src_metadata", x.size(0), epWorldSize, topkIdx.numel(), x.device());
    EnsureMoeContext(cclBufferSize, epWorldSize);
    int64_t rankNumPerServer = ResolveRankNumPerServer(epWorldSize);
    int64_t topoType = ResolveTopoType(epWorldSize, rankNumPerServer);

    c10::optional<at::Tensor> topkWeightsOpt = topkWeights;
    ACLNN_CMD(aclnnMoeEpCombine, moeContextTensor_, x, topkIdx, recvSrcMetadata, numRecvTokensPerExpert, topkWeightsOpt,
              epWorldSize, epRankId, numExperts, numMaxTokensPerRank, moeCclBufferSize_, topoType, rankNumPerServer,
              moeCommAlg_);
}

Mc2Api::ElasticBuffer::CombineEpilogueTensorList Mc2Api::ElasticBuffer::MoeEpCombineEpilogue(
    const at::Tensor& x, const at::Tensor& topkIdx, const at::Tensor& recvSrcMetadata,
    const c10::optional<at::Tensor>& topkWeights, int64_t epWorldSize, int64_t epRankId, int64_t numExperts,
    int64_t numMaxTokensPerRank, int64_t cclBufferSize, at::Tensor& combinedX,
    const c10::optional<at::Tensor>& combinedTopkWeightsOpt)
{
    TORCH_CHECK(x.dim() == DIM_TWO, "x dims must be 2, but got ", x.dim());
    TORCH_CHECK(topkIdx.dim() == DIM_TWO, "topk_idx dims must be 2, but got ", topkIdx.dim());
    CheckMoeEpMetadataTensor(recvSrcMetadata, "recv_src_metadata", x.size(0), epWorldSize, topkIdx.numel(), x.device());
    EnsureMoeContext(cclBufferSize, epWorldSize);
    int64_t rankNumPerServer = ResolveRankNumPerServer(epWorldSize);
    int64_t topoType = ResolveTopoType(epWorldSize, rankNumPerServer);

    bool hasTopkWeights = topkWeights.has_value();
    at::Tensor combinedTopkWeightsTensor = combinedTopkWeightsOpt.has_value() ?
                                               *combinedTopkWeightsOpt :
                                               at::empty({1}, combinedX.options().dtype(at::kFloat));
    ACLNN_CMD(aclnnMoeEpCombineEpilogue, moeContextTensor_, x, topkIdx, recvSrcMetadata, topkWeights, epWorldSize,
              epRankId, numExperts, numMaxTokensPerRank, moeCclBufferSize_, hasTopkWeights, topoType, rankNumPerServer,
              moeCommAlg_, combinedX, combinedTopkWeightsTensor);

    c10::optional<at::Tensor> combinedTopkWeightsOutput;
    if (hasTopkWeights) {
        combinedTopkWeightsOutput = *combinedTopkWeightsOpt;
    }
    return std::make_tuple(combinedX, combinedTopkWeightsOutput);
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<OpApi::HostPinnedCounter>(m, "HostPinnedCounter")
        .def(pybind11::init<>())
        .def("spin_wait", &OpApi::HostPinnedCounter::SpinWait)
        .def("reset", &OpApi::HostPinnedCounter::Reset)
        .def("device_ptr", &OpApi::HostPinnedCounter::DevicePtr)
        .def("host_ptr", &OpApi::HostPinnedCounter::HostPtr);

    pybind11::class_<Mc2Api::ElasticBuffer>(m, "ElasticBuffer")
        .def(pybind11::init<const std::optional<std::string>&, int64_t, int64_t, bool, bool, const std::string&,
                            const std::string&, int64_t, int64_t>(),
             pybind11::arg("groupName"), pybind11::arg("numCpuBytes"), pybind11::arg("numMaxTokensPerRank") = 0,
             pybind11::arg("withGrad") = false, pybind11::arg("explicitlyDestroy") = false,
             pybind11::arg("netLayer") = "auto", pybind11::arg("moeCommAlg") = Mc2Api::MOE_COMM_ALG_DEFAULT,
             pybind11::arg("allowHybridMode") = 0, pybind11::arg("worldSize") = 0)
        .def("get_ranks_per_supernode",
             [](const Mc2Api::ElasticBuffer& self) -> int64_t {
                 return static_cast<int64_t>(self.GetRanksPerSupernode());
             })
        .def("engram_write", &Mc2Api::ElasticBuffer::EngramWrite, pybind11::arg("storage").noconvert(),
             pybind11::arg("sf") = pybind11::none(), pybind11::arg("zeroCopy"))
        .def_static("engram_fetch",
                    static_cast<Mc2Api::ElasticBuffer::EngramFetchOutput (*)(
                        const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, const at::Tensor&, int64_t)>(
                        &Mc2Api::ElasticBuffer::EngramFetch),
                    pybind11::arg("context"), pybind11::arg("indices"), pybind11::arg("hidden_size"),
                    pybind11::arg("num_entries"), pybind11::arg("dtype"), pybind11::arg("sf_table"),
                    pybind11::arg("hybrid_mode"))
        .def_static("engram_fetch_train",
                    static_cast<Mc2Api::ElasticBuffer::EngramFetchTrainOutput (*)(
                        const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, const at::Tensor&,
                        at::ScalarType, int64_t, const at::Tensor&, int64_t, int64_t, int64_t, int64_t)>(
                        &Mc2Api::ElasticBuffer::EngramFetchTrain),
                    pybind11::arg("context"), pybind11::arg("indices"), pybind11::arg("hidden_size"),
                    pybind11::arg("num_entries"), pybind11::arg("dtype"), pybind11::arg("local_storage_addr"),
                    pybind11::arg("sf_dtype"), pybind11::arg("sf_packs"), pybind11::arg("sf_table_addr"),
                    pybind11::arg("num_max_tokens_per_rank"), pybind11::arg("comm_buffer_size"),
                    pybind11::arg("rank_size"), pybind11::arg("ranks_per_supernode") = 0)
        .def_static("engram_fetch_wait", &Mc2Api::ElasticBuffer::EngramFetchWait, pybind11::arg("context"),
                    pybind11::arg("fetched"))
        .def_static("engram_fetch_grad_op",
                    static_cast<Mc2Api::ElasticBuffer::EngramFetchGradOutput (*)(
                        const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t)>(
                        &Mc2Api::ElasticBuffer::EngramFetchGrad),
                    pybind11::arg("context"), pybind11::arg("gradFetched").noconvert(),
                    pybind11::arg("perm").noconvert(), pybind11::arg("sendCounts").noconvert(),
                    pybind11::arg("recvCounts").noconvert(), pybind11::arg("recvLocalEntry").noconvert(),
                    pybind11::arg("numRecv").noconvert(), pybind11::arg("numEntries"), pybind11::arg("commBufferSize"),
                    pybind11::arg("numMaxTokensPerRank"), pybind11::arg("rankSize"),
                    pybind11::arg("ranksPerSupernode") = 0)
        .def("engram_barrier", &Mc2Api::ElasticBuffer::EngramBarrier, pybind11::arg("useCommStream") = true,
             pybind11::arg("withCpuSync") = false)
        .def("destroy", &Mc2Api::ElasticBuffer::Destroy)
        .def("get_context_tensor", &Mc2Api::ElasticBuffer::GetContextTensor)
        .def("engram_context_initialized", &Mc2Api::ElasticBuffer::IsEngramContextInitialized)
        .def("get_local_storage_addr", &Mc2Api::ElasticBuffer::GetLocalStorageAddrTensor)
        .def("get_local_sf_table_addr", &Mc2Api::ElasticBuffer::GetLocalSfTableAddrTensor)
        .def("get_comm_buffer_size", &Mc2Api::ElasticBuffer::GetCommBufferSize)
        .def("get_rank_size", &Mc2Api::ElasticBuffer::GetRankSize)
        .def("moe_ep_dispatch", &Mc2Api::ElasticBuffer::MoeEpDispatch)
        .def("moe_ep_dispatch_epilogue", &Mc2Api::ElasticBuffer::MoeEpDispatchEpilogue)
        .def("moe_ep_combine", &Mc2Api::ElasticBuffer::MoeEpCombine)
        .def("moe_ep_combine_epilogue", &Mc2Api::ElasticBuffer::MoeEpCombineEpilogue)
        .def_static("get_engram_storage_size_hint", &Mc2Api::ElasticBuffer::GetEngramStorageSizeHint,
                    pybind11::arg("numEntries"), pybind11::arg("hiddenSize"), pybind11::arg("dtype") = at::kBFloat16);
}
