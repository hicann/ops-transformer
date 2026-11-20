/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <algorithm>
#include <array>
#include <dlfcn.h>
#include <new>
#include <numeric>
#include <map>
#include <cstring>
#include <string>
#include <vector>
#include "securec.h"
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "acl/acl_rt.h"
#include "../../../op_api/aclnn_distribute_barrier.h"
#include "../../../op_api/aclnn_distribute_barrier_v2.h"
#include "../../../op_api/distribute_barrier_base.h"
#include "aclnnInner_distribute_barrier_extend.h"
#include "../../../../common/utils/op_mc2_def.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

#include "../../../../common/op_api/mc2_context.h"

using namespace op;
using namespace std;

namespace {
const aclTensor* g_forwardedContext = nullptr;
const aclTensor* g_forwardedX = nullptr;
const aclTensor* g_forwardedTimeout = nullptr;
const aclTensor* g_forwardedElasticInfo = nullptr;
std::string g_forwardedGroup;
int64_t g_forwardedWorldSize = 0;
uint32_t g_extendCalls = 0;
uint32_t g_extendSecondCalls = 0;
uint32_t g_contextBuildCalls = 0;
const aclTensor* g_builtContext = nullptr;
aclnnStatus g_contextStatus = ACLNN_SUCCESS;

struct BarrierCaptureRuntime;
thread_local BarrierCaptureRuntime* g_barrierCaptureRuntime = nullptr;

struct BarrierCaptureRuntime {
    BarrierCaptureRuntime()
    {
        g_barrierCaptureRuntime = this;
    }
    ~BarrierCaptureRuntime()
    {
        g_barrierCaptureRuntime = nullptr;
    }
    aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
    aclError clearStatus = ACL_SUCCESS;
    uint32_t clears = 0;
};
} // namespace

extern "C" aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode* mode)
{
    if (g_barrierCaptureRuntime == nullptr) {
        static auto original = reinterpret_cast<decltype(&aclmdlRICaptureThreadExchangeMode)>(
            dlsym(RTLD_NEXT, "aclmdlRICaptureThreadExchangeMode"));
        return original(mode);
    }
    std::swap(*mode, g_barrierCaptureRuntime->mode);
    return ACL_SUCCESS;
}

extern "C" aclError aclrtMemset(void* memory, size_t capacity, int32_t value, size_t bytes)
{
    if (g_barrierCaptureRuntime == nullptr) {
        static auto original = reinterpret_cast<decltype(&aclrtMemset)>(dlsym(RTLD_NEXT, "aclrtMemset"));
        return original(memory, capacity, value, bytes);
    }
    auto& runtime = *g_barrierCaptureRuntime;
    ++runtime.clears;
    if (runtime.mode != ACL_MODEL_RI_CAPTURE_MODE_RELAXED) {
        return ACL_ERROR_RT_CAPTURE_MODE_NOT_SUPPORT;
    }
    return runtime.clearStatus == ACL_SUCCESS && memset_s(memory, capacity, value, bytes) != EOK ?
               ACL_ERROR_RT_PARAM_INVALID :
               runtime.clearStatus;
}

namespace Mc2Aclnn {
aclnnStatus Mc2Context::GetBarrierContextTensor(const char*, int64_t, aclTensor*& mc2Context)
{
    static auto context = TensorDesc({Mc2Barrier::ContextElems(16, 1)}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    ++g_contextBuildCalls;
    mc2Context = context.get();
    g_builtContext = mc2Context;
    return g_contextStatus;
}
} // namespace Mc2Aclnn

extern "C" aclnnStatus aclnnInnerDistributeBarrierExtendGetWorkspaceSize(const aclTensor* context, aclTensor* x,
                                                                         const aclTensor* timeout,
                                                                         const aclTensor* elasticInfo, char* group,
                                                                         int64_t worldSize, uint64_t* workspaceSize,
                                                                         aclOpExecutor** executor)
{
    ++g_extendCalls;
    g_forwardedContext = context;
    g_forwardedX = x;
    g_forwardedTimeout = timeout;
    g_forwardedElasticInfo = elasticInfo;
    g_forwardedGroup = group;
    g_forwardedWorldSize = worldSize;
    *workspaceSize = 0;
    *executor = nullptr;
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnInnerDistributeBarrierExtend(void*, uint64_t, aclOpExecutor*, aclrtStream)
{
    ++g_extendSecondCalls;
    return ACLNN_SUCCESS;
}

class L2AclnnDistributeBarrierTest : public testing::Test {
protected:
    void SetUp() override
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
        g_extendCalls = 0;
        g_extendSecondCalls = 0;
        g_contextBuildCalls = 0;
        g_contextStatus = ACLNN_SUCCESS;
    }

    void TearDown() override
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
    }

    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
        cout << "L2AclnnDistributeBarrierTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AclnnDistributeBarrierTest TearDown" << endl;
    }
};

namespace Mc2Aclnn {
class Mc2BarrierContextTest : public testing::Test {
protected:
    using Builder = Mc2Aclnn::Mc2Context::BarrierContext;
    struct CachedMemory {
        uint64_t bytes = 0;
        std::vector<uint64_t> data;
    };

    void SetUp() override
    {
        owner_.epRankSize_ = ranks_.size();
        owner_.HcclGetRankSize = [](HcclComm comm, uint32_t* size) {
            auto& test = *static_cast<Mc2BarrierContextTest*>(comm);
            *size = test.rankSize_;
            return test.rankStatus_;
        };
        owner_.HcclGetRankId = [](HcclComm, uint32_t* rank) {
            *rank = 0;
            return HCCL_SUCCESS;
        };
        owner_.HcclRankGraphGetLayers = [](HcclComm comm, uint32_t** layers, uint32_t* count) {
            auto& test = *static_cast<Mc2BarrierContextTest*>(comm);
            static uint32_t ids[] = {1, 0};
            ++test.layerQueries_;
            *layers = ids;
            *count = 2;
            return HCCL_SUCCESS;
        };
        owner_.HcclRankGraphGetRanksByLayer = [](HcclComm comm, uint32_t, uint32_t** ranks, uint32_t* count) {
            auto& test = *static_cast<Mc2BarrierContextTest*>(comm);
            *ranks = test.ranks_.data();
            *count = test.ranks_.size();
            return HCCL_SUCCESS;
        };
        owner_.HcclRankGraphGetLinks = [](HcclComm comm, uint32_t layer, uint32_t src, uint32_t dst, CommLink** links,
                                          uint32_t* count) {
            return static_cast<Mc2BarrierContextTest*>(comm)->GetLinks(layer, src, dst, links, count);
        };
    }

    void SetRankSize(uint32_t size)
    {
        owner_.epRankSize_ = size;
    }

    void MockContextStorage()
    {
        owner_.HcclEngineCtxGet = [](HcclComm comm, const char* tag, CommEngine engine, void** data, uint64_t* bytes) {
            auto& cache = static_cast<Mc2BarrierContextTest*>(comm)->storage_;
            auto it = cache.find({engine, tag});
            if (it == cache.end()) {
                return HCCL_E_NOT_FOUND;
            }
            *data = it->second.data.data();
            *bytes = it->second.bytes;
            return HCCL_SUCCESS;
        };
        owner_.HcclEngineCtxCreate = [](HcclComm comm, const char* tag, CommEngine engine, uint64_t bytes,
                                        void** data) {
            auto& test = *static_cast<Mc2BarrierContextTest*>(comm);
            if (test.storage_.count({engine, tag}) != 0) {
                return HCCL_E_PARA;
            }
            auto& memory = test.storage_[{engine, tag}];
            memory.bytes = bytes;
            memory.data.resize((bytes + sizeof(uint64_t) - 1U) / sizeof(uint64_t), 1234U);
            *data = memory.data.data();
            return HCCL_SUCCESS;
        };
        MockContextCopy();
    }

    void MockContextCopy()
    {
        owner_.HcclEngineCtxCopy = [](HcclComm comm, CommEngine engine, const char* tag, const void* data,
                                      uint64_t bytes, uint64_t offset) {
            auto& test = *static_cast<Mc2BarrierContextTest*>(comm);
            if (test.copyStatus_ != HCCL_SUCCESS) {
                return test.copyStatus_;
            }
            const auto found = test.storage_.find({engine, tag});
            if (found == test.storage_.end() || data == nullptr) {
                return HCCL_E_PARA;
            }
            auto& memory = found->second;
            const uint64_t capacity = memory.bytes;
            if (offset > capacity || bytes > capacity - offset) {
                return HCCL_E_PARA;
            }
            return memcpy_s(reinterpret_cast<uint8_t*>(memory.data.data()) + offset, capacity - offset, data, bytes) ==
                           EOK ?
                       HCCL_SUCCESS :
                       HCCL_E_PARA;
        };
    }

    void MockChannels()
    {
        owner_.HcclChannelAcquire = [](HcclComm comm, CommEngine engine, const HcclChannelDesc* descs, uint32_t count,
                                       ChannelHandle* handles) {
            auto& test = *static_cast<Mc2BarrierContextTest*>(comm);
            EXPECT_EQ(engine, CommEngine::COMM_ENGINE_AIV);
            EXPECT_LE(count, Mc2Barrier::CHANNEL_BATCH_SIZE);
            ++test.acquireCalls_;
            for (uint32_t i = 0; i < count; ++i) {
                handles[i] = descs[i].remoteRank + 1U;
                EXPECT_EQ(test.channels_.count(handles[i]), 0U);
                test.channels_[handles[i]] = descs[i].channelProtocol;
            }
            return HCCL_SUCCESS;
        };
        owner_.HcclChannelGetRemoteMems = [](HcclComm, ChannelHandle channel, uint32_t* count, CommMem** memories,
                                             char*** tags) {
            static CommMem memory{};
            static char tag[] = "barrier_buffer";
            static char* tagPtr = tag;
            memory.addr = reinterpret_cast<void*>(0x100000000ULL + channel * 0x200000ULL);
            memory.size = 0x200000ULL;
            *count = 1;
            *memories = &memory;
            *tags = &tagPtr;
            return HCCL_SUCCESS;
        };
    }

    Builder::State& InitCachedState(const std::string& tag, uint32_t local, uint32_t rail)
    {
        auto& memory = storage_[{CommEngine::COMM_ENGINE_CPU, tag}];
        memory.bytes = sizeof(Builder::State);
        memory.data.resize(sizeof(Builder::State) / sizeof(uint64_t));
        return *new (memory.data.data()) Builder::State{nullptr, local, rail};
    }

    uint32_t Physical(uint32_t rank)
    {
        return permuted_ ? rank * 3U % ranks_.size() : rank;
    }

    CommLink MakeLink(uint32_t src, uint32_t dst, CommProtocol protocol)
    {
        CommLink link{};
        link.linkAttr.linkProtocol = protocol;
        link.srcEndpointDesc.protocol = link.dstEndpointDesc.protocol = protocol;
        auto& source = link.srcEndpointDesc.loc;
        auto& target = link.dstEndpointDesc.loc;
        source.locType = target.locType = ENDPOINT_LOC_TYPE_DEVICE;
        source.device.devPhyId = Physical(src) % localSize_;
        target.device.devPhyId = Physical(dst) % localSize_;
        source.device.superDevId = missingIdentity_ ? Mc2Barrier::NO_RANK : source.device.devPhyId;
        target.device.superDevId = missingIdentity_ ? Mc2Barrier::NO_RANK : target.device.devPhyId;
        return link;
    }

    HcclResult GetLinks(uint32_t layer, uint32_t src, uint32_t dst, CommLink** links, uint32_t* count)
    {
        candidates_.clear();
        const uint32_t sourcePhysicalRank = Physical(src);
        const uint32_t destinationPhysicalRank = Physical(dst);
        if (dst != missingPeer_ && (layer == 0 || fallback_)) {
            if (sourcePhysicalRank / localSize_ == destinationPhysicalRank / localSize_) {
                const bool ub = layer == 0 && !(src == 0 && dst == missingUbPeer_);
                candidates_.push_back(MakeLink(src, dst, ub ? COMM_PROTOCOL_UB_MEM : COMM_PROTOCOL_ROCE));
            } else if (sourcePhysicalRank % localSize_ == destinationPhysicalRank % localSize_) {
                if (layer == 1) {
                    candidates_.push_back(MakeLink(src, dst, COMM_PROTOCOL_ROCE));
                } else {
                    for (CommProtocol protocol : protocols_) {
                        candidates_.push_back(MakeLink(src, dst, protocol));
                    }
                }
            }
        }
        *links = candidates_.data();
        *count = candidates_.size();
        return HCCL_SUCCESS;
    }

    Mc2Aclnn::Mc2Context owner_;
    std::map<std::pair<CommEngine, std::string>, CachedMemory> storage_;
    std::map<ChannelHandle, CommProtocol> channels_;
    uint32_t acquireCalls_ = 0;
    HcclResult copyStatus_ = HCCL_SUCCESS;
    std::vector<uint32_t> ranks_{7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<CommProtocol> protocols_{COMM_PROTOCOL_ROCE, COMM_PROTOCOL_UBC_CTP};
    std::vector<CommLink> candidates_;
    uint32_t localSize_ = 4;
    uint32_t rankSize_ = 8;
    HcclResult rankStatus_ = HCCL_SUCCESS;
    uint32_t missingUbPeer_ = Mc2Barrier::NO_RANK;
    uint32_t missingPeer_ = Mc2Barrier::NO_RANK;
    uint32_t layerQueries_ = 0;
    bool permuted_ = false;
    bool missingIdentity_ = false;
    bool fallback_ = true;
};

TEST_F(Mc2BarrierContextTest, SelectsNativeStaticGraphForSingleUbDomainWithoutAllocatingChannels)
{
    fallback_ = false;
    missingIdentity_ = true;
    for (uint32_t size : {2U, 8U, 16U}) {
        SCOPED_TRACE(size);
        rankSize_ = localSize_ = size;
        ranks_.resize(size);
        std::iota(ranks_.begin(), ranks_.end(), 0U);
        Builder builder{owner_, this};
        EXPECT_TRUE(builder.SupportsStaticGraph(size));
        EXPECT_EQ(builder.local.size(), size);
        EXPECT_EQ(builder.rail.size(), 1U);
        EXPECT_EQ(acquireCalls_, 0U);
        EXPECT_TRUE(storage_.empty());
    }
}

TEST_F(Mc2BarrierContextTest, KeepsIncompleteTopologyOffNativeStaticGraph)
{
    localSize_ = rankSize_;
    missingPeer_ = 1;
    Builder incomplete{owner_, this};
    EXPECT_FALSE(incomplete.SupportsStaticGraph(rankSize_));
    Builder mismatch{owner_, this};
    EXPECT_FALSE(mismatch.SupportsStaticGraph(rankSize_ - 1));
    rankStatus_ = HCCL_E_INTERNAL;
    Builder unavailable{owner_, this};
    EXPECT_FALSE(unavailable.SupportsStaticGraph(rankSize_));
}

TEST_F(Mc2BarrierContextTest, InitializesRealHostRankSizeWithoutLegacyLimit)
{
    for (uint32_t size : {1024U, 1025U, 204800U}) {
        SCOPED_TRACE(size);
        rankSize_ = size;
        Builder builder{owner_, this};
        EXPECT_EQ(builder.Init(size), ACLNN_SUCCESS);
        EXPECT_EQ(builder.rank, 0U);
    }
    Builder builder{owner_, this};
    EXPECT_EQ(builder.Init(1024), ACLNN_ERR_PARAM_INVALID);
    for (uint32_t size : {0U, 1U, 204801U}) {
        rankSize_ = size;
        EXPECT_EQ(builder.Init(size), ACLNN_ERR_PARAM_INVALID);
    }
    rankStatus_ = HCCL_E_INTERNAL;
    EXPECT_EQ(builder.Init(1024), ACLNN_ERR_INNER);
}

TEST_F(Mc2BarrierContextTest, RejectsAsymmetricFallbackWithMatchingTopologyProduct)
{
    ranks_ = {11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    SetRankSize(ranks_.size());
    missingUbPeer_ = 1;
    Builder builder{owner_, this};
    EXPECT_EQ(builder.Discover(), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(Mc2BarrierContextTest, MergesLayersAndSortsPermutedRanks)
{
    permuted_ = true;
#if HCOMM_VERSION_NUM >= HCOMM_BARRIER_RTP_MIN_VERSION
    protocols_.push_back(COMM_PROTOCOL_UB_RTP);
#endif
    Builder builder{owner_, this};
    builder.rank = 3;
    ASSERT_EQ(builder.Discover(), ACLNN_SUCCESS);
    EXPECT_EQ(builder.links.at(7).linkAttr.linkProtocol, COMM_PROTOCOL_UBC_CTP);
    EXPECT_EQ(builder.links.at(1).linkAttr.linkProtocol, COMM_PROTOCOL_UB_MEM);
    std::vector<uint64_t> storage((Mc2Barrier::ContextBytes(4, 2) + sizeof(uint64_t) - 1U) / sizeof(uint64_t),
                                  ~uint64_t{0});
    auto& context = *new (storage.data()) Mc2Barrier::Context;
    context.localSize = 4;
    context.railSize = 2;
    builder.InitPeers(context);
    EXPECT_EQ(builder.local, (std::vector<uint32_t>{0, 1, 3, 6}));
    EXPECT_EQ(builder.rail, (std::vector<uint32_t>{3, 7}));
    EXPECT_EQ(context.localSlot, 2U);
    EXPECT_EQ(context.railSlot, 0U);
    auto* peers = reinterpret_cast<Mc2Barrier::Peer*>(&context + 1);
    EXPECT_EQ(peers[context.localSlot].rank, builder.rank);
    EXPECT_EQ(peers[context.localSlot].window, 0U);
    EXPECT_EQ(peers[context.localSlot].handle, 0U);
    EXPECT_EQ(peers[context.localSlot].protocol, Mc2Barrier::Protocol::UBMEM);
    EXPECT_EQ(peers[4 + context.railSlot].rank, builder.rank);
}

#if HCOMM_VERSION_NUM >= HCOMM_BARRIER_RTP_MIN_VERSION
TEST_F(Mc2BarrierContextTest, DiscoversUbgRailOnlyOnLayerTwo)
{
    ranks_.resize(32);
    std::iota(ranks_.begin(), ranks_.end(), 0U);
    rankSize_ = 32;
    localSize_ = 16;
    missingIdentity_ = true;
    SetRankSize(rankSize_);
    owner_.HcclRankGraphGetLayers = [](HcclComm, uint32_t** layers, uint32_t* count) {
        static uint32_t ids[] = {0, 1, 2};
        *layers = ids;
        *count = 3;
        return HCCL_SUCCESS;
    };
    owner_.HcclRankGraphGetRanksByLayer = [](HcclComm comm, uint32_t layer, uint32_t** ranks, uint32_t* count) {
        auto& test = *static_cast<Mc2BarrierContextTest*>(comm);
        *ranks = test.ranks_.data();
        *count = layer == 2 ? test.rankSize_ : test.localSize_;
        return HCCL_SUCCESS;
    };
    owner_.HcclRankGraphGetLinks = [](HcclComm comm, uint32_t layer, uint32_t src, uint32_t dst, CommLink** links,
                                      uint32_t* count) {
        auto& test = *static_cast<Mc2BarrierContextTest*>(comm);
        test.candidates_.clear();
        if (layer == 0 && src < 16 && dst < 16) {
            test.candidates_.push_back(test.MakeLink(src, dst, COMM_PROTOCOL_UB_MEM));
        } else if (layer == 2 && dst != test.missingPeer_ && ((src == 0 && dst == 16) || (src == 16 && dst == 0))) {
            test.candidates_.push_back(test.MakeLink(src, dst, COMM_PROTOCOL_UB_RTP));
        }
        *links = test.candidates_.data();
        *count = test.candidates_.size();
        return HCCL_SUCCESS;
    };
    Builder builder{owner_, this};
    ASSERT_EQ(builder.Discover(), ACLNN_SUCCESS);
    std::sort(builder.local.begin(), builder.local.end());
    EXPECT_EQ(builder.local, (std::vector<uint32_t>(ranks_.begin(), ranks_.begin() + 16)));
    EXPECT_EQ(builder.rail, (std::vector<uint32_t>{0, 16}));
    EXPECT_EQ(builder.selfCard, Mc2Barrier::NO_RANK);
    EXPECT_EQ(builder.links.at(16).linkAttr.linkProtocol, COMM_PROTOCOL_UB_RTP);

    missingPeer_ = 16;
    Builder missing{owner_, this};
    EXPECT_EQ(missing.Discover(), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(missing.local.size(), 16U);
    EXPECT_EQ(missing.rail.size(), 1U);
    EXPECT_EQ(missing.links.size(), 15U);
    EXPECT_EQ(missing.selfCard, Mc2Barrier::NO_RANK);
}
#endif

TEST_F(Mc2BarrierContextTest, PreservesTransportAndSelectsDeviceBackend)
{
    std::vector<CommProtocol> protocols{COMM_PROTOCOL_UBC_CTP, COMM_PROTOCOL_ROCE};
#if HCOMM_VERSION_NUM >= HCOMM_BARRIER_RTP_MIN_VERSION
    protocols.push_back(COMM_PROTOCOL_UB_RTP);
#endif
    missingIdentity_ = true;
    fallback_ = false;
    for (CommProtocol protocol : protocols) {
        SCOPED_TRACE(static_cast<int32_t>(protocol));
        protocols_ = {protocol};
        Builder builder{owner_, this};
        EXPECT_FALSE(builder.SupportsStaticGraph(rankSize_));
        ASSERT_EQ(builder.local.size(), localSize_);
        ASSERT_EQ(builder.rail.size(), 2U);
        EXPECT_EQ(acquireCalls_, 0U);
        std::vector<uint64_t> storage((Mc2Barrier::ContextBytes(4, 2) + sizeof(uint64_t) - 1U) / sizeof(uint64_t));
        auto& context = *new (storage.data()) Mc2Barrier::Context;
        context.localSize = 4;
        context.railSize = 2;
        builder.InitPeers(context);
        HcclMemHandle memory = nullptr;
        std::vector<HcclChannelDesc> descs;
        ASSERT_EQ(builder.DescribeChannels(context, memory, descs), ACLNN_SUCCESS);
        EXPECT_EQ(descs[1].channelProtocol, COMM_PROTOCOL_UB_MEM);
        EXPECT_EQ(descs[5].channelProtocol, protocol);
        auto* peers = reinterpret_cast<Mc2Barrier::Peer*>(&context + 1);
        EXPECT_EQ(peers[1].protocol, Mc2Barrier::Protocol::UBMEM);
        EXPECT_EQ(peers[5].protocol,
                  protocol == COMM_PROTOCOL_ROCE ? Mc2Barrier::Protocol::ROCE : Mc2Barrier::Protocol::URMA);
    }
}

#if HCOMM_VERSION_NUM >= HCOMM_BARRIER_RTP_MIN_VERSION
TEST_F(Mc2BarrierContextTest, PreservesCtpRtpRocePriority)
{
    fallback_ = false;
    protocols_ = {COMM_PROTOCOL_ROCE, COMM_PROTOCOL_UB_RTP, COMM_PROTOCOL_UBC_CTP};
    for (CommProtocol expected : {COMM_PROTOCOL_UBC_CTP, COMM_PROTOCOL_UB_RTP, COMM_PROTOCOL_ROCE}) {
        SCOPED_TRACE(static_cast<int32_t>(expected));
        Builder builder{owner_, this};
        ASSERT_EQ(builder.Discover(), ACLNN_SUCCESS);
        EXPECT_EQ(builder.links.at(4).linkAttr.linkProtocol, expected);
        protocols_.pop_back();
    }
}
#endif

TEST_F(Mc2BarrierContextTest, ReusesContextAfterTopologyBecomesLocalOnly)
{
    localSize_ = 16;
    ranks_.resize(32);
    std::iota(ranks_.begin(), ranks_.end(), 0U);
    SetRankSize(ranks_.size());
    missingIdentity_ = true;
    fallback_ = false;
#if HCOMM_VERSION_NUM >= HCOMM_BARRIER_RTP_MIN_VERSION
    protocols_ = {COMM_PROTOCOL_UB_RTP};
#else
    protocols_ = {COMM_PROTOCOL_UBC_CTP};
#endif
    MockContextStorage();
    MockChannels();
    Builder initial{owner_, this};
    initial.rank = 14;
    ASSERT_EQ(initial.Discover(), ACLNN_SUCCESS);
    const uint64_t bytes = Mc2Barrier::ContextBytes(16, 2);
    std::vector<uint64_t> memory((bytes + sizeof(uint64_t) - 1U) / sizeof(uint64_t));
    auto& context = *new (memory.data()) Mc2Barrier::Context;
    context.worldSize = ranks_.size();
    context.rank = initial.rank;
    context.localSize = 16;
    context.railSize = 2;
    std::vector<uint64_t> window(Mc2Barrier::WindowBytes(16, 2) / sizeof(uint64_t), 1234U);
    context.window = reinterpret_cast<uint64_t>(window.data());
    initial.InitPeers(context);
    ASSERT_EQ(initial.BuildChannels(context, "barrier_buffer", reinterpret_cast<void*>(1)), ACLNN_SUCCESS);
    EXPECT_EQ(channels_.at(31), protocols_.front());
    EXPECT_EQ(channels_.at(1), COMM_PROTOCOL_UB_MEM);
    const auto* peers = reinterpret_cast<const Mc2Barrier::Peer*>(&context + 1);
    EXPECT_EQ(peers[17].protocol, Mc2Barrier::Protocol::URMA);
    EXPECT_NE(peers[17].window, 0U);
    auto& state = InitCachedState("ready", 16, 2);
    void* cached = nullptr;
    ASSERT_EQ(initial.CacheContext("ready", context, state, cached, bytes), ACLNN_SUCCESS);
    missingPeer_ = 30;
    Builder cold{owner_, this};
    cold.rank = 14;
    EXPECT_EQ(cold.Discover(), ACLNN_ERR_PARAM_INVALID);
    const uint32_t queries = layerQueries_;
    uint64_t cachedBytes = 0;
    ASSERT_EQ(cold.GetContext("ready", cached, cachedBytes), ACLNN_SUCCESS);
    EXPECT_EQ(cached, state.context);
    EXPECT_EQ(cachedBytes, bytes);
    EXPECT_EQ(layerQueries_, queries);
    EXPECT_EQ(window.front(), 1234U);
    EXPECT_EQ(std::memcmp(cached, &context, bytes), 0);
}

TEST_F(Mc2BarrierContextTest, FailedCopyStaysUnpublished)
{
    MockContextStorage();
    copyStatus_ = HCCL_E_INTERNAL;
    Builder builder{owner_, this};
    std::vector<uint64_t> memory((Mc2Barrier::ContextBytes(4, 1) + sizeof(uint64_t) - 1U) / sizeof(uint64_t));
    auto& complete = *new (memory.data()) Mc2Barrier::Context;
    complete.localSize = 4;
    complete.railSize = 1;
    void* cached = nullptr;
    uint64_t bytes = Mc2Barrier::ContextBytes(4, 1);
    auto& state = InitCachedState("failed", 4, 1);
    EXPECT_EQ(builder.CacheContext("failed", complete, state, cached, bytes), ACLNN_ERR_INNER);
    EXPECT_EQ(state.context, nullptr);
    const auto queries = layerQueries_;
    EXPECT_EQ(builder.GetContext("failed", cached, bytes), ACLNN_ERR_INNER);
    storage_.erase({CommEngine::COMM_ENGINE_AIV, "failed"});
    EXPECT_EQ(builder.GetContext("failed", cached, bytes), ACLNN_ERR_INNER);
    EXPECT_EQ(storage_.size(), 1U);
    EXPECT_EQ(layerQueries_, queries);
    EXPECT_EQ(acquireCalls_, 0U);
}

TEST_F(Mc2BarrierContextTest, InitializesWindowDuringGlobalCaptureAndRestoresMode)
{
    MockContextStorage();
    MockChannels();
    BarrierCaptureRuntime runtime;
    owner_.HcclCommMemReg = [](HcclComm, const char*, const CommMem*, HcclMemHandle* handle) {
        EXPECT_EQ(g_barrierCaptureRuntime->mode, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);
        *handle = reinterpret_cast<HcclMemHandle>(1);
        return HCCL_SUCCESS;
    };
    Builder builder{owner_, this};
    void* cached = nullptr;
    uint64_t bytes = 0;
    ASSERT_EQ(builder.GetContext("barrier", cached, bytes), ACLNN_SUCCESS);
    EXPECT_EQ(runtime.mode, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);
    auto* context = static_cast<Mc2Barrier::Context*>(cached);
    auto* window = reinterpret_cast<uint64_t*>(context->window);
    EXPECT_TRUE(std::all_of(window, window + context->windowBytes / sizeof(uint64_t),
                            [](uint64_t value) { return value == 0; }));
    window[0] = 1234U;
    ASSERT_EQ(builder.GetContext("barrier", cached, bytes), ACLNN_SUCCESS);
    EXPECT_EQ(window[0], 1234U);
    EXPECT_EQ(runtime.clears, 1U);
    runtime.clearStatus = ACL_ERROR_RT_PARAM_INVALID;
    EXPECT_EQ(builder.GetContext("failed_capture", cached, bytes), ACLNN_ERR_INNER);
    EXPECT_EQ(runtime.mode, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);
    EXPECT_EQ(runtime.clears, 2U);
}

TEST_F(Mc2BarrierContextTest, SplitsFullScaleRailIntoChannelBatches)
{
    MockChannels();
    constexpr uint32_t count = 12800;
    constexpr uint32_t self = 7777;
    std::vector<Mc2Barrier::Peer> peers(count);
    std::vector<HcclChannelDesc> descs(count);
    for (uint32_t i = 0; i < count; ++i) {
        peers[i].rank = i;
        descs[i].remoteRank = i;
        descs[i].channelProtocol = COMM_PROTOCOL_UBC_CTP;
    }
    Builder builder{owner_, this};
    builder.rank = self;
    ASSERT_EQ(builder.AcquireGroup(peers.data(), count, self, descs.data(), "barrier_buffer", 1024), ACLNN_SUCCESS);
    EXPECT_EQ(channels_.size(), count - 1U);
    EXPECT_EQ(acquireCalls_, 25U);
    EXPECT_EQ(peers[self].handle, 0U);
    EXPECT_EQ(peers[0].handle, 1U);
    EXPECT_EQ(peers[count - 1].handle, count);
}

TEST_F(Mc2BarrierContextTest, RejectsUnverifiedMixedTransportBeforeAllocatingResources)
{
    Builder builder{owner_, this};
    builder.allowMixedProtocols = false;
    EXPECT_EQ(builder.Discover(), ACLNN_ERR_PARAM_INVALID);
    EXPECT_TRUE(storage_.empty());
    EXPECT_EQ(acquireCalls_, 0U);
    localSize_ = ranks_.size();
    Builder single{owner_, this};
    single.allowMixedProtocols = false;
    EXPECT_EQ(single.Discover(), ACLNN_SUCCESS);
}

} // namespace Mc2Aclnn

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierFirstApi)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrier, INPUT(xRef, "test_distribute_barrier", worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierFirstApiNullptrGroup)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrier, INPUT(xRef, nullptr, worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierFirstApiGroupMin)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrier, INPUT(xRef, "", worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2FirstApi)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2, INPUT(xRef, nullptr, nullptr, "test_distribute_barrier", worldSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2FirstApiNullptrGroup)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2, INPUT(xRef, nullptr, nullptr, nullptr, worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2FirstApiGroupMin)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2, INPUT(xRef, nullptr, nullptr, "", worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierGroupTooLong)
{
    TensorDesc xRefDesc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto xRef = xRefDesc.ToAclType();

    for (size_t groupLength : {HCCL_GROUP_NAME_MAX, HCCL_GROUP_NAME_MAX + 1}) {
        SCOPED_TRACE(groupLength);
        std::string group(groupLength, 'a');
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        EXPECT_EQ(aclnnDistributeBarrierGetWorkspaceSize(xRef.get(), group.c_str(), 16, &workspaceSize, &executor),
                  ACLNN_ERR_PARAM_INVALID);
        EXPECT_EQ(aclnnDistributeBarrierV2GetWorkspaceSize(xRef.get(), nullptr, nullptr, group.c_str(), 16,
                                                           &workspaceSize, &executor),
                  ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(L2AclnnDistributeBarrierTest, TestBarrierCheckParamsValidGroupLengthBoundaries)
{
    TensorDesc xRefDesc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto xRef = xRefDesc.ToAclType();

    for (size_t groupLength : {size_t{1}, HCCL_GROUP_NAME_MAX - 1}) {
        SCOPED_TRACE(groupLength);
        std::string group(groupLength, 'a');
        EXPECT_EQ(BarrierCheckParams(xRef.get(), group.c_str()), ACLNN_SUCCESS);
    }
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2FirstApiTimeOut)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc timeOut = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2, INPUT(xRef, timeOut, nullptr, "test_distribute_barrier", worldSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2FirstApiElasticInfo)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc elasticInfo = TensorDesc({36}, ACL_INT32, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2,
                        INPUT(xRef, nullptr, elasticInfo, "test_distribute_barrier", worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierSecondApi)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrier, INPUT(xRef, "test_distribute_barrier", worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    if (aclRet == ACLNN_SUCCESS && executor != nullptr) {
        aclRet = aclnnDistributeBarrier(nullptr, workspaceSize, executor, nullptr);
    }
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2SecondApi)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2, INPUT(xRef, nullptr, nullptr, "test_distribute_barrier", worldSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    if (aclRet == ACLNN_SUCCESS && executor != nullptr) {
        aclRet = aclnnDistributeBarrierV2(nullptr, workspaceSize, executor, nullptr);
    }
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierSecondApiTimeOut)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc timeOut = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2, INPUT(xRef, timeOut, nullptr, "test_distribute_barrier", worldSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    if (aclRet == ACLNN_SUCCESS && executor != nullptr) {
        aclRet = aclnnDistributeBarrierV2(nullptr, workspaceSize, executor, nullptr);
    }
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierSecondApiElasticInfo)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc elasticInfo = TensorDesc({36}, ACL_INT32, ACL_FORMAT_ND);
    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2,
                        INPUT(xRef, nullptr, elasticInfo, "test_distribute_barrier", worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    if (aclRet == ACLNN_SUCCESS && executor != nullptr) {
        aclRet = aclnnDistributeBarrierV2(nullptr, workspaceSize, executor, nullptr);
    }
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierSecondApiTimeOutAndElasticInfo)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc timeOut = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc elasticInfo = TensorDesc({36}, ACL_INT32, ACL_FORMAT_ND);
    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2,
                        INPUT(xRef, timeOut, elasticInfo, "test_distribute_barrier", worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    if (aclRet == ACLNN_SUCCESS && executor != nullptr) {
        aclRet = aclnnDistributeBarrierV2(nullptr, workspaceSize, executor, nullptr);
    }
}

// ========== 新增测试用例：直接调用第二段接口覆盖未覆盖代码 ==========

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierSecondApiDirect)
{
    // 直接调用第二段接口，覆盖 aclnnDistributeBarrier 和 aclnnDistributeBarrierBase 函数体
    aclnnStatus aclRet = aclnnDistributeBarrier(nullptr, 0, nullptr, nullptr);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2SecondApiDirect)
{
    // 直接调用第二段接口，覆盖 aclnnDistributeBarrierV2 和 aclnnDistributeBarrierBase 函数体
    aclnnStatus aclRet = aclnnDistributeBarrierV2(nullptr, 0, nullptr, nullptr);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// ========== 参数校验测试用例 ==========

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierNullptrXRef)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = aclnnDistributeBarrierGetWorkspaceSize(nullptr, "test_group", 16, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2NullptrXRef)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = aclnnDistributeBarrierV2GetWorkspaceSize(nullptr, nullptr, nullptr, "test_group", 16,
                                                                  &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ========== dtype 测试用例 ==========

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierDtypeFloat32)
{
    TensorDesc xRefDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xRefPtr = xRefDesc.ToAclType();

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet =
        aclnnDistributeBarrierGetWorkspaceSize(xRefPtr.get(), "test_distribute_barrier", 16, &workspaceSize, &executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierDtypeBfloat16)
{
    TensorDesc xRefDesc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto xRefPtr = xRefDesc.ToAclType();

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet =
        aclnnDistributeBarrierGetWorkspaceSize(xRefPtr.get(), "test_distribute_barrier", 16, &workspaceSize, &executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierDtypeInt32)
{
    TensorDesc xRefDesc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto xRefPtr = xRefDesc.ToAclType();

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet =
        aclnnDistributeBarrierGetWorkspaceSize(xRefPtr.get(), "test_distribute_barrier", 16, &workspaceSize, &executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, A5V1V2BuildsInternalContext)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto x = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto timeout = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    auto elastic = TensorDesc({36}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    for (int64_t world : {16L, 204800L}) {
        SCOPED_TRACE(world);
        ASSERT_EQ(aclnnDistributeBarrierGetWorkspaceSize(x.get(), "barrier", world, &bytes, &executor), ACLNN_SUCCESS);
        EXPECT_EQ(g_forwardedContext, g_builtContext);
        EXPECT_EQ(g_forwardedWorldSize, world);
        EXPECT_EQ(g_forwardedGroup, "barrier");
        const auto* optionalTimeout = world == 16 ? timeout.get() : nullptr;
        const auto* optionalElastic = world == 16 ? elastic.get() : nullptr;
        ASSERT_EQ(aclnnDistributeBarrierV2GetWorkspaceSize(x.get(), optionalTimeout, optionalElastic, "barrier", world,
                                                           &bytes, &executor),
                  ACLNN_SUCCESS);
        EXPECT_EQ(g_forwardedContext, g_builtContext);
        EXPECT_EQ(g_forwardedWorldSize, world);
        EXPECT_EQ(g_forwardedX, x.get());
        EXPECT_EQ(g_forwardedTimeout, optionalTimeout);
        EXPECT_EQ(g_forwardedElasticInfo, optionalElastic);
    }
    EXPECT_EQ(g_contextBuildCalls, 4U);
    EXPECT_EQ(g_extendCalls, 4U);
    EXPECT_EQ(aclnnDistributeBarrier(nullptr, bytes, executor, nullptr), ACLNN_SUCCESS);
    EXPECT_EQ(aclnnDistributeBarrierV2(nullptr, bytes, executor, nullptr), ACLNN_SUCCESS);
    EXPECT_EQ(g_extendSecondCalls, 2U);
}

TEST_F(L2AclnnDistributeBarrierTest, A5ContextFailureStopsLaunch)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto x = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    for (aclnnStatus status : {ACLNN_ERR_PARAM_INVALID, ACLNN_ERR_INNER}) {
        g_contextStatus = status;
        EXPECT_EQ(aclnnDistributeBarrierV2GetWorkspaceSize(x.get(), nullptr, nullptr, "barrier", 16, &bytes, &executor),
                  status);
    }
    EXPECT_EQ(g_contextBuildCalls, 2U);
    EXPECT_EQ(g_extendCalls, 0U);
}
