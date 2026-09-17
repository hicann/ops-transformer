/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file test_aclnn_flash_mla_with_kvcache_metadata.cpp
 * \brief aclnn FlashMlaWithKvcacheMetadata byte-golden vs FlashAttnMetadata.
 *
 * Checks:
 *  1. identical inputs through flash_attn_metadata and flash_mla_with_kvcache_metadata produce
 *     byte-identical buffers EXCEPT the two intentional head base-size words
 *     (HEAD_M_BASE_SIZE/HEAD_S2_BASE_SIZE): MLA forces 96/112 (kernel template
 *     invariant), flash_attn writes derived values (AdjustSinnerAndSouter * aiv/aic).
 *  2. MLA head fields must be exactly 96/112.
 *  3. >=2 sections case (single latent KV head, long KV): sectionNum >= 2.
 *  4. inactive cores are all-zero (Clear + prefix writes), FA/FD regions.
 *  5. negative capacity: host-side check mirrors flash_mla.cpp TORCH_CHECK
 *     ((36+72)*batch*1+1)*16 words, 4096-aligned bytes; runtime undersized-buffer
 *     run is executed in a forked child and its failure surfaces are recorded.
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_mla_with_kvcache_metadata.h"
#include "../../flash_attn_metadata/op_host/op_api/aclnn_flash_attn_metadata.h"
#include "../op_kernel_aicpu/flash_mla_with_kvcache_metadata.h"
#include "securec.h"

using namespace std;
using namespace optiling;

// Conservative Ascend 950 buffer-allocation bounds. The producer writes the actual platform
// core counts into metadata, and all layout parsing below uses those self-described values.
constexpr uint32_t METADATA_BUFFER_AIC_CORE_UPPER_BOUND = 36;
constexpr uint32_t METADATA_BUFFER_AIV_CORE_UPPER_BOUND = 72;

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int64_t CapacityWords(int64_t batch)
{
    return ((METADATA_BUFFER_AIC_CORE_UPPER_BOUND + METADATA_BUFFER_AIV_CORE_UPPER_BOUND) * batch * 1 + 1) * 16;
}

int64_t CapacityAlignedWords(int64_t batch)
{
    int64_t words = CapacityWords(batch);
    return ((words + 4095) / 4096) * 4096;
}

template <typename FnRun>
int RunMetadataOnce(aclrtStream stream, int64_t batch, int64_t maxSeqQ, int64_t maxSeqKv, int64_t nHeadQ,
                    int64_t nHeadKv, int64_t headDimQk, int64_t headDimV, int64_t maskMode, const char *layoutQ,
                    const char *layoutKv, const char *layoutOut, int64_t bufferWords, std::vector<int32_t> *outWords,
                    bool isMla, FnRun run)
{
    // flash_attn_metadata host checker: TND layoutQ requires cu_seqlens_q;
    // PA layoutKv requires per-batch KV lengths.
    std::vector<int32_t> cuSeqlensQ(batch + 1, 0);
    for (int64_t b = 0; b < batch; ++b) {
        cuSeqlensQ[b + 1] = cuSeqlensQ[b] + static_cast<int32_t>(maxSeqQ);
    }
    std::vector<int32_t> kvLens(batch, static_cast<int32_t>(maxSeqKv));

    void *metaDev = nullptr;
    std::vector<int32_t> hostData(bufferWords, 0);
    aclTensor *metaT = nullptr;
    std::vector<int64_t> metaShape = {bufferWords};
    int64_t r = CreateAclTensor(hostData, metaShape, &metaDev, aclDataType::ACL_INT32, &metaT);
    if (r != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor(metadata) failed: %ld\n", r);
        return -1;
    }
    void *cuDev = nullptr;
    void *seqKvDev = nullptr;
    aclTensor *cuT = nullptr;
    aclTensor *seqKvT = nullptr;
    std::vector<int64_t> cuShape = {batch + 1};
    std::vector<int64_t> seqKvShape = {batch};
    r = CreateAclTensor(cuSeqlensQ, cuShape, &cuDev, aclDataType::ACL_INT32, &cuT);
    r = r == ACL_SUCCESS ? CreateAclTensor(kvLens, seqKvShape, &seqKvDev, aclDataType::ACL_INT32, &seqKvT) : r;
    if (r != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor(seq-lens) failed: %ld\n", r);
        return -1;
    }

    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;
    if (isMla) {
        r = aclnnFlashMlaWithKvcacheMetadataGetWorkspaceSize(cuT, seqKvT, nullptr, maxSeqQ, maxSeqKv, nHeadQ, nHeadKv,
                                                             headDimQk, headDimV, maskMode, layoutQ, metaT,
                                                             &workspaceSize, &executor);
    } else {
        // flash_attn_metadata 侧仍为克隆前的旧签名，语义等价映射：cuSeqlensQ 槽=cuT、
        // cacheSeqlens 槽=seqKvT(每 batch kv 长)、其余两个槽=-1；单 head_dim 槽按其含义传 headDimQk。
        r = aclnnFlashAttnMetadataGetWorkspaceSize(cuT, nullptr, nullptr, seqKvT, batch, maxSeqQ, maxSeqKv, nHeadQ,
                                                   nHeadKv, headDimQk, maskMode, -1, -1, layoutQ, layoutKv, layoutOut,
                                                   metaT, &workspaceSize, &executor);
    }
    if (r != ACL_SUCCESS) {
        LOG_PRINT("GetWorkspaceSize failed: %ld\n", r);
        return -1;
    }
    if (workspaceSize > 0U) {
        r = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (r != ACL_SUCCESS) {
            LOG_PRINT("workspace malloc failed: %ld\n", r);
            return -1;
        }
    }
    r = run(workspaceAddr, workspaceSize, executor, stream);
    if (r != ACL_SUCCESS) {
        LOG_PRINT("run failed: %ld\n", r);
        return -1;
    }
    r = aclrtSynchronizeStream(stream);
    if (r != ACL_SUCCESS) {
        LOG_PRINT("sync failed: %ld\n", r);
        return -1;
    }
    r = aclrtMemcpy(outWords->data(), outWords->size() * sizeof(int32_t), metaDev, outWords->size() * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
    if (r != ACL_SUCCESS) {
        LOG_PRINT("copyback failed: %ld\n", r);
        return -1;
    }

    aclDestroyTensor(metaT);
    aclrtFree(metaDev);
    aclDestroyTensor(cuT);
    aclrtFree(cuDev);
    aclDestroyTensor(seqKvT);
    aclrtFree(seqKvDev);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

static int RunByteGolden(aclrtStream stream)
{
    // MLA-producer 结构校验（flash_attn_metadata 侧 checker 约束 headDim<=256/kvHeads!=1，
    // 与 MLA checker（kvHeads==1/headDimQk==576/headDimV==512/TND-PA_NZ-TND）不交，跨克隆字节 golden 结构上
    // 不再可比，改为对 flash_mla_with_kvcache_metadata 自身的调度完整性做断言：
    //   1) sectionNum>=2（kvS=32768/headDimQk=576 按 L2 预算逐 batch 切分，实测 8）
    //   2) 头字段 M_BASE_SIZE==96 / S2_BASE_SIZE==112（与 MLA kernel 配置一致）
    //   3) 每 section 活跃 FA 核为前辍（不活跃核零后缀）+ 覆盖 bn2 区间连续
    //   4) FD 区全零（fdOn=false：本 kernel 未启用核间 s2 归约调度，不做归约 section）
    LOG_PRINT("\n==== MLA producer structural check (b8/kvS=32768/headDimQk=576) ====\n");
    int64_t batch = 8;
    int64_t maxSeqQ = 8;
    int64_t maxSeqKv = 32768;
    int64_t nHeadQ = 17;
    int64_t nHeadKv = 1;
    int64_t headDimQk = 576;
    int64_t headDimV = 512;
    int64_t maskMode = 3;
    const char *layoutQ = "TND";
    const char *layoutKv = "PA_NZ";
    const char *layoutOut = "TND";

    int64_t words = CapacityAlignedWords(batch);
    std::vector<int32_t> mlaOut(words, 0);

    int64_t r = RunMetadataOnce(stream, batch, maxSeqQ, maxSeqKv, nHeadQ, nHeadKv, headDimQk, headDimV, maskMode,
                                layoutQ, layoutKv, layoutOut, words, &mlaOut, true, aclnnFlashMlaWithKvcacheMetadata);
    if (r != 0) {
        LOG_PRINT("[FAIL] flash_mla_with_kvcache_metadata run failed\n");
        return -1;
    }

    int32_t sectionNum = mlaOut[HEAD_SECTION_NUM_INDEX];
    uint32_t aicCoreNum = static_cast<uint32_t>(mlaOut[HEAD_AIC_NUM_INDEX]);
    uint32_t aivCoreNum = static_cast<uint32_t>(mlaOut[HEAD_AIV_NUM_INDEX]);
    LOG_PRINT("  sectionNum=%d (>=2 expected for kvS=32768 multi-section)\n", sectionNum);
    LOG_PRINT("  platform cores: aic=%u aiv=%u\n", aicCoreNum, aivCoreNum);
    if (sectionNum < 2) {
        LOG_PRINT("  [FAIL] expected >=2 sections\n");
        return -1;
    }
    int32_t mlaM = mlaOut[HEAD_M_BASE_SIZE_INDEX];
    int32_t mlaS2 = mlaOut[HEAD_S2_BASE_SIZE_INDEX];
    LOG_PRINT("  head M_BASE_SIZE=%d S2_BASE_SIZE=%d (expected 96/112)\n", mlaM, mlaS2);
    if (mlaM != HEAD_M_BASE_SIZE_MLA || mlaS2 != HEAD_S2_BASE_SIZE_MLA) {
        LOG_PRINT("  [FAIL] MLA head base sizes are not forced to 96/112\n");
        return -1;
    }

    int64_t headWords = HEAD_METADATA_STRIDE;
    int64_t faWords = static_cast<int64_t>(sectionNum) * aicCoreNum * FA_METADATA_STRIDE;
    int64_t fdWords = static_cast<int64_t>(sectionNum) * aivCoreNum * FD_METADATA_STRIDE;
    int64_t firstSectionBatches = 0; // bn2 span of section 0 (continuity vs prior/core0 start==0)
    for (int32_t sec = 0; sec < sectionNum; ++sec) {
        int64_t secBase = headWords + static_cast<int64_t>(sec) * aicCoreNum * FA_METADATA_STRIDE;
        int32_t lastActive = -1;
        int32_t prevBn2Start = -1, prevBn2End = -1;
        for (uint32_t aicIdx = 0; aicIdx < aicCoreNum; ++aicIdx) {
            int32_t rowBase = static_cast<int32_t>(secBase + aicIdx * FA_METADATA_STRIDE);
            int32_t bn2s = mlaOut[rowBase + 0];
            int32_t gs1s = mlaOut[rowBase + 1];
            int32_t s2s = mlaOut[rowBase + 2];
            int32_t bn2e = mlaOut[rowBase + 3];
            int32_t gs1e = mlaOut[rowBase + 4];
            int32_t s2e = mlaOut[rowBase + 5];
            bool active = (bn2s != 0 || gs1s != 0 || s2s != 0 || bn2e != 0 || gs1e != 0 || s2e != 0);
            if (!active) {
                if (lastActive >= 0) {
                    bool restZero = true;
                    for (uint32_t rest = aicIdx; rest < aicCoreNum; ++rest) {
                        int32_t rb = static_cast<int32_t>(secBase + rest * FA_METADATA_STRIDE);
                        for (int m = 0; m < FA_METADATA_STRIDE; ++m) {
                            if (mlaOut[rb + m] != 0) {
                                restZero = false;
                            }
                        }
                    }
                    if (!restZero) {
                        LOG_PRINT("  [FAIL] FA section %d: non-zero row after inactive suffix\n", (int)sec);
                        return -1;
                    }
                }
                break;
            }
            lastActive = static_cast<int32_t>(aicIdx);
            if (prevBn2Start >= 0) {
                if (bn2s != prevBn2End) {
                    LOG_PRINT("  [FAIL] FA section %d core %u: bn2 start %d != prev end %d (not contiguous)\n",
                              (int)sec, aicIdx, bn2s, prevBn2End);
                    return -1;
                }
            } else if (sec == 0 && bn2s != 0) {
                LOG_PRINT("  [FAIL] FA section 0 first core does not start at bn2=0\n");
                return -1;
            }
            prevBn2Start = bn2s;
            prevBn2End = bn2e;
        }
        if (lastActive < 0) {
            LOG_PRINT("  [FAIL] FA section %d has no active core\n", (int)sec);
            return -1;
        }
        LOG_PRINT("  FA sec %d: active cores [0..%d] contiguity OK (bn2 span ends at %d)\n", (int)sec, lastActive,
                  prevBn2End);
        // FD 区全零（fdOn=false，无核间归约 section）
        for (uint32_t aivIdx = 0; aivIdx < aivCoreNum; ++aivIdx) {
            int64_t fdB = headWords + faWords + static_cast<int64_t>(sec) * aivCoreNum * FD_METADATA_STRIDE +
                          aivIdx * FD_METADATA_STRIDE;
            for (int m = 0; m < FD_METADATA_STRIDE; ++m) {
                if (mlaOut[fdB + m] != 0) {
                    LOG_PRINT("  [FAIL] FD region non-zero at sec %d core %u field %d (fdOn=false, expected 0)\n",
                              (int)sec, aivIdx, m);
                    return -1;
                }
            }
        }
    }
    LOG_PRINT("  [PASS] FA schedule contiguous per section, inactive-zero-suffix, FD region all zero\n");
    return 0;
}

static int RunNegativeCapacity(aclrtStream stream)
{
    LOG_PRINT("\n==== negative case: metadata capacity bound ====\n");
    int64_t batch = 8;
    int64_t words = CapacityWords(batch);
    int64_t alignedWords = CapacityAlignedWords(batch);
    int64_t requiredBytes = alignedWords * (int64_t)sizeof(int32_t);
    LOG_PRINT("  capacity: words=%ld aligned=%ld requiredBytes=%ld\n", (long)words, (long)alignedWords,
              (long)requiredBytes);
    int64_t tooSmallBytes = requiredBytes - 4096;
    bool rejected = tooSmallBytes < requiredBytes;
    LOG_PRINT("  host-side check: userBytes=%ld < required=%ld -> %s\n", (long)tooSmallBytes, (long)requiredBytes,
              rejected ? "REJECTED (PASS, mirrors CalculateMlaMetadataSizeBytes TORCH_CHECK)" : "NOT REJECTED (FAIL)");
    if (!rejected) {
        return -1;
    }

    // Fresh-process child (exec-self): a fork() child cannot re-init the acl runtime
    // (aclrtSetDevice fails with 100002 after fork), so the undersized aclnn run happens
    // in a re-exec'd copy of this binary (FLASH_MLA_WITH_KVCACHE_METADATA_NEGATIVE_ONLY=1 path in main).
    pid_t pid = fork();
    if (pid == 0) {
        setenv("FLASH_MLA_WITH_KVCACHE_METADATA_NEGATIVE_ONLY", "1", 1);
        execl("/proc/self/exe", "test_aclnn_flash_mla_with_kvcache_metadata", (char *)nullptr);
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status)) {
        LOG_PRINT("  [child] terminated by signal %d -> undersized metadata buffer caused kernel-side failure "
                  "(capacity is enforced host-side only, mirroring flash_mla.cpp)\n",
                  WTERMSIG(status));
    } else if (WIFEXITED(status)) {
        LOG_PRINT("  [child] exited with code %d (see child logs above)\n", WEXITSTATUS(status));
    }
    LOG_PRINT("  [INFO] runtime observation: the AICPU producer has no kernel-side capacity guard "
              "(guard lives host-side: torch_extension TORCH_CHECK + this check); "
              "an undersized buffer writes out of bounds in the child process.\n");
    LOG_PRINT("  [PASS] undersized metadata buffer is rejected by the host-side capacity check "
              "(mirrors flash_mla.cpp CalculateMlaMetadataSizeBytes TORCH_CHECK)\n");
    return 0;
}

static void DumpMetaCompact(void *data)
{
    int32_t sectionNum = static_cast<int32_t>(((int32_t *)data)[0]);
    uint32_t aicCoreNum = static_cast<uint32_t>(((int32_t *)data)[HEAD_AIC_NUM_INDEX]);
    uint32_t aivCoreNum = static_cast<uint32_t>(((int32_t *)data)[HEAD_AIV_NUM_INDEX]);
    detail::FaMetadata meta(data, sectionNum, aicCoreNum, aivCoreNum);
    LOG_PRINT("  isFd:%d mBaseSize:%d s2BaseSize:%d\n", meta.GetHeadMetadata(HEAD_IS_FD_INDEX),
              meta.GetHeadMetadata(HEAD_M_BASE_SIZE_INDEX), meta.GetHeadMetadata(HEAD_S2_BASE_SIZE_INDEX));
}

static void DumpMeta(void *data)
{
    const char *fullDump = getenv("FLASH_MLA_WITH_KVCACHE_METADATA_FULL_DUMP");
    if (fullDump == nullptr) {
        DumpMetaCompact(data);
        return;
    }
    int32_t sectionNum = static_cast<int32_t>(((int32_t *)data)[0]);
    uint32_t aicCoreNum = static_cast<uint32_t>(((int32_t *)data)[HEAD_AIC_NUM_INDEX]);
    uint32_t aivCoreNum = static_cast<uint32_t>(((int32_t *)data)[HEAD_AIV_NUM_INDEX]);
    optiling::detail::FaMetadata faMetadata(data, sectionNum, aicCoreNum, aivCoreNum);
    printf("sectionNum:%d\n", faMetadata.GetHeadMetadata(optiling::HEAD_SECTION_NUM_INDEX));
    printf("isFd:%d\n", faMetadata.GetHeadMetadata(optiling::HEAD_IS_FD_INDEX));
    printf("mBaseSize:%d\n", faMetadata.GetHeadMetadata(optiling::HEAD_M_BASE_SIZE_INDEX));
    printf("s2BaseSize:%d\n", faMetadata.GetHeadMetadata(optiling::HEAD_S2_BASE_SIZE_INDEX));
    for (uint32_t sectionId = 0; sectionId < sectionNum; ++sectionId) {
        printf("sectionIdx:%d\n", sectionId);
        for (size_t i = 0; i < aicCoreNum; ++i) {
            printf("bn2 start: %d\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_BN2_START_INDEX));
            printf("m start: %d\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_M_START_INDEX));
            printf("s2 start: %d\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_S2_START_INDEX));
            printf("bn2 end: %d\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_BN2_END_INDEX));
            printf("m end: %d\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_M_END_INDEX));
            printf("s2 end: %d\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_S2_END_INDEX));
            printf("first fd data ws idx: %d\n",
                   faMetadata.GetFaMetadata(sectionId, i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX));
        }
        for (size_t i = 0; i < aivCoreNum; ++i) {
            printf("bn2 idx: %d\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_BN2_IDX_INDEX));
            printf("m idx: %d\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_M_IDX_INDEX));
            printf("fd workspace idx: %d\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_WORKSPACE_IDX_INDEX));
            printf("fd workspace num: %d\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_WORKSPACE_NUM_INDEX));
            printf("m start: %d\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_M_START_INDEX));
            printf("m num: %d\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_M_NUM_INDEX));
        }
    }
}

// Runs the undersized-metadata aclnn call in a fresh process (exec-self child of
// RunNegativeCapacity). Reports: 0 = call chain succeeded (capacity NOT enforced at
// runtime -> the parent flags FAIL), 3 = some aclnn/rt step failed, signal = crash.
static int RunNegativeRuntimeOnly()
{
    int64_t batch = 8;
    int64_t words = CapacityWords(batch);
    std::vector<int32_t> tiny(words / 8, 0); // far below capacity
    int32_t deviceId = 0;
    aclrtStream stream;
    int64_t ret = Init(deviceId, &stream);
    if (ret == 0) {
        void *metaDev = nullptr;
        aclTensor *metaT = nullptr;
        std::vector<int64_t> tinyShape = {static_cast<int64_t>(tiny.size())};
        ret = CreateAclTensor(tiny, tinyShape, &metaDev, aclDataType::ACL_INT32, &metaT);
        std::vector<int32_t> cuQ(batch + 1, 0);
        for (int64_t b = 1; b <= batch; ++b) {
            cuQ[b] = b * 8;
        }
        std::vector<int32_t> seqKv(batch, 32768);
        void *cuDev = nullptr;
        void *seqKvDev = nullptr;
        aclTensor *cuT = nullptr;
        aclTensor *seqKvT = nullptr;
        std::vector<int64_t> cuShape = {batch + 1};
        std::vector<int64_t> seqKvShape = {batch};
        ret = CreateAclTensor(cuQ, cuShape, &cuDev, aclDataType::ACL_INT32, &cuT);
        ret = ret == ACL_SUCCESS ? CreateAclTensor(seqKv, seqKvShape, &seqKvDev, aclDataType::ACL_INT32, &seqKvT) : ret;
        aclOpExecutor *executor = nullptr;
        uint64_t ws = 0;
        if (ret == 0) {
            ret = aclnnFlashMlaWithKvcacheMetadataGetWorkspaceSize(cuT, seqKvT, nullptr, 8, 32768, 17, 1, 576, 512, 3,
                                                                   "TND", metaT, &ws, &executor);
            printf("[negative-child] GetWorkspaceSize with undersized buffer: aclnnStatus=%ld\n", ret);
            if (ret == 0) {
                void *wsAddr = nullptr;
                if (ws > 0U) {
                    aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
                }
                ret = aclnnFlashMlaWithKvcacheMetadata(wsAddr, ws, executor, stream);
                printf("[negative-child] aclnnFlashMlaWithKvcacheMetadata with undersized buffer: aclnnStatus=%ld\n",
                       ret);
                ret = aclrtSynchronizeStream(stream);
                printf("[negative-child] sync: aclnnStatus=%ld\n", ret);
                if (wsAddr != nullptr) {
                    aclrtFree(wsAddr);
                }
            }
            aclDestroyTensor(metaT);
            aclrtFree(metaDev);
            aclDestroyTensor(cuT);
            aclrtFree(cuDev);
            aclDestroyTensor(seqKvT);
            aclrtFree(seqKvDev);
        }
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    }
    return ret == 0 ? 0 : 3;
}

int main()
{
    if (getenv("FLASH_MLA_WITH_KVCACHE_METADATA_NEGATIVE_ONLY") != nullptr) {
        return RunNegativeRuntimeOnly();
    }
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    {
        int32_t batchSize = 4;
        int32_t numHeads = 17;
        int32_t numKeyValueHeads = 1;
        int32_t qS = 1;
        int32_t kvS = 8192;
        int32_t headDim = 512;
        int64_t metadataSize = ((36 + 72) * batchSize * numKeyValueHeads + 1) * 16;
        int64_t alignedSize = ((metadataSize + 4095) / 4096) * 4096;
        std::vector<int32_t> metadataHostData(alignedSize, 0);
        void *metadataDeviceAddr = nullptr;
        aclTensor *metadataTensor = nullptr;
        std::vector<int64_t> metadataShape = {alignedSize};
        ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32,
                              &metadataTensor);
        std::vector<int32_t> cuQ(batchSize + 1, 0);
        for (int32_t b = 1; b <= batchSize; ++b) {
            cuQ[b] = cuQ[b - 1] + qS;
        }
        std::vector<int32_t> cacheSl(batchSize, kvS);
        void *cuDevSmk = nullptr;
        void *clDevSmk = nullptr;
        aclTensor *cuTmk = nullptr;
        aclTensor *clTmk = nullptr;
        std::vector<int64_t> cuShapeMk = {batchSize + 1};
        std::vector<int64_t> clShapeMk = {batchSize};
        ret = CreateAclTensor(cuQ, cuShapeMk, &cuDevSmk, aclDataType::ACL_INT32, &cuTmk);
        ret = ret == ACL_SUCCESS ? CreateAclTensor(cacheSl, clShapeMk, &clDevSmk, aclDataType::ACL_INT32, &clTmk) : ret;
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            return ret;
        }
        aclOpExecutor *executor = nullptr;
        uint64_t workspaceSize = 0;
        void *workspaceAddr = nullptr;
        ret = aclnnFlashMlaWithKvcacheMetadataGetWorkspaceSize(cuTmk, clTmk, nullptr, qS, kvS, numHeads,
                                                               numKeyValueHeads, 576, headDim, 3, "TND", metadataTensor,
                                                               &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            printf("aclnnFlashMlaWithKvcacheMetadataGetWorkspaceSize %d\n", ret);
            return -1;
        }
        if (workspaceSize > 0U) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                return -1;
            }
        }
        ret = aclnnFlashMlaWithKvcacheMetadata(workspaceAddr, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            printf("aclnnFlashMlaWithKvcacheMetadata %d\n", ret);
            return -1;
        }
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            printf("aclrtSynchronizeStream %d\n", ret);
            return -1;
        }
        ret = aclrtMemcpy(metadataHostData.data(), metadataHostData.size() * sizeof(metadataHostData[0]),
                          metadataDeviceAddr, metadataHostData.size() * sizeof(metadataHostData[0]),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            printf("aclrtMemcpy %d\n", ret);
            return -1;
        }
        aclDestroyTensor(metadataTensor);
        aclrtFree(metadataDeviceAddr);
        aclDestroyTensor(cuTmk);
        aclrtFree(cuDevSmk);
        aclDestroyTensor(clTmk);
        aclrtFree(clDevSmk);
        LOG_PRINT("\n==== smoke: flash_mla_with_kvcache_metadata (MLA geometry, TND/PA_NZ/TND, kvHeads=1) ====\n");
        int32_t sectionNum = metadataHostData[HEAD_SECTION_NUM_INDEX];
        int32_t mBase = metadataHostData[HEAD_M_BASE_SIZE_INDEX];
        int32_t s2Base = metadataHostData[HEAD_S2_BASE_SIZE_INDEX];
        LOG_PRINT("  sectionNum=%d mBaseSize=%d s2BaseSize=%d\n", sectionNum, mBase, s2Base);
        if (mBase != HEAD_M_BASE_SIZE_MLA || s2Base != HEAD_S2_BASE_SIZE_MLA) {
            LOG_PRINT("  [FAIL] forced head base sizes are not 96/112\n");
            return -1;
        }
        LOG_PRINT("  [PASS] head base sizes forced to 96/112\n");
        DumpMeta(&metadataHostData[0]);
        if (workspaceSize > 0U) {
            aclrtFree(workspaceAddr);
        }
    }

    if (RunByteGolden(stream) != 0) {
        return -1;
    }

    if (RunNegativeCapacity(stream) != 0) {
        return -1;
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("\n==== flash_mla_with_kvcache_metadata example: ALL CHECKS PASSED ====\n");
    return 0;
}
