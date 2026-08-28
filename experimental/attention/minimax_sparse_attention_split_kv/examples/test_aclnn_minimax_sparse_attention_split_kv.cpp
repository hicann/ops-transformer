/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * On-device smoke + golden check for MinimaxSparseAttentionSplitKv (innerPrecise=0/4).
 * Paged KV cache, TND / BNSD / BSND contiguous. Dense local-block CSR, causal prefill.
 * softmaxLseFlag=false for the main suite; a smaller LSE suite checks fp32 lse.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_minimax_sparse_attention_split_kv.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

constexpr int64_t kHeadDim = 128;
constexpr float kAbsTol = 2e-2f;
constexpr float kRelTol = 2e-2f;

enum class LayoutKind {
    TND,
    BNSD,
    BSND
};

struct CaseSpec {
    int64_t seq;
    int64_t blockSize;
    int64_t numHeads;
    int64_t kvHeads;
    int64_t topK;
    const char *name;
    bool pagedKv;
    LayoutKind layout = LayoutKind::TND;
    // Dummy requests prepended with q_len=kv_len=0. TND/PA query T stays `seq`
    // (packed); actual_seq_lengths becomes [0]*padBatches + [seq].
    int64_t padBatches = 0;
};

const char *LayoutStr(const CaseSpec &spec)
{
    if (spec.layout == LayoutKind::BNSD) {
        return "BNSD";
    }
    if (spec.layout == LayoutKind::BSND) {
        return "BSND";
    }
    return "TND";
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t n = 1;
    for (auto d : shape) {
        n *= d;
    }
    return n;
}

uint16_t FloatToBf16(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t lsb = (x >> 16) & 1U;
    x += 0x7FFFU + lsb;
    return static_cast<uint16_t>(x >> 16);
}

float Bf16ToFloat(uint16_t b)
{
    uint32_t x = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    LOG_PRINT("aclInit start, deviceId=%d\n", deviceId);
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclrtSetDevice start\n");
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclrtCreateStream start\n");
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("Init done\n");
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i) + 1] * strides[static_cast<size_t>(i) + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void FillOnes(std::vector<uint16_t> &dst)
{
    const uint16_t one = FloatToBf16(1.0f);
    std::fill(dst.begin(), dst.end(), one);
}

void FillRandom(std::vector<uint16_t> &dst, uint32_t seed)
{
    uint32_t s = seed;
    for (size_t i = 0; i < dst.size(); ++i) {
        s = s * 1664525U + 1013904223U;
        float v = static_cast<float>(s & 0xFFFFU) / 65535.0f * 2.0f - 1.0f;
        dst[i] = FloatToBf16(v);
    }
}

size_t QElem(const CaseSpec &spec, int64_t t, int64_t h, int64_t d)
{
    if (spec.layout == LayoutKind::BNSD) {
        return static_cast<size_t>(((h * spec.seq + t) * kHeadDim) + d);
    }
    return static_cast<size_t>(((t * spec.numHeads + h) * kHeadDim) + d);
}

size_t KvElem(const CaseSpec &spec, int64_t tok, int64_t kvh, int64_t d)
{
    if (spec.layout == LayoutKind::BNSD) {
        return static_cast<size_t>(((kvh * spec.seq + tok) * kHeadDim) + d);
    }
    return static_cast<size_t>(((tok * spec.kvHeads + kvh) * kHeadDim) + d);
}

size_t LseElem(const CaseSpec &spec, int64_t t, int64_t h)
{
    if (spec.layout == LayoutKind::BNSD) {
        return static_cast<size_t>(h * spec.seq + t);
    }
    return static_cast<size_t>(t * spec.numHeads + h);
}

// innerPrecise=0 golden: fp32 softmax, P rounded to bf16 before PV.
// pagedKv: K/V stored as [1, blockSize, kvHeads, D]; else TND [seq, kvHeads, D],
// BNSD [1, kvHeads, seq, D], or BSND [1, seq, kvHeads, D].
void CpuGolden(const CaseSpec &spec, const std::vector<uint16_t> &q, const std::vector<uint16_t> &k,
               const std::vector<uint16_t> &v, std::vector<float> &outFp32, std::vector<float> &lseFp32, float scale,
               bool roundPToBf16)
{
    const int64_t groupSize = spec.numHeads / spec.kvHeads;
    outFp32.assign(static_cast<size_t>(spec.seq * spec.numHeads * kHeadDim), 0.0f);
    lseFp32.assign(static_cast<size_t>(spec.seq * spec.numHeads), 0.0f);
    for (int64_t t = 0; t < spec.seq; ++t) {
        const int64_t causalLen = t + 1;
        for (int64_t h = 0; h < spec.numHeads; ++h) {
            const int64_t kvh = h / groupSize;
            std::vector<float> scores(static_cast<size_t>(causalLen), 0.0f);
            float rowMax = -1e30f;
            for (int64_t c = 0; c < causalLen; ++c) {
                float s = 0.0f;
                const int64_t kvTok = spec.pagedKv ? (0 * spec.blockSize + c) : c;
                for (int64_t d = 0; d < kHeadDim; ++d) {
                    float qv = Bf16ToFloat(q[QElem(spec, t, h, d)]);
                    float kv = Bf16ToFloat(k[KvElem(spec, kvTok, kvh, d)]);
                    s += qv * kv;
                }
                s *= scale;
                scores[static_cast<size_t>(c)] = s;
                rowMax = std::max(rowMax, s);
            }
            float rowSum = 0.0f;
            for (int64_t c = 0; c < causalLen; ++c) {
                float p = std::exp(scores[static_cast<size_t>(c)] - rowMax);
                if (roundPToBf16) {
                    p = Bf16ToFloat(FloatToBf16(p));
                }
                scores[static_cast<size_t>(c)] = p;
                rowSum += p;
            }
            lseFp32[LseElem(spec, t, h)] = std::log(rowSum) + rowMax;
            for (int64_t d = 0; d < kHeadDim; ++d) {
                float o = 0.0f;
                for (int64_t c = 0; c < causalLen; ++c) {
                    const int64_t kvTok = spec.pagedKv ? (0 * spec.blockSize + c) : c;
                    float vv = Bf16ToFloat(v[KvElem(spec, kvTok, kvh, d)]);
                    o += scores[static_cast<size_t>(c)] * vv;
                }
                outFp32[QElem(spec, t, h, d)] = o / rowSum;
            }
        }
    }
}

bool CloseEnough(float a, float b)
{
    float diff = std::fabs(a - b);
    float scale = std::max(std::fabs(a), std::fabs(b));
    return diff <= kAbsTol + kRelTol * scale;
}

int CompareOut(const CaseSpec &spec, const std::vector<uint16_t> &got, const std::vector<float> &golden,
               const char *tag)
{
    int64_t n = spec.seq * spec.numHeads * kHeadDim;
    int64_t mismatch = 0;
    float maxAbs = 0.0f;
    int64_t nanCount = 0;
    int64_t firstMis = -1;
    int64_t lastMis = -1;
    std::vector<int64_t> headHit(static_cast<size_t>(spec.seq * spec.numHeads), 0);
    for (int64_t i = 0; i < n; ++i) {
        float g = Bf16ToFloat(got[static_cast<size_t>(i)]);
        if (!std::isfinite(g)) {
            ++nanCount;
            ++mismatch;
            if (firstMis < 0) {
                firstMis = i;
            }
            lastMis = i;
            headHit[static_cast<size_t>(i / kHeadDim)]++;
            continue;
        }
        float diff = std::fabs(g - golden[static_cast<size_t>(i)]);
        maxAbs = std::max(maxAbs, diff);
        if (!CloseEnough(g, golden[static_cast<size_t>(i)])) {
            ++mismatch;
            if (mismatch <= 8) {
                LOG_PRINT("  mismatch[%ld] got=%f golden=%f abs=%f\n", i, g, golden[static_cast<size_t>(i)], diff);
            }
            if (firstMis < 0) {
                firstMis = i;
            }
            lastMis = i;
            headHit[static_cast<size_t>(i / kHeadDim)]++;
        }
    }
    LOG_PRINT("[%s] elems=%ld mismatch=%ld nan/inf=%ld maxAbsErr=%f first3=[%f, %f, %f] misRange=[%ld, %ld]\n", tag, n,
              mismatch, nanCount, maxAbs, Bf16ToFloat(got[0]), Bf16ToFloat(got[1]), Bf16ToFloat(got[2]), firstMis,
              lastMis);
    int printedHeads = 0;
    for (int64_t th = 0; th < spec.seq * spec.numHeads; ++th) {
        if (headHit[static_cast<size_t>(th)] == 0) {
            continue;
        }
        LOG_PRINT("  hole token=%ld head=%ld count=%ld\n", th / spec.numHeads, th % spec.numHeads,
                  headHit[static_cast<size_t>(th)]);
        if (++printedHeads >= 16) {
            LOG_PRINT("  ... more heads omitted\n");
            break;
        }
    }
    return (mismatch == 0 && nanCount == 0) ? 0 : 1;
}

int CompareLse(const CaseSpec &spec, const std::vector<float> &got, const std::vector<float> &golden, const char *tag)
{
    int64_t n = spec.seq * spec.numHeads;
    int64_t mismatch = 0;
    float maxAbs = 0.0f;
    int64_t nanCount = 0;
    for (int64_t i = 0; i < n; ++i) {
        float g = got[static_cast<size_t>(i)];
        if (!std::isfinite(g) || !std::isfinite(golden[static_cast<size_t>(i)])) {
            ++nanCount;
            ++mismatch;
            continue;
        }
        float diff = std::fabs(g - golden[static_cast<size_t>(i)]);
        maxAbs = std::max(maxAbs, diff);
        if (!CloseEnough(g, golden[static_cast<size_t>(i)])) {
            ++mismatch;
            if (mismatch <= 8) {
                LOG_PRINT("  lse mismatch[%ld] got=%f golden=%f abs=%f\n", i, g, golden[static_cast<size_t>(i)], diff);
            }
        }
    }
    LOG_PRINT("[%s] lse elems=%ld mismatch=%ld nan/inf=%ld maxAbsErr=%f first3=[%f, %f, %f]\n", tag, n, mismatch,
              nanCount, maxAbs, got[0], got[1], got[2]);
    return (mismatch == 0 && nanCount == 0) ? 0 : 1;
}

struct CaseTensors {
    aclTensor *query = nullptr;
    aclTensor *key = nullptr;
    aclTensor *value = nullptr;
    aclTensor *blockTable = nullptr;
    aclTensor *k2qRowPtr = nullptr;
    aclTensor *k2qQIndices = nullptr;
    aclTensor *k2qSlotIndices = nullptr;
    aclTensor *actualSeqQ = nullptr;
    aclTensor *actualSeqKv = nullptr;
    aclTensor *attentionOut = nullptr;
    aclTensor *softmaxLse = nullptr;
    void *queryAddr = nullptr;
    void *keyAddr = nullptr;
    void *valueAddr = nullptr;
    void *blockTableAddr = nullptr;
    void *k2qRowPtrAddr = nullptr;
    void *k2qQAddr = nullptr;
    void *k2qSlotAddr = nullptr;
    void *seqQAddr = nullptr;
    void *seqKvAddr = nullptr;
    void *outAddr = nullptr;
    void *lseAddr = nullptr;
    void *workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<uint16_t> qHost;
    std::vector<uint16_t> kHost;
    std::vector<uint16_t> vHost;
    std::vector<uint16_t> outHost;
    std::vector<float> lseHost;
};

int BuildCase(const CaseSpec &spec, CaseTensors &c, bool ones)
{
    std::vector<int64_t> qShape;
    std::vector<int64_t> kvShape;
    if (spec.pagedKv) {
        qShape = {spec.seq, spec.numHeads, kHeadDim};
        kvShape = {1, spec.blockSize, spec.kvHeads, kHeadDim};
    } else if (spec.layout == LayoutKind::BNSD) {
        qShape = {1, spec.numHeads, spec.seq, kHeadDim};
        kvShape = {1, spec.kvHeads, spec.seq, kHeadDim};
    } else if (spec.layout == LayoutKind::BSND) {
        qShape = {1, spec.seq, spec.numHeads, kHeadDim};
        kvShape = {1, spec.seq, spec.kvHeads, kHeadDim};
    } else {
        qShape = {spec.seq, spec.numHeads, kHeadDim};
        kvShape = {spec.seq, spec.kvHeads, kHeadDim};
    }
    const std::vector<int64_t> btShape = {1 + spec.padBatches, 1};
    const std::vector<int64_t> rowPtrShape = {spec.kvHeads, 2};
    const std::vector<int64_t> csrShape = {spec.kvHeads, spec.seq * spec.topK};
    const std::vector<int64_t> seqShape = {1 + spec.padBatches};

    c.qHost.assign(static_cast<size_t>(GetShapeSize(qShape)), 0);
    c.kHost.assign(static_cast<size_t>(GetShapeSize(kvShape)), 0);
    c.vHost.assign(static_cast<size_t>(GetShapeSize(kvShape)), 0);
    c.outHost.assign(static_cast<size_t>(GetShapeSize(qShape)), 0);
    if (ones) {
        FillOnes(c.qHost);
        FillOnes(c.kHost);
        FillOnes(c.vHost);
    } else {
        FillRandom(c.qHost, 1);
        FillRandom(c.kHost, 2);
        FillRandom(c.vHost, 3);
    }

    std::vector<int32_t> rowPtr(static_cast<size_t>(spec.kvHeads * 2));
    for (int64_t h = 0; h < spec.kvHeads; ++h) {
        rowPtr[static_cast<size_t>(h * 2)] = 0;
        rowPtr[static_cast<size_t>(h * 2 + 1)] = static_cast<int32_t>(spec.seq);
    }
    std::vector<int32_t> qIdx(static_cast<size_t>(spec.kvHeads * spec.seq * spec.topK), 0);
    std::vector<int32_t> slotIdx(qIdx.size(), 0);
    for (int64_t h = 0; h < spec.kvHeads; ++h) {
        for (int32_t i = 0; i < static_cast<int32_t>(spec.seq); ++i) {
            qIdx[static_cast<size_t>(h * spec.seq * spec.topK + i)] = i;
        }
    }
    std::vector<int32_t> seqQ(static_cast<size_t>(spec.padBatches), 0);
    seqQ.push_back(static_cast<int32_t>(spec.seq));
    std::vector<int32_t> seqKv = seqQ;

    int ret = CreateAclTensor(c.qHost, qShape, &c.queryAddr, ACL_BF16, &c.query);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(c.kHost, kvShape, &c.keyAddr, ACL_BF16, &c.key);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(c.vHost, kvShape, &c.valueAddr, ACL_BF16, &c.value);
    CHECK_RET(ret == 0, return ret);
    if (spec.pagedKv) {
        std::vector<int32_t> blockTable(static_cast<size_t>(1 + spec.padBatches), 0);
        ret = CreateAclTensor(blockTable, btShape, &c.blockTableAddr, ACL_INT32, &c.blockTable);
        CHECK_RET(ret == 0, return ret);
    } else {
        c.blockTable = nullptr;
        c.blockTableAddr = nullptr;
    }
    ret = CreateAclTensor(rowPtr, rowPtrShape, &c.k2qRowPtrAddr, ACL_INT32, &c.k2qRowPtr);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(qIdx, csrShape, &c.k2qQAddr, ACL_INT32, &c.k2qQIndices);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(slotIdx, csrShape, &c.k2qSlotAddr, ACL_INT32, &c.k2qSlotIndices);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(seqQ, seqShape, &c.seqQAddr, ACL_INT32, &c.actualSeqQ);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(seqKv, seqShape, &c.seqKvAddr, ACL_INT32, &c.actualSeqKv);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(c.outHost, qShape, &c.outAddr, ACL_BF16, &c.attentionOut);
    CHECK_RET(ret == 0, return ret);
    std::vector<int64_t> lseShape;
    if (spec.layout == LayoutKind::BNSD) {
        lseShape = {1, spec.numHeads, spec.seq, 1};
    } else if (spec.layout == LayoutKind::BSND) {
        lseShape = {1, spec.seq, spec.numHeads, 1};
    } else {
        lseShape = {spec.seq, spec.numHeads, 1};
    }
    c.lseHost.assign(static_cast<size_t>(GetShapeSize(lseShape)), 0.0f);
    ret = CreateAclTensor(c.lseHost, lseShape, &c.lseAddr, ACL_FLOAT, &c.softmaxLse);
    CHECK_RET(ret == 0, return ret);
    return 0;
}

void DestroyCase(CaseTensors &c)
{
    aclDestroyTensor(c.query);
    aclDestroyTensor(c.key);
    aclDestroyTensor(c.value);
    if (c.blockTable != nullptr) {
        aclDestroyTensor(c.blockTable);
    }
    aclDestroyTensor(c.k2qRowPtr);
    aclDestroyTensor(c.k2qQIndices);
    aclDestroyTensor(c.k2qSlotIndices);
    aclDestroyTensor(c.actualSeqQ);
    aclDestroyTensor(c.actualSeqKv);
    aclDestroyTensor(c.attentionOut);
    if (c.softmaxLse != nullptr) {
        aclDestroyTensor(c.softmaxLse);
    }
    aclrtFree(c.queryAddr);
    aclrtFree(c.keyAddr);
    aclrtFree(c.valueAddr);
    if (c.blockTableAddr != nullptr) {
        aclrtFree(c.blockTableAddr);
    }
    aclrtFree(c.k2qRowPtrAddr);
    aclrtFree(c.k2qQAddr);
    aclrtFree(c.k2qSlotAddr);
    aclrtFree(c.seqQAddr);
    aclrtFree(c.seqKvAddr);
    aclrtFree(c.outAddr);
    if (c.lseAddr != nullptr) {
        aclrtFree(c.lseAddr);
    }
    if (c.workspaceAddr != nullptr) {
        aclrtFree(c.workspaceAddr);
    }
}

int RunOnce(aclrtStream stream, const CaseSpec &spec, bool ones, int64_t innerPrecise, bool softmaxLseFlag,
            const char *tag)
{
    CaseTensors c;
    int ret = BuildCase(spec, c, ones);
    CHECK_RET(ret == 0, return ret);

    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    aclOpExecutor *executor = nullptr;
    ret = aclnnMinimaxSparseAttentionSplitKvGetWorkspaceSize(
        c.query, c.key, c.value, c.blockTable, c.k2qRowPtr, c.k2qQIndices, c.k2qSlotIndices, c.actualSeqQ,
        c.actualSeqKv, spec.kvHeads, static_cast<double>(scale), spec.blockSize, spec.topK, innerPrecise,
        softmaxLseFlag, LayoutStr(spec), c.attentionOut, softmaxLseFlag ? c.softmaxLse : nullptr, &c.workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] GetWorkspaceSize failed. ERROR: %d\n", tag, ret); DestroyCase(c);
              return ret);
    LOG_PRINT("[%s] workspaceSize=%llu T=%ld BS=%ld Hq/Hkv=%ld/%ld lse=%d\n", tag,
              static_cast<unsigned long long>(c.workspaceSize), spec.seq, spec.blockSize, spec.numHeads, spec.kvHeads,
              softmaxLseFlag ? 1 : 0);

    if (c.workspaceSize > 0) {
        ret = aclrtMalloc(&c.workspaceAddr, c.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); DestroyCase(c);
                  return ret);
    }

    ret = aclnnMinimaxSparseAttentionSplitKv(c.workspaceAddr, c.workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] launch failed. ERROR: %d\n", tag, ret); DestroyCase(c); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] sync failed. ERROR: %d\n", tag, ret); DestroyCase(c); return ret);

    auto outBytes = static_cast<size_t>(spec.seq * spec.numHeads * kHeadDim) * sizeof(uint16_t);
    ret = aclrtMemcpy(c.outHost.data(), outBytes, c.outAddr, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out failed. ERROR: %d\n", ret); DestroyCase(c); return ret);

    std::vector<float> golden;
    std::vector<float> goldenLse;
    CpuGolden(spec, c.qHost, c.kHost, c.vHost, golden, goldenLse, scale, innerPrecise == 0);
    int cmp = CompareOut(spec, c.outHost, golden, tag);
    if (softmaxLseFlag) {
        auto lseBytes = static_cast<size_t>(spec.seq * spec.numHeads) * sizeof(float);
        ret = aclrtMemcpy(c.lseHost.data(), lseBytes, c.lseAddr, lseBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy lse failed. ERROR: %d\n", ret); DestroyCase(c); return ret);
        cmp |= CompareLse(spec, c.lseHost, goldenLse, tag);
    }
    DestroyCase(c);
    return cmp;
}

} // namespace

int main()
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int32_t deviceId = 0;
    if (const char *env = std::getenv("ASCEND_DEVICE_ID")) {
        deviceId = static_cast<int32_t>(std::atoi(env));
    }
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const CaseSpec kCases[] = {
        // baseline: former hole at token 2 / head 4 (M-row 36 of a 64-row tile)
        {32, 32, 16, 1, 1, "base_t32_g16", true},
        // P_ZN_CHUNK leftovers: 1 / 2 / 3 groups => M=16 / 32 / 48
        {4, 32, 16, 1, 1, "m16_one_group", true},
        {8, 32, 16, 1, 1, "m32_one_chunk", true},
        {12, 32, 16, 1, 1, "m48_chunk_32_16", true},
        // tail batch after full 4-group tiles
        {17, 32, 16, 1, 1, "tail_t17", true},
        // GQA and other groupSize (still M=64 high-prec tiles)
        {32, 32, 32, 2, 1, "gqa_h32_kv2", true},
        {32, 32, 8, 1, 1, "g8", true},
        // larger K / production-ish blockSize
        {64, 64, 16, 1, 1, "bs64_t64", true},
        {32, 128, 16, 1, 1, "bs128_t32", true},
        {16, 64, 64, 4, 1, "prod_gqa_h64_kv4", true},
        // TND non-kvcache: key/value [T, kvHeads, D], no block_table
        {32, 32, 16, 1, 1, "tnd_t32_g16", false},
        {32, 32, 32, 2, 1, "tnd_gqa_h32_kv2", false},
        {32, 128, 16, 1, 1, "tnd_bs128_t32", false},
        {16, 64, 64, 4, 1, "tnd_prod_gqa_h64_kv4", false},
        // TND/PA dummy request in front: q_len=kv_len=0 (packedRow 0 is the real batch)
        {32, 32, 16, 1, 1, "pad0_pa_t32", true, LayoutKind::TND, 1},
        {32, 32, 16, 1, 1, "pad0_tnd_t32", false, LayoutKind::TND, 1},
        // BNSD contiguous training: query/key/value [1, N, S, D], no block_table
        {32, 32, 16, 1, 1, "bnsd_t32_g16", false, LayoutKind::BNSD},
        {32, 32, 32, 2, 1, "bnsd_gqa_h32_kv2", false, LayoutKind::BNSD},
        {16, 64, 64, 4, 1, "bnsd_prod_gqa_h64_kv4", false, LayoutKind::BNSD},
        // BSND contiguous: query/key/value [1, S, N, D], no block_table
        {32, 32, 16, 1, 1, "bsnd_t32_g16", false, LayoutKind::BSND},
        {32, 32, 32, 2, 1, "bsnd_gqa_h32_kv2", false, LayoutKind::BSND},
        {16, 64, 64, 4, 1, "bsnd_prod_gqa_h64_kv4", false, LayoutKind::BSND},
    };

    const char *ipEnv = std::getenv("MSA_INNER_PRECISE");
    const bool run0 = (ipEnv == nullptr) || (std::string(ipEnv).find('0') != std::string::npos);
    const bool run4 = (ipEnv == nullptr) || (std::string(ipEnv).find('4') != std::string::npos);
    const char *filter = std::getenv("MSA_CASE");

    int fail = 0;
    int ran = 0;
    for (const auto &spec : kCases) {
        if (filter != nullptr && std::string(spec.name).find(filter) == std::string::npos) {
            continue;
        }
        if (run4) {
            char tag[128];
            std::snprintf(tag, sizeof(tag), "%s_ones_p4", spec.name);
            fail |= RunOnce(stream, spec, true, 4, false, tag);
            ++ran;
            std::snprintf(tag, sizeof(tag), "%s_rand_p4", spec.name);
            fail |= RunOnce(stream, spec, false, 4, false, tag);
            ++ran;
        }
        if (run0) {
            char tag[128];
            std::snprintf(tag, sizeof(tag), "%s_ones_p0", spec.name);
            fail |= RunOnce(stream, spec, true, 0, false, tag);
            ++ran;
            std::snprintf(tag, sizeof(tag), "%s_rand_p0", spec.name);
            fail |= RunOnce(stream, spec, false, 0, false, tag);
            ++ran;
        }
    }

    const CaseSpec kLseCases[] = {
        {32, 32, 16, 1, 1, "base_t32_g16", true},
        {32, 32, 32, 2, 1, "gqa_h32_kv2", true},
        {32, 32, 16, 1, 1, "tnd_t32_g16", false},
        {16, 64, 64, 4, 1, "prod_gqa_h64_kv4", true},
        {32, 32, 16, 1, 1, "bnsd_t32_g16", false, LayoutKind::BNSD},
        {32, 32, 16, 1, 1, "bsnd_t32_g16", false, LayoutKind::BSND},
    };
    for (const auto &spec : kLseCases) {
        if (filter != nullptr && std::string(spec.name).find(filter) == std::string::npos) {
            continue;
        }
        if (run4) {
            char tag[128];
            std::snprintf(tag, sizeof(tag), "%s_rand_p4_lse", spec.name);
            fail |= RunOnce(stream, spec, false, 4, true, tag);
            ++ran;
        }
        if (run0) {
            char tag[128];
            std::snprintf(tag, sizeof(tag), "%s_ones_p0_lse", spec.name);
            fail |= RunOnce(stream, spec, true, 0, true, tag);
            ++ran;
        }
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    if (fail != 0) {
        LOG_PRINT("MinimaxSparseAttentionSplitKv device test FAILED (%d runs)\n", ran);
        return 1;
    }
    LOG_PRINT("MinimaxSparseAttentionSplitKv device test PASSED (%d runs)\n", ran);
    return 0;
}
