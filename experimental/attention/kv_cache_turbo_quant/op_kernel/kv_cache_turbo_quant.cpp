/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#include "kv_cache_turbo_quant_tiling.h"

using namespace AscendC;

template <HardEvent event>
__aicore__ inline void Sync()
{
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

template <uint32_t BITS>
class KvCacheTurboQuant {
public:
    __aicore__ inline void Init(GM_ADDR kv, GM_ADDR rotation, GM_ADDR qjl, GM_ADDR idxOut, GM_ADDR qjlOut,
                                GM_ADDR normOut, GM_ADDR gammaOut, uint32_t vectors, uint32_t blocks,
                                uint32_t idxStride, uint32_t qjlStride, uint32_t rotationDtype = 0,
                                uint32_t qjlDtype = 0, uint32_t outputDtype = 0)
    {
        vectors_ = vectors;
        blocks_ = blocks;
        idxStride_ = idxStride;
        qjlStride_ = qjlStride;
        rotationDtype_ = rotationDtype;
        qjlDtype_ = qjlDtype;
        outputDtype_ = outputDtype;
        input_.SetGlobalBuffer((__gm__ bfloat16_t*)kv);
        rotation_.SetGlobalBuffer((__gm__ float*)rotation);
        qjl_.SetGlobalBuffer((__gm__ float*)qjl);
        rotationBf_.SetGlobalBuffer((__gm__ bfloat16_t*)rotation);
        qjlBf_.SetGlobalBuffer((__gm__ bfloat16_t*)qjl);
        idxOut_.SetGlobalBuffer((__gm__ uint8_t*)idxOut);
        qjlOut_.SetGlobalBuffer((__gm__ uint8_t*)qjlOut);
        normOut_.SetGlobalBuffer((__gm__ bfloat16_t*)normOut);
        gammaOut_.SetGlobalBuffer((__gm__ bfloat16_t*)gammaOut);
        normOutHalf_.SetGlobalBuffer((__gm__ half*)normOut);
        gammaOutHalf_.SetGlobalBuffer((__gm__ half*)gammaOut);

        pipe_.InitBuffer(matrixBuf_, 2 * 128 * 128 * sizeof(float));
        pipe_.InitBuffer(fpBuf_, 784 * sizeof(float));
        pipe_.InitBuffer(bfBuf_, 128 * sizeof(bfloat16_t));
        pipe_.InitBuffer(packedBuf_, 96);

        auto matrices = matrixBuf_.Get<float>();
        rotationLocal_ = matrices;
        qjlLocal_ = matrices[128 * 128];
        auto fp = fpBuf_.Get<float>();
        x_ = fp;
        rotated_ = fp[128];
        residual_ = fp[256];
        projected_ = fp[384];
        work_ = fp[512];
        reduced_ = fp[640];
        meta_ = fp[648];
        matrixRow_ = fp[656];
        inputBf_ = bfBuf_.Get<bfloat16_t>();
        outputHalf_ = bfBuf_.Get<half>();
        auto packed = packedBuf_.Get<uint8_t>();
        idxPacked_ = packed;
        qjlPacked_ = packed[64];

        if (rotationDtype_ == 0)
            DataCopy(rotationLocal_, rotation_, 128 * 128);
        if (qjlDtype_ == 0)
            DataCopy(qjlLocal_, qjl_, 128 * 128);
        if (rotationDtype_ == 0 || qjlDtype_ == 0)
            Sync<HardEvent::MTE2_V>();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t vector = GetBlockIdx(); vector < vectors_; vector += blocks_) {
            Encode(vector);
        }
    }

    static __aicore__ inline float Centroid(uint32_t index)
    {
        if constexpr (BITS == 2) {
            constexpr float values[4] = {-0.1335033178f, -0.04002048075f, 0.04002048075f, 0.1335033178f};
            return values[index];
        } else if constexpr (BITS == 3) {
            constexpr float values[8] = {-0.19020693f,   -0.1187859178f, -0.06682205945f, -0.02166347019f,
                                         0.02166347019f, 0.06682205945f, 0.1187859178f,   0.19020693f};
            return values[index];
        } else {
            constexpr float values[16] = {-0.2414890379f,  -0.1828317791f,  -0.1429702938f,  -0.1109927073f,
                                          -0.08325428516f, -0.05802082643f, -0.03428063914f, -0.01134236995f,
                                          0.01134236995f,  0.03428063914f,  0.05802082643f,  0.08325428516f,
                                          0.1109927073f,   0.1429702938f,   0.1828317791f,   0.2414890379f};
            return values[index];
        }
    }

    static __aicore__ inline uint32_t Quantize(float value)
    {
        if constexpr (BITS == 2) {
            constexpr float bounds[3] = {-0.08676189929f, 0.0f, 0.08676189929f};
            uint32_t index = 0;
            for (uint32_t i = 0; i < 3; ++i)
                index += value > bounds[i];
            return index;
        } else if constexpr (BITS == 3) {
            constexpr float bounds[7] = {-0.1544964164f, -0.09280399233f, -0.04424276203f, 0.0f,
                                         0.04424276203f, 0.09280399233f,  0.1544964164f};
            uint32_t index = 0;
            for (uint32_t i = 0; i < 7; ++i)
                index += value > bounds[i];
            return index;
        } else {
            constexpr float bounds[15] = {-0.2121604085f,  -0.162901029f,   -0.1269814968f,  -0.09712349623f,
                                          -0.07063755393f, -0.04615073279f, -0.02281150408f, 0.0f,
                                          0.02281150408f,  0.04615073279f,  0.07063755393f,  0.09712349623f,
                                          0.1269814968f,   0.162901029f,    0.2121604085f};
            uint32_t index = 0;
            for (uint32_t i = 0; i < 15; ++i)
                index += value > bounds[i];
            return index;
        }
    }

private:
    __aicore__ inline float SumSquares(const LocalTensor<float>& value)
    {
        Mul(work_, value, value, 128);
        PipeBarrier<PIPE_V>();
        Add(work_, work_, work_[64], 64);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(reduced_, work_, 64, 1, 1, 1, 8);
        Sync<HardEvent::V_S>();
        return reduced_.GetValue(0);
    }

    __aicore__ inline void MatVec(const LocalTensor<float>& matrix, const LocalTensor<float>& vector,
                                  const LocalTensor<float>& output)
    {
        for (uint32_t row = 0; row < 128; ++row) {
            Mul(work_, vector, matrix[row * 128], 128);
            PipeBarrier<PIPE_V>();
            Add(work_, work_, work_[64], 64);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(output[row], work_, 64, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void MatVecBf(const GlobalTensor<bfloat16_t>& matrix, const LocalTensor<float>& vector,
                                    const LocalTensor<float>& output)
    {
        for (uint32_t row = 0; row < 128; ++row) {
            DataCopy(inputBf_, matrix[row * 128], 128);
            Sync<HardEvent::MTE2_V>();
            Cast(matrixRow_, inputBf_, RoundMode::CAST_NONE, 128);
            PipeBarrier<PIPE_V>();
            Mul(work_, vector, matrixRow_, 128);
            PipeBarrier<PIPE_V>();
            Add(work_, work_, work_[64], 64);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(output[row], work_, 64, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Encode(uint32_t vector)
    {
        constexpr uint32_t idxBytes = 16 * BITS;
        DataCopy(inputBf_, input_[static_cast<uint64_t>(vector) * 128], 128);
        Sync<HardEvent::MTE2_V>();
        Cast(x_, inputBf_, RoundMode::CAST_NONE, 128);
        PipeBarrier<PIPE_V>();
        float norm = sqrt(SumSquares(x_));
        if (norm > 0) {
            Duplicate(work_, norm, 128);
            PipeBarrier<PIPE_V>();
            Div(x_, x_, work_, 128);
        } else {
            Duplicate(x_, 0.0f, 128);
        }
        PipeBarrier<PIPE_V>();

        if (rotationDtype_ == 0)
            MatVec(rotationLocal_, x_, rotated_);
        else
            MatVecBf(rotationBf_, x_, rotated_);
        Sync<HardEvent::V_S>();
        for (uint32_t group = 0; group < 16; ++group) {
            uint32_t packed = 0;
            for (uint32_t i = 0; i < 8; ++i) {
                uint32_t coordinate = group * 8 + i;
                float value = rotated_.GetValue(coordinate);
                uint32_t index = Quantize(value);
                residual_.SetValue(coordinate, value - Centroid(index));
                packed |= index << (i * BITS);
            }
            for (uint32_t byte = 0; byte < BITS; ++byte) {
                idxPacked_.SetValue(group * BITS + byte, static_cast<uint8_t>(packed >> (byte * 8)));
            }
        }
        Sync<HardEvent::S_V>();
        float residualNorm = sqrt(SumSquares(residual_));
        float gamma = norm * residualNorm;

        if (residualNorm > 0) {
            Duplicate(work_, residualNorm, 128);
            PipeBarrier<PIPE_V>();
            Div(residual_, residual_, work_, 128);
            PipeBarrier<PIPE_V>();
        }
        if (qjlDtype_ == 0)
            MatVec(qjlLocal_, residual_, projected_);
        else
            MatVecBf(qjlBf_, residual_, projected_);
        Sync<HardEvent::V_S>();
        for (uint32_t byte = 0; byte < 16; ++byte) {
            uint8_t packed = 0;
            for (uint32_t bit = 0; bit < 8; ++bit) {
                packed |= static_cast<uint8_t>((projected_.GetValue(byte * 8 + bit) >= 0) << bit);
            }
            qjlPacked_.SetValue(byte, packed);
        }
        meta_.SetValue(0, norm);
        meta_.SetValue(1, gamma);
        Sync<HardEvent::S_V>();
        UnaryRepeatParams castParams{1, 1, 1, 1};
        if (outputDtype_ == 0) {
            Cast(inputBf_, meta_, RoundMode::CAST_RINT, 2, 1, castParams);
        } else {
            Cast(outputHalf_, meta_, RoundMode::CAST_NONE, 2, 1, castParams);
        }
        Sync<HardEvent::V_S>();
        Sync<HardEvent::V_S>();
        auto metaBits =
            outputDtype_ == 0 ? inputBf_.ReinterpretCast<uint16_t>() : outputHalf_.ReinterpretCast<uint16_t>();
        metaBits.SetValue(16, metaBits.GetValue(1));
        Sync<HardEvent::S_MTE3>();
        DataCopyExtParams idxCopy{1, idxBytes, 0, 0, 0};
        DataCopyPad(idxOut_[static_cast<uint64_t>(vector) * idxStride_], idxPacked_, idxCopy);
        DataCopyExtParams qjlCopy{1, 16, 0, 0, 0};
        DataCopyPad(qjlOut_[static_cast<uint64_t>(vector) * qjlStride_], qjlPacked_, qjlCopy);
        // Scalar metadata is logically contiguous for every execution path.
        // An unaligned scalar copy avoids the padded 16-element private layout
        // formerly used only by the direct Torch launcher.
        DataCopyExtParams scalarCopy{1, sizeof(bfloat16_t), 0, 0, 0};
        if (outputDtype_ == 0) {
            DataCopyPad(normOut_[vector], inputBf_, scalarCopy);
            DataCopyPad(gammaOut_[vector], inputBf_[16], scalarCopy);
        } else {
            DataCopyPad(normOutHalf_[vector], outputHalf_, scalarCopy);
            DataCopyPad(gammaOutHalf_[vector], outputHalf_[16], scalarCopy);
        }
        Sync<HardEvent::MTE3_S>();
        Sync<HardEvent::MTE3_MTE2>();
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> matrixBuf_, fpBuf_, bfBuf_, packedBuf_;
    LocalTensor<float> rotationLocal_, qjlLocal_;
    LocalTensor<float> x_, rotated_, residual_, projected_, work_, reduced_, meta_, matrixRow_;
    LocalTensor<bfloat16_t> inputBf_;
    LocalTensor<half> outputHalf_;
    LocalTensor<uint8_t> idxPacked_, qjlPacked_;
    GlobalTensor<bfloat16_t> input_, normOut_, gammaOut_;
    GlobalTensor<float> rotation_, qjl_;
    GlobalTensor<bfloat16_t> rotationBf_, qjlBf_;
    GlobalTensor<half> normOutHalf_, gammaOutHalf_;
    GlobalTensor<uint8_t> idxOut_, qjlOut_;
    uint32_t vectors_ = 0;
    uint32_t blocks_ = 0;
    uint32_t idxStride_ = 0;
    uint32_t qjlStride_ = 0;
    uint32_t rotationDtype_ = 0, qjlDtype_ = 0, outputDtype_ = 0;
};

// Generic fallback for the verified non-128 dimension combinations. A single
// matrix row is staged in UB, keeping UB usage bounded for dimensions up to
// 256 while preserving the optimized 128x128 mixed path above.
template <uint32_t BITS, typename Meta = bfloat16_t, bool HALF_OUT = false>
class KvCacheTurboQuantGeneric {
public:
    __aicore__ inline void Init(GM_ADDR kv, GM_ADDR rotation, GM_ADDR qjl, GM_ADDR idxOut, GM_ADDR qjlOut,
                                GM_ADDR normOut, GM_ADDR gammaOut, uint32_t vectors, uint32_t blocks, uint32_t headDim,
                                uint32_t qjlDim, uint32_t idxStride, uint32_t qjlStride, float codebookScale)
    {
        vectors_ = vectors;
        blocks_ = blocks;
        headDim_ = headDim;
        qjlDim_ = qjlDim;
        idxStride_ = idxStride;
        qjlStride_ = qjlStride;
        codebookScale_ = codebookScale;
        input_.SetGlobalBuffer((__gm__ bfloat16_t*)kv);
        rotation_.SetGlobalBuffer((__gm__ float*)rotation);
        qjl_.SetGlobalBuffer((__gm__ float*)qjl);
        idxOut_.SetGlobalBuffer((__gm__ uint8_t*)idxOut);
        qjlOut_.SetGlobalBuffer((__gm__ uint8_t*)qjlOut);
        normOut_.SetGlobalBuffer((__gm__ Meta*)normOut);
        gammaOut_.SetGlobalBuffer((__gm__ Meta*)gammaOut);

        pipe_.InitBuffer(matrixBuf_, 256 * sizeof(float));
        pipe_.InitBuffer(fpBuf_, 8 * 256 * sizeof(float));
        pipe_.InitBuffer(bfBuf_, 320 * sizeof(bfloat16_t));
        pipe_.InitBuffer(packedBuf_, 160);
        matrixRow_ = matrixBuf_.Get<float>();
        auto fp = fpBuf_.Get<float>();
        x_ = fp;
        rotated_ = fp[256];
        residual_ = fp[512];
        projected_ = fp[768];
        meta_ = fp[1288];
        inputBf_ = bfBuf_.Get<bfloat16_t>();
        metaOut_ = bfBuf_.Get<Meta>();
        auto packed = packedBuf_.Get<uint8_t>();
        idxPacked_ = packed;
        qjlPacked_ = packed[128];
    }

    __aicore__ inline void Process()
    {
        for (uint32_t vector = GetBlockIdx(); vector < vectors_; vector += blocks_) {
            Encode(vector);
        }
    }

private:
    __aicore__ inline float SumSquares(const LocalTensor<float>& value, uint32_t count)
    {
        float sum = 0.0f;
        for (uint32_t i = 0; i < count; ++i) {
            float v = value.GetValue(i);
            sum += v * v;
        }
        return sum;
    }

    __aicore__ inline void MatVec(const GlobalTensor<float>& matrix, const LocalTensor<float>& vector,
                                  const LocalTensor<float>& output, uint32_t rows, uint32_t cols)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            DataCopy(matrixRow_, matrix[static_cast<uint64_t>(row) * cols], cols);
            Sync<HardEvent::MTE2_V>();
            float sum = 0.0f;
            for (uint32_t col = 0; col < cols; ++col) {
                sum += matrixRow_.GetValue(col) * vector.GetValue(col);
            }
            output.SetValue(row, sum);
        }
    }

    __aicore__ inline void Encode(uint32_t vector)
    {
        const uint32_t idxBytes = (headDim_ * BITS + 7) / 8;
        const uint32_t qjlBytes = (qjlDim_ + 7) / 8;
        DataCopy(inputBf_, input_[static_cast<uint64_t>(vector) * headDim_], headDim_);
        Sync<HardEvent::MTE2_V>();
        Cast(x_, inputBf_, RoundMode::CAST_NONE, headDim_);
        Sync<HardEvent::V_S>();
        float norm = sqrt(SumSquares(x_, headDim_));
        if (norm > 0.0f) {
            for (uint32_t i = 0; i < headDim_; ++i) {
                x_.SetValue(i, x_.GetValue(i) / norm);
            }
        } else {
            for (uint32_t i = 0; i < headDim_; ++i)
                x_.SetValue(i, 0.0f);
        }
        MatVec(rotation_, x_, rotated_, headDim_, headDim_);

        for (uint32_t byte = 0; byte < idxBytes; ++byte)
            idxPacked_.SetValue(byte, 0);
        for (uint32_t coordinate = 0; coordinate < headDim_; ++coordinate) {
            uint32_t index = KvCacheTurboQuant<BITS>::Quantize(rotated_.GetValue(coordinate) / codebookScale_);
            residual_.SetValue(
                coordinate, rotated_.GetValue(coordinate) - KvCacheTurboQuant<BITS>::Centroid(index) * codebookScale_);
            uint32_t bit = coordinate * BITS;
            for (uint32_t b = 0; b < BITS; ++b) {
                if ((index >> b) & 1U) {
                    uint32_t dst = (bit + b) / 8;
                    uint32_t shift = (bit + b) % 8;
                    idxPacked_.SetValue(dst, static_cast<uint8_t>(idxPacked_.GetValue(dst) | (1U << shift)));
                }
            }
        }

        float residualNorm = sqrt(SumSquares(residual_, headDim_));
        float gamma = norm * residualNorm;
        if (residualNorm > 0.0f) {
            for (uint32_t i = 0; i < headDim_; ++i) {
                residual_.SetValue(i, residual_.GetValue(i) / residualNorm);
            }
        } else {
            for (uint32_t i = 0; i < headDim_; ++i)
                residual_.SetValue(i, 0.0f);
        }
        MatVec(qjl_, residual_, projected_, qjlDim_, headDim_);

        for (uint32_t byte = 0; byte < qjlBytes; ++byte)
            qjlPacked_.SetValue(byte, 0);
        for (uint32_t coordinate = 0; coordinate < qjlDim_; ++coordinate) {
            if (projected_.GetValue(coordinate) >= 0.0f) {
                qjlPacked_.SetValue(coordinate / 8, static_cast<uint8_t>(qjlPacked_.GetValue(coordinate / 8) |
                                                                         (1U << (coordinate % 8))));
            }
        }

        meta_.SetValue(0, norm);
        meta_.SetValue(1, gamma);
        Sync<HardEvent::S_V>();
        if constexpr (HALF_OUT) {
            Cast(metaOut_, meta_, RoundMode::CAST_NONE, 2);
        } else {
            Cast(metaOut_, meta_, RoundMode::CAST_RINT, 2);
        }
        Sync<HardEvent::V_S>();
        auto metaBits = metaOut_.template ReinterpretCast<uint16_t>();
        metaBits.SetValue(16, metaBits.GetValue(1));
        Sync<HardEvent::S_MTE3>();
        DataCopyExtParams idxCopy{1, idxBytes, 0, 0, 0};
        DataCopyPad(idxOut_[static_cast<uint64_t>(vector) * idxStride_], idxPacked_, idxCopy);
        DataCopyExtParams qjlCopy{1, qjlBytes, 0, 0, 0};
        DataCopyPad(qjlOut_[static_cast<uint64_t>(vector) * qjlStride_], qjlPacked_, qjlCopy);
        DataCopyExtParams scalarCopy{1, sizeof(Meta), 0, 0, 0};
        DataCopyPad(normOut_[vector], metaOut_, scalarCopy);
        DataCopyPad(gammaOut_[vector], metaOut_[16], scalarCopy);
        Sync<HardEvent::MTE3_S>();
        Sync<HardEvent::MTE3_MTE2>();
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> matrixBuf_, fpBuf_, bfBuf_, packedBuf_;
    LocalTensor<float> matrixRow_;
    LocalTensor<float> x_, rotated_, residual_, projected_, meta_;
    LocalTensor<bfloat16_t> inputBf_;
    LocalTensor<Meta> metaOut_;
    LocalTensor<uint8_t> idxPacked_, qjlPacked_;
    GlobalTensor<bfloat16_t> input_;
    GlobalTensor<Meta> normOut_, gammaOut_;
    GlobalTensor<float> rotation_, qjl_;
    GlobalTensor<uint8_t> idxOut_, qjlOut_;
    uint32_t vectors_ = 0;
    uint32_t blocks_ = 0;
    uint32_t headDim_ = 0;
    uint32_t qjlDim_ = 0;
    uint32_t idxStride_ = 0;
    uint32_t qjlStride_ = 0;
    float codebookScale_ = 1.0f;
};

#include "tq_pipeline_vector.cpp"

using PipelineA = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using PipelineB = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using PipelineC = MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulConfig TQ_FRAMEWORK_MM_CONFIG = GetNormalConfig();
constexpr MatmulApiStaticTiling TQ_FRAMEWORK_MM_TILING =
    GetMatmulApiTiling<PipelineA, PipelineB, PipelineC, PipelineC>(TQ_FRAMEWORK_MM_CONFIG);

__aicore__ inline void FrameworkProject(GM_ADDR input, GM_ADDR matrix, GM_ADDR output,
                                        const tiling::TCubeTiling* cubeTiling, uint32_t rows, uint32_t tiles,
                                        uint32_t aicBlocks)
{
    TPipe pipe;
    Matmul<PipelineA, PipelineB, PipelineC, PipelineC, TQ_FRAMEWORK_MM_TILING> mm;
    GlobalTensor<float> a, b, c;
    a.SetGlobalBuffer((__gm__ float*)input);
    b.SetGlobalBuffer((__gm__ float*)matrix);
    c.SetGlobalBuffer((__gm__ float*)output);
    mm.Init(cubeTiling, &pipe);
    for (uint32_t tile = GetBlockIdx(); tile < tiles; tile += aicBlocks) {
        mm.SetOrgShape(rows, 128, 128);
        mm.SetSingleShape(rows, 128, 128);
        mm.SetTensorA(a[static_cast<uint64_t>(tile) * rows * 128], false);
        mm.SetTensorB(b, true);
        mm.IterateAll(c[static_cast<uint64_t>(tile) * rows * 128]);
        mm.End();
    }
}

template <typename Meta = bfloat16_t, bool HALF_OUT = false>
__aicore__ inline void RunFrameworkPipeline(GM_ADDR kv, GM_ADDR rotation, GM_ADDR qjl, GM_ADDR inputFp, GM_ADDR norms,
                                            GM_ADDR rotated, GM_ADDR residual, GM_ADDR projected, GM_ADDR idx,
                                            GM_ADDR qjlOut, GM_ADDR normOut, GM_ADDR gammaOut,
                                            const optiling::KvCacheTurboQuantTilingData& data)
{
    uint32_t aicBlocks = data.blocks / 2;
    uint32_t tiles = data.vectors / 128;
    if ASCEND_IS_AIV {
        TqPrepare<64> op;
        op.Init(kv, inputFp, norms, data.vectors, data.blocks);
        op.Process();
        SyncAll();
        NotifyEvent<PIPE_MTE3>(6);
        PipeBarrier<PIPE_ALL>();
    }
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(6);
        FrameworkProject(inputFp, rotation, rotated, &data.cubeTiling, 128, tiles, aicBlocks);
        NotifyEvent<PIPE_FIX>(5);
        PipeBarrier<PIPE_ALL>();
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(5);
        SyncAll();
        PipeBarrier<PIPE_ALL>();
        if (data.mseBits == 2) {
            TqQuantize<2, 64, Meta, HALF_OUT> op;
            op.Init(rotated, norms, residual, idx, normOut, gammaOut, data.vectors, data.blocks, data.idxStride);
            op.Process();
        } else if (data.mseBits == 3) {
            TqQuantize<3, 64, Meta, HALF_OUT> op;
            op.Init(rotated, norms, residual, idx, normOut, gammaOut, data.vectors, data.blocks, data.idxStride);
            op.Process();
        } else {
            TqQuantize<4, 64, Meta, HALF_OUT> op;
            op.Init(rotated, norms, residual, idx, normOut, gammaOut, data.vectors, data.blocks, data.idxStride);
            op.Process();
        }
        SyncAll();
        NotifyEvent<PIPE_MTE3>(6);
        PipeBarrier<PIPE_ALL>();
    }
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(6);
        FrameworkProject(residual, qjl, projected, &data.cubeTiling, 128, tiles, aicBlocks);
        NotifyEvent<PIPE_FIX>(5);
        PipeBarrier<PIPE_ALL>();
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(5);
        SyncAll();
        PipeBarrier<PIPE_ALL>();
        TqSignPack<64> op;
        op.Init(projected, qjlOut, data.vectors, data.blocks);
        op.Process();
    }
}

template <typename Meta = bfloat16_t, bool HALF_OUT = false>
__aicore__ inline void RunFrameworkSmall(GM_ADDR kv, GM_ADDR rotation, GM_ADDR qjl, GM_ADDR inputFp, GM_ADDR norms,
                                         GM_ADDR rotated, GM_ADDR residual, GM_ADDR projected, GM_ADDR idx,
                                         GM_ADDR qjlOut, GM_ADDR normOut, GM_ADDR gammaOut,
                                         const optiling::KvCacheTurboQuantTilingData& data)
{
    if ASCEND_IS_AIV {
        TqPrepare<8> op;
        op.Init(kv, inputFp, norms, data.vectors, data.blocks);
        op.Process();
        SyncAll();
        NotifyEvent<PIPE_MTE3>(6);
        PipeBarrier<PIPE_ALL>();
    }
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(6);
        FrameworkProject(inputFp, rotation, rotated, &data.cubeTiling, 8, (data.vectors + 7) / 8, 1);
        NotifyEvent<PIPE_FIX>(5);
        PipeBarrier<PIPE_ALL>();
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(5);
        SyncAll();
        PipeBarrier<PIPE_ALL>();
        if (data.mseBits == 2) {
            TqQuantize<2, 8, Meta, HALF_OUT> op;
            op.Init(rotated, norms, residual, idx, normOut, gammaOut, data.vectors, data.blocks, data.idxStride);
            op.Process();
        } else if (data.mseBits == 3) {
            TqQuantize<3, 8, Meta, HALF_OUT> op;
            op.Init(rotated, norms, residual, idx, normOut, gammaOut, data.vectors, data.blocks, data.idxStride);
            op.Process();
        } else {
            TqQuantize<4, 8, Meta, HALF_OUT> op;
            op.Init(rotated, norms, residual, idx, normOut, gammaOut, data.vectors, data.blocks, data.idxStride);
            op.Process();
        }
        SyncAll();
        NotifyEvent<PIPE_MTE3>(6);
        PipeBarrier<PIPE_ALL>();
    }
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(6);
        FrameworkProject(residual, qjl, projected, &data.cubeTiling, 8, (data.vectors + 7) / 8, 1);
        NotifyEvent<PIPE_FIX>(5);
        PipeBarrier<PIPE_ALL>();
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(5);
        SyncAll();
        PipeBarrier<PIPE_ALL>();
        TqSignPack<8> op;
        op.Init(projected, qjlOut, data.vectors, data.blocks);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void kv_cache_turbo_quant(GM_ADDR kv, GM_ADDR rotation, GM_ADDR qjl, GM_ADDR idxOut,
                                                           GM_ADDR qjlOut, GM_ADDR normOut, GM_ADDR gammaOut,
                                                           GM_ADDR workspace, GM_ADDR tilingData)
{
    REGISTER_TILING_DEFAULT(optiling::KvCacheTurboQuantTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(optiling::KvCacheTurboQuantTilingData, data, tilingData);

    if (data.vectors == 0)
        return;

    // Keep the task-book FP32/BF16 -> BF16 fast path unchanged. Alternate
    // matrix/metadata dtypes use the dtype-aware 128-dimensional Vector path.
    if (data.mode == 4) {
        if ASCEND_IS_AIC {
            return;
        }
        if (data.mseBits == 2) {
            KvCacheTurboQuant<2> op;
            op.Init(kv, rotation, qjl, idxOut, qjlOut, normOut, gammaOut, data.vectors, data.blocks, data.idxStride,
                    data.qjlStride, data.rotationDtype, data.qjlDtype, data.outputDtype);
            op.Process();
        } else if (data.mseBits == 3) {
            KvCacheTurboQuant<3> op;
            op.Init(kv, rotation, qjl, idxOut, qjlOut, normOut, gammaOut, data.vectors, data.blocks, data.idxStride,
                    data.qjlStride, data.rotationDtype, data.qjlDtype, data.outputDtype);
            op.Process();
        } else {
            KvCacheTurboQuant<4> op;
            op.Init(kv, rotation, qjl, idxOut, qjlOut, normOut, gammaOut, data.vectors, data.blocks, data.idxStride,
                    data.qjlStride, data.rotationDtype, data.qjlDtype, data.outputDtype);
            op.Process();
        }
        return;
    }

    if (data.mode == 3) {
        if ASCEND_IS_AIC {
            return;
        }
        if (data.outputDtype == 1) {
            if (data.mseBits == 2) {
                KvCacheTurboQuantGeneric<2, half, true> op;
                op.Init(kv, rotation, qjl, idxOut, qjlOut, normOut, gammaOut, data.vectors, data.blocks, data.headDim,
                        data.qjlDim, data.idxStride, data.qjlStride, data.codebookScale);
                op.Process();
            } else if (data.mseBits == 3) {
                KvCacheTurboQuantGeneric<3, half, true> op;
                op.Init(kv, rotation, qjl, idxOut, qjlOut, normOut, gammaOut, data.vectors, data.blocks, data.headDim,
                        data.qjlDim, data.idxStride, data.qjlStride, data.codebookScale);
                op.Process();
            } else {
                KvCacheTurboQuantGeneric<4, half, true> op;
                op.Init(kv, rotation, qjl, idxOut, qjlOut, normOut, gammaOut, data.vectors, data.blocks, data.headDim,
                        data.qjlDim, data.idxStride, data.qjlStride, data.codebookScale);
                op.Process();
            }
        } else {
            if (data.mseBits == 2) {
                KvCacheTurboQuantGeneric<2> op;
                op.Init(kv, rotation, qjl, idxOut, qjlOut, normOut, gammaOut, data.vectors, data.blocks, data.headDim,
                        data.qjlDim, data.idxStride, data.qjlStride, data.codebookScale);
                op.Process();
            } else if (data.mseBits == 3) {
                KvCacheTurboQuantGeneric<3> op;
                op.Init(kv, rotation, qjl, idxOut, qjlOut, normOut, gammaOut, data.vectors, data.blocks, data.headDim,
                        data.qjlDim, data.idxStride, data.qjlStride, data.codebookScale);
                op.Process();
            } else {
                KvCacheTurboQuantGeneric<4> op;
                op.Init(kv, rotation, qjl, idxOut, qjlOut, normOut, gammaOut, data.vectors, data.blocks, data.headDim,
                        data.qjlDim, data.idxStride, data.qjlStride, data.codebookScale);
                op.Process();
            }
        }
        return;
    }

    if ASCEND_IS_AIC {
        if (data.mode == 0)
            return;
    }

    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    uint64_t paddedVectors = (uint64_t(data.vectors) + 7) / 8 * 8;
    uint64_t vectorElements = paddedVectors * data.headDim;
    GM_ADDR inputFp = userWorkspace;
    GM_ADDR norms = inputFp + vectorElements * sizeof(float);
    GM_ADDR rotated = norms + paddedVectors * sizeof(float);
    GM_ADDR residual = rotated + vectorElements * sizeof(float);
    GM_ADDR projected = residual + vectorElements * sizeof(float);

    if (data.mode == 1) {
        if (data.outputDtype == 1) {
            RunFrameworkSmall<half, true>(kv, rotation, qjl, inputFp, norms, rotated, residual, projected, idxOut,
                                          qjlOut, normOut, gammaOut, data);
        } else {
            RunFrameworkSmall(kv, rotation, qjl, inputFp, norms, rotated, residual, projected, idxOut, qjlOut, normOut,
                              gammaOut, data);
        }
    } else if (data.mode == 2) {
        if (data.outputDtype == 1) {
            RunFrameworkPipeline<half, true>(kv, rotation, qjl, inputFp, norms, rotated, residual, projected, idxOut,
                                             qjlOut, normOut, gammaOut, data);
        } else {
            RunFrameworkPipeline(kv, rotation, qjl, inputFp, norms, rotated, residual, projected, idxOut, qjlOut,
                                 normOut, gammaOut, data);
        }
    } else if ASCEND_IS_AIV {
        if (data.mseBits == 2) {
            KvCacheTurboQuant<2> op;
            op.Init(kv, rotation, qjl, idxOut, qjlOut, normOut, gammaOut, data.vectors, data.blocks, data.idxStride,
                    data.qjlStride, data.rotationDtype, data.qjlDtype, data.outputDtype);
            op.Process();
        } else if (data.mseBits == 3) {
            KvCacheTurboQuant<3> op;
            op.Init(kv, rotation, qjl, idxOut, qjlOut, normOut, gammaOut, data.vectors, data.blocks, data.idxStride,
                    data.qjlStride, data.rotationDtype, data.qjlDtype, data.outputDtype);
            op.Process();
        } else {
            KvCacheTurboQuant<4> op;
            op.Init(kv, rotation, qjl, idxOut, qjlOut, normOut, gammaOut, data.vectors, data.blocks, data.idxStride,
                    data.qjlStride, data.rotationDtype, data.qjlDtype, data.outputDtype);
            op.Process();
        }
    }
}
