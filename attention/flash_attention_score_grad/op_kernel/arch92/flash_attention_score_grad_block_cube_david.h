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
 * \file flash_attention_score_grad_block_cube_david.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_DAVID_H
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_DAVID_H

namespace FagBaseApi {
constexpr FixpipeConfig FAG_CFG_ROW_MAJOR_UB = {
    CO2Layout::ROW_MAJOR, true}; // ROW_MAJOR: 使能NZ2ND，输出数据格式为ND格式; true: 用于用户指定目的地址的位置是否是UB
constexpr FixpipeConfig FAG_CFG_ROW_MAJOR_GM = {
    CO2Layout::ROW_MAJOR,
    false}; // ROW_MAJOR: 使能NZ2ND，输出数据格式为ND格式; true: 用于用户指定目的地址的位置是否是UB

TEMPLATES_DEF
class FAGBlockCube {
public:
    constexpr static bool IS_FP8_INPUT = is_f8_v<T1>;
    constexpr static bool IS_FP16_INPUT = is_f16_v<T1>;
    constexpr static bool IS_FP32_INPUT = is_f32_v<T1>;
    constexpr static uint32_t CUBE_BASEM = (uint32_t)s1TemplateType;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;
    constexpr static uint32_t BASE_DQ_SIZE = CUBE_BASEM * HEAD_DIM_ALIGN;
    constexpr static uint32_t BASE_DKV_SIZE = CUBE_BASEN * HEAD_DIM_ALIGN;
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE = 32 / sizeof(T1);

    constexpr static uint32_t VECTOR_BASEM = 64;
    constexpr static uint32_t VECTOR_BASEN = 128;
    constexpr static uint32_t ROPE_D_128 = 128;
    // 同步
    static constexpr uint64_t DS_READY = 8;
    static constexpr uint64_t POUT_READY = 9;
    static constexpr uint64_t DS_RELEASE = 10;
    static constexpr uint64_t POUT_RELEASE = 11;
    static constexpr uint64_t ALL_DONE = 12;
    // L1划分
    constexpr static uint32_t Q_DX_L1_BUF_NUM =
        IS_FP32_INPUT ? 2 :
                        (IS_FP8_INPUT ? (HEAD_DIM_ALIGN <= 256 ? 4 : 2) :
                                        (HEAD_DIM_ALIGN <= 128 ? 4 : 2)); // D小于128 Q DX共用4块buff，Dk Dv做L1复用 4
    constexpr static uint32_t KV_L1_BUF_NUM =
        IS_FP32_INPUT ? 2 :
                        (IS_FP8_INPUT ? (HEAD_DIM_ALIGN <= 256 ? 3 : 2) :
                                        (HEAD_DIM_ALIGN <= 192 ? 3 : 2)); // D小于128 K V 共用3块buff，Dq做L1复用 3
    constexpr static uint32_t Q_DX_L1_BUF_SIZE =
        IS_FP32_INPUT ? 128 * 128 :
                        (HEAD_DIM_ALIGN <= 256 ? CUBE_BASEM * HEAD_DIM_ALIGN : CUBE_BASEM * 256); // 128 * 256
    constexpr static uint32_t KV_L1_BUF_SIZE =
        IS_FP32_INPUT ? 128 * 128 : (HEAD_DIM_ALIGN <= 256 ? CUBE_BASEN * HEAD_DIM_ALIGN : CUBE_BASEN * 256);
    constexpr static uint32_t DS_P_L1_BUF_SIZE = CUBE_BASEM * CUBE_BASEN;

    constexpr static uint32_t L0AB_SIZE = 64 * 1024;
    constexpr static uint32_t L0AB_BUF_NUM = 2; // L0A L0B默认开DB
    constexpr static uint32_t L0AB_DB_SIZE = L0AB_SIZE / L0AB_BUF_NUM;
    constexpr static uint32_t L0C_BUF_NUM = 4; // L0C开4 buf
    constexpr static uint32_t L0C_DBBUF_SIZE = 256 * 1024 / L0C_BUF_NUM;
    constexpr static bool IS_DKV_RESIDENT_L0C =
        IS_DKV_RESIDENT_L0C(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) && !IS_FP32_INPUT && (DETER_SPARSE_TYPE == 0);
    constexpr static uint32_t L0C_SHARED_BUF_NUM = IS_DKV_RESIDENT_L0C ? 2 : L0C_BUF_NUM;
    constexpr static uint32_t L0C_DK_BUF_IDX = 2;
    constexpr static uint32_t L0C_DV_BUF_IDX = 3;

    constexpr static uint32_t UB_BUF_NUM = IS_FP32_INPUT ? 2 : 3;
    constexpr static bool IS_TSCM_REUSE =
        IS_FP32_INPUT ? false : (IS_FP8_INPUT ? (HEAD_DIM_ALIGN <= 256) : (HEAD_DIM_ALIGN <= 128));
    constexpr static bool IS_TSCM_REUSE_K =
        IS_FP32_INPUT ? false : (IS_FP8_INPUT ? (HEAD_DIM_ALIGN <= 256) : (HEAD_DIM_ALIGN <= 192));
    constexpr static bool IS_TSCM_FULL_LOAD = IS_FP32_INPUT ? false : (HEAD_DIM_ALIGN <= 256); // L1基本块全载

    using DsType = std::conditional_t<is_e4m3_v<T1>, bfloat16_t, T1>;
    constexpr static bool ENABLE_UNITFLAG =
        HEAD_DIM_ALIGN <= static_cast<uint16_t>(DTemplateType::Aligned768) && !IS_DETER_OLD(DETER_SPARSE_TYPE);
    constexpr static uint32_t UNITFLAG_EN_OUTER_LAST = 3;
    constexpr static uint32_t UNITFLAG_DISABLE = 0;
    constexpr static uint32_t UNITFLAG_ENABLE = 2;
    constexpr static uint32_t BYTE_BLOCK = 32;

    uint32_t qDxBufferIdx = 0;
    uint32_t kvBuffIdx = 0;
    uint32_t qBuffIdx = 0;
    uint32_t dxBuffIdx = 0;
    uint32_t kBuffIdx = 0;
    uint32_t vBuffIdx = 0;
    uint32_t ubBufIdx = 0;

    uint32_t mmL0APingPong = 0;
    uint32_t mmL0BPingPong = 0;
    uint32_t mmL0CPingPong = 0;

    TPipe* pipe;
    FagTilingType tilingData;

    // l1 buffer manage
    // input
    GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, queryRopeGm, keyRopeGm;

    // L1
    TBuf<TPosition::A1> tBufL1Ds;
    LocalTensor<DsType> L1DsTensor;

    TBuf<TPosition::A1> tBufL1P;
    LocalTensor<T1> L1PTensor;

    TBuf<TPosition::A1> tBufL1QDx[Q_DX_L1_BUF_NUM];
    LocalTensor<T1> L1QDxTensor[Q_DX_L1_BUF_NUM];
    BufferManager<TPosition::A1, Q_DX_L1_BUF_NUM> L1QDxBufferManager;

    TBuf<TPosition::B1> tBufL1KV[KV_L1_BUF_NUM];
    LocalTensor<T1> L1KVTensor[KV_L1_BUF_NUM];
    BufferManager<TPosition::B1, KV_L1_BUF_NUM> L1KVBufferManager;

    // L0
    TBuf<TPosition::A2> tBufL0A[L0AB_BUF_NUM];
    LocalTensor<T1> L0ATensorArr[L0AB_BUF_NUM];
    LocalTensor<DsType> L0ATensorArrDsType[L0AB_BUF_NUM];
    BufferManager<TPosition::A2, L0AB_BUF_NUM> l0aBufferManager;

    TBuf<TPosition::B2> tBufL0B[L0AB_BUF_NUM];
    LocalTensor<T1> L0BTensorArr[L0AB_BUF_NUM];
    BufferManager<TPosition::B2, L0AB_BUF_NUM> l0bBufferManager;

    TBuf<TPosition::CO1> tBufL0C[L0C_BUF_NUM];
    LocalTensor<T2> L0CTensorArr[L0C_BUF_NUM];
    BufferManager<TPosition::CO1, L0C_BUF_NUM> l0cBufferManager;

    // 同步事件
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint64_t QKDXV_RES_READY[3] = {0, 1, 2};
    static constexpr uint64_t QKDXV_RES_RELEASE[3] = {3, 4, 5};

    __aicore__ inline FAGBlockCube(){};
    __aicore__ inline ~FAGBlockCube();
    __aicore__ inline void SetCubeBlockParams(TPipe* pipe, FagTilingType tilingData);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR queryRope,
                                            GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace);
    __aicore__ inline void InitCubeBuffer();
    __aicore__ inline void FreeAlloc(int64_t innerMm12LoopIdx);
    __aicore__ inline void FreeBuffer();
    template <typename AType, typename BType, const bool IS_TRANS_A, const bool IS_TRANS_B, const bool IS_DKDV>
    __aicore__ inline void SingleMatmul(LocalTensor<AType> L1ATensor, LocalTensor<BType> L1BTensor,
                                        LocalTensor<T2> L0CTensor, uint32_t m, uint32_t n, uint32_t k, uint32_t srcA,
                                        uint32_t srcB, uint32_t offsetA, uint32_t offsetB, bool isInitVal,
                                        bool isSplitK = false, bool isLastLoopK = false, bool isS2IdxNoChange = false,
                                        uint32_t unitFlag = 0);
    __aicore__ inline void CopyGmToL1(LocalTensor<T1> dstTensor, GlobalTensor<T1> srcTensor, uint32_t nValue,
                                      uint32_t dValue, uint32_t srcStride, uint32_t dstStride, uint64_t srcOffset,
                                      uint64_t dstOffset);
    __aicore__ inline void IterateMmQKAndDyV(LocalTensor<T2>& mm1ResTensor, LocalTensor<T2>& mm2ResTensor,
                                             FagRunInfo& runInfo, FagRunInfo& nextRunInfo, FagConstInfo& constInfo,
                                             int8_t innerLoopNum, int8_t innerLoopIdx, int8_t baseUbBufIdx);
    __aicore__ inline void IterateMmDyV(LocalTensor<T2>& mm1ResTensor, FagRunInfo& runInfo, FagConstInfo& constInfo,
                                        LocalTensor<T2>& L0CTensor, uint32_t mIdx, uint32_t mSize, bool isLast); // mm1
    __aicore__ inline void IterateMmQK(LocalTensor<T2>& mm2ResTensor, FagRunInfo& runInfo, FagConstInfo& constInfo,
                                       LocalTensor<T2>& L0CTensor, uint32_t mIdx, uint32_t mSize, bool isLast); // mm2
    __aicore__ inline void IterateMmDsK(FagRunInfo& runInfo, FagConstInfo& constInfo,
                                        GlobalTensor<float>& dqWorkSpaceGm, bool isDqNeedDeter = false,
                                        int64_t cBlockIdx = 0, GlobalTensor<T2>* deterGmPtr = nullptr,
                                        bool deterPpFlag = false); // mm3 dq
    __aicore__ inline void IterateMmDsQ(FagRunInfo& runInfo, FagConstInfo& constInfo,
                                        GlobalTensor<float>& dkWorkSpaceGm, bool isDkDvNeedDeter = false,
                                        int64_t cBlockIdx = 0, GlobalTensor<T2>* deterGmPtr = nullptr,
                                        bool deterPpFlag = false); // mm4 dk
    __aicore__ inline void IterateMmPDy(
        FagRunInfo& runInfo, FagConstInfo& constInfo,
        GlobalTensor<std::conditional_t<SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK, T1, float>>& dvWorkSpaceGm,
        bool isDkDvNeedDeter = false, int64_t cBlockIdx = 0, GlobalTensor<T2>* deterGmPtr = nullptr,
        bool deterPpFlag = false); // mm5 dv
    template <typename DataType>
    __aicore__ inline void LoadDataToL0(LocalTensor<DataType> dstTensor, LocalTensor<DataType> srcTensor, uint32_t m,
                                        uint32_t k, uint32_t srcM, uint64_t srcOffset, bool isTrans);
    template <typename AType, typename BType, bool IsDkDv = false>
    __aicore__ inline void ComputeBaseMm(LocalTensor<T2>& L0CTensor, LocalTensor<AType>& L0ATensor,
                                         LocalTensor<BType>& L0BTensor, uint32_t m, uint32_t n, uint64_t k,
                                         bool isInitVal, bool isS2IdxNoChange = false, uint32_t unitFlag = 0);
    __aicore__ inline void CopyL0CToUB(LocalTensor<T2>& mmResUb, LocalTensor<T2>& L0CTensor, uint32_t m, uint32_t n,
                                       uint32_t unitFlag = 0);
    template <bool IS_FP8_TRANS_A>
    __aicore__ inline void CopyL0CToGm(GlobalTensor<T2>& dstTensor, LocalTensor<T2>& L0CTensor, uint32_t m, uint32_t n,
                                       uint64_t dstStride, uint64_t dstOffset, float deqScaleValue = 0.0,
                                       uint32_t unitFlag = 0, bool isAtomic = false);
    __aicore__ inline void CopyL0CToGm(GlobalTensor<T1>& dstTensor, LocalTensor<T2>& L0CTensor, uint32_t m, uint32_t n,
                                       uint64_t dstStride, uint64_t dstOffset, uint32_t unitFlag = 0);
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline FAGBlockCube<TEMPLATE_ARGS>::~FAGBlockCube() {}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::SetCubeBlockParams(TPipe* pipe, FagTilingType tilingData)
{
    this->pipe = pipe;
    this->tilingData = tilingData;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                                                     GM_ADDR dy, GM_ADDR queryRope, GM_ADDR keyRope,
                                                                     GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                                                     GM_ADDR workspace)
{
    queryGm.SetGlobalBuffer((__gm__ T1*)query);
    keyGm.SetGlobalBuffer((__gm__ T1*)key);
    valueGm.SetGlobalBuffer((__gm__ T1*)value);
    dxGm.SetGlobalBuffer((__gm__ T1*)dy);
    queryRopeGm.SetGlobalBuffer((__gm__ T1*)queryRope);
    keyRopeGm.SetGlobalBuffer((__gm__ T1*)keyRope);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::InitCubeBuffer()
{
    L1QDxBufferManager.Init();
    L1KVBufferManager.Init();
    l0aBufferManager.Init();
    l0bBufferManager.Init();
    l0cBufferManager.Init();

    // L1
    pipe->InitBuffer(tBufL1Ds, DS_P_L1_BUF_SIZE * sizeof(DsType));
    L1DsTensor = tBufL1Ds.template Get<DsType>();

    pipe->InitBuffer(tBufL1P, DS_P_L1_BUF_SIZE * sizeof(T1));
    L1PTensor = tBufL1P.template Get<T1>();

    for (int32_t i = 0; i < Q_DX_L1_BUF_NUM; ++i) {
        pipe->InitBuffer(tBufL1QDx[i], Q_DX_L1_BUF_SIZE * sizeof(T1));
        L1QDxTensor[i] = tBufL1QDx[i].template Get<T1>();
    }

    for (int32_t i = 0; i < KV_L1_BUF_NUM; ++i) {
        pipe->InitBuffer(tBufL1KV[i], KV_L1_BUF_SIZE * sizeof(T1));
        L1KVTensor[i] = tBufL1KV[i].template Get<T1>();
    }

    // L0
    for (int32_t i = 0; i < L0AB_BUF_NUM; ++i) {
        pipe->InitBuffer(tBufL0A[i], L0AB_DB_SIZE);
        L0ATensorArr[i] = tBufL0A[i].template Get<T1>();
        L0ATensorArrDsType[i] = tBufL0A[i].template Get<DsType>();
        pipe->InitBuffer(tBufL0B[i], L0AB_DB_SIZE);
        L0BTensorArr[i] = tBufL0B[i].template Get<T1>();
    }
    for (int32_t i = 0; i < L0C_BUF_NUM; ++i) {
        pipe->InitBuffer(tBufL0C[i], L0C_DBBUF_SIZE);
        L0CTensorArr[i] = tBufL0C[i].template Get<T2>();
    }
}
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::FreeAlloc(int64_t innerMm12LoopIdx)
{
    for (int32_t i = 0; i < Min(UB_BUF_NUM, innerMm12LoopIdx); i++) {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(QKDXV_RES_RELEASE[i]);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::FreeBuffer()
{
    L1QDxBufferManager.Destory();
    L1KVBufferManager.Destory();
    l0aBufferManager.Destory();
    l0bBufferManager.Destory();
    l0cBufferManager.Destory();
}

TEMPLATES_DEF_NO_DEFAULT
template <typename AType, typename BType, const bool IS_TRANS_A, const bool IS_TRANS_B, const bool IS_DKDV>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::SingleMatmul(
    LocalTensor<AType> L1ATensor, LocalTensor<BType> L1BTensor, LocalTensor<T2> L0CTensor, uint32_t m, uint32_t n,
    uint32_t k, uint32_t srcA, uint32_t srcB, uint32_t offsetA, uint32_t offsetB, bool isInitVal, bool isSplitK,
    bool isLastLoopK, bool isS2IdxNoChange, uint32_t unitFlag)
{
    /// NOTE: 对于FP32,FP4类型，下方的下标计算需要额外考虑
    LocalTensor<AType> L0ATensor;
    if constexpr (std::is_same_v<AType, DsType>) {
        L0ATensor = L0ATensorArrDsType[l0aBufferManager.GetBuffer(mmL0APingPong)];
    } else {
        L0ATensor = L0ATensorArr[l0aBufferManager.GetBuffer(mmL0APingPong)];
    }
    if constexpr (IS_TRANS_A) {
        LoadDataToL0(L0ATensor, L1ATensor, k, m, srcA, offsetA, true); // s1, s2
    } else {
        LoadDataToL0(L0ATensor, L1ATensor, m, k, srcA, offsetA, false); // s1, s2
    }
    l0aBufferManager.EnQue(mmL0APingPong);
    l0aBufferManager.DeQue(mmL0APingPong);

    LocalTensor<BType> L0BTensor = L0BTensorArr[l0bBufferManager.GetBuffer(mmL0BPingPong)];
    if constexpr (IS_TRANS_B) {
        LoadDataToL0(L0BTensor, L1BTensor, k, n, srcB, offsetB, true); // s1 D
    } else {
        LoadDataToL0(L0BTensor, L1BTensor, n, k, srcB, offsetB, false); // s1 D
    }
    l0bBufferManager.EnQue(mmL0BPingPong);
    l0bBufferManager.DeQue(mmL0BPingPong);

    if (isSplitK && unitFlag != 0) {
        unitFlag = (unitFlag == UNITFLAG_EN_OUTER_LAST) && isLastLoopK ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE;
    }
    if constexpr (IS_FP8_INPUT) {
        if constexpr (is_f8_v<AType> && IS_TRANS_A) {
            ComputeBaseMm(L0CTensor, L0ATensor, L0BTensor, ((m + 31) >> 5) << 5, n, k, isInitVal);
        } else {
            ComputeBaseMm(L0CTensor, L0ATensor, L0BTensor, m, n, k, isInitVal);
        }
    } else if constexpr (IS_FP16_INPUT) {
        ComputeBaseMm<AType, BType, IS_DKDV>(L0CTensor, L0ATensor, L0BTensor, m, n, k, isInitVal, isS2IdxNoChange,
                                             unitFlag);
    } else if constexpr (IS_FP32_INPUT) {
        ComputeBaseMm<AType, BType, IS_DKDV>(L0CTensor, L0ATensor, L0BTensor, m, n, k, isInitVal, isS2IdxNoChange,
                                             unitFlag);
    }

    l0aBufferManager.ReleaseBuffer(mmL0APingPong);
    l0bBufferManager.ReleaseBuffer(mmL0BPingPong);
    mmL0APingPong = 1 - mmL0APingPong;
    mmL0BPingPong = 1 - mmL0BPingPong;
}

TEMPLATES_DEF_NO_DEFAULT
template <typename AType, typename BType, bool IsDkDv>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::ComputeBaseMm(LocalTensor<T2>& L0CTensor,
                                                                  LocalTensor<AType>& L0ATensor,
                                                                  LocalTensor<BType>& L0BTensor, uint32_t m, uint32_t n,
                                                                  uint64_t k, bool isInitVal, bool isS2IdxNoChange,
                                                                  uint32_t unitFlag)
{
    MmadParams mmadParams;
    mmadParams.m = m; // 左矩阵Height(row)
    mmadParams.n = n;
    mmadParams.k = k;
    if (mmadParams.m == 1) { // m等于1会默认开GEMV模式，文档上没有写怎么关闭GEMV，所以规避当作矩阵计算
        mmadParams.m = 16;
    }
    mmadParams.cmatrixInitVal = isInitVal; // 配置C矩阵初始值是否为0。默认值true。
    mmadParams.cmatrixSource = false;      // 配置C矩阵初始值是否来源于C2（存放Bias的硬件缓存区）
    if constexpr (IsDkDv && IS_DKV_RESIDENT_L0C) {
        mmadParams.cmatrixInitVal = isInitVal ? !isS2IdxNoChange : false;
    }
    mmadParams.unitFlag = unitFlag;
    Mmad(L0CTensor, L0ATensor, L0BTensor, mmadParams); // 矩阵乘并将结果写入L0C
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::CopyGmToL1(LocalTensor<T1> dstTensor, GlobalTensor<T1> srcTensor,
                                                               uint32_t nValue, uint32_t dValue, uint32_t srcStride,
                                                               uint32_t dstStride, uint64_t srcOffset,
                                                               uint64_t dstOffset)
{
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;             // ND矩阵的个数
    nd2nzParams.nValue = nValue;       // 单个ND矩阵的实际行数, 单位为元素个数
    nd2nzParams.dValue = dValue;       // 单个ND矩阵的列数, 单位为元素个数
    nd2nzParams.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
    nd2nzParams.srcDValue = srcStride; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
    nd2nzParams.dstNzC0Stride =
        (dstStride + 15) >> 4 << 4; // 转换为NZ矩阵后，源ND矩阵上相邻Block起始地址之间的偏移, 单位为Block个数;
    nd2nzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
    nd2nzParams.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素数量
    DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], nd2nzParams, true);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmQKAndDyV(LocalTensor<T2>& mm1ResTensor,
                                                                      LocalTensor<T2>& mm2ResTensor,
                                                                      FagRunInfo& runInfo, FagRunInfo& nextRunInfo,
                                                                      FagConstInfo& constInfo, int8_t innerLoopNum,
                                                                      int8_t innerLoopIdx, int8_t baseUbBufIdx)
{
    ubBufIdx = baseUbBufIdx; // TODO 临时措施
    if (innerLoopIdx == 0) {
        // 只在innerLoop的第一次调用
        if constexpr (IS_TSCM_REUSE_K) {
            vBuffIdx = 2;
            if (!runInfo.isS2IdxNoChange || IS_TSCM_FULL_LOAD) {
                kBuffIdx = 1 - kBuffIdx;
            }
            runInfo.kvPingPong = kBuffIdx;
        } else if constexpr (IS_TSCM_FULL_LOAD) {
            // runInfo.isS2IdxNoChange = false;
            kvBuffIdx = (kvBuffIdx + 1) % KV_L1_BUF_NUM;
            runInfo.kvPingPong = kvBuffIdx;
            kvBuffIdx = (kvBuffIdx + 1) % KV_L1_BUF_NUM;
            vBuffIdx = kvBuffIdx;
        }

        if constexpr (IS_TSCM_REUSE) {
            dxBuffIdx = runInfo.taskPingPong;
            qBuffIdx = runInfo.taskPingPong + 2;
        } else if constexpr (IS_TSCM_FULL_LOAD) {
            qDxBufferIdx = (qDxBufferIdx + 1) % Q_DX_L1_BUF_NUM;
            qBuffIdx = qDxBufferIdx;
            qDxBufferIdx = (qDxBufferIdx + 1) % Q_DX_L1_BUF_NUM;
            dxBuffIdx = qDxBufferIdx;
        }
    }

    bool isInnerBlockLast = (innerLoopIdx == innerLoopNum - 1);
    uint32_t mSize =
        (isInnerBlockLast) ? (runInfo.commonRunInfo.s1RealSize - innerLoopIdx * VECTOR_BASEM) : VECTOR_BASEM;
    uint32_t sharedL0CIdx = mmL0CPingPong % L0C_SHARED_BUF_NUM;
    LocalTensor<T2> L0CTensorQk;
    if constexpr (ENABLE_UNITFLAG) {
        L0CTensorQk = L0CTensorArr[sharedL0CIdx];
    } else {
        L0CTensorQk = L0CTensorArr[l0cBufferManager.GetBuffer(sharedL0CIdx)];
    }
    LocalTensor<T2> L0CTensorDxV = L0CTensorArr[sharedL0CIdx][VECTOR_BASEM * CUBE_BASEN];
    IterateMmQK(mm2ResTensor, runInfo, constInfo, L0CTensorQk, innerLoopIdx, mSize, isInnerBlockLast);
    IterateMmDyV(mm1ResTensor, runInfo, constInfo, L0CTensorDxV, innerLoopIdx, mSize, isInnerBlockLast);
    if constexpr (!ENABLE_UNITFLAG) {
        l0cBufferManager.ReleaseBuffer(sharedL0CIdx);
    }
    mmL0CPingPong = (mmL0CPingPong + 1) % L0C_SHARED_BUF_NUM;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDyV(LocalTensor<T2>& mm1ResTensor, FagRunInfo& runInfo,
                                                                 FagConstInfo& constInfo, LocalTensor<T2>& L0CTensor,
                                                                 uint32_t mIdx, uint32_t mSize, bool isLast)
{
    ///////////////////////////////////////////////////////////////
    // MM1: dx@v
    ///////////////////////////////////////////////////////////////
    LocalTensor<T1> L1ATensor;
    LocalTensor<T1> L1BTensor;
    if constexpr (IS_TSCM_FULL_LOAD) {
        qDxBufferIdx = dxBuffIdx;
        kvBuffIdx = vBuffIdx;
        L1ATensor = L1QDxTensor[qDxBufferIdx];
        L1BTensor = L1KVTensor[kvBuffIdx];
        if (mIdx == 0) {
            L1KVBufferManager.GetBuffer(kvBuffIdx); // 可能会插一个空同步
            if (!runInfo.isS2IdxNoChange || IS_TSCM_FULL_LOAD) {
                CopyGmToL1(L1BTensor, valueGm, runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.dSizeV,
                           constInfo.commonConstInfo.mm1Kb, runInfo.commonRunInfo.s2RealSize, runInfo.valueGmOffset, 0);
            }
            L1QDxBufferManager.GetBuffer(qDxBufferIdx);
        }
        if (mIdx % 2 == 0) {
            uint32_t nValue = runInfo.commonRunInfo.s1RealSize - mIdx * VECTOR_BASEM > VECTOR_BASEM * 2 ?
                                  VECTOR_BASEM * 2 :
                                  runInfo.commonRunInfo.s1RealSize - mIdx * VECTOR_BASEM;
            uint64_t srcOffset = runInfo.dxGmOffset + mIdx * VECTOR_BASEM * constInfo.commonConstInfo.mm1Ka;
            uint64_t dstOffset = mIdx * VECTOR_BASEM * FRACTAL_NZ_C0_SIZE;
            CopyGmToL1(L1ATensor, dxGm, nValue, constInfo.commonConstInfo.dSizeV, constInfo.commonConstInfo.mm1Ka,
                       runInfo.commonRunInfo.s1RealSize, srcOffset, dstOffset);
            L1QDxBufferManager.EnQue(qDxBufferIdx);
            L1QDxBufferManager.DeQue(qDxBufferIdx);
        }
    }

    uint32_t kL1ABase = IS_FP32_INPUT ? 128 : 512;
    uint32_t kL1BBase = IS_FP32_INPUT ? 128 : 256;
    uint32_t kL1ALoopTimes = (constInfo.commonConstInfo.dSizeV + kL1ABase - 1) / kL1ABase;
    bool mmadFirst = true;
    bool isSplitK = false;
    for (uint32_t kL1AIdx = 0; kL1AIdx < kL1ALoopTimes; kL1AIdx++) {
        isSplitK = isSplitK || (kL1ALoopTimes != 1);
        uint32_t kL1ASize =
            (kL1AIdx == kL1ALoopTimes - 1) ? (constInfo.commonConstInfo.dSizeV - kL1AIdx * kL1ABase) : kL1ABase;
        // dx to L1
        if constexpr (!IS_TSCM_FULL_LOAD) {
            int64_t dxOffset =
                runInfo.dxGmOffset + mIdx * VECTOR_BASEM * constInfo.commonConstInfo.mm1Ka + kL1AIdx * kL1ABase;
            qDxBufferIdx = 1 - qDxBufferIdx;
            L1ATensor = L1QDxTensor[L1QDxBufferManager.GetBuffer(qDxBufferIdx)];
            CopyGmToL1(L1ATensor, dxGm, mSize, kL1ASize, constInfo.commonConstInfo.mm1Ka, mSize, dxOffset, 0);
            L1QDxBufferManager.EnQue(qDxBufferIdx);
            L1QDxBufferManager.DeQue(qDxBufferIdx);
        }

        uint32_t l1BLoopTimes = (kL1ASize + kL1BBase - 1) / kL1BBase;   // 只切k D>256会切
        for (uint32_t kL1BIdx = 0; kL1BIdx < l1BLoopTimes; kL1BIdx++) { // L1B循环
            isSplitK = isSplitK || (l1BLoopTimes != 1);
            // v to L1
            uint32_t kL1BSize = (kL1BIdx == l1BLoopTimes - 1) ? (kL1ASize - kL1BIdx * kL1BBase) : kL1BBase;
            if constexpr (!IS_TSCM_FULL_LOAD) { // L1 全载
                kvBuffIdx = 1 - kvBuffIdx;
                L1BTensor = L1KVTensor[kvBuffIdx];
                L1KVBufferManager.GetBuffer(kvBuffIdx);
                int64_t valueOffset = runInfo.valueGmOffset + kL1AIdx * kL1ABase + kL1BIdx * kL1BBase;
                CopyGmToL1(L1BTensor, valueGm, runInfo.commonRunInfo.s2RealSize, kL1BSize,
                           constInfo.commonConstInfo.mm1Kb, runInfo.commonRunInfo.s2RealSize, valueOffset,
                           0); // todo D大于256需要改offset
                L1KVBufferManager.EnQue(kvBuffIdx);
                L1KVBufferManager.DeQue(kvBuffIdx);
            }

            // L0 按64*128切分 切k
            uint32_t l0BaseK = IS_FP32_INPUT ? 64 : 128;
            uint32_t kLoopTimes = (kL1BSize + l0BaseK - 1) / l0BaseK;
            l0BaseK = ((kL1BSize + kLoopTimes - 1) / kLoopTimes * sizeof(T1) + 31) / 32 * 32 / sizeof(T1);
            for (uint32_t kL0Idx = 0; kL0Idx < kLoopTimes; kL0Idx++) {
                isSplitK = isSplitK || (kLoopTimes != 1);
                uint32_t kL0Size = (kL0Idx == kLoopTimes - 1) ? (kL1BSize - kL0Idx * l0BaseK) : l0BaseK;

                uint32_t srcM = mSize;
                uint32_t offsetA = (kL1BIdx * kL1BBase + kL0Idx * l0BaseK) * AlignTo16(mSize);
                if constexpr (IS_TSCM_FULL_LOAD) {
                    srcM = runInfo.commonRunInfo.s1RealSize;
                    offsetA = mIdx * VECTOR_BASEM * FRACTAL_NZ_C0_SIZE +
                              (kL1AIdx * kL1ASize + kL1BIdx * kL1BBase + kL0Idx * l0BaseK) * AlignTo16(srcM);
                }
                uint32_t offsetB = kL0Idx * runInfo.commonRunInfo.s2AlignedSize * l0BaseK;
                uint32_t unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
                bool isLastLoopK =
                    (kL1AIdx == kL1ALoopTimes - 1) && (kL1BIdx == l1BLoopTimes - 1) && (kL0Idx == kLoopTimes - 1);
                SingleMatmul<T1, T1, false, false, false>(L1ATensor, L1BTensor, L0CTensor, mSize,
                                                          runInfo.commonRunInfo.s2RealSize, kL0Size, // m, n, k
                                                          srcM, runInfo.commonRunInfo.s2RealSize,    // A, B src
                                                          offsetA, offsetB,                          // A, B offset
                                                          mmadFirst, isSplitK, isLastLoopK, false, unitFlag);
                mmadFirst = false;

                if (isLastLoopK) {
                    if constexpr (!ENABLE_UNITFLAG) {
                        l0cBufferManager.EnQue(mmL0CPingPong);
                        l0cBufferManager.DeQue(mmL0CPingPong);
                    }
                    CopyL0CToUB(mm1ResTensor, L0CTensor, mSize, runInfo.commonRunInfo.s2RealSize, unitFlag);
                }
            }
            if constexpr (!IS_TSCM_FULL_LOAD) {
                L1KVBufferManager.ReleaseBuffer(kvBuffIdx);
            } else {
                if (isLast) {
                    L1KVBufferManager.ReleaseBuffer(kvBuffIdx); // v后续不复用，在最后一次循环释放掉
                }
            }
        }
        if constexpr (!IS_TSCM_FULL_LOAD) {
            L1QDxBufferManager.ReleaseBuffer(qDxBufferIdx);
        } else if constexpr (!IS_TSCM_REUSE) { // 128 < D <=256
            if (isLast) {
                L1QDxBufferManager.ReleaseBuffer(qDxBufferIdx);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmQK(LocalTensor<T2>& mm2ResTensor, FagRunInfo& runInfo,
                                                                FagConstInfo& constInfo, LocalTensor<T2>& L0CTensor,
                                                                uint32_t mIdx, uint32_t mSize, bool isLast)
{
    ///////////////////////////////////////////////////////////////
    // MM2: q@k
    ///////////////////////////////////////////////////////////////
    LocalTensor<T1> L1ATensor;
    LocalTensor<T1> L1BTensor;
    if constexpr (IS_TSCM_FULL_LOAD) {
        qDxBufferIdx = qBuffIdx;
        kvBuffIdx = runInfo.kvPingPong;
        L1ATensor = L1QDxTensor[qDxBufferIdx];
        L1BTensor = L1KVTensor[kvBuffIdx];
        if (mIdx == 0) {
            if (!runInfo.isS2IdxNoChange || IS_TSCM_FULL_LOAD) {
                L1KVBufferManager.GetBuffer(kvBuffIdx);
                if constexpr (IS_ROPE) {
                    CopyGmToL1(L1BTensor, keyGm, runInfo.commonRunInfo.s2RealSize, ROPE_D_128, constInfo.mm2Kb,
                               runInfo.commonRunInfo.s2RealSize, runInfo.keyGmOffset, 0);

                    CopyGmToL1(L1BTensor, keyRopeGm, runInfo.commonRunInfo.s2RealSize, constInfo.dRopeSize,
                               constInfo.mm2Kb / 2, runInfo.commonRunInfo.s2RealSize, runInfo.keyRopeGmOffset,
                               runInfo.commonRunInfo.s2AlignedSize * ROPE_D_128);
                } else {
                    CopyGmToL1(L1BTensor, keyGm, runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.dSize,
                               constInfo.mm2Kb, runInfo.commonRunInfo.s2RealSize, runInfo.keyGmOffset, 0);
                }
            }
            L1QDxBufferManager.GetBuffer(qDxBufferIdx);
        }
        if (mIdx % 2 == 0) {
            uint32_t nValue = runInfo.commonRunInfo.s1RealSize - mIdx * VECTOR_BASEM > VECTOR_BASEM * 2 ?
                                  VECTOR_BASEM * 2 :
                                  runInfo.commonRunInfo.s1RealSize - mIdx * VECTOR_BASEM;
            uint64_t querySrcOffset = runInfo.queryGmOffset + mIdx * VECTOR_BASEM * constInfo.mm2Ka;
            uint64_t dstOffset = mIdx * VECTOR_BASEM * FRACTAL_NZ_C0_SIZE;
            if constexpr (IS_ROPE) {
                uint64_t queryRopeSrcOffset = runInfo.queryRopeGmOffset + mIdx * VECTOR_BASEM * constInfo.mm2Ka / 2;
                CopyGmToL1(L1ATensor, queryGm, nValue, ROPE_D_128, constInfo.mm2Ka, runInfo.commonRunInfo.s1RealSize,
                           querySrcOffset, dstOffset);

                CopyGmToL1(L1ATensor, queryRopeGm, nValue, constInfo.dRopeSize, constInfo.mm2Ka / 2,
                           runInfo.commonRunInfo.s1RealSize, queryRopeSrcOffset,
                           (AlignTo16(runInfo.commonRunInfo.s1RealSize) * ROPE_D_128 + dstOffset));
            } else {
                CopyGmToL1(L1ATensor, queryGm, nValue, constInfo.commonConstInfo.dSize, constInfo.mm2Ka,
                           runInfo.commonRunInfo.s1RealSize, querySrcOffset, dstOffset);
            }
            L1QDxBufferManager.EnQue(qDxBufferIdx);
            L1QDxBufferManager.DeQue(qDxBufferIdx);
        }
    }

    uint32_t kL1ABase = IS_FP32_INPUT ? 128 : 512;
    uint32_t kL1BBase = IS_FP32_INPUT ? 128 : 256;
    uint32_t kL1ALoopTimes = (constInfo.commonConstInfo.dSize + kL1ABase - 1) / kL1ABase;
    bool mmadFirst = true;
    bool isSplitK = false;
    for (uint32_t kL1AIdx = 0; kL1AIdx < kL1ALoopTimes; kL1AIdx++) {
        isSplitK = isSplitK || (kL1ALoopTimes != 1);
        // q to L1
        uint32_t kL1ASize =
            (kL1AIdx == kL1ALoopTimes - 1) ? (constInfo.commonConstInfo.dSize - kL1AIdx * kL1ABase) : kL1ABase;
        if constexpr (!IS_TSCM_FULL_LOAD) { // L1全载
            qDxBufferIdx = 1 - qDxBufferIdx;
            L1ATensor = L1QDxTensor[L1QDxBufferManager.GetBuffer(qDxBufferIdx)];
            int64_t qOffset = runInfo.queryGmOffset + mIdx * VECTOR_BASEM * constInfo.mm2Ka + kL1AIdx * kL1ABase;
            CopyGmToL1(L1ATensor, queryGm, mSize, kL1ASize, constInfo.mm2Ka, mSize, qOffset, 0);
            L1QDxBufferManager.EnQue(qDxBufferIdx);
            L1QDxBufferManager.DeQue(qDxBufferIdx);
        }

        uint32_t l1BLoopTimes = (kL1ASize + kL1BBase - 1) / kL1BBase;   // 只切k D>256会切
        for (uint32_t kL1BIdx = 0; kL1BIdx < l1BLoopTimes; kL1BIdx++) { // L1B循环
            isSplitK = isSplitK || (l1BLoopTimes != 1);
            // k to L1
            uint32_t kL1BSize = (kL1BIdx == l1BLoopTimes - 1) ? (kL1ASize - kL1BIdx * kL1BBase) : kL1BBase;
            if constexpr (!IS_TSCM_FULL_LOAD) {
                kvBuffIdx = 1 - kvBuffIdx;
                L1BTensor = L1KVTensor[L1KVBufferManager.GetBuffer(kvBuffIdx)];
                int64_t keyOffset = runInfo.keyGmOffset + kL1AIdx * kL1ABase + kL1BIdx * kL1BBase;
                CopyGmToL1(L1BTensor, keyGm, runInfo.commonRunInfo.s2RealSize, kL1BSize, constInfo.mm2Kb,
                           runInfo.commonRunInfo.s2RealSize, keyOffset, 0);
                L1KVBufferManager.EnQue(kvBuffIdx);
                L1KVBufferManager.DeQue(kvBuffIdx);
            }

            // L0 按64*128切分

            uint32_t l0BaseK = IS_FP32_INPUT ? 64 : 128;
            uint32_t kLoopTimes = (kL1BSize + l0BaseK - 1) / l0BaseK;
            l0BaseK = ((kL1BSize + kLoopTimes - 1) / kLoopTimes * sizeof(T1) + 31) / 32 * 32 / sizeof(T1);
            for (uint32_t kL0Idx = 0; kL0Idx < kLoopTimes; kL0Idx++) {
                isSplitK = isSplitK || (kLoopTimes != 1);
                uint32_t kL0Size = (kL0Idx == kLoopTimes - 1) ? (kL1BSize - kL0Idx * l0BaseK) : l0BaseK;

                uint32_t srcM = mSize;
                uint32_t offsetA = (kL1BIdx * kL1BBase + kL0Idx * l0BaseK) * AlignTo16(mSize);
                if constexpr (IS_TSCM_FULL_LOAD) {
                    srcM = runInfo.commonRunInfo.s1RealSize;
                    offsetA = mIdx * VECTOR_BASEM * FRACTAL_NZ_C0_SIZE +
                              (kL1AIdx * kL1ASize + kL1BIdx * kL1BBase + kL0Idx * l0BaseK) * AlignTo16(srcM);
                }

                uint32_t offsetB = kL0Idx * runInfo.commonRunInfo.s2AlignedSize * l0BaseK;
                uint32_t unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
                bool isLastLoopK =
                    (kL1AIdx == kL1ALoopTimes - 1) && (kL1BIdx == l1BLoopTimes - 1) && (kL0Idx == kLoopTimes - 1);
                SingleMatmul<T1, T1, false, false, false>(L1ATensor, L1BTensor, L0CTensor, mSize,
                                                          runInfo.commonRunInfo.s2RealSize, kL0Size, // m, n, k
                                                          srcM, runInfo.commonRunInfo.s2RealSize,    // A, B src
                                                          offsetA, offsetB,                          // A, B offset
                                                          mmadFirst, isSplitK, isLastLoopK, false, unitFlag);
                mmadFirst = false;

                if (isLastLoopK) {
                    if constexpr (!ENABLE_UNITFLAG) {
                        l0cBufferManager.EnQue(mmL0CPingPong);
                        l0cBufferManager.DeQue(mmL0CPingPong);
                    }
                    CopyL0CToUB(mm2ResTensor, L0CTensor, mSize, runInfo.commonRunInfo.s2RealSize, unitFlag);
                }
            }

            if constexpr (!IS_TSCM_FULL_LOAD) {
                L1KVBufferManager.ReleaseBuffer(kvBuffIdx);
            } else if constexpr (!IS_TSCM_REUSE_K) {
                if (isLast) {
                    L1KVBufferManager.ReleaseBuffer(kvBuffIdx);
                }
            }
        }
        if constexpr (!IS_TSCM_FULL_LOAD) {
            L1QDxBufferManager.ReleaseBuffer(qDxBufferIdx);
        } else if constexpr (!IS_TSCM_REUSE) {
            if (isLast) {
                L1QDxBufferManager.ReleaseBuffer(qDxBufferIdx);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsK(FagRunInfo& runInfo, FagConstInfo& constInfo,
                                                                 GlobalTensor<float>& dqWorkSpaceGm, bool isDqNeedDeter,
                                                                 int64_t cBlockIdx, GlobalTensor<T2>* deterGmPtr,
                                                                 bool deterPpFlag)
{
    constexpr bool isTransA = false;
    constexpr bool isTransB = true;
    uint64_t dstStride = constInfo.commonConstInfo.n2GD;
    if constexpr (!IS_TND && !IS_BN2_MULTIBLK) {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            dstStride = constInfo.commonConstInfo.dSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            dstStride = constInfo.commonConstInfo.bN2GD;
        }
    } else if constexpr (IS_BN2_MULTIBLK) {
        dstStride = constInfo.commonConstInfo.dSize;
    }
    int64_t dqOffset;
    uint32_t l1LoopSize = IS_FP32_INPUT ? 128 : 256;
    uint32_t l1LoopTimes = (constInfo.commonConstInfo.dSize + l1LoopSize - 1) / l1LoopSize;

    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(DS_READY);

    for (uint32_t i = 0; i < l1LoopTimes; i++) {
        if constexpr (IS_TSCM_REUSE_K) { // 1
            kvBuffIdx = runInfo.kvPingPong;
        } else {
            kvBuffIdx = 1 - kvBuffIdx;
        }

        LocalTensor<T1> L1BTensor = L1KVTensor[kvBuffIdx]; // key L1复用
        int32_t nL1Size = (i == l1LoopTimes - 1) ? (constInfo.commonConstInfo.dSize - i * l1LoopSize) : l1LoopSize;

        if constexpr (!IS_TSCM_REUSE_K) {
            int64_t keyGmOffset = runInfo.keyGmOffset + i * l1LoopSize;
            L1BTensor = L1KVTensor[L1KVBufferManager.GetBuffer(kvBuffIdx)];
            CopyGmToL1(L1BTensor, keyGm, runInfo.commonRunInfo.s2RealSize, nL1Size, constInfo.mm2Kb,
                       runInfo.commonRunInfo.s2RealSize, keyGmOffset, 0);
            L1KVBufferManager.EnQue(kvBuffIdx);
            L1KVBufferManager.DeQue(kvBuffIdx);
        }

        // L0 按128*128切分
        uint32_t l0BaseM = IS_FP32_INPUT ? 64 : 128;
        uint32_t l0BaseN = IS_FP32_INPUT ? 64 : 128;
        uint32_t mLoopTimes = (runInfo.commonRunInfo.s1RealSize + l0BaseM - 1) / l0BaseM;
        if (!constInfo.deterConstInfo.noNeedDeter && isDqNeedDeter) { // deter流程按照完整基本块处理
            mLoopTimes = (CUBE_BASEM + l0BaseM - 1) / l0BaseM;
        }
        uint32_t nLoopTimes = (nL1Size + l0BaseN - 1) / l0BaseN;
        l0BaseN = ((nL1Size + nLoopTimes - 1) / nLoopTimes * sizeof(T1) + 31) / 32 * 32 / sizeof(T1);
        uint32_t l0LoopTimes = mLoopTimes * nLoopTimes;
        LocalTensor<T2> L0CTensor;
        for (uint32_t j = 0; j < l0LoopTimes; j++) {
            uint32_t mL0Idx = j / nLoopTimes;
            uint32_t nL0Idx = j % nLoopTimes;
            uint32_t mL0Size =
                (mL0Idx == mLoopTimes - 1) ? (runInfo.commonRunInfo.s1RealSize - mL0Idx * l0BaseM) : l0BaseM;
            if (!constInfo.deterConstInfo.noNeedDeter && isDqNeedDeter) {
                mL0Size = l0BaseM;
            }
            uint32_t nL0Size = (nL0Idx == nLoopTimes - 1) ? (nL1Size - nL0Idx * l0BaseN) : l0BaseN;
            L0CTensor =
                ENABLE_UNITFLAG ? L0CTensorArr[mmL0CPingPong] : L0CTensorArr[l0cBufferManager.GetBuffer(mmL0CPingPong)];
            uint32_t offsetA = mL0Idx * l0BaseM * 32 / sizeof(DsType);
            uint32_t offsetB = nL0Idx * runInfo.commonRunInfo.s2AlignedSize * l0BaseN;
            uint32_t unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
            SingleMatmul<DsType, T1, isTransA, isTransB, false>(L1DsTensor, L1BTensor, L0CTensor, mL0Size, nL0Size,
                                                                runInfo.commonRunInfo.s2RealSize,             // m,n,k
                                                                CUBE_BASEM, runInfo.commonRunInfo.s2RealSize, // A,B src
                                                                offsetA, offsetB, // A,B offset
                                                                true, false, false, false, unitFlag);

            if constexpr (!ENABLE_UNITFLAG) {
                l0cBufferManager.EnQue(mmL0CPingPong);
                l0cBufferManager.DeQue(mmL0CPingPong);
            }
            if constexpr (SPLIT_AXIS == BN2) {
                if constexpr (IS_BN2_MULTIBLK) {
                    dqOffset = GetBlockIdx() * constInfo.commonConstInfo.s1Size * HEAD_DIM_ALIGN +
                               runInfo.queryGmOffsetForBn2Blk + mL0Idx * l0BaseM * constInfo.commonConstInfo.dSize +
                               i * l1LoopSize + nL0Idx * l0BaseN;
                } else {
                    dqOffset = runInfo.taskPingPong * CUBE_BASEM * HEAD_DIM_ALIGN +
                               mL0Idx * l0BaseM * constInfo.commonConstInfo.dSize + i * l1LoopSize + nL0Idx * l0BaseN;
                    dstStride = constInfo.commonConstInfo.dSize;
                }
            } else if constexpr (IS_TND) {
                dqOffset = runInfo.queryGmOffset + mL0Idx * l0BaseM * constInfo.commonConstInfo.n2GD + i * l1LoopSize +
                           nL0Idx * l0BaseN;
            } else {
                if (constInfo.commonConstInfo.layoutType == BNGSD) {
                    dqOffset = runInfo.queryGmOffset + mL0Idx * l0BaseM * constInfo.commonConstInfo.dSize +
                               i * l1LoopSize + nL0Idx * l0BaseN;
                } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                    dqOffset = runInfo.queryGmOffset + mL0Idx * l0BaseM * constInfo.commonConstInfo.bN2GD +
                               i * l1LoopSize + nL0Idx * l0BaseN;
                } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                    dqOffset = runInfo.queryGmOffset + mL0Idx * l0BaseM * constInfo.commonConstInfo.n2GD +
                               i * l1LoopSize + nL0Idx * l0BaseN;
                }
            }
            if constexpr (IS_ROPE) {
                dqOffset = dqOffset + runInfo.queryRopeGmOffset;
            }
            if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
                if (!constInfo.deterConstInfo.noNeedDeter && isDqNeedDeter) {
                    int64_t deterDqOff =
                        cBlockIdx * BASE_DQ_SIZE +
                        deterPpFlag * (BASE_DQ_SIZE + 2 * BASE_DKV_SIZE) * constInfo.deterConstInfo.usedCubeCoreNum +
                        mL0Idx * l0BaseM * HEAD_DIM_ALIGN + i * l1LoopSize + nL0Idx * l0BaseN;
                    CopyL0CToGm</*FP8_TRANSA*/ false>(*deterGmPtr, L0CTensor, mL0Size, nL0Size, HEAD_DIM_ALIGN,
                                                      deterDqOff, 0.0, false, /*IS_ATOMIC*/ false);
                    l0cBufferManager.ReleaseBuffer(mmL0CPingPong);
                    mmL0CPingPong = (mmL0CPingPong + 1) % L0C_BUF_NUM;
                    continue;
                }
            }
            if constexpr (SPLIT_AXIS == BN2) {
                bool needAtomic = IS_BN2_MULTIBLK ? !runInfo.isFirstS2Outer : false;
                CopyL0CToGm</*FP8_TRANSA*/ false>(dqWorkSpaceGm, L0CTensor, mL0Size, nL0Size, dstStride, dqOffset, 0.0,
                                                  unitFlag, needAtomic);
            } else {
                if (IS_FP8_INPUT) {
                    float deqValue = runInfo.quantScaleInfo.deqScaleKValue;
                    CopyL0CToGm</*FP8_TRANSA*/ is_f8_v<DsType> && isTransA>(dqWorkSpaceGm, L0CTensor, mL0Size, nL0Size,
                                                                            dstStride, dqOffset, deqValue, false,
                                                                            /*IS_ATOMIC*/ true);
                } else {
                    CopyL0CToGm<false>(dqWorkSpaceGm, L0CTensor, mL0Size, nL0Size, dstStride, dqOffset, 0.0, unitFlag,
                                       /*IS_ATOMIC*/ true);
                }
            }
            if constexpr (!ENABLE_UNITFLAG) {
                l0cBufferManager.ReleaseBuffer(mmL0CPingPong);
            }
            mmL0CPingPong = (mmL0CPingPong + 1) % L0C_SHARED_BUF_NUM;
        }
        if constexpr (!IS_TSCM_REUSE_K) {
            L1KVBufferManager.ReleaseBuffer(kvBuffIdx);
        } else {
            if (!runInfo.nextS2IdxNoChange || IS_TSCM_FULL_LOAD) {
                L1KVBufferManager.ReleaseBuffer(kvBuffIdx);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsQ(FagRunInfo& runInfo, FagConstInfo& constInfo,
                                                                 GlobalTensor<float>& dkWorkSpaceGm,
                                                                 bool isDkDvNeedDeter, int64_t cBlockIdx,
                                                                 GlobalTensor<T2>* deterGmPtr, bool deterPpFlag)
{
    constexpr bool isTransA = true;
    constexpr bool isTransB = true;
    uint64_t dstStride = constInfo.commonConstInfo.n2D;
    if constexpr (!IS_TND && !IS_BN2_MULTIBLK) {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            dstStride = constInfo.commonConstInfo.dSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            dstStride = constInfo.commonConstInfo.bN2D;
        }
    } else if constexpr (IS_BN2_MULTIBLK) {
        dstStride = constInfo.commonConstInfo.dSize;
    }
    int64_t dkOffset;
    uint32_t l1LoopSize = IS_FP32_INPUT ? 64 : 256;
    uint32_t l1LoopTimes = (constInfo.commonConstInfo.dSize + l1LoopSize - 1) / l1LoopSize;
    bool isSplitK = false;
    for (uint32_t i = 0; i < l1LoopTimes; i++) {
        if constexpr (IS_TSCM_REUSE) {
            qDxBufferIdx = runInfo.taskPingPong + 2;
        } else {
            qDxBufferIdx = (qDxBufferIdx + 1) % Q_DX_L1_BUF_NUM;
        }
        LocalTensor<T1> L1BTensor = L1QDxTensor[qDxBufferIdx]; // Q L1复用
        int32_t nL1Size = (i == l1LoopTimes - 1) ? constInfo.commonConstInfo.dSize - i * l1LoopSize : l1LoopSize;

        if constexpr (!IS_TSCM_REUSE) { // 非复用场景，重新搬运Q
            int64_t queryGmOffset = runInfo.queryGmOffset + i * l1LoopSize;
            L1QDxBufferManager.GetBuffer(qDxBufferIdx);
            if constexpr (IS_ROPE) {
                queryGmOffset = runInfo.queryGmOffset;
                uint64_t queryRopeOffset = runInfo.queryRopeGmOffset;
                CopyGmToL1(L1BTensor, queryGm, runInfo.commonRunInfo.s1RealSize, ROPE_D_128, constInfo.mm2Ka,
                           runInfo.commonRunInfo.s1RealSize, queryGmOffset, 0);
                CopyGmToL1(L1BTensor, queryRopeGm, runInfo.commonRunInfo.s1RealSize, constInfo.dRopeSize,
                           constInfo.mm2Ka / 2, runInfo.commonRunInfo.s1RealSize, queryRopeOffset,
                           AlignTo16(runInfo.commonRunInfo.s1RealSize) * ROPE_D_128);
            } else {
                CopyGmToL1(L1BTensor, queryGm, runInfo.commonRunInfo.s1RealSize, nL1Size, constInfo.mm2Ka,
                           runInfo.commonRunInfo.s1RealSize, queryGmOffset, 0);
            }
            L1QDxBufferManager.EnQue(qDxBufferIdx);
            L1QDxBufferManager.DeQue(qDxBufferIdx);
        }

        // L0 Matmul: (s1=128, s2=128) ^ T x (s1=128, d=128)
        uint32_t l0BaseM = 128;                      // i.e. s2.
        uint32_t l0BaseN = IS_FP32_INPUT ? 64 : 128; // i.e. d.
        uint32_t l0BaseK = IS_FP32_INPUT ? 64 : 128; // i.e. s1.
        uint32_t mLoopTimes = (runInfo.commonRunInfo.s2RealSize + l0BaseM - 1) / l0BaseM;
        uint32_t nLoopTimes = (nL1Size + l0BaseN - 1) / l0BaseN;
        uint32_t kLoopTimes = (runInfo.commonRunInfo.s1RealSize + l0BaseK - 1) / l0BaseK;
        isSplitK = isSplitK || (kLoopTimes != 1);
        uint32_t l0NKLoopTimes = nLoopTimes * kLoopTimes;
        if (!constInfo.deterConstInfo.noNeedDeter && isDkDvNeedDeter) {
            mLoopTimes = (VECTOR_BASEN + l0BaseM - 1) / l0BaseM;
            l0BaseM = ((VECTOR_BASEN + mLoopTimes - 1) / mLoopTimes * sizeof(T1) + BYTE_BLOCK - 1) / BYTE_BLOCK *
                      BYTE_BLOCK / sizeof(T1);
        } else {
            l0BaseM = ((runInfo.commonRunInfo.s2RealSize + mLoopTimes - 1) / mLoopTimes * sizeof(T1) + BYTE_BLOCK - 1) /
                      BYTE_BLOCK * BYTE_BLOCK / sizeof(T1);
        }
        l0BaseN = ((nL1Size + nLoopTimes - 1) / nLoopTimes * sizeof(T1) + 31) / 32 * 32 / sizeof(T1);
        LocalTensor<T2> L0CTensor;
        // M-Loop
        for (uint32_t mL0Idx = 0; mL0Idx < mLoopTimes; mL0Idx++) {
            uint32_t mL0Size =
                (mL0Idx == mLoopTimes - 1) ? (runInfo.commonRunInfo.s2RealSize - mL0Idx * l0BaseM) : l0BaseM;
            if (!constInfo.deterConstInfo.noNeedDeter && isDkDvNeedDeter) {
                mL0Size = l0BaseM;
            }
            // N-Loop & K-Loop
            bool mmadFirst = true;
            for (uint32_t j = 0; j < l0NKLoopTimes; j++) {
                uint32_t nL0Idx = j / kLoopTimes;
                uint32_t kL0Idx = j % kLoopTimes;
                uint32_t kL0Size =
                    (kL0Idx == kLoopTimes - 1) ? (runInfo.commonRunInfo.s1RealSize - kL0Idx * l0BaseK) : l0BaseK;
                uint32_t nL0Size = (nL0Idx == nLoopTimes - 1) ? (nL1Size - nL0Idx * l0BaseN) : l0BaseN;
                uint32_t l0cIdx = IS_DKV_RESIDENT_L0C ? L0C_DK_BUF_IDX : mmL0CPingPong;
                if (kL0Idx == 0) {
                    L0CTensor =
                        ENABLE_UNITFLAG ? L0CTensorArr[l0cIdx] : L0CTensorArr[l0cBufferManager.GetBuffer(l0cIdx)];
                    mmadFirst = true;
                }
                uint32_t offsetA = kL0Idx * l0BaseK * 32 / sizeof(DsType) +
                                   mL0Idx * AlignTo16(runInfo.commonRunInfo.s2RealSize) * l0BaseM;
                uint32_t offsetB = kL0Idx * l0BaseK * FRACTAL_NZ_C0_SIZE +
                                   nL0Idx * AlignTo16(runInfo.commonRunInfo.s1RealSize) * l0BaseN;
                bool isFixpOut = true;
                if constexpr (IS_DKV_RESIDENT_L0C && !(SPLIT_AXIS == BN2)) {
                    isFixpOut = !runInfo.nextS2IdxNoChange;
                }
                if constexpr (IS_DKV_RESIDENT_L0C && IS_BN2_MULTIBLK) {
                    isFixpOut = !runInfo.nextS2IdxNoChange;
                }
                uint32_t unitFlag =
                    !ENABLE_UNITFLAG ? UNITFLAG_DISABLE : (isFixpOut ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE);
                bool isLastLoopK = (kL0Idx == kLoopTimes - 1);
                SingleMatmul<DsType, T1, isTransA, isTransB, true>(
                    L1DsTensor, L1BTensor, L0CTensor, mL0Size, nL0Size,
                    kL0Size,                                      // m, n, k
                    CUBE_BASEM, runInfo.commonRunInfo.s1RealSize, // A, B src
                    offsetA, offsetB,                             // A, B offset
                    mmadFirst, isSplitK, isLastLoopK, runInfo.isS2IdxNoChange, unitFlag);
                mmadFirst = false;

                if (isLastLoopK) {
                    if constexpr (!ENABLE_UNITFLAG) {
                        l0cBufferManager.EnQue(l0cIdx);
                        l0cBufferManager.DeQue(l0cIdx);
                    }
                    /* dK: (s2 x d), dK-offset = m * d + k
                     * where: m = mL0Idx * loBaseM, k = i * 256 + nL0Idx * l0BaseN
                     */
                    if constexpr (SPLIT_AXIS == BN2) {
                        if constexpr (IS_BN2_MULTIBLK) {
                            dkOffset = GetBlockIdx() * constInfo.commonConstInfo.s2Size * HEAD_DIM_ALIGN +
                                       runInfo.keyGmOffsetForBn2Blk + /*v-offset*/ (i * l1LoopSize + nL0Idx * l0BaseN) +
                                       /*m-offset*/ (mL0Idx * l0BaseM * constInfo.commonConstInfo.dSize);
                        } else {
                            dkOffset = runInfo.taskPingPong * CUBE_BASEN * HEAD_DIM_ALIGN + i * 256 + nL0Idx * l0BaseN;
                            dstStride = constInfo.commonConstInfo.dSize;
                        }
                    } else {
                        dkOffset = runInfo.keyGmOffset + /*k-offset*/ (i * l1LoopSize + nL0Idx * l0BaseN) +
                                   /*m-offset*/ (mL0Idx * l0BaseM * constInfo.commonConstInfo.dSize);
                    }
                    if constexpr (IS_ROPE) {
                        dkOffset = dkOffset + runInfo.keyRopeGmOffset;
                    }
                    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
                        if (!constInfo.deterConstInfo.noNeedDeter && isDkDvNeedDeter) {
                            int64_t deterDkOff = cBlockIdx * BASE_DKV_SIZE +
                                                 BASE_DQ_SIZE * constInfo.deterConstInfo.usedCubeCoreNum +
                                                 deterPpFlag * (BASE_DQ_SIZE + 2 * BASE_DKV_SIZE) *
                                                     constInfo.deterConstInfo.usedCubeCoreNum +
                                                 mL0Idx * l0BaseM * HEAD_DIM_ALIGN + i * l1LoopSize + nL0Idx * l0BaseN;
                            CopyL0CToGm</*FP8_TRANSA*/ false>(*deterGmPtr, L0CTensor, mL0Size, nL0Size, HEAD_DIM_ALIGN,
                                                              deterDkOff, 0.0, false, false);
                            if constexpr (!ENABLE_UNITFLAG) {
                                l0cBufferManager.ReleaseBuffer(mmL0CPingPong);
                            }
                            mmL0CPingPong = (mmL0CPingPong + 1) % L0C_BUF_NUM;
                            continue;
                        }
                    }
                    if constexpr (SPLIT_AXIS == BN2) {
                        if constexpr (IS_BN2_MULTIBLK) {
                            if (isFixpOut) {
                                bool needAtomic = !IS_DKV_RESIDENT_L0C && runInfo.isS2IdxNoChange;
                                CopyL0CToGm</*FP8_TRANSA*/ false>(dkWorkSpaceGm, L0CTensor, mL0Size, nL0Size, dstStride,
                                                                  dkOffset, 0.0, unitFlag, needAtomic);
                            }
                        } else {
                            CopyL0CToGm</*FP8_TRANSA*/ false>(dkWorkSpaceGm, L0CTensor, mL0Size, nL0Size, dstStride,
                                                              dkOffset, 0.0, unitFlag, false);
                        }
                    } else {
                        if (IS_FP8_INPUT) {
                            float deqValue = runInfo.quantScaleInfo.deqScaleQValue;
                            CopyL0CToGm</*FP8_TRANSA*/ is_f8_v<DsType> && isTransA>(
                                dkWorkSpaceGm, L0CTensor, mL0Size, nL0Size, dstStride, dkOffset, deqValue, false, true);
                        } else if (isFixpOut) {
                            CopyL0CToGm</*FP8_TRANSA*/ false>(dkWorkSpaceGm, L0CTensor, mL0Size, nL0Size, dstStride,
                                                              dkOffset, 0.0, unitFlag, true);
                        }
                    }
                    if constexpr (!ENABLE_UNITFLAG) {
                        l0cBufferManager.ReleaseBuffer(l0cIdx);
                    }
                    if constexpr (!IS_DKV_RESIDENT_L0C) {
                        mmL0CPingPong = (mmL0CPingPong + 1) % L0C_BUF_NUM;
                    }
                }
            }
        }
        L1QDxBufferManager.ReleaseBuffer(qDxBufferIdx);
    }
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(DS_RELEASE);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmPDy(
    FagRunInfo& runInfo, FagConstInfo& constInfo,
    GlobalTensor<std::conditional_t<SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK, T1, float>>& dvWorkSpaceGm,
    bool isDkDvNeedDeter, int64_t cBlockIdx, GlobalTensor<T2>* deterGmPtr, bool deterPpFlag)
{
    constexpr bool isTransA = true;
    constexpr bool isTransB = true;
    uint64_t dstStride = constInfo.commonConstInfo.n2Dv;
    if constexpr (!IS_TND && !IS_BN2_MULTIBLK) {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            dstStride = constInfo.commonConstInfo.dSizeV;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            dstStride = constInfo.commonConstInfo.bN2Dv;
        }
    } else if constexpr (IS_BN2_MULTIBLK) {
        dstStride = constInfo.commonConstInfo.dSizeV;
    }
    int64_t dvOffset;
    uint32_t l1LoopSize = IS_FP32_INPUT ? 64 : 256;
    uint32_t l1LoopTimes = (constInfo.commonConstInfo.dSizeV + l1LoopSize - 1) / l1LoopSize; // n切64 (d)
    bool isSplitK = false;
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(POUT_READY);
    for (uint32_t i = 0; i < l1LoopTimes; i++) {
        if constexpr (IS_TSCM_REUSE) {
            qDxBufferIdx = runInfo.taskPingPong;
        } else {
            qDxBufferIdx = (qDxBufferIdx + 1) % Q_DX_L1_BUF_NUM;
        }
        LocalTensor<T1> L1BTensor = L1QDxTensor[qDxBufferIdx]; // Q L1复用
        int32_t nL1Size = (i == l1LoopTimes - 1) ? constInfo.commonConstInfo.dSizeV - i * l1LoopSize : l1LoopSize;

        if constexpr (!IS_TSCM_REUSE) { // 非复用场景，重新搬运Q
            L1QDxBufferManager.GetBuffer(qDxBufferIdx);
            int64_t dxGmOffset = runInfo.dxGmOffset + i * l1LoopSize;
            CopyGmToL1(L1BTensor, dxGm, runInfo.commonRunInfo.s1RealSize, nL1Size, constInfo.commonConstInfo.mm1Ka,
                       runInfo.commonRunInfo.s1RealSize, dxGmOffset, 0);
            L1QDxBufferManager.EnQue(qDxBufferIdx);
            L1QDxBufferManager.DeQue(qDxBufferIdx);
        }

        // L0 Matmul: (s1=128, s2=128) ^ T x (s1=128, d=128)
        uint32_t l0BaseM = 128;                      // i.e. s2.
        uint32_t l0BaseN = IS_FP32_INPUT ? 64 : 128; // i.e. d.
        uint32_t l0BaseK = IS_FP32_INPUT ? 64 : 128; // i.e. s1.
        uint32_t mLoopTimes = (runInfo.commonRunInfo.s2RealSize + l0BaseM - 1) / l0BaseM;
        uint32_t nLoopTimes = (nL1Size + l0BaseN - 1) / l0BaseN;
        uint32_t kLoopTimes = (runInfo.commonRunInfo.s1RealSize + l0BaseK - 1) / l0BaseK;
        isSplitK = isSplitK || (kLoopTimes != 1);
        if (!constInfo.deterConstInfo.noNeedDeter && isDkDvNeedDeter) {
            mLoopTimes = (VECTOR_BASEN + l0BaseM - 1) / l0BaseM;
            l0BaseM = ((VECTOR_BASEN + mLoopTimes - 1) / mLoopTimes * sizeof(T1) + BYTE_BLOCK - 1) / BYTE_BLOCK *
                      BYTE_BLOCK / sizeof(T1);
        } else {
            l0BaseM = ((runInfo.commonRunInfo.s2RealSize + mLoopTimes - 1) / mLoopTimes * sizeof(T1) + BYTE_BLOCK - 1) /
                      BYTE_BLOCK * BYTE_BLOCK / sizeof(T1);
        }
        l0BaseN = ((nL1Size + nLoopTimes - 1) / nLoopTimes * sizeof(T1) + 31) / 32 * 32 / sizeof(T1);
        uint32_t l0LoopTimes = kLoopTimes * nLoopTimes;
        LocalTensor<T2> L0CTensor;
        // M-Loop
        for (uint32_t mL0Idx = 0; mL0Idx < mLoopTimes; mL0Idx++) {
            uint32_t mL0Size =
                (mL0Idx == mLoopTimes - 1) ? (runInfo.commonRunInfo.s2RealSize - mL0Idx * l0BaseM) : l0BaseM;
            if (!constInfo.deterConstInfo.noNeedDeter && isDkDvNeedDeter) {
                mL0Size = l0BaseM;
            }
            // N-Loop & K-Loop
            bool mmadFirst = true;
            for (uint32_t j = 0; j < l0LoopTimes; j++) {
                uint32_t nL0Idx = j / kLoopTimes;
                uint32_t kL0Idx = j % kLoopTimes;
                uint32_t kL0Size =
                    (kL0Idx == kLoopTimes - 1) ? (runInfo.commonRunInfo.s1RealSize - kL0Idx * l0BaseK) : l0BaseK;
                uint32_t nL0Size = (nL0Idx == nLoopTimes - 1) ? (nL1Size - nL0Idx * l0BaseN) : l0BaseN;

                uint32_t l0cIdx = IS_DKV_RESIDENT_L0C ? L0C_DV_BUF_IDX : mmL0CPingPong;
                if (kL0Idx == 0) {
                    L0CTensor =
                        ENABLE_UNITFLAG ? L0CTensorArr[l0cIdx] : L0CTensorArr[l0cBufferManager.GetBuffer(l0cIdx)];
                    mmadFirst = true;
                }
                uint32_t offsetA = kL0Idx * l0BaseK * FRACTAL_NZ_C0_SIZE +
                                   mL0Idx * AlignTo16(runInfo.commonRunInfo.s2RealSize) * l0BaseM;
                uint32_t offsetB = kL0Idx * l0BaseK * FRACTAL_NZ_C0_SIZE +
                                   nL0Idx * AlignTo16(runInfo.commonRunInfo.s1RealSize) * l0BaseN;
                bool isFixpOut = true;
                if constexpr (IS_DKV_RESIDENT_L0C && !(SPLIT_AXIS == BN2)) {
                    isFixpOut = !runInfo.nextS2IdxNoChange;
                }
                if constexpr (IS_BN2_MULTIBLK && IS_DKV_RESIDENT_L0C) {
                    isFixpOut = !runInfo.nextS2IdxNoChange;
                }
                uint32_t unitFlag =
                    !ENABLE_UNITFLAG ? UNITFLAG_DISABLE : (isFixpOut ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE);
                bool isLastLoopK = (kL0Idx == kLoopTimes - 1);
                SingleMatmul<T1, T1, isTransA, isTransB, true>(L1PTensor, L1BTensor, L0CTensor, mL0Size, nL0Size,
                                                               kL0Size,                                      // m, n, k
                                                               CUBE_BASEM, runInfo.commonRunInfo.s1RealSize, // A, B src
                                                               offsetA, offsetB, // A, B offset
                                                               mmadFirst, isSplitK, isLastLoopK,
                                                               runInfo.isS2IdxNoChange, unitFlag);
                mmadFirst = false;

                if (isLastLoopK) {
                    if constexpr (!ENABLE_UNITFLAG) {
                        l0cBufferManager.EnQue(l0cIdx);
                        l0cBufferManager.DeQue(l0cIdx);
                    }
                    if constexpr (IS_BN2_MULTIBLK) {
                        dvOffset = GetBlockIdx() * constInfo.commonConstInfo.s2Size * HEAD_DIM_ALIGN +
                                   runInfo.valueGmOffsetForBn2Blk + /*k-offset*/ (i * l1LoopSize + nL0Idx * l0BaseN) +
                                   /*m-offset*/ (mL0Idx * l0BaseM * HEAD_DIM_ALIGN);
                    } else {
                        dvOffset = runInfo.valueGmOffset + /*k-offset*/ (i * l1LoopSize + nL0Idx * l0BaseN) +
                                   /*m-offset*/ (mL0Idx * l0BaseM * constInfo.commonConstInfo.dSizeV);
                    }
                    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
                        if (!constInfo.deterConstInfo.noNeedDeter && isDkDvNeedDeter) {
                            int64_t deterDvOff =
                                cBlockIdx * BASE_DKV_SIZE +
                                (BASE_DQ_SIZE + BASE_DKV_SIZE) * constInfo.deterConstInfo.usedCubeCoreNum +
                                deterPpFlag * (BASE_DQ_SIZE + 2 * BASE_DKV_SIZE) *
                                    constInfo.deterConstInfo.usedCubeCoreNum +
                                mL0Idx * l0BaseM * HEAD_DIM_ALIGN + i * l1LoopSize + nL0Idx * l0BaseN;
                            CopyL0CToGm</*FP8_TRANSA*/ false>(*deterGmPtr, L0CTensor, mL0Size, nL0Size, HEAD_DIM_ALIGN,
                                                              deterDvOff, 0.0, false, false);
                            l0cBufferManager.ReleaseBuffer(mmL0CPingPong);
                            mmL0CPingPong = (mmL0CPingPong + 1) % L0C_BUF_NUM;
                            continue;
                        }
                    }
                    if constexpr (SPLIT_AXIS == BN2) {
                        if constexpr (IS_BN2_MULTIBLK) {
                            if (isFixpOut) {
                                bool needAtomic = (!IS_DKV_RESIDENT_L0C) && runInfo.isS2IdxNoChange;
                                CopyL0CToGm</*FP8_TRANSA*/ false>(dvWorkSpaceGm, L0CTensor, mL0Size, nL0Size, dstStride,
                                                                  dvOffset, 0.0, unitFlag, needAtomic);
                            }
                        } else {
                            CopyL0CToGm(dvWorkSpaceGm, L0CTensor, mL0Size, nL0Size, dstStride, dvOffset, unitFlag);
                        }
                    } else {
                        if (IS_FP8_INPUT) {
                            float deqValue = runInfo.quantScaleInfo.deqScaleDyValue;
                            CopyL0CToGm</*FP8_TRANSA*/ is_f8_v<T1> && isTransA>(
                                dvWorkSpaceGm, L0CTensor, mL0Size, nL0Size, dstStride, dvOffset, deqValue, false, true);
                        } else if (isFixpOut) {
                            CopyL0CToGm</*FP8_TRANSA*/ false>(dvWorkSpaceGm, L0CTensor, mL0Size, nL0Size, dstStride,
                                                              dvOffset, 0.0, unitFlag, true);
                        }
                    }
                    if constexpr (!ENABLE_UNITFLAG) {
                        l0cBufferManager.ReleaseBuffer(l0cIdx);
                    }
                    if constexpr (!IS_DKV_RESIDENT_L0C) {
                        mmL0CPingPong = (mmL0CPingPong + 1) % L0C_BUF_NUM;
                    }
                }
            }
        }
        L1QDxBufferManager.ReleaseBuffer(qDxBufferIdx);
    }
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(POUT_RELEASE);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename DataType>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::LoadDataToL0(LocalTensor<DataType> dstTensor,
                                                                 LocalTensor<DataType> srcTensor, uint32_t m,
                                                                 uint32_t k, uint32_t srcM, uint64_t srcOffset,
                                                                 bool isTrans)
{
    /// NOTE: 对于FP32,FP4类型，下方的下标计算需要额外考虑
    if constexpr (is_f16_v<DataType>) {
        LoadData2DParamsV2 loadData2DParams; // 基础API LoadData的参数结构体
        loadData2DParams.mStartPosition = 0; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
        loadData2DParams.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B
        loadData2DParams.ifTranspose = isTrans; // 是否启用转置功能
        loadData2DParams.mStep =
            (m + 15) >> 4; // 以M*K矩阵为例, 源矩阵M轴方向搬运长度, 单位为16 element, 取值范围：mStep∈[0, 255]
        loadData2DParams.kStep = (k * sizeof(DataType) + 31) /
                                 32; // 以M*K矩阵为例, 源矩阵K轴方向搬运长度, 单位为32B, 取值范围：nStep∈[0, 255]
        loadData2DParams.srcStride =
            (srcM + 15) >>
            4; // 以M*K矩阵为例，在源矩阵K方向上，前一个分形起始地址与后一个分形起始地址的间隔, 单位：512B
        loadData2DParams.dstStride = isTrans ? ((k + 15) >> 4) : loadData2DParams.mStep;
        LoadData(dstTensor, srcTensor[srcOffset], loadData2DParams);
    } else if constexpr (is_f8_v<DataType>) {
        LoadData2DParamsV2 loadData2DParams;
        loadData2DParams.mStartPosition = 0;
        loadData2DParams.kStartPosition = 0;
        loadData2DParams.ifTranspose = isTrans;
        loadData2DParams.mStep = isTrans ? (((m + 31) >> 5) << 1) : ((m + 15) >> 4);
        loadData2DParams.kStep = (k * sizeof(DataType) + 31) / 32;
        loadData2DParams.srcStride = (srcM + 15) >> 4;
        loadData2DParams.dstStride = isTrans ? (((k + 31) >> 5) << 1) : loadData2DParams.mStep;
        LoadData(dstTensor, srcTensor[srcOffset], loadData2DParams);
    } else if constexpr (is_f32_v<DataType>) {
        LoadData2DParamsV2 loadData2DParams;
        loadData2DParams.mStartPosition = 0;
        loadData2DParams.kStartPosition = 0;
        loadData2DParams.ifTranspose = isTrans;
        loadData2DParams.mStep = (m + 15) >> 4;
        loadData2DParams.kStep = ((k + 7) >> 3 << 3) >> 3;
        if (isTrans) {
            loadData2DParams.kStep = (loadData2DParams.kStep + 1) & ~1;
        }
        loadData2DParams.srcStride = (srcM + 15) >> 4;
        loadData2DParams.dstStride = isTrans ? ((k + 15) >> 4) : loadData2DParams.mStep;
        LoadData(dstTensor, srcTensor[srcOffset], loadData2DParams);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::CopyL0CToUB(LocalTensor<T2>& mmResUb, LocalTensor<T2>& L0CTensor,
                                                                uint32_t m, uint32_t n, uint32_t unitFlag)
{
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    fixpipeParams.nSize = (n + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小(32B对齐，float类型8对齐)
    fixpipeParams.mSize = m; // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小
    fixpipeParams.srcStride =
        ((fixpipeParams.mSize + 15) / 16) *
        16; // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T)
    fixpipeParams.dstStride = VECTOR_BASEN; // mmResUb上两行之间的间隔，单位：element。
    fixpipeParams.dualDstCtl = 0; // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.subBlockId = 0;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    fixpipeParams.unitFlag = unitFlag;

    Fixpipe<T2, T2, FAG_CFG_ROW_MAJOR_UB>(mmResUb, L0CTensor, fixpipeParams); // 将matmul结果从L0C搬运到UB
}

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FP8_TRANS_A>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::CopyL0CToGm(GlobalTensor<T2>& dstTensor, LocalTensor<T2>& L0CTensor,
                                                                uint32_t m, uint32_t n, uint64_t dstStride,
                                                                uint64_t dstOffset, float deqScaleValue,
                                                                uint32_t unitFlag, bool isAtomic)
{
    // 对于 FP8 的 TRANS_A 场景，srcStride 是 32element 对齐的
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = n; // L0C上的bmm1结果矩阵N方向的size大小 32B对齐
    fixpipeParams.mSize = m; // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小
    if constexpr (IS_FP8_TRANS_A) {
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 31) >> 5) << 5;
    } else {
        // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T)
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
    }
    fixpipeParams.dstStride = dstStride; // mmResUb上两行之间的间隔，单位：element。
    if (IS_FP8_INPUT) {
        fixpipeParams.quantPre = QuantMode_t::QF322F32_PRE;
        fixpipeParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&deqScaleValue));
    }
    fixpipeParams.dualDstCtl = 0; // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    fixpipeParams.unitFlag = unitFlag;

    if (isAtomic) {
        SetAtomicAdd<T2>();
    }
    Fixpipe<T2, T2, FAG_CFG_ROW_MAJOR_GM>(dstTensor[dstOffset], L0CTensor, fixpipeParams); // L0c -> GM
    if (isAtomic) {
        SetAtomicNone();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::CopyL0CToGm(GlobalTensor<T1>& dstTensor, LocalTensor<T2>& L0CTensor,
                                                                uint32_t m, uint32_t n, uint64_t dstStride,
                                                                uint64_t dstOffset, uint32_t unitFlag)
{
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = n; // L0C上的bmm1结果矩阵N方向的size大小 32B对齐
    fixpipeParams.mSize = m; // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小
    fixpipeParams.srcStride =
        ((fixpipeParams.mSize + 15) / 16) *
        16; // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T)
    fixpipeParams.dstStride = dstStride; // mmResUb上两行之间的间隔，单位：element。
    if constexpr (IsSameType<T1, half>::value) {
        fixpipeParams.quantPre = QuantMode_t::F322F16;
    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
        fixpipeParams.quantPre = QuantMode_t::F322BF16;
    }
    fixpipeParams.dualDstCtl = 0; // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    fixpipeParams.unitFlag = unitFlag;

    Fixpipe<T1, T2, FAG_CFG_ROW_MAJOR_GM>(dstTensor[dstOffset], L0CTensor, fixpipeParams); // L0c -> GM
}

TEMPLATES_DEF
class FAGBlockCubeDummy {
public:
    __aicore__ inline FAGBlockCubeDummy(){};
    __aicore__ inline void SetCubeBlockParams(TPipe* pipe, FagTilingType tilingData) {};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR queryRope,
                                            GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace) {};
    __aicore__ inline void InitCubeBuffer() {};
    __aicore__ inline void FreeAlloc(int64_t innerMm12LoopIdx) {};
    __aicore__ inline void FreeBuffer() {};
    template <typename AType, typename BType, const bool IS_TRANS_A, const bool IS_TRANS_B, const bool IS_DKDV>
    __aicore__ inline void SingleMatmul(LocalTensor<AType> L1ATensor, LocalTensor<BType> L1BTensor,
                                        LocalTensor<T2> L0CTensor, uint32_t m, uint32_t n, uint32_t k, uint32_t srcA,
                                        uint32_t srcB, uint32_t offsetA, uint32_t offsetB, bool isInitVal,
                                        bool isSplitK, bool isLastLoopK = false, bool isS2IdxNoChange = false,
                                        uint32_t unitFlag = 0) {};
    __aicore__ inline void CopyGmToL1(LocalTensor<T1> dstTensor, GlobalTensor<T1> srcTensor, uint32_t nValue,
                                      uint32_t dValue, uint32_t srcStride, uint32_t dstStride, uint64_t srcOffset,
                                      uint64_t dstOffset) {};
    __aicore__ inline void IterateMmQKAndDyV(LocalTensor<T2>& mm1ResTensor, LocalTensor<T2>& mm2ResTensor,
                                             FagRunInfo& runInfo, FagRunInfo& nextRunInfo, FagConstInfo& constInfo,
                                             int8_t innerLoopNum, int8_t innerLoopIdx, int8_t baseUbBufIdx) {};
    __aicore__ inline void IterateMmDyV(LocalTensor<T2>& mm1ResTensor, FagRunInfo& runInfo, FagConstInfo& constInfo,
                                        LocalTensor<T2>& L0CTensor, uint32_t mIdx, uint32_t mSize,
                                        bool isLast) {}; // mm1
    __aicore__ inline void IterateMmQK(LocalTensor<T2>& mm2ResTensor, FagRunInfo& runInfo, FagConstInfo& constInfo,
                                       LocalTensor<T2>& L0CTensor, uint32_t mIdx, uint32_t mSize,
                                       bool isLast) {}; // mm2
    __aicore__ inline void IterateMmDsK(FagRunInfo& runInfo, FagConstInfo& constInfo,
                                        GlobalTensor<float>& dqWorkSpaceGm, bool isDqNeedDeter = false,
                                        int64_t cBlockIdx = 0, GlobalTensor<T2>* deterGmPtr = nullptr,
                                        bool deterPpFlag = false) {}; // mm3 dq
    __aicore__ inline void IterateMmDsQ(FagRunInfo& runInfo, FagConstInfo& constInfo,
                                        GlobalTensor<float>& dkWorkSpaceGm, bool isDkDvNeedDeter = false,
                                        int64_t cBlockIdx = 0, GlobalTensor<T2>* deterGmPtr = nullptr,
                                        bool deterPpFlag = false) {}; // mm4 dk
    __aicore__ inline void IterateMmPDy(
        FagRunInfo& runInfo, FagConstInfo& constInfo,
        GlobalTensor<std::conditional_t<SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK, T1, float>>& dvWorkSpaceGm,
        bool isDkDvNeedDeter = false, int64_t cBlockIdx = 0, GlobalTensor<T2>* deterGmPtr = nullptr,
        bool deterPpFlag = false) {}; // mm5 dv
    template <typename DataType>
    __aicore__ inline void LoadDataToL0(LocalTensor<DataType> dstTensor, LocalTensor<DataType> srcTensor, uint32_t m,
                                        uint32_t k, uint32_t srcM, uint64_t srcOffset, bool isTrans) {};
    template <typename AType, typename BType, bool IsDkDv = false>
    __aicore__ inline void ComputeBaseMm(LocalTensor<T2>& L0CTensor, LocalTensor<AType>& L0ATensor,
                                         LocalTensor<BType>& L0BTensor, uint32_t m, uint32_t n, uint64_t k,
                                         bool isInitVal, bool isS2IdxNoChange = false, uint32_t unitFlag = 0) {};
    __aicore__ inline void CopyL0CToUB(LocalTensor<T2>& mmResUb, LocalTensor<T2>& L0CTensor, uint32_t m, uint32_t n,
                                       uint32_t unitFlag = 0) {};
    template <bool IS_FP8_TRANS_A>
    __aicore__ inline void CopyL0CToGm(GlobalTensor<T2>& dstTensor, LocalTensor<T2>& L0CTensor, uint32_t m, uint32_t n,
                                       uint64_t dstStride, uint64_t dstOffset, float deqScaleValue,
                                       uint32_t unitFlag = 0, bool isAtomic = false) {};
    __aicore__ inline void CopyL0CToGm(GlobalTensor<T1>& dstTensor, LocalTensor<T2>& L0CTensor, uint32_t m, uint32_t n,
                                       uint64_t dstStride, uint64_t dstOffset, uint32_t unitFlag = 0) {};
};

template <typename T>
struct CubeBlockTraits; // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    };

DEFINE_CUBE_BLOCK_TRAITS(FAGBlockCube);
DEFINE_CUBE_BLOCK_TRAITS(FAGBlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)

} // namespace FagBaseApi
#endif
