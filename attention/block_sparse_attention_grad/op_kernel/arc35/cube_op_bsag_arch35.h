/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
using namespace AscendC;

namespace BSA_ARC35 {

template <typename BSA_TYPE>
class CubeOp {
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    static constexpr uint32_t INPUT_LAYOUT = BSA_TYPE::input_layout;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;

private:
    int32_t batch_num_;
    int32_t q_seq_len_;
    int32_t kv_seq_len_;
    int32_t q_group_;
    int32_t q_head_num_;
    int32_t kv_head_num_;
    int32_t head_dim_;
    int32_t head_dim_align_;
    int32_t event_ping_pong_flag{0};
    uint32_t q_stride_;
    uint32_t kv_stride_;
    // l1_tensor
    LocalTensor<INPUT_TYPE> query_l1_tensor_ping_;
    LocalTensor<INPUT_TYPE> query_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> key_l1_tensor_ping_;
    LocalTensor<INPUT_TYPE> key_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> value_l1_tensor_ping_;
    LocalTensor<INPUT_TYPE> value_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> dy_l1_tensor_ping_;
    LocalTensor<INPUT_TYPE> dy_l1_tensor_pong_;
    // l0_tensor
    LocalTensor<INPUT_TYPE> l0_a_tensor_ping_;
    LocalTensor<INPUT_TYPE> l0_a_tensor_pong_;
    LocalTensor<INPUT_TYPE> l0_b_tensor_ping_;
    LocalTensor<INPUT_TYPE> l0_b_tensor_pong_;
    LocalTensor<float> l0_c_tensor_ping_;
    LocalTensor<float> l0_c_tensor_pong_;
    // dk\dvL0C
    LocalTensor<float> l0_c_dk_tensor_;
    LocalTensor<float> l0_c_dv_tensor_;
    static constexpr int32_t L1_SIZE = 512 * 1024;
    static constexpr int32_t L0_AB_SIZE = 64 * 1024;
    static constexpr int32_t L0_C_SIZE = 256 * 1024;
    static constexpr int32_t BASE_BLOCK_SIZE = 128 * 128;
    TEventID event_ping_ = EVENT_ID3;
    TEventID event_pong_ = EVENT_ID4;

public:
    __aicore__ inline CubeOp(){};
    __aicore__ inline void Init(const TILING_CLASS *tilingData, TPipe *tPipe, TBuf<TPosition::A1> &l1_buffer,
                                uint32_t l1_offset)
    {
        this->batch_num_ = tilingData->batchNum;
        this->q_seq_len_ = tilingData->qSeqLen;
        this->kv_seq_len_ = tilingData->kvSeqLen;
        this->q_group_ = tilingData->qGroup;
        this->q_head_num_ = tilingData->qHeadNum;
        this->kv_head_num_ = tilingData->kvHeadNum;
        this->head_dim_ = tilingData->headDim;
        this->head_dim_align_ = RoundUp(head_dim_, 16);
        uint32_t base_m = tilingData->baseM;
        uint32_t base_n = tilingData->baseN;
        if constexpr (INPUT_LAYOUT == BSND) {
            q_stride_ = q_head_num_ * head_dim_;
            kv_stride_ = kv_head_num_ * head_dim_;
        } else if constexpr (INPUT_LAYOUT == BNSD) {
            q_stride_ = head_dim_;
            kv_stride_ = head_dim_;
        } else if constexpr (INPUT_LAYOUT == TND) {
            q_stride_ = q_head_num_ * head_dim_;
            kv_stride_ = kv_head_num_ * head_dim_;
        }

        TBuf<TPosition::A2> l0_a_buffer_;
        TBuf<TPosition::B2> l0_b_buffer_;
        TBuf<TPosition::CO1> l0_c_buffer_;
        tPipe->InitBuffer(l0_a_buffer_, L0_AB_SIZE);
        tPipe->InitBuffer(l0_b_buffer_, L0_AB_SIZE);
        tPipe->InitBuffer(l0_c_buffer_, L0_C_SIZE);
        // query_l1
        query_l1_tensor_ping_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align_, l1_offset);
        l1_offset += base_m * head_dim_align_ * sizeof(INPUT_TYPE);
        query_l1_tensor_pong_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align_, l1_offset);
        l1_offset += base_m * head_dim_align_ * sizeof(INPUT_TYPE);
        // key_l1
        key_l1_tensor_ping_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_n * head_dim_align_, l1_offset);
        l1_offset += base_n * head_dim_align_ * sizeof(INPUT_TYPE);
        key_l1_tensor_pong_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_n * head_dim_align_, l1_offset);
        l1_offset += base_n * head_dim_align_ * sizeof(INPUT_TYPE);
        // value_l1
        value_l1_tensor_ping_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_n * head_dim_align_, l1_offset);
        l1_offset += base_m * head_dim_align_ * sizeof(INPUT_TYPE);
        value_l1_tensor_pong_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_n * head_dim_align_, l1_offset);
        l1_offset += base_m * head_dim_align_ * sizeof(INPUT_TYPE);
        // dy_l1
        dy_l1_tensor_ping_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_n * head_dim_align_, l1_offset);
        l1_offset += base_n * head_dim_align_ * sizeof(INPUT_TYPE);
        dy_l1_tensor_pong_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_n * head_dim_align_, l1_offset);
        l1_offset += base_n * head_dim_align_ * sizeof(INPUT_TYPE);
        // l0
        l0_a_tensor_ping_ = l0_a_buffer_.GetWithOffset<INPUT_TYPE>(32 * 1024 / sizeof(INPUT_TYPE), 0);
        l0_a_tensor_pong_ = l0_a_buffer_.GetWithOffset<INPUT_TYPE>(32 * 1024 / sizeof(INPUT_TYPE), 32 * 1024);
        l0_b_tensor_ping_ = l0_b_buffer_.GetWithOffset<INPUT_TYPE>(32 * 1024 / sizeof(INPUT_TYPE), 0);
        l0_b_tensor_pong_ = l0_b_buffer_.GetWithOffset<INPUT_TYPE>(32 * 1024 / sizeof(INPUT_TYPE), 32 * 1024);
        l0_c_tensor_ping_ = l0_c_buffer_.GetWithOffset<float>(64 * 1024 / sizeof(float), 0);
        l0_c_tensor_pong_ = l0_c_buffer_.GetWithOffset<float>(64 * 1024 / sizeof(float), 64 * 1024);
        l0_c_dk_tensor_ = l0_c_buffer_.GetWithOffset<float>(64 * 1024 / sizeof(float), 128 * 1024);
        l0_c_dv_tensor_ = l0_c_buffer_.GetWithOffset<float>(64 * 1024 / sizeof(float), 192 * 1024);
        SET_FLAG(M, MTE1, event_ping_);
        SET_FLAG(M, MTE1, event_pong_);
        SET_FLAG(FIX, M, event_ping_);
        SET_FLAG(FIX, M, event_pong_);
    }

    __aicore__ inline void Destroy()
    {
        WAIT_FLAG(M, MTE1, event_ping_);
        WAIT_FLAG(M, MTE1, event_pong_);
        WAIT_FLAG(FIX, M, event_ping_);
        WAIT_FLAG(FIX, M, event_pong_);
    }

    __aicore__ inline void SendMatmulQK(const GlobalTensor<INPUT_TYPE> &queryGm, const GlobalTensor<INPUT_TYPE> &keyGm,
                                        const LocalTensor<float> &mm1OutUb, const RunTimeInfo &runTimeInfo,
                                        const uint32_t ping_pong_idx)
    {
        LocalTensor<INPUT_TYPE> l1_a_tensor = ping_pong_idx == 0 ? query_l1_tensor_ping_ : query_l1_tensor_pong_;
        LocalTensor<INPUT_TYPE> l1_b_tensor =
            runTimeInfo.kv_ping_pong_idx == 0 ? key_l1_tensor_ping_ : key_l1_tensor_pong_;

        ComputeMM12(queryGm[runTimeInfo.queryGmOffset], keyGm[runTimeInfo.keyGmOffset], l1_a_tensor, l1_b_tensor,
                    mm1OutUb, runTimeInfo);
        event_ping_pong_flag = 1 - event_ping_pong_flag;
    }

    __aicore__ inline void SendMatmulDyV(const GlobalTensor<INPUT_TYPE> &dyGm, const GlobalTensor<INPUT_TYPE> &valueGm,
                                         LocalTensor<float> &mm2OutUb, const RunTimeInfo &runTimeInfo,
                                         const uint32_t ping_pong_idx)
    {
        LocalTensor<INPUT_TYPE> l1_a_tensor = ping_pong_idx == 0 ? dy_l1_tensor_ping_ : dy_l1_tensor_pong_;
        LocalTensor<INPUT_TYPE> l1_b_tensor =
            runTimeInfo.kv_ping_pong_idx == 0 ? value_l1_tensor_ping_ : value_l1_tensor_pong_;

        ComputeMM12(dyGm[runTimeInfo.queryGmOffset], valueGm[runTimeInfo.keyGmOffset], l1_a_tensor, l1_b_tensor,
                    mm2OutUb, runTimeInfo);
        event_ping_pong_flag = 1 - event_ping_pong_flag;
    }

    __aicore__ inline void SendMatmulDq(const LocalTensor<INPUT_TYPE> &ds_l1_tensor, const GlobalTensor<float> &outGm,
                                        const RunTimeInfo &runTimeInfo, const uint32_t ping_pong_idx)
    {
        LocalTensor<INPUT_TYPE> l1_b_tensor =
            runTimeInfo.kv_ping_pong_idx == 0 ? key_l1_tensor_ping_ : key_l1_tensor_pong_;

        ComputeMMDQ(ds_l1_tensor, l1_b_tensor, outGm[runTimeInfo.queryGmOffset], runTimeInfo);
        event_ping_pong_flag = 1 - event_ping_pong_flag;
    }

    __aicore__ inline void SendMatmulDk(const LocalTensor<INPUT_TYPE> &ds_l1_tensor, const GlobalTensor<float> &outGm,
                                        const RunTimeInfo &runTimeInfo, const uint32_t ping_pong_idx)
    {
        LocalTensor<INPUT_TYPE> l1_b_tensor = ping_pong_idx == 0 ? query_l1_tensor_ping_ : query_l1_tensor_pong_;
        ComputeMMDKV<DK>(ds_l1_tensor, l1_b_tensor, outGm[runTimeInfo.keyGmOffset], runTimeInfo);
        event_ping_pong_flag = 1 - event_ping_pong_flag;
    }

    __aicore__ inline void SendMatmulDv(const LocalTensor<INPUT_TYPE> &p_l1_tensor, const GlobalTensor<float> &outGm,
                                        const RunTimeInfo &runTimeInfo, const uint32_t ping_pong_idx)
    {
        LocalTensor<INPUT_TYPE> l1_b_tensor = ping_pong_idx == 0 ? dy_l1_tensor_ping_ : dy_l1_tensor_pong_;
        ComputeMMDKV<DV>(p_l1_tensor, l1_b_tensor, outGm[runTimeInfo.keyGmOffset], runTimeInfo);
        event_ping_pong_flag = 1 - event_ping_pong_flag;
    }

private:
    __aicore__ inline void ComputeMM12(const GlobalTensor<INPUT_TYPE> &leftGm, const GlobalTensor<INPUT_TYPE> &rightGm,
                                       const LocalTensor<INPUT_TYPE> &l1_a_tensor,
                                       const LocalTensor<INPUT_TYPE> &l1_b_tensor, const LocalTensor<float> &outUb,
                                       const RunTimeInfo &runTimeInfo)
    {
        int32_t mProcess = runTimeInfo.s1Len;
        int32_t nProcess = runTimeInfo.s2Len;
        int32_t mProcessAlign = runTimeInfo.s1LenAlign;
        int32_t nProcessAlign = runTimeInfo.s2LenAlign;
        LocalTensor<INPUT_TYPE> l0_a_tensor = event_ping_pong_flag ? l0_a_tensor_ping_ : l0_a_tensor_pong_;
        LocalTensor<INPUT_TYPE> l0_b_tensor = event_ping_pong_flag ? l0_b_tensor_ping_ : l0_b_tensor_pong_;
        LocalTensor<float> l0_c_tensor = event_ping_pong_flag ? l0_c_tensor_ping_ : l0_c_tensor_pong_;
        TEventID evnet_id = event_ping_pong_flag ? event_ping_ : event_pong_;

        WAIT_FLAG(M, MTE1, evnet_id);
        load_data_gm_2_l0<true>(l0_a_tensor, l1_a_tensor, leftGm, mProcess, head_dim_, mProcessAlign, head_dim_align_,
                                q_stride_);
        if (runTimeInfo.need_copy_kv) {
            load_data_gm_2_l0_trans<false>(l0_b_tensor, l1_b_tensor, rightGm, nProcess, head_dim_, nProcessAlign,
                                           head_dim_align_, kv_stride_);
        } else {
            load_data_l1_2_l0_nz(l0_b_tensor, l1_b_tensor, nProcessAlign, head_dim_align_);
        }

        SET_FLAG(MTE1, M, EVENT_ID0);
        WAIT_FLAG(MTE1, M, EVENT_ID0);

        WAIT_FLAG(FIX, M, evnet_id);
        MmadParams madParams;
        madParams.m = mProcess == 1 ? 2 : mProcess;
        madParams.n = nProcess;
        madParams.k = head_dim_;
        madParams.cmatrixInitVal = true;
        madParams.unitFlag = 3;
        AscendC::Mmad(l0_c_tensor, l0_a_tensor, l0_b_tensor, madParams);
        AscendC::PipeBarrier<PIPE_M>();
        SET_FLAG(M, MTE1, evnet_id);

        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.mSize = mProcessAlign;
        fixpipeParams.nSize = nProcessAlign;
        fixpipeParams.srcStride = mProcessAlign;
        fixpipeParams.dstStride = nProcessAlign;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.unitFlag = 3;
        constexpr static FixpipeConfig ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};
        AscendC::Fixpipe<float, float, ROW_MAJOR_UB>(outUb, l0_c_tensor, fixpipeParams);
        SET_FLAG(FIX, M, evnet_id);
    }

    template <uint32_t TYPE>
    __aicore__ inline void ComputeMMDKV(const LocalTensor<INPUT_TYPE> &l1_a_tensor,
                                        const LocalTensor<INPUT_TYPE> &l1_b_tensor, const GlobalTensor<float> &outGm,
                                        const RunTimeInfo &runTimeInfo)
    {
        int32_t mProcess = runTimeInfo.s1Len;
        int32_t nProcess = runTimeInfo.s2Len;
        int32_t mProcessAlign = runTimeInfo.s1LenAlign;
        int32_t nProcessAlign = runTimeInfo.s2LenAlign;
        LocalTensor<INPUT_TYPE> l0_a_tensor = event_ping_pong_flag ? l0_a_tensor_ping_ : l0_a_tensor_pong_;
        LocalTensor<INPUT_TYPE> l0_b_tensor = event_ping_pong_flag ? l0_b_tensor_ping_ : l0_b_tensor_pong_;
        LocalTensor<float> l0_c_tensor;
        if constexpr (TYPE == DK) {
            l0_c_tensor = l0_c_dk_tensor_;
        } else {
            l0_c_tensor = l0_c_dv_tensor_;
        }
        TEventID evnet_id = event_ping_pong_flag ? event_ping_ : event_pong_;

        WAIT_FLAG(M, MTE1, evnet_id);
        load_data_l1_2_l0_zn(l0_a_tensor, l1_a_tensor, mProcessAlign, nProcessAlign);
        load_data_l1_2_l0_zn(l0_b_tensor, l1_b_tensor, mProcessAlign, head_dim_align_);
        SET_FLAG(MTE1, M, EVENT_ID0);
        WAIT_FLAG(MTE1, M, EVENT_ID0);

        WAIT_FLAG(FIX, M, evnet_id);
        MmadParams madParams;
        madParams.m = nProcess == 1 ? 2 : nProcess;
        madParams.n = head_dim_;
        madParams.k = mProcess;
        madParams.cmatrixInitVal = runTimeInfo.need_copy_kv;
        madParams.unitFlag = runTimeInfo.is_singlekv_last ? 3 : 2;
        AscendC::Mmad(l0_c_tensor, l0_a_tensor, l0_b_tensor, madParams);
        AscendC::PipeBarrier<PIPE_M>();
        SET_FLAG(M, MTE1, evnet_id);

        if (runTimeInfo.is_singlekv_last) {
            AscendC::FixpipeParamsV220 fixpipeParamsV220;
            fixpipeParamsV220.mSize = nProcess;
            fixpipeParamsV220.nSize = head_dim_;
            fixpipeParamsV220.srcStride = nProcessAlign;
            fixpipeParamsV220.dstStride = kv_stride_;
            fixpipeParamsV220.unitFlag = 3;
            MM345CopyOut<true>(outGm, l0_c_tensor, fixpipeParamsV220);
        }
        SET_FLAG(FIX, M, evnet_id);
    }

    __aicore__ inline void ComputeMMDQ(const LocalTensor<INPUT_TYPE> &l1_a_tensor,
                                       const LocalTensor<INPUT_TYPE> &l1_b_tensor, const GlobalTensor<float> &outGm,
                                       const RunTimeInfo &runTimeInfo)
    {
        int32_t mProcess = runTimeInfo.s1Len;
        int32_t nProcess = runTimeInfo.s2Len;
        int32_t mProcessAlign = runTimeInfo.s1LenAlign;
        int32_t nProcessAlign = runTimeInfo.s2LenAlign;
        LocalTensor<INPUT_TYPE> l0_a_tensor = event_ping_pong_flag ? l0_a_tensor_ping_ : l0_a_tensor_pong_;
        LocalTensor<INPUT_TYPE> l0_b_tensor = event_ping_pong_flag ? l0_b_tensor_ping_ : l0_b_tensor_pong_;
        LocalTensor<float> l0_c_tensor = event_ping_pong_flag ? l0_c_tensor_ping_ : l0_c_tensor_pong_;
        TEventID evnet_id = event_ping_pong_flag ? event_ping_ : event_pong_;

        WAIT_FLAG(M, MTE1, evnet_id);
        load_data_l1_2_l0_nz(l0_a_tensor, l1_a_tensor, mProcessAlign, nProcessAlign);
        load_data_l1_2_l0_zn(l0_b_tensor, l1_b_tensor, nProcessAlign, head_dim_align_);
        SET_FLAG(MTE1, M, EVENT_ID0);
        WAIT_FLAG(MTE1, M, EVENT_ID0);

        WAIT_FLAG(FIX, M, evnet_id);
        MmadParams madParams;
        madParams.m = mProcess == 1 ? 2 : mProcess;
        madParams.n = head_dim_;
        madParams.k = nProcess;
        madParams.cmatrixInitVal = true;
        madParams.unitFlag = 3;
        AscendC::Mmad(l0_c_tensor, l0_a_tensor, l0_b_tensor, madParams);
        AscendC::PipeBarrier<PIPE_M>();
        SET_FLAG(M, MTE1, evnet_id);

        AscendC::FixpipeParamsV220 fixpipeParamsV220;
        fixpipeParamsV220.mSize = mProcess;
        fixpipeParamsV220.nSize = head_dim_;
        fixpipeParamsV220.srcStride = mProcessAlign;
        fixpipeParamsV220.dstStride = q_stride_;
        fixpipeParamsV220.unitFlag = 3;
        MM345CopyOut<true>(outGm, l0_c_tensor, fixpipeParamsV220);
        SET_FLAG(FIX, M, evnet_id);
    }

    __aicore__ inline void load_data_gm_2_l1(const LocalTensor<INPUT_TYPE> &dstL1Tensor,
                                             const GlobalTensor<INPUT_TYPE> &srcGmTensor, const int32_t mSize,
                                             const int32_t kSize, const int32_t mSizeAlign, const int32_t kSizeAlign,
                                             const int32_t srcStride)
    {
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = mSize;
        nd2nzPara.dValue = kSize;
        nd2nzPara.srcDValue = srcStride;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzC0Stride = mSizeAlign;
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.dstNzMatrixStride = 0;
        AscendC::DataCopy(dstL1Tensor, srcGmTensor, nd2nzPara);
    }

    __aicore__ inline void load_data_l1_2_l0_nz(const LocalTensor<INPUT_TYPE> &dstL0Tensor,
                                                const LocalTensor<INPUT_TYPE> &srcL1Tensor, const int32_t mSizeAlign,
                                                const int32_t kSizeAlign)
    {
        /*
         * Load 左矩阵非转置 or 右矩阵转置
         */
        AscendC::LoadData2DParamsV2 loadData2dParams;
        loadData2dParams.mStartPosition = 0;
        loadData2dParams.kStartPosition = 0;
        loadData2dParams.mStep = mSizeAlign / 16;
        loadData2dParams.kStep = kSizeAlign / 16;
        loadData2dParams.srcStride = mSizeAlign / 16;
        loadData2dParams.dstStride = mSizeAlign / 16;
        loadData2dParams.ifTranspose = false;

        AscendC::LoadData(dstL0Tensor, srcL1Tensor, loadData2dParams);
    }

    __aicore__ inline void load_data_l1_2_l0_zn(const LocalTensor<INPUT_TYPE> &dstL0Tensor,
                                                const LocalTensor<INPUT_TYPE> &srcL1Tensor, const int32_t mSizeAlign,
                                                const int32_t kSizeAlign)
    {
        /*
         * Load 左矩阵转置 or 右矩阵非转置
         */
        AscendC::LoadData2DParamsV2 loadData2dParams;
        loadData2dParams.kStartPosition = 0;
        loadData2dParams.mStep = 1;
        loadData2dParams.kStep = kSizeAlign / 16;
        loadData2dParams.srcStride = mSizeAlign / 16;
        loadData2dParams.dstStride = 1;
        loadData2dParams.ifTranspose = true;

        for (int32_t i = 0; i < mSizeAlign / 16; i++) {
            loadData2dParams.mStartPosition = i;
            AscendC::LoadData(dstL0Tensor[i * kSizeAlign * 16], srcL1Tensor, loadData2dParams);
        }
    }

    template <bool A_MATRIX>
    __aicore__ inline void
    load_data_gm_2_l0(const LocalTensor<INPUT_TYPE> &dstL0Tensor, const LocalTensor<INPUT_TYPE> &dstL1Tensor,
                      const GlobalTensor<INPUT_TYPE> &srcGmTensor, const int32_t mSize, const int32_t kSize,
                      const int32_t mSizeAlign, const int32_t kSizeAlign, const int32_t srcKStride)
    {
        load_data_gm_2_l1(dstL1Tensor, srcGmTensor, mSize, kSize, mSizeAlign, kSizeAlign, srcKStride);
        SET_FLAG(MTE2, MTE1, EVENT_ID0);
        WAIT_FLAG(MTE2, MTE1, EVENT_ID0);

        if constexpr (A_MATRIX) {
            load_data_l1_2_l0_nz(dstL0Tensor, dstL1Tensor, mSizeAlign, kSizeAlign);
        } else {
            load_data_l1_2_l0_zn(dstL0Tensor, dstL1Tensor, mSizeAlign, kSizeAlign);
        }
    }

    template <bool A_MATRIX>
    __aicore__ inline void
    load_data_gm_2_l0_trans(const LocalTensor<INPUT_TYPE> &dstL0Tensor, const LocalTensor<INPUT_TYPE> &dstL1Tensor,
                            const GlobalTensor<INPUT_TYPE> &srcGmTensor, const int32_t mSize, const int32_t kSize,
                            const int32_t mSizeAlign, const int32_t kSizeAlign, const int32_t srcKStride)
    {
        load_data_gm_2_l1(dstL1Tensor, srcGmTensor, mSize, kSize, mSizeAlign, kSizeAlign, srcKStride);
        SET_FLAG(MTE2, MTE1, EVENT_ID0);
        WAIT_FLAG(MTE2, MTE1, EVENT_ID0);

        if constexpr (A_MATRIX) {
            load_data_l1_2_l0_zn(dstL0Tensor, dstL1Tensor, mSizeAlign, kSizeAlign);
        } else {
            load_data_l1_2_l0_nz(dstL0Tensor, dstL1Tensor, mSizeAlign, kSizeAlign);
        }
    }

    template <bool ATMOIC_ADD>
    __aicore__ inline void MM345CopyOut(const GlobalTensor<float> &dstTensor, const LocalTensor<float> &srcTensor,
                                        const AscendC::FixpipeParamsV220 &fixpipeParamsV220)
    {
        if constexpr (ATMOIC_ADD) {
            AscendC::SetAtomicType<float>();
        }

        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(dstTensor, srcTensor, fixpipeParamsV220);

        if constexpr (ATMOIC_ADD) {
            AscendC::SetAtomicNone();
        }
    }
};

} // namespace BSA_ARC35
