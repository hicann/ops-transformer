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
 * \file compressor_grad_cast_scatter.h
 * \brief CompressorGrad arch22 AIV kernels:
 *        Phase0 CastKernel: x/wkv/wgate -> fp16 blocked-NZ workspace；dn_kv/dn_sc/d_ape 与
 *          d_ape partial 槽清零
 *        Phase1 ScatterKernel: 逐块 softmax 反向 + 梯度路由 + d_ape 本核 partial 槽写入
 *        Phase1.5 ApeReduceKernel: 每核定长区间按固定核序 Kahan 归并 d_ape
 *        BSH: nblk=S/cr；TH: 逐 batch 块数由 cu_seqlens/seqused/start_pos GM 值确定
 *
 * dn workspace（blocked-NZ）：tile (t/128, c/128) 位于 C-strip 主序
 * cTile*(T_pad/128)+tTile；tile 内 group g=cLocal/16 于 g*2048，行内 r*16+cLocal%16。
 * fp16: [hi|lo] 交错，hi=fp16(dn)、lo=fp16(dn-fp32(hi))；bf16: 仅 hi 平面。
 */

#ifndef COMPRESSOR_GRAD_CAST_SCATTER_H
#define COMPRESSOR_GRAD_CAST_SCATTER_H

#include "kernel_operator.h"
#include "compressor_grad_common_arch22.h"
#include "compressor_grad_tiling_data.h"

using namespace AscendC;

// =========================================================================
// Phase 0 (AIV): cast + zero fill
//   x(T,H) -> f16 blocked-NZ：tile (kb,nb) 于 GM 索引 nb*tTiles+kb，
//     128*128 元素连续，tile 内 (c/16)*2048 + r*16 + c%16（Nd2Nz L1 镜像）
//   wkv/wgate(C,H) -> f16 blocked-NZ：tile (kb,nb) 于 nb*(C/128)+kb
//   dn_kv/dn_sc(T_pad,C) f16 清零；d_ape(cr,C) f32 清零；d_ape partial f32 线性清零
// ========================================================================
constexpr uint32_t CG_CAST_VEC = 2048; // elements per vector chunk
constexpr uint32_t CG_X_TILE = 128;    // blocked-NZ tile edge (rows and cols)

template <typename ioT>
class CompressorGradCastKernel {
public:
    __aicore__ inline CompressorGradCastKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR wkv, GM_ADDR wgate, GM_ADDR dApe, GM_ADDR userWs,
                                const optiling::CompressorGradTilingDataA3* tilingData, TPipe* pipe)
    {
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ ioT*>(x));
        wkvGM_.SetGlobalBuffer(reinterpret_cast<__gm__ ioT*>(wkv));
        wgateGM_.SetGlobalBuffer(reinterpret_cast<__gm__ ioT*>(wgate));
        xF16GM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->xF16WsOff));
        wkvF16GM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->wkvF16WsOff));
        wgateF16GM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->wgateF16WsOff));
        dnKvGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->dnKvWsOff));
        dnScGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->dnScWsOff));
        dApeGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dApe));
        tLen_ = static_cast<uint32_t>(tilingData->tLen);
        hSize_ = static_cast<uint32_t>(tilingData->hiddenSize);
        cSize_ = static_cast<uint32_t>(tilingData->totalHeadDim);
        crSize_ = static_cast<uint32_t>(tilingData->cmpRatio);
        tPad_ = static_cast<uint32_t>(tilingData->tPad);
        coreNum_ = static_cast<uint32_t>(tilingData->castCoreNum);
        // dnPlanes 是 host 侧记录的 dn 平面真值；必须与 dtype 模板推导的平面数
        // （fp16 HiLo = 2，bf16 = 1）一致，否则 ZeroDn 的平面寻址会错写 GM。
        assert(tilingData->dnPlanes == (std::is_same_v<ioT, half> ? 2 : 1), "dnPlanes != dtype-derived plane count");
        pipe->InitBuffer(inBuf_, CG_X_TILE * CG_X_TILE * sizeof(ioT));
        pipe->InitBuffer(f32Buf_, CG_X_TILE * CG_X_TILE * sizeof(float));
        pipe->InitBuffer(f16Buf_, CG_X_TILE * CG_X_TILE * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        uint32_t cid = GetBlockIdx();
        if (cid >= coreNum_) {
            return;
        }
        CastXNz(cid);
        CastWeightNz(cid);
        ZeroDn(cid);
        if (cid == 0) {
            ZeroApe();
        }
    }

private:
    // Stage one 128x128 ioT tile (strided 256B rows from the natural x),
    // cast ioT -> f32 (full-tile contiguous) -> f16 with per-16-column-group
    // strided repeats that place each group at its NZ slot, then flush the
    // tile as one 32KB contiguous write.
    __aicore__ inline void CastXNz(uint32_t cid)
    {
        uint32_t tTiles = tPad_ / CG_X_TILE; // tPad % 128 == 0 (host pads)
        uint32_t nHm = hSize_ / CG_X_TILE;   // H % 128 == 0 (host checks)
        uint32_t totalTiles = tTiles * nHm;
        uint32_t tilesPerCore = CeilDivU32(totalTiles, coreNum_);
        uint32_t t0 = cid * tilesPerCore;
        uint32_t t1 = MinU32(t0 + tilesPerCore, totalTiles);
        LocalTensor<ioT> inLocal = inBuf_.Get<ioT>();
        LocalTensor<float> f32Local = f32Buf_.Get<float>();
        LocalTensor<half> nzLocal = f16Buf_.Get<half>();
        for (uint32_t t = t0; t < t1; t++) {
            uint32_t nb = t / tTiles;
            uint32_t kb = t - nb * tTiles;
            uint32_t rowBase = kb * CG_X_TILE;
            uint32_t validRows = (rowBase < tLen_) ? MinU32(CG_X_TILE, tLen_ - rowBase) : 0;
            SetWaitFlag<HardEvent::V_MTE2>(); // prior tile's casts done reading inLocal
            if (validRows < CG_X_TILE) {
                Duplicate(inLocal, static_cast<ioT>(0), static_cast<int32_t>(CG_X_TILE * CG_X_TILE));
                SetWaitFlag<HardEvent::V_MTE2>();
            }
            if (validRows > 0) {
                DataCopyExtParams cp{static_cast<uint16_t>(validRows), static_cast<uint32_t>(CG_X_TILE * sizeof(ioT)),
                                     static_cast<uint32_t>((hSize_ - CG_X_TILE) * sizeof(ioT)), 0, 0};
                DataCopyPadExtParams<ioT> pp{false, 0, 0, static_cast<ioT>(0)};
                DataCopyPad(inLocal, xGM_[static_cast<uint64_t>(rowBase) * hSize_ + nb * CG_X_TILE], cp, pp);
            }
            SetWaitFlag<HardEvent::MTE2_V>();
            SetWaitFlag<HardEvent::MTE3_V>(); // prior tile's GM write of nzLocal drained
            Cast(f32Local, inLocal, RoundMode::CAST_NONE, static_cast<int32_t>(CG_X_TILE * CG_X_TILE));
            // f32 tile (row-major 128x128) -> NZ slots: group g (cols 16g..)
            // lands at g*2048 with its 128 rows packed contiguously.
            UnaryRepeatParams gather{1, 1, 1, static_cast<uint8_t>(CG_X_TILE / 8)};
            for (uint32_t g = 0; g < CG_X_TILE / 16; g++) {
                Cast(nzLocal[g * CG_X_TILE * 16], f32Local[g * 16], RoundMode::CAST_NONE, 16, CG_X_TILE, gather);
            }
            SetWaitFlag<HardEvent::V_MTE3>();
            DataCopy(xF16GM_[static_cast<uint64_t>(t) * CG_X_TILE * CG_X_TILE], nzLocal,
                     static_cast<int32_t>(CG_X_TILE * CG_X_TILE));
        }
        SetWaitFlag<HardEvent::MTE3_V>(); // drain last tile write: CastWeightNz/ZeroDn reuse f16Buf_
        SetWaitFlag<HardEvent::V_MTE2>(); // drain last tile casts: CastWeightNz reloads inBuf_
    }

    // wkv/wgate (C,H) ioT -> f16 blocked-NZ tiles. Same writer pipeline
    // as CastXNz; C % 128 == 0 so tiles are always full (no pad path). One
    // tile index space covers both tensors: t < cTiles*nHm -> wkv, else
    // wgate; tile (kb, nb) lands at in-tensor index tIn = nb*cTiles + kb.
    __aicore__ inline void CastWeightNz(uint32_t cid)
    {
        uint32_t cTiles = cSize_ / CG_X_TILE; // C % 128 == 0 (host checks)
        uint32_t nHm = hSize_ / CG_X_TILE;    // H % 128 == 0
        uint32_t perTensor = cTiles * nHm;
        uint32_t totalTiles = 2U * perTensor;
        uint32_t tilesPerCore = CeilDivU32(totalTiles, coreNum_);
        uint32_t t0 = cid * tilesPerCore;
        uint32_t t1 = MinU32(t0 + tilesPerCore, totalTiles);
        LocalTensor<ioT> inLocal = inBuf_.Get<ioT>();
        LocalTensor<float> f32Local = f32Buf_.Get<float>();
        LocalTensor<half> nzLocal = f16Buf_.Get<half>();
        for (uint32_t t = t0; t < t1; t++) {
            bool gate = (t >= perTensor);
            uint32_t tIn = gate ? (t - perTensor) : t;
            uint32_t nb = tIn / cTiles;
            uint32_t kb = tIn - nb * cTiles;
            const GlobalTensor<ioT>& src = gate ? wgateGM_ : wkvGM_;
            const GlobalTensor<half>& dst = gate ? wgateF16GM_ : wkvF16GM_;
            SetWaitFlag<HardEvent::V_MTE2>(); // prior tile's casts done reading inLocal
            DataCopyExtParams cp{static_cast<uint16_t>(CG_X_TILE), static_cast<uint32_t>(CG_X_TILE * sizeof(ioT)),
                                 static_cast<uint32_t>((hSize_ - CG_X_TILE) * sizeof(ioT)), 0, 0};
            DataCopyPadExtParams<ioT> pp{false, 0, 0, static_cast<ioT>(0)};
            DataCopyPad(inLocal, src[static_cast<uint64_t>(kb) * CG_X_TILE * hSize_ + nb * CG_X_TILE], cp, pp);
            SetWaitFlag<HardEvent::MTE2_V>();
            SetWaitFlag<HardEvent::MTE3_V>(); // prior tile's GM write of nzLocal drained
            Cast(f32Local, inLocal, RoundMode::CAST_NONE, static_cast<int32_t>(CG_X_TILE * CG_X_TILE));
            UnaryRepeatParams gather{1, 1, 1, static_cast<uint8_t>(CG_X_TILE / 8)};
            for (uint32_t g = 0; g < CG_X_TILE / 16; g++) {
                Cast(nzLocal[g * CG_X_TILE * 16], f32Local[g * 16], RoundMode::CAST_NONE, 16, CG_X_TILE, gather);
            }
            SetWaitFlag<HardEvent::V_MTE3>();
            DataCopy(dst[static_cast<uint64_t>(tIn) * CG_X_TILE * CG_X_TILE], nzLocal,
                     static_cast<int32_t>(CG_X_TILE * CG_X_TILE));
        }
        SetWaitFlag<HardEvent::MTE3_V>(); // drain last tile write: ZeroDn reuses f16Buf_
        SetWaitFlag<HardEvent::V_MTE2>(); // drain last tile casts
    }

    __aicore__ inline void ZeroDn(uint32_t cid)
    {
        // dn workspaces are HiLo dual-plane for fp16 inputs (dn = hi + lo);
        // bf16 keeps the single-plane layout. All planes must start zeroed.
        constexpr uint32_t planes = std::is_same_v<ioT, half> ? 2U : 1U;
        // plane element count (tPad * cSize) can exceed 2^32 on extreme
        // shapes: all tile arithmetic stays in 64-bit
        const uint64_t planeElems = static_cast<uint64_t>(tPad_) * cSize_;
        const uint64_t hTiles = (planeElems + CG_CAST_VEC - 1) / CG_CAST_VEC;
        uint64_t totalTiles = 2ULL * planes * hTiles;
        uint32_t perCore = CeilDivU32(static_cast<uint32_t>(totalTiles), coreNum_);
        LocalTensor<half> f16Local = f16Buf_.Get<half>();
        for (uint32_t v = 0; v < perCore; v++) {
            uint64_t tile = static_cast<uint64_t>(cid) * perCore + v;
            if (tile >= totalTiles) {
                break;
            }
            uint32_t idx = static_cast<uint32_t>(tile / hTiles);
            uint64_t elem = (tile % hTiles) * CG_CAST_VEC;
            uint32_t cnt = static_cast<uint32_t>(MinU64(CG_CAST_VEC, planeElems - elem));
            Duplicate(f16Local, static_cast<half>(0), cnt);
            SetWaitFlag<HardEvent::V_MTE3>();
            DataCopyExtParams outCp{1, static_cast<uint32_t>(cnt * sizeof(half)), 0, 0, 0};
            uint64_t base = static_cast<uint64_t>(idx % planes) * planeElems + elem;
            if (idx < planes) {
                DataCopyPad(dnKvGM_[base], f16Local, outCp);
            } else {
                DataCopyPad(dnScGM_[base], f16Local, outCp);
            }
            SetWaitFlag<HardEvent::MTE3_V>();
        }
    }

    __aicore__ inline void ZeroApe()
    {
        LocalTensor<float> f32Local = f32Buf_.Get<float>();
        for (uint32_t off = 0; off < crSize_ * cSize_; off += CG_CAST_VEC) {
            uint32_t cnt = MinU32(CG_CAST_VEC, crSize_ * cSize_ - off);
            Duplicate(f32Local, 0.0f, cnt);
            SetWaitFlag<HardEvent::V_MTE3>();
            DataCopyExtParams outCp{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
            DataCopyPad(dApeGM_[off], f32Local, outCp);
            SetWaitFlag<HardEvent::MTE3_V>();
        }
    }

    GlobalTensor<ioT> xGM_, wkvGM_, wgateGM_;
    GlobalTensor<half> xF16GM_, wkvF16GM_, wgateF16GM_, dnKvGM_, dnScGM_;
    GlobalTensor<float> dApeGM_;
    uint32_t tLen_, hSize_, cSize_, crSize_, tPad_, coreNum_;
    TBuf<TPosition::VECCALC> inBuf_;
    TBuf<TPosition::VECCALC> f32Buf_;
    TBuf<TPosition::VECCALC> f16Buf_;
};

// =========================================================================
// Phase 1 (AIV): scatter
//   每 (块, D 段)：dK=dC*sm；dSp=dC*kv；col=sum sm*dSp；dZ=sm*(dSp-col)；
//   路由 cur/prev 槽到 dn_kv/dn_sc；dZ 按行累加 per-core d_ape，段末写本核 partial 槽。
//   sm/kv 每段整块装入；colsum 用 4 路 MulAddDst；行 staging 到 (cr,seg)
//   fp16 slot tile，每 tile 一次 2D DataCopyPad 写出。
// ========================================================================
template <typename ioT>
class CompressorGradScatterKernel {
public:
    __aicore__ inline CompressorGradScatterKernel() {}

    __aicore__ inline void Init(GM_ADDR dCmpKv, GM_ADDR smScore, GM_ADDR kvIn, GM_ADDR cuSeqlens, GM_ADDR seqUsed,
                                GM_ADDR startPos, GM_ADDR userWs,
                                const optiling::CompressorGradTilingDataA3* tilingData, TPipe* pipe)
    {
        dCkGM_.SetGlobalBuffer(reinterpret_cast<__gm__ ioT*>(dCmpKv));
        smGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(smScore));
        kvGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(kvIn));
        dnKvGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->dnKvWsOff));
        dnScGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->dnScWsOff));
        apePartGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(userWs + tilingData->apePartWsOff));
        bSize_ = static_cast<uint32_t>(tilingData->batchSize);
        sSize_ = static_cast<uint32_t>(tilingData->seqSize);
        tLen_ = static_cast<uint32_t>(tilingData->tLen);
        dSize_ = static_cast<uint32_t>(tilingData->headDim);
        crSize_ = static_cast<uint32_t>(tilingData->cmpRatio);
        coff_ = static_cast<uint32_t>(tilingData->coff);
        nblk_ = static_cast<uint32_t>(tilingData->nblk);
        nblkIn_ = static_cast<uint32_t>(tilingData->nblkIn);
        nRows_ = static_cast<uint32_t>(tilingData->cmpRowCnt);
        cSize_ = static_cast<uint32_t>(tilingData->totalHeadDim);
        segLen_ = static_cast<uint32_t>(tilingData->segLen);
        blocksPerCore_ = static_cast<uint32_t>(tilingData->blocksPerCore);
        scatterCoreNum_ = static_cast<uint32_t>(tilingData->scatterCoreNum);
        isTh_ = (tilingData->layout == 1);
        // seqused/start_pos 的 GM 值决定每 batch 块数与路由（A5 语义）：
        //   TH: hasSq_/hasSp_ 由运行时指针判定（走查元数据 LoadBatchMeta 用）
        //   BSH: 由 tiling 的 seqUsedValid/startPosValid flag 判定（desc 实例化，
        //        值语义路径）；flag=0 时不绑定 GM 指针（两可选输入均不传的默认
        //        路径走 shape 推导）
        if (isTh_) {
            hasSq_ = (seqUsed != nullptr);
            hasSp_ = (startPos != nullptr);
            cuGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cuSeqlens));
        } else {
            hasSq_ = (tilingData->seqUsedValid != 0);
            hasSp_ = (tilingData->startPosValid != 0);
        }
        if (hasSq_) {
            sqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqUsed));
        }
        if (hasSp_) {
            spGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(startPos));
        }
        // blocked-NZ dn: T-tiles per C-strip (tPad is 128-aligned)
        dnTilesT_ = CeilDivU32(static_cast<uint32_t>(tilingData->tPad), 128U);
        uint32_t seg = segLen_;
        // fp16: HiLo dual-plane dn staging; bf16: hi plane only.
        constexpr uint32_t planes = std::is_same_v<ioT, half> ? 2U : 1U;
        // 与 cast kernel 同一契约：dnPlanes（host 真值）必须等于模板平面数，
        // 否则 staging slot / FlushSlotPlane 的 dn 平面寻址会错写 GM。
        assert(static_cast<uint32_t>(tilingData->dnPlanes) == planes, "dnPlanes != dtype-derived plane count");
        // prev slot tile only exists for coff==2.
        const uint32_t slots = (coff_ == 2) ? 2U : 1U;
        // dn rows are staged per (cur/prev) slot x plane as (cr, seg) fp16 tiles
        // and flushed with one 2D DataCopyPad per tile instead of per-row copies.
        // fp32 region: ape 2*cr + sm/kv/dkv 3*N + colAcc 4 + dC32 1
        // [+ hBTile cr, fp16 HiLo roundtrip only]
        uint32_t f32Elems = (2U * crSize_ + 3U * nRows_ + 5U) * seg + (planes - 1U) * crSize_ * seg;
        uint32_t bufBytes =
            f32Elems * sizeof(float) + seg * sizeof(ioT) + slots * planes * crSize_ * seg * sizeof(half);
        pipe->InitBuffer(buf_, bufBytes);
        apeCur_ = buf_.Get<float>();
        apePrev_ = buf_.Get<float>()[crSize_ * seg];
        uint32_t off = 2 * crSize_ * seg; // in float elements
        sm_ = buf_.Get<float>()[off];
        kvBlk_ = buf_.Get<float>()[off + nRows_ * seg];
        dkvBlk_ = buf_.Get<float>()[off + 2 * nRows_ * seg];
        off += 3 * nRows_ * seg;
        colAcc_ = buf_.Get<float>()[off]; // 4 x seg multi-accumulator
        dC32_ = buf_.Get<float>()[off + 4 * seg];
        off += 5 * seg;
        if constexpr (std::is_same_v<ioT, half>) {
            hBTile_ = buf_.Get<float>()[off];
            off += crSize_ * seg;
        }
        uint32_t byteOff = off * sizeof(float);
        dCIn_ = buf_.Get<uint8_t>().ReinterpretCast<ioT>()[byteOff / sizeof(ioT)];
        uint32_t halfOff = (byteOff + seg * sizeof(ioT)) / sizeof(half);
        // tiles laid out plane-major: [hi cur][hi prev?] then [lo cur][lo prev?]
        stageHiCur_ = buf_.Get<half>()[halfOff];
        stageHiPrev_ = buf_.Get<half>()[halfOff + crSize_ * seg];
        stageLoCur_ = buf_.Get<half>()[halfOff + slots * crSize_ * seg];
        stageLoPrev_ = buf_.Get<half>()[halfOff + (slots + 1) * crSize_ * seg];
        if (isTh_) {
            // TH 逐 batch 走查元数据：cu_seqlens(B+1) + seqused(B) + start_pos(B)
            // int32，一次 MTE2 搬入 UB（走查标量读 UB）；
            // 容量 (spOff + B) 个 int32，三段各自 32B 对齐（已计入 scatter UB 预算）
            // Sub-array offsets must be 32B-aligned (DataCopyPad UB constraint):
            // round each region start up to 8*int32 (32 bytes)
            const uint32_t cuInts = bSize_ + 1U;
            const uint32_t sqOff = (cuInts + 7U) & ~7U;           // align8
            const uint32_t spOff = sqOff + ((bSize_ + 7U) & ~7U); // align8
            pipe->InitBuffer(metaBuf_, (spOff + bSize_) * sizeof(int32_t));
            cuMeta_ = metaBuf_.Get<int32_t>();
            sqMeta_ = cuMeta_[sqOff];
            spMeta_ = cuMeta_[spOff];
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t cid = GetBlockIdx();
        if (cid >= scatterCoreNum_) {
            return;
        }
        if (isTh_) {
            LoadBatchMeta();
        }
        for (uint32_t d0 = 0; d0 < dSize_; d0 += segLen_) {
            Duplicate(apeCur_, 0.0f, crSize_ * segLen_);
            Duplicate(apePrev_, 0.0f, crSize_ * segLen_);

            if (isTh_) {
                ScatterRangeTh(cid, d0);
            } else {
                uint32_t nbTotal = bSize_ * nblk_;
                for (uint32_t t = 0; t < blocksPerCore_; t++) {
                    uint32_t blk = cid * blocksPerCore_ + t;
                    if (blk >= nbTotal) {
                        break;
                    }
                    uint32_t b = blk / nblk_;
                    uint32_t j = blk % nblk_;
                    // 值语义（BSH）：nblk_ 是全 batch 上界（tiling 按值算或
                    // 默认 S/cr），各 batch 实际完整块数 = (sp+sq)/cr；不参与的块直接
                    // 跳过（其 dCmpKv/kv/sm 行可能是前向未写入的 padding，读入会污染
                    // 输出）。跳过不触碰任何 staging/流水线状态，在途拷贝由后续块的
                    // V_MTE2/MTE3_V 等待点收敛。
                    const uint32_t sp = BlockAnchorOff(b);
                    if (j >= UsedBlocks(b, sp)) {
                        continue;
                    }
                    // sp≠0 时块锚点移到全局对齐块：cur 槽 token 行 [j*cr-sp, (j+1)*cr-sp)，
                    // 首块(j==0)/次块(j==1) 的前 sp 个 staged 行是 cache 历史行，
                    // 不落 dn、不累 d_ape（golden: cache 行不路由）
                    const uint32_t fc = (j == 0U) ? sp : 0U;
                    const uint32_t pfc = (j == 1U) ? sp : 0U;
                    const uint32_t tokOff = (j == 0U) ? 0U : j * crSize_ - sp;
                    ProcessBlockSegment(d0, static_cast<uint64_t>(b) * nblkIn_ + j, static_cast<uint64_t>(b) * sSize_,
                                        tokOff, fc, pfc, j >= 1U);
                }
            }
            // store this segment's d_ape partials into the core's private slot
            // (layout [cid][cr][C], no atomic: the cross-core reduce is fixed-order)
            uint32_t curCol = (coff_ - 1) * dSize_ + d0;
            const uint64_t slot = static_cast<uint64_t>(cid) * crSize_ * cSize_;
            SetWaitFlag<HardEvent::V_MTE3>();
            DataCopyExtParams cpApe{static_cast<uint16_t>(crSize_), static_cast<uint32_t>(segLen_ * sizeof(float)), 0,
                                    static_cast<uint32_t>((cSize_ - segLen_) * sizeof(float)), 0};
            DataCopyPad(apePartGM_[slot + curCol], apeCur_, cpApe);
            if (coff_ == 2) {
                DataCopyPad(apePartGM_[slot + d0], apePrev_, cpApe);
            }
            SetWaitFlag<HardEvent::MTE3_V>();
            SetWaitFlag<HardEvent::V_MTE2>();
        }
    }

private:
    // TH 元数据搬入：cu_seqlens/seqused/start_pos 一次 MTE2 拷入 UB，走查用
    // 标量读（seqused/start_pos 缺省时跳过，走查回退 cu 差分 / sp=0）。
    // 同步必须用 MTE2_S：GetValue 走 Scalar 管道，MTE2_V 只保证 V 管道可见。
    __aicore__ inline void LoadBatchMeta()
    {
        DataCopyPadExtParams<int32_t> ppI{false, 0, 0, 0};
        DataCopyExtParams cpCu{1, static_cast<uint32_t>((bSize_ + 1U) * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(cuMeta_, cuGm_, cpCu, ppI);
        if (hasSq_) {
            DataCopyExtParams cpSq{1, static_cast<uint32_t>(bSize_ * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(sqMeta_, sqGm_, cpSq, ppI);
        }
        if (hasSp_) {
            DataCopyExtParams cpSp{1, static_cast<uint32_t>(bSize_ * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(spMeta_, spGm_, cpSp, ppI);
        }
        SetWaitFlag<HardEvent::MTE2_S>();
    }

    // TH 块路由：本核负责全局块空间 [blkLo, blkHi)，总块数由走查收敛。
    //   used_b = seqused ? seqused[b] : cu[b+1]-cu[b]；sp_b = start_pos ? : 0
    //   nblk_b = (sp+used)/cr - sp/cr（首块可为部分块，尾部不足一块不计）
    //   块 (b,j)：dC/sm/kv 行 = rowBase + j；cur 槽 dn 行 = cu[b]+tokOff_j+(r-fc_j)
    //     fc_0 = sp%cr, tokOff_0 = 0；j>=1: fc=0, tokOff_j=(cr-sp%cr)+(j-1)*cr
    //     prev 槽 (coff=2, j>=1)：行 [pfc,cr) -> dn [cu[b]+tokOff_j-cr+pfc, tokOff_j)
    // 输入契约（见 README/aclnn）：cu_seqlens 首元素 0、单调不减、末元素 <= T；
    //   seqused ∈ [0, cu[b+1]-cu[b]]；start_pos >= 0（可大于本 batch 窗口长度）。
    // 值域违规（cu <0/>T/非单调、seqused 越界、sp 负值）不拒绝执行：
    //   违规 batch 记 0 块跳过（输出保持 Phase0 全零）；seqused clamp 到 [0,span]；
    //   sp 负值按 0；块序号截断到 nblkIn_。合法输入下行为不变。
    __aicore__ inline void ScatterRangeTh(uint32_t cid, uint32_t d0)
    {
        const uint32_t blkLo = cid * blocksPerCore_;
        const uint32_t blkHi = blkLo + blocksPerCore_;
        uint32_t rowBase = 0; // 有效块前缀（= 当前 batch 首块的 dC 行号）
        for (uint32_t b = 0; b < bSize_ && rowBase < blkHi; b++) {
            const int64_t cuBaseI = static_cast<int64_t>(cuMeta_.GetValue(b));
            const int64_t cuNextI = static_cast<int64_t>(cuMeta_.GetValue(b + 1));
            if (cuBaseI < 0 || cuBaseI > static_cast<int64_t>(tLen_) || cuNextI < cuBaseI ||
                cuNextI > static_cast<int64_t>(tLen_)) {
                continue; // 违规 batch：0 块（dn 保持 Phase0 全零），rowBase 不增
            }
            const uint32_t cuBase = static_cast<uint32_t>(cuBaseI);
            const uint32_t spanRaw = static_cast<uint32_t>(cuNextI - cuBaseI);
            int64_t usedI = hasSq_ ? static_cast<int64_t>(sqMeta_.GetValue(b)) : static_cast<int64_t>(spanRaw);
            usedI = (usedI < 0) ? 0 : ((usedI > static_cast<int64_t>(spanRaw)) ? static_cast<int64_t>(spanRaw) : usedI);
            int64_t spI = hasSp_ ? static_cast<int64_t>(spMeta_.GetValue(b)) : 0;
            spI = (spI < 0) ? 0 : spI; // 负值按 0；正值不 clamp（合法可远大于 span）
            const uint32_t used = static_cast<uint32_t>(usedI);
            const uint32_t sp = static_cast<uint32_t>(spI);
            const uint32_t cmpEnd = (sp + used) / crSize_;
            const uint32_t cmpStart = sp / crSize_;
            const uint32_t nblkB = (cmpEnd > cmpStart) ? (cmpEnd - cmpStart) : 0U;
            if (nblkB > 0U && rowBase + nblkB > blkLo) {
                const uint32_t j0 = (rowBase > blkLo ? rowBase : blkLo) - rowBase;
                const uint32_t jCap = (rowBase < nblkIn_) ? (nblkIn_ - rowBase) : 0U;
                const uint32_t j1 = MinU32(MinU32(blkHi - rowBase, nblkB), jCap);
                const uint32_t fc0 = sp - cmpStart * crSize_; // sp % cr：首块 cache 行数
                const uint32_t head = crSize_ - fc0;          // 首块实际路由 token 数
                for (uint32_t j = j0; j < j1; j++) {
                    const uint32_t fc = (j == 0U) ? fc0 : 0U;
                    const uint32_t tokOff = (j == 0U) ? 0U : head + (j - 1U) * crSize_;
                    const uint32_t pfc = (j == 1U) ? fc0 : 0U;
                    ProcessBlockSegment(d0, static_cast<uint64_t>(rowBase) + j, static_cast<uint64_t>(cuBase), tokOff,
                                        fc, pfc, j >= 1U);
                }
            }
            rowBase += nblkB;
        }
    }

    // sp = startPos[b] % cr —— 全局对齐块锚点相对本 chunk 起点的偏移（BSH 值语义）。
    // hasSp_=0（张量未提供）时恒 0；负值经 uint 回绕仍 < cr（垃圾容错，不越界）。
    __aicore__ inline uint32_t BlockAnchorOff(uint32_t b) const
    {
        if (!hasSp_) {
            return 0;
        }
        const int32_t spRaw = spGm_.GetValue(b);
        return static_cast<uint32_t>(spRaw) % crSize_;
    }

    // batch b 实际完整块数 = (sp + sq) / cr，sq 缺省取 sSize_（全序列）；
    // clamp [0, S] 兜底垃圾值。缺省必须取 sSize_ 而非 0：取 0 会使默认路径
    // （不传 seqused 的调用）UsedBlocks==0，全部块被跳过 → dn 全零 →
    // d_x/d_wkv/d_wgate 全零。
    __aicore__ inline uint32_t UsedBlocks(uint32_t b, uint32_t sp) const
    {
        const int64_t sqRaw = hasSq_ ? static_cast<int64_t>(sqGm_.GetValue(b)) : static_cast<int64_t>(sSize_);
        const int64_t sq =
            (sqRaw < 0) ? 0 : ((sqRaw > static_cast<int64_t>(sSize_)) ? static_cast<int64_t>(sSize_) : sqRaw);
        return static_cast<uint32_t>((static_cast<int64_t>(sp) + sq) / crSize_);
    }

    // 布局参数化的单块处理：blkBase = dC/sm/kv 行号（BSH: b*nblkIn+j；TH: 全局
    // 有效块序号）；dnBase = batch 首 token 行（BSH: b*S；TH: cu[b]）；tokOff/fc/
    // pfc/hasPrev 描述窗口路由（BSH 恒为 j*cr/0/0/j>=1）
    __aicore__ inline void ProcessBlockSegment(uint32_t d0, uint64_t blkBase, uint64_t dnBase, uint32_t tokOff,
                                               uint32_t fc, uint32_t pfc, bool hasPrev)
    {
        const uint32_t seg = segLen_;
        const uint32_t n = nRows_;
        // one fp32 row / one fp16 tile row in 32B blocks (seg in {16, 32, 64})
        const uint8_t rowBlk = static_cast<uint8_t>(seg / 8);
        // stage-in: sm + kv blocks (N, SEG) + dC row (SEG) in one MTE2 batch
        {
            DataCopyExtParams cpBlk{static_cast<uint16_t>(n), static_cast<uint32_t>(seg * sizeof(float)),
                                    static_cast<uint32_t>((dSize_ - seg) * sizeof(float)), 0, 0};
            DataCopyPadExtParams<float> ppF{true, 0, 0, 0.0f};
            DataCopyExtParams cpDc{1, static_cast<uint32_t>(seg * sizeof(ioT)), 0, 0, 0};
            DataCopyPadExtParams<ioT> ppDc{true, 0, 0, static_cast<ioT>(0)};
            SetWaitFlag<HardEvent::V_MTE2>();
            DataCopyPad(sm_, smGM_[blkBase * n * dSize_ + d0], cpBlk, ppF);
            DataCopyPad(kvBlk_, kvGM_[blkBase * n * dSize_ + d0], cpBlk, ppF);
            DataCopyPad(dCIn_, dCkGM_[blkBase * dSize_ + d0], cpDc, ppDc);
            Duplicate(colAcc_, 0.0f, 4 * seg); // overlaps the in-flight loads
            SetWaitFlag<HardEvent::MTE2_V>();
            Cast(dC32_, dCIn_, RoundMode::CAST_NONE, seg);
        }
        // colraw lanes: colAcc_[k] = sum_{r = k mod 4} sm_r (*) kv_r (fp32 fma,
        // single-rounding; dC factored out, applied once by pass 2's Mul)
        SetWaitFlag<HardEvent::MTE3_V>(); // prev block's dn_sc flush drained (staging free)
        {
            BinaryRepeatParams lane{1, 1, 1, 0, static_cast<uint8_t>(4 * rowBlk), static_cast<uint8_t>(4 * rowBlk)};
            for (uint32_t k = 0; k < 4 && k < n; k++) {
                MulAddDst(colAcc_[k * seg], sm_[k * seg], kvBlk_[k * seg], seg, (n - k + 3) >> 2, lane);
            }
        }
        // pass 1: dkv = sm (*) dC (dC broadcast per row), staged into dn_kv tiles
        {
            BinaryRepeatParams bcast{1, 1, 1, rowBlk, rowBlk, 0};
            // repeatTime is uint8_t (<=255): slab N>255 (cr=128 x coff=2 -> N=256)
            // into <=255-repeat chunks; src1RepStride=0 keeps dC broadcast from
            // its base every chunk, so chunking is value-identical.
            for (uint32_t r0 = 0; r0 < n; r0 += 255) {
                const uint8_t reps = static_cast<uint8_t>(MinU32(n - r0, 255));
                Mul(dkvBlk_[r0 * seg], sm_[r0 * seg], dC32_, seg, reps, bcast);
            }
        }
        StageTiles(dkvBlk_);
        FlushDn(dnKvGM_, dnBase, tokOff, d0, fc, pfc, hasPrev);
        // colsum finalize + pass 2 block math overlap the dn_kv flush
        // (none of them touch the staging tiles)
        Add(colAcc_, colAcc_, colAcc_[seg], seg);
        Add(colAcc_[2 * seg], colAcc_[2 * seg], colAcc_[3 * seg], seg);
        Add(colAcc_, colAcc_, colAcc_[2 * seg], seg);
        // pass 2 (in-place on kvBlk_): dz = sm * dC * (kv - colRaw)
        {
            BinaryRepeatParams bcast{1, 1, 1, rowBlk, rowBlk, 0};
            // same uint8_t repeatTime guard as pass 1: <=255-repeat row slabs;
            // per-row independent ops, chunking keeps values bit-identical.
            for (uint32_t r0 = 0; r0 < n; r0 += 255) {
                const uint8_t reps = static_cast<uint8_t>(MinU32(n - r0, 255));
                Sub(kvBlk_[r0 * seg], kvBlk_[r0 * seg], colAcc_, seg, reps, bcast);
            }
            for (uint32_t r0 = 0; r0 < n; r0 += 255) {
                const uint8_t reps = static_cast<uint8_t>(MinU32(n - r0, 255));
                Mul(kvBlk_[r0 * seg], kvBlk_[r0 * seg], dC32_, seg, reps, bcast);
            }
            Mul(kvBlk_, sm_, kvBlk_, static_cast<int32_t>(n * seg));
        }
        // ape accumulation: cur rows [fc, cr) -> apeCur_, prev rows [pfc, cr) -> apePrev_
        // (ape 行号 = 槽内 token 行号 r（golden 恒等映射）；fc>0 的 cache 行不累加，
        //  累加器该行保持 0，段末整块 atomic 写出无害)
        if (coff_ == 1) {
            if (fc == 0U) {
                Add(apeCur_, apeCur_, kvBlk_, static_cast<int32_t>(n * seg));
            } else {
                // rows [fc, cr) are contiguous fp32: one flat add
                Add(apeCur_[fc * seg], apeCur_[fc * seg], kvBlk_[fc * seg], static_cast<int32_t>((crSize_ - fc) * seg));
            }
        } else {
            BinaryRepeatParams apePrm{1, 1, 1, rowBlk, rowBlk, static_cast<uint8_t>(2 * rowBlk)};
            Add(apeCur_[fc * seg], apeCur_[fc * seg], kvBlk_[(2 * fc + 1) * seg], seg, crSize_ - fc, apePrm);
            if (hasPrev) {
                Add(apePrev_[pfc * seg], apePrev_[pfc * seg], kvBlk_[2 * pfc * seg], seg, crSize_ - pfc, apePrm);
            }
        }
        SetWaitFlag<HardEvent::MTE3_V>(); // dn_kv flush drained (staging free for pass 2)
        StageTiles(kvBlk_);
        FlushDn(dnScGM_, dnBase, tokOff, d0, fc, pfc, hasPrev);
        // no trailing MTE3_V: the dn_sc flush drains under the next block's
        // loads; it is awaited at the top of the next ProcessBlockSegment (and
        // at segment end by the d_ape flush's MTE3_V).
    }

    // 将行主序 fp32 块 (N, seg) 的各行 staged 到 cur/prev slot tile（blocked-NZ
    // group 主序）：group g 于 stage + g*cr*16，行连续打包。
    // coff==1: 行 1:1 映射 cur；coff==2: 奇行 -> cur、偶行 -> prev。
    // fp16 另产出 lo tile：lo = fp16(row - fp32(hi))；bf16 仅 hi tile。
    __aicore__ inline void StageTiles(const LocalTensor<float>& blk)
    {
        const uint32_t seg = segLen_;
        const uint8_t rowBlk = static_cast<uint8_t>(seg / 8);     // fp32 row stride, 32B units
        const uint8_t row2Blk = static_cast<uint8_t>(2 * rowBlk); // fp32 two-row stride
        UnaryRepeatParams gather{1, 1, 1, rowBlk};                // f32 rows -> f16 NZ rows
        UnaryRepeatParams gatherOdd{1, 1, 1, row2Blk};            // f32 odd/even rows -> f16
        UnaryRepeatParams roundtrip{1, 1, rowBlk, 1};             // f16 NZ rows -> f32 rows
        if (coff_ == 1) {
            for (uint32_t g = 0; g < seg / 16; g++) {
                Cast(stageHiCur_[g * crSize_ * 16], blk[g * 16], RoundMode::CAST_NONE, 16, crSize_, gather);
            }
            if constexpr (std::is_same_v<ioT, half>) {
                for (uint32_t g = 0; g < seg / 16; g++) {
                    Cast(hBTile_[g * 16], stageHiCur_[g * crSize_ * 16], RoundMode::CAST_NONE, 16, crSize_, roundtrip);
                }
                Sub(hBTile_, blk, hBTile_, static_cast<int32_t>(crSize_ * seg));
                for (uint32_t g = 0; g < seg / 16; g++) {
                    Cast(stageLoCur_[g * crSize_ * 16], hBTile_[g * 16], RoundMode::CAST_NONE, 16, crSize_, gather);
                }
            }
        } else {
            for (uint32_t g = 0; g < seg / 16; g++) {
                Cast(stageHiCur_[g * crSize_ * 16], blk[seg + g * 16], RoundMode::CAST_NONE, 16, crSize_, gatherOdd);
                Cast(stageHiPrev_[g * crSize_ * 16], blk[g * 16], RoundMode::CAST_NONE, 16, crSize_, gatherOdd);
            }
            if constexpr (std::is_same_v<ioT, half>) {
                BinaryRepeatParams subPrm{1, 1, 1, rowBlk, row2Blk, rowBlk};
                for (uint32_t g = 0; g < seg / 16; g++) {
                    Cast(hBTile_[g * 16], stageHiCur_[g * crSize_ * 16], RoundMode::CAST_NONE, 16, crSize_, roundtrip);
                }
                Sub(hBTile_, blk[seg], hBTile_, seg, crSize_, subPrm);
                for (uint32_t g = 0; g < seg / 16; g++) {
                    Cast(stageLoCur_[g * crSize_ * 16], hBTile_[g * 16], RoundMode::CAST_NONE, 16, crSize_, gather);
                }
                for (uint32_t g = 0; g < seg / 16; g++) {
                    Cast(hBTile_[g * 16], stageHiPrev_[g * crSize_ * 16], RoundMode::CAST_NONE, 16, crSize_, roundtrip);
                }
                Sub(hBTile_, blk, hBTile_, seg, crSize_, subPrm);
                for (uint32_t g = 0; g < seg / 16; g++) {
                    Cast(stageLoPrev_[g * crSize_ * 16], hBTile_[g * 16], RoundMode::CAST_NONE, 16, crSize_, gather);
                }
            }
        }
    }

    // 将一个 staged slot tile（一个 HiLo 平面）写入 blocked-NZ dn workspace：
    // dn 行 [r0,r0+nRFlush) x 列 [c0,c0+seg) 取 staged 行 [rowOff,+nRFlush)，
    // 覆盖 tile (rt,ct) 的 NZ group g0..g0+seg/16-1。组内目的行连续（行距 16 元素），
    // 常规情形为一次 2D DataCopyPad（seg/16 个 burst，源连续，目的 4KB 行距）。
    // Tile 单元 [hi|lo]（fp16 64KB）或仅 hi（bf16 32KB），单元 stride 随平面数。
    __aicore__ inline void FlushSlotPlane(const GlobalTensor<half>& dn, const LocalTensor<half>& stage, uint32_t plane,
                                          uint64_t r0, uint64_t c0, uint32_t rowOff, uint32_t nRFlush)
    {
        const uint32_t seg = segLen_;
        const uint32_t ct = static_cast<uint32_t>(c0 / 128U);
        const uint32_t g0 = static_cast<uint32_t>((c0 % 128U) / 16U);
        const uint64_t r1 = r0 + nRFlush;
        constexpr uint64_t dnUnit = static_cast<uint64_t>(std::is_same_v<ioT, half> ? 2U : 1U) * 128U * 128U;
        for (uint64_t rt = r0 / 128U, rtEnd = (r1 + 127U) / 128U; rt < rtEnd; rt++) {
            uint64_t rFirst = (r0 > rt * 128U) ? r0 : rt * 128U;
            uint64_t rLast = (r1 < (rt + 1U) * 128U) ? r1 : (rt + 1U) * 128U;
            uint32_t nR = static_cast<uint32_t>(rLast - rFirst);
            // staged source rows are [rowOff + (rFirst - r0), +nR) of the (cr, seg)
            // tile (rowOff = fc/pfc partial-window cache rows)
            uint32_t sOff = rowOff + static_cast<uint32_t>(rFirst - r0);
            uint64_t unit = (static_cast<uint64_t>(ct) * dnTilesT_ + rt) * dnUnit;
            uint64_t dstBase = unit + static_cast<uint64_t>(plane) * 16384U + (rFirst - rt * 128U) * 16U;
            if (sOff == 0U && nR == crSize_) {
                // common case (cr divides 128, full window): one 2D copy, contiguous src
                DataCopyExtParams cp{static_cast<uint16_t>(seg / 16U), nR * 32U, 0, 128U * 32U - nR * 32U, 0};
                DataCopyPad(dn[dstBase + g0 * 2048U], stage, cp);
            } else {
                // partial window (TH first-block cache rows, sOff > 0) or
                // row-straddling tail (cr not a divisor of 128): the staged
                // groups are row-sliced, so walk them one contiguous burst
                // per group (nonzero srcStride on the sliced-2D pad-copy
                // path is avoided: it is timing-sensitive on 910_93)
                for (uint32_t g = 0; g < seg / 16U; g++) {
                    DataCopyExtParams cp1{1, nR * 32U, 0, 0, 0};
                    DataCopyPad(dn[dstBase + (g0 + g) * 2048U], stage[(g * crSize_ + sOff) * 16U], cp1);
                }
            }
        }
    }

    // 写出 staged slot tile：cur 行 [fc,cr) -> dn token [dnBase+tokOff, +cr-fc)
    // 列 [(coff-1)*D+d0, +seg)；prev (coff==2, hasPrev) 行 [pfc,cr) ->
    // dn token [dnBase+tokOff-cr+pfc, dnBase+tokOff) 列 [d0, +seg)。
    // 每 slot 每 HiLo 平面一次 FlushSlotPlane（bf16 仅 hi）。
    // BSH: dnBase=b*S, tokOff=j*cr, fc=pfc=0；TH: dnBase=cu[b], tokOff/fc/pfc 由 sp 决定。
    __aicore__ inline void FlushDn(const GlobalTensor<half>& dn, uint64_t dnBase, uint32_t tokOff, uint32_t d0,
                                   uint32_t fc, uint32_t pfc, bool hasPrev)
    {
        uint64_t curRow = dnBase + tokOff;
        uint64_t curCol = static_cast<uint64_t>(coff_ - 1U) * dSize_ + d0;
        SetWaitFlag<HardEvent::V_MTE3>();
        FlushSlotPlane(dn, stageHiCur_, 0U, curRow, curCol, fc, crSize_ - fc);
        if constexpr (std::is_same_v<ioT, half>) {
            FlushSlotPlane(dn, stageLoCur_, 1U, curRow, curCol, fc, crSize_ - fc);
        }
        if (coff_ == 2 && hasPrev) {
            uint64_t prevRow = dnBase + tokOff - crSize_ + pfc;
            FlushSlotPlane(dn, stageHiPrev_, 0U, prevRow, d0, pfc, crSize_ - pfc);
            if constexpr (std::is_same_v<ioT, half>) {
                FlushSlotPlane(dn, stageLoPrev_, 1U, prevRow, d0, pfc, crSize_ - pfc);
            }
        }
    }

    GlobalTensor<ioT> dCkGM_;
    GlobalTensor<float> smGM_, kvGM_, apePartGM_;
    GlobalTensor<int32_t> cuGm_, sqGm_, spGm_;
    GlobalTensor<half> dnKvGM_, dnScGM_;
    uint32_t bSize_, sSize_, tLen_, dSize_, crSize_, coff_, nblk_, nblkIn_, nRows_, cSize_, segLen_;
    uint32_t blocksPerCore_, scatterCoreNum_, dnTilesT_;
    bool isTh_ = false;
    bool hasSq_ = false;
    bool hasSp_ = false;
    TBuf<TPosition::VECCALC> buf_;
    TBuf<TPosition::VECCALC> metaBuf_;
    LocalTensor<float> apeCur_, apePrev_, sm_, kvBlk_, dkvBlk_, colAcc_, dC32_, hBTile_;
    LocalTensor<ioT> dCIn_;
    LocalTensor<half> stageHiCur_, stageHiPrev_, stageLoCur_, stageLoPrev_;
    LocalTensor<int32_t> cuMeta_, sqMeta_, spMeta_;
};

// =========================================================================
// Phase 1.5 (AIV): d_ape 跨核归约
//   partial 槽布局 [cid][cr][C]；每核处理 cr*C 的连续区间 [e0,e1)，按固定核序
//   k=0..N-1 做 Kahan 补偿累加，结果普通写覆盖 d_ape（Phase0 已清零）。
//   acc/comp/x 三个 fp32 缓冲合计 48KB：块长 L 按 UB 预算取
//   48KB/(4*(N+2)) 并向下取 8 的倍数，使全部 N 个槽一次 2D 装入。
// ========================================================================
constexpr uint32_t CG_APE_REDUCE_VEC = 4096;       // element cap per UB chunk
constexpr uint32_t CG_APE_REDUCE_UB = 48U * 1024U; // acc + comp + x buffers

class CompressorGradApeReduceKernel {
public:
    __aicore__ inline CompressorGradApeReduceKernel() {}

    __aicore__ inline void Init(GM_ADDR dApe, GM_ADDR userWs, const optiling::CompressorGradTilingDataA3* tilingData,
                                TPipe* pipe)
    {
        partGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(userWs + tilingData->apePartWsOff));
        dApeGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dApe));
        crSize_ = static_cast<uint32_t>(tilingData->cmpRatio);
        cSize_ = static_cast<uint32_t>(tilingData->totalHeadDim);
        coreNum_ = static_cast<uint32_t>(tilingData->scatterCoreNum);
        // 区间按 8 元素（32B）切分：cr*C 必须为 8 的倍数（headDim 为 128/512 恒成立，
        // 显式断言防参数域放开后尾部元素漏归约）
        assert((static_cast<uint64_t>(crSize_) * cSize_) % 8U == 0U, "d_ape total elements must be 8-aligned");
        // 48KB budget over acc/comp (L each) + x (N slots x L), 8-element rows
        chunkLen_ = static_cast<uint32_t>(CG_APE_REDUCE_UB / (sizeof(float) * (coreNum_ + 2U))) & ~7U;
        chunkLen_ = MinU32(chunkLen_, CG_APE_REDUCE_VEC);
        pipe->InitBuffer(buf_, static_cast<uint32_t>((2U * chunkLen_ + coreNum_ * chunkLen_) * sizeof(float)));
        acc_ = buf_.Get<float>();
        comp_ = buf_.Get<float>()[chunkLen_];
        xbuf_ = buf_.Get<float>()[2 * chunkLen_];
    }

    __aicore__ inline void Process()
    {
        const uint32_t cid = GetBlockIdx();
        if (cid >= coreNum_) {
            return;
        }
        const uint64_t total = static_cast<uint64_t>(crSize_) * cSize_;
        // contiguous [e0, e1) per core, remainder to the leading cores; the
        // split is done in 8-element units so every range stays 32B-aligned
        const uint64_t totalU = total / 8U;
        const uint64_t perU = totalU / coreNum_;
        const uint64_t remU = totalU % coreNum_;
        const uint64_t e0 = (static_cast<uint64_t>(cid) * perU + MinU64(cid, remU)) * 8U;
        const uint64_t e1 = e0 + (perU + (cid < remU ? 1U : 0U)) * 8U;
        for (uint64_t e = e0; e < e1; e += chunkLen_) {
            const uint32_t n = static_cast<uint32_t>(MinU64(chunkLen_, e1 - e));
            Duplicate(acc_, 0.0f, n);
            Duplicate(comp_, 0.0f, n);
            // one 2D load: all N slots x n elements, rows packed in UB
            SetWaitFlag<HardEvent::V_MTE2>();
            DataCopyExtParams cp{static_cast<uint16_t>(coreNum_), static_cast<uint32_t>(n * sizeof(float)),
                                 static_cast<uint32_t>((total - n) * sizeof(float)), 0, 0};
            DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
            DataCopyPad(xbuf_, partGM_[e], cp, pp);
            SetWaitFlag<HardEvent::MTE2_V>();
            // fixed-order Kahan: y = x - c; t = s + y; c = (t - s) - y; s = t
            for (uint32_t k = 0; k < coreNum_; k++) {
                LocalTensor<float> x = xbuf_[k * n];
                Sub(x, x, comp_, n);        // y = x - c
                Add(comp_, acc_, x, n);     // t = s + y
                Sub(comp_, comp_, acc_, n); // t - s
                Sub(comp_, comp_, x, n);    // c = (t - s) - y
                Add(acc_, acc_, x, n);      // s = t
            }
            SetWaitFlag<HardEvent::V_MTE3>();
            DataCopyExtParams outCp{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
            DataCopyPad(dApeGM_[e], acc_, outCp);
            SetWaitFlag<HardEvent::MTE3_V>();
        }
    }

private:
    GlobalTensor<float> partGM_, dApeGM_;
    uint32_t crSize_ = 0, cSize_ = 0, coreNum_ = 0, chunkLen_ = 0;
    TBuf<TPosition::VECCALC> buf_;
    LocalTensor<float> acc_, comp_, xbuf_;
};

#endif // COMPRESSOR_GRAD_CAST_SCATTER_H
