/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file CopyInL1.h
 * \brief
 */
#ifndef COPYINL1_H
#define COPYINL1_H

enum class KVLAYOUT {
    BNBD, // [blockNums, headNum, blockSize, headDim]
    BBH,  // [blockNums, blockSize, headNum * headDim]
    NZ    // [blockNums, headNum, d1, blockSize, d0], d1 = headDim / d0, d0 = 32 (block byte) / sizeof(KV_T)
};

struct CopyParam {
    uint32_t width;
    uint32_t height;
    uint32_t orgWidth;
};

struct PAShape {
    uint32_t blockNum;
    uint32_t blockSize;
    uint32_t headNum;             // 一般为kv的head num
    uint32_t headDim;             // mla下rope为64， 非rope为512
    uint32_t maxblockNumPerBatch; // block table 每一行的最大个数
    uint32_t actHeadDim;          // 实际拷贝col大小，考虑到N切块 s*d， 对应d
    uint32_t copyRowNum;
    uint32_t copyRowNumAlign;
    uint32_t pageStride;
    uint32_t rowStride;
};

struct Position {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint64_t s2Offset;
    uint32_t dIdx; // N轴被切，对应D轴被切
};

// 场景：key、value GM to L1
// GM按ND格式存储
// L1按NZ格式存储
// GM 的行、列、列 stride 由输入布局决定，可为 D 或 N*D
template <typename L1Type>
__aicore__ inline void DataCopyGmNDToL1(LocalTensor<L1Type> &l1Tensor, GlobalTensor<L1Type> &gmTensor, uint32_t rowAct,
                                        uint32_t rowAlign,
                                        uint32_t col,       // D
                                        uint32_t colStride) // D or N*D
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = rowAct; // 行数

    nd2nzPara.dValue = col;
    nd2nzPara.srcDValue = colStride;
    nd2nzPara.dstNzC0Stride = rowAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}

template <typename L1Type>
__aicore__ inline void DataCopyGmScaleNDToL1(LocalTensor<L1Type> &l1Tensor, GlobalTensor<L1Type> &gmTensor,
                                             uint32_t rowAct, uint32_t rowAlign,
                                             uint32_t col,       // D
                                             uint32_t colStride) // D or N*D
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = rowAct;

    nd2nzPara.dValue = col;
    nd2nzPara.srcDValue = colStride;
    nd2nzPara.dstNzC0Stride = rowAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = nd2nzPara.nValue;

    LocalTensor<bfloat16_t> l1TensorCast = l1Tensor.template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gmTensorCast;
    gmTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t *)(gmTensor.GetPhyAddr())));
    DataCopy(l1TensorCast, gmTensorCast, nd2nzPara);
}

template <typename L1Type>
__aicore__ inline void DataCopyGmScaleDNToL1(LocalTensor<L1Type> &l1Tensor, GlobalTensor<L1Type> &gmTensor,
                                             uint32_t rowAct, uint32_t rowAlign, uint32_t col, uint32_t colStride)
{
    Dn2NzParams dn2nzPara;
    dn2nzPara.dnNum = 1;
    dn2nzPara.nValue = col >> 1;
    dn2nzPara.dValue = rowAct;
    dn2nzPara.srcDValue = colStride >> 1;
    dn2nzPara.dstNzC0Stride = dn2nzPara.nValue;
    dn2nzPara.dstNzNStride = 1;
    dn2nzPara.srcDnMatrixStride = 0;
    dn2nzPara.dstNzMatrixStride = dn2nzPara.nValue;

    LocalTensor<bfloat16_t> l1TensorCast = l1Tensor.template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gmTensorCast;
    gmTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t *)(gmTensor.GetPhyAddr())));
    DataCopy(l1TensorCast, gmTensorCast, dn2nzPara);
}

template <typename L1Type>
__aicore__ inline void DataCopyGmNZToL1(LocalTensor<L1Type> &l1Tensor, GlobalTensor<L1Type> &gmTensor, uint32_t rowAct,
                                        uint32_t dstRowStride, uint32_t srcRowStride, uint32_t col)
{
    uint32_t blockElementCnt = 32U / sizeof(L1Type);
    if constexpr (IsSameType<L1Type, int4b_t>::value) {
        blockElementCnt = 64U;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = col / blockElementCnt;
    intriParams.blockLen = rowAct;
    intriParams.dstStride = dstRowStride;
    intriParams.srcStride = srcRowStride;
    DataCopy(l1Tensor, gmTensor, intriParams);
}

template <typename L1Type>
__aicore__ inline void GmCopyInToL1PA(LocalTensor<L1Type> &l1Tensor, GlobalTensor<L1Type> &gmTensor,
                                      GlobalTensor<int32_t> &blockTableGm, KVLAYOUT kvLayout, const PAShape &shape,
                                      const Position &startPos)
{
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = static_cast<uint64_t>(startPos.bIdx) * shape.maxblockNumPerBatch; // 块表的基偏移量
    uint64_t curS2Idx = startPos.s2Offset;
    uint32_t blockElementCnt = 32U / sizeof(L1Type); // 每个块的元素数量
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize; // 获取block table上的索引
        uint64_t remainRowCnt = curS2Idx % shape.blockSize;  // 获取在单个块上超出的行数
        uint64_t idInBlockTable =
            blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上获取的编号
        // 计算可以拷贝行数
        uint32_t copyRowCnt = shape.blockSize - remainRowCnt; // 一次只能处理一个Block
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt; // 一个block未拷满
        }
        uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim; // PA的偏移
        if (kvLayout == KVLAYOUT::NZ) {
            offset += static_cast<uint64_t>(startPos.n2Idx) * shape.blockSize * shape.headDim +
                      remainRowCnt * blockElementCnt + static_cast<uint64_t>(startPos.dIdx) * shape.blockSize;

            LocalTensor<L1Type> tmpNopeDstTensor = l1Tensor[copyFinishRowCnt * blockElementCnt];
            GlobalTensor<L1Type> tmpNopeSrcTensor = gmTensor[offset];
            DataCopyGmNZToL1(tmpNopeDstTensor, tmpNopeSrcTensor, copyRowCnt, (shape.copyRowNumAlign - copyRowCnt),
                             (shape.blockSize - copyRowCnt), shape.actHeadDim);
        } else {
            uint64_t dStride = shape.headDim;
            if (kvLayout == KVLAYOUT::BBH) {
                offset += static_cast<uint64_t>(startPos.n2Idx) * shape.headDim +
                          remainRowCnt * shape.headDim * shape.headNum + startPos.dIdx;
                dStride = shape.headDim * shape.headNum;
            } else {
                offset = idInBlockTable * shape.pageStride +
                         static_cast<uint64_t>(startPos.n2Idx) * shape.blockSize * shape.rowStride +
                         remainRowCnt * shape.rowStride + startPos.dIdx;
                dStride = shape.rowStride;
            }

            uint32_t dValue = shape.actHeadDim;
            uint32_t srcDValue = dStride;
            LocalTensor<L1Type> tmpNopeDstTensor = l1Tensor[copyFinishRowCnt * blockElementCnt];
            GlobalTensor<L1Type> tmpNopeSrcTensor = gmTensor[offset];
            DataCopyGmNDToL1(tmpNopeDstTensor, tmpNopeSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);
        }
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

template <typename INPUT_T>
__aicore__ inline void CopyToL1Nd2Nz(const LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmTensor,
                                     uint32_t nValue, uint32_t dValue, uint32_t srcDValue)
{
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.ndNum = 1;             // ND矩阵的个数
    gm2L1Nd2NzParams.nValue = nValue;       // 单个ND矩阵的实际行数，单位为元素个数
    gm2L1Nd2NzParams.dValue = dValue;       // 单个ND矩阵的实际列数，单位为元素个数
    gm2L1Nd2NzParams.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
    gm2L1Nd2NzParams.srcDValue = srcDValue; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102)
    if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                  IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value) {
        gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 31) >> 5 << 5;
    } else {
        gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 15) >> 4 << 4;
    }
#else
    gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 15) >> 4 << 4; // NZ矩阵相邻Block起始地址之间的偏移， 单位为Block个数
#endif
    gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
    gm2L1Nd2NzParams.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}

template <typename INPUT_T>
__aicore__ inline void MxCopyScaleToL1Nd2Nz(const LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmTensor,
                                            uint32_t nValue, uint32_t dValue, uint32_t srcDValue,
                                            uint32_t dstNzC0Stride)
{
    // MXFP8 scale 为 fp8_e8m0，但 scale 搬运路径按 bf16 粒度执行。
    // value scale 的 GM shape 是 [physicalBlock, N, blockSize/64, DV, 2]，最后一维两个 e8m0
    // reinterpret 成一个 bf16 lane，因此压缩的是 D 方向，block 内 token group 行数不能除以 2。
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.ndNum = 1;
    gm2L1Nd2NzParams.nValue = nValue;
    gm2L1Nd2NzParams.dValue = dValue >> 1;
    gm2L1Nd2NzParams.srcNdMatrixStride = 0;
    gm2L1Nd2NzParams.srcDValue = srcDValue >> 1;
    gm2L1Nd2NzParams.dstNzC0Stride = dstNzC0Stride;
    gm2L1Nd2NzParams.dstNzNStride = 1;
    gm2L1Nd2NzParams.dstNzMatrixStride = gm2L1Nd2NzParams.nValue;

    LocalTensor<bfloat16_t> l1TensorCast = l1Tensor.template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gmTensorCast;
    gmTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t *)(gmTensor.GetPhyAddr())));
    DataCopy(l1TensorCast, gmTensorCast, gm2L1Nd2NzParams);
}

template <typename INPUT_T>
__aicore__ inline void MxCopyScaleToL1Dn2Nz(const LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmTensor,
                                            uint32_t nValue, uint32_t dValue, uint32_t srcDValue)
{
    // DN2NZ 版本用于跟随 BMM1 转置操作数的 scale。
    // QScale GM shape 为 [T,N,D/64,2]；KScale GM shape 为 [blockNum,N,blockSize,D/64,2]。
    // reinterpret 为 bf16 后，
    // 最后一维两个 e8m0 元素变成一个拷贝元素。这样可以保证 Q/K 按转置语义
    // 被 MX LoadData 消费时，scale layout 与数据 layout 对齐。
    Dn2NzParams gm2L1Dn2NzParams;
    gm2L1Dn2NzParams.dnNum = 1;
    gm2L1Dn2NzParams.nValue = nValue >> 1;
    gm2L1Dn2NzParams.dValue = dValue;
    gm2L1Dn2NzParams.srcDnMatrixStride = 0;
    gm2L1Dn2NzParams.srcDValue = srcDValue >> 1;
    gm2L1Dn2NzParams.dstNzC0Stride = nValue >> 1;
    gm2L1Dn2NzParams.dstNzNStride = 1;
    gm2L1Dn2NzParams.dstNzMatrixStride = gm2L1Dn2NzParams.nValue;

    LocalTensor<bfloat16_t> l1TensorCast = l1Tensor.template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gmTensorCast;
    gmTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t *)(gmTensor.GetPhyAddr())));
    DataCopy(l1TensorCast, gmTensorCast, gm2L1Dn2NzParams);
}

template <typename L1Type>
__aicore__ inline void MxGmFlatScaleCopyInToL1PAForDN(LocalTensor<L1Type> &l1Tensor, GlobalTensor<L1Type> &gmTensor,
                                                      GlobalTensor<int32_t> &blockTableGm, const PAShape &shape,
                                                      const Position &startPos)
{
    // PA scale helper，用于 K per-token-group scale：
    //   GM 逻辑 shape：[physicalBlock, N, blockSize, D/64, 2]
    //   PA 查表流程：逻辑 token 偏移 -> block_table -> 物理 block id + block 内 token offset
    //   L1 拷贝方式：DN2NZ，因为对应的数据操作数按转置方式加载。
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = static_cast<uint64_t>(startPos.bIdx) * shape.maxblockNumPerBatch;
    uint64_t curS2Idx = startPos.s2Offset;
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize;
        uint32_t remainRowCnt = static_cast<uint32_t>(curS2Idx % shape.blockSize);
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyRowCnt = shape.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt;
        }

        uint64_t offset = idInBlockTable * shape.pageStride +
                          static_cast<uint64_t>(startPos.n2Idx) * shape.blockSize * shape.rowStride +
                          static_cast<uint64_t>(remainRowCnt) * shape.rowStride + startPos.dIdx;
        MxCopyScaleToL1Dn2Nz(l1Tensor[copyFinishRowCnt * shape.actHeadDim], gmTensor[offset], shape.actHeadDim,
                             copyRowCnt, shape.rowStride);
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

template <typename L1Type>
__aicore__ inline void MxGmFlatScaleCopyInToL1PAForND(LocalTensor<L1Type> &l1Tensor, GlobalTensor<L1Type> &gmTensor,
                                                      GlobalTensor<int32_t> &blockTableGm, const PAShape &shape,
                                                      const Position &startPos)
{
    // PA scale helper，用于 value per-channel-group scale：
    // GM 逻辑 shape：[physicalBlock, N, blockSize/64, DV, 2]
    // 每个拷贝行覆盖 block 内 64 个 value token。
    // 这里使用 ND2NZ，因为 V 在 BMM2 中作为非转置右操作数被消费。
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = static_cast<uint64_t>(startPos.bIdx) * shape.maxblockNumPerBatch;
    uint64_t curS2Idx = startPos.s2Offset;
    constexpr uint32_t blockElementCnt = 32U / sizeof(L1Type);
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize;
        uint32_t remainRowCnt = static_cast<uint32_t>(curS2Idx % shape.blockSize);
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyRowCnt = shape.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt;
        }

        uint64_t offset = idInBlockTable * shape.pageStride +
                          static_cast<uint64_t>(startPos.n2Idx) * shape.blockSize * shape.rowStride +
                          static_cast<uint64_t>(remainRowCnt) * shape.rowStride + startPos.dIdx;
        MxCopyScaleToL1Nd2Nz(l1Tensor[copyFinishRowCnt * blockElementCnt], gmTensor[offset], copyRowCnt,
                             shape.actHeadDim, shape.rowStride, shape.copyRowNumAlign);
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

template <typename INPUT_T>
__aicore__ inline void CopyToL1Nd2NzGS1Merge(const LocalTensor<INPUT_T> &l1Tensor,
                                             const GlobalTensor<INPUT_T> &gmTensor, uint32_t ndNum, uint32_t nValue,
                                             uint32_t dValue, uint32_t srcNdMatrixStride, uint32_t srcDValue,
                                             uint32_t dstNzC0Stride) // BSNGD 合轴拷贝
{
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.ndNum = ndNum;                         // ND矩阵的个数
    gm2L1Nd2NzParams.nValue = nValue;                       // 单个ND矩阵的实际行数，单位为元素个数
    gm2L1Nd2NzParams.dValue = dValue;                       // 单个ND矩阵的实际列数，单位为元素个数
    gm2L1Nd2NzParams.srcNdMatrixStride = srcNdMatrixStride; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
    gm2L1Nd2NzParams.srcDValue = srcDValue; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102)
    if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                  IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value) {
        gm2L1Nd2NzParams.dstNzC0Stride =
            (dstNzC0Stride + 31) >> 5 << 5; // NZ矩阵相邻Block起始地址之间的偏移，单位为Block个数，32对齐
    } else {
        gm2L1Nd2NzParams.dstNzC0Stride =
            (dstNzC0Stride + 15) >> 4 << 4; // NZ矩阵相邻Block起始地址之间的偏移，单位为Block个数，16对齐
    }
#else
    gm2L1Nd2NzParams.dstNzC0Stride =
        (dstNzC0Stride + 15) >> 4 << 4; // NZ矩阵相邻Block起始地址之间的偏移，单位为Block个数，16对齐
#endif
    gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
    gm2L1Nd2NzParams.dstNzMatrixStride =
        nValue * 32 / sizeof(INPUT_T); // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}
#endif
