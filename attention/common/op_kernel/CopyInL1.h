/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file CopyInL1.h
 * \brief
 */
#ifndef COPYINL1_H
#define COPYINL1_H

enum class KVLAYOUT
{
    BNBD, // [blockNums, headNum, blockSize, headDim]
    BBH // [blockNums, blockSize, headNum * headDim]
};

struct CopyParam{
    uint32_t width;
    uint32_t height;
    uint32_t orgWidth;
};

struct PAShape{
    uint32_t blockNum;
    uint32_t blockSize;
    uint32_t headNum; // 一般为kv的head num
    uint32_t headDim; // mla下rope为64， 非rope为512
    uint32_t maxblockNumPerBatch; // block table 每一行的最大个数
    uint32_t actHeadDim; // 实际拷贝col大小，考虑到N切块 s*d， 对应d
    uint32_t copyRowNum;
    uint32_t copyRowNumAlign;
};

struct Position{
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Offset;
    uint32_t dIdx; // N轴被切，对应D轴被切
};

template<typename L1Type>
__aicore__ inline void GmCopyInToL1(LocalTensor<L1Type>& L1Tensor, GlobalTensor<L1Type>& GmTensor, const CopyParam& mmCopyParam)
{
    Nd2NzParams Gm2L1Nd2NzParams;
    Gm2L1Nd2NzParams.ndNum = 1; // ND矩阵的个数
    Gm2L1Nd2NzParams.nValue = mmCopyParam.height; // 单个ND矩阵的实际行数，单位为元素个数
    Gm2L1Nd2NzParams.dValue = mmCopyParam.width; // 单个ND矩阵的实际列数(vD)，单位为元素个数
    Gm2L1Nd2NzParams.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
    Gm2L1Nd2NzParams.srcDValue = mmCopyParam.orgWidth; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
    Gm2L1Nd2NzParams.dstNzC0Stride = (Gm2L1Nd2NzParams.nValue + 15) >> 4 << 4; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移， 单位为Block个数
    Gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
    Gm2L1Nd2NzParams.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
    DataCopy(L1Tensor, GmTensor, Gm2L1Nd2NzParams);
}

// 场景：key、value GM to L1
// GM按ND格式存储
// L1按NZ格式存储
// GM的行、列、列的stride（D or ND）BNSD 和 BSH的区别
template<typename L1Type>
__aicore__ inline void DataCopyGmNDToL1(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                        uint32_t rowAct, 
                                        uint32_t rowAlign,
                                        uint32_t col, // D
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

template<typename L1Type>
__aicore__ inline void GmCopyInToL1HasRopePA(LocalTensor<L1Type>& nopeTensor, LocalTensor<L1Type>& ropeTensor,
                                GlobalTensor<L1Type>& nopeGmTensor, GlobalTensor<L1Type>& ropeGmTensor,
                                GlobalTensor<int32_t>& blockTableGm, KVLAYOUT KvLayout, 
                                const PAShape &shape,
                                const PAShape &ropeShape,
                                const Position &startPos)
{
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch; // 块表的基偏移量
    uint32_t curS2Idx = startPos.s2Offset;
    uint32_t blockElementCnt = 32 / sizeof(L1Type); // 每个块的元素数量
    // ropeshape的M方向与nopeshape保持一样， 此处只判断nopeshape的
    while(copyFinishRowCnt < shape.copyRowNum){
        uint64_t blockIdOffset = curS2Idx / shape.blockSize; // 获取block table上的索引
        uint64_t remainRowCnt = curS2Idx % shape.blockSize; // 获取在单个块上超出的行数
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上获取的编号
        //计算可以拷贝行数
        uint32_t copyRowCnt = shape.blockSize - remainRowCnt; // 一次只能处理一个Block
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum){
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt; // 一个block未拷满
        }
        uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim; // PA的偏移
        uint64_t keyRopeOffset = idInBlockTable * ropeShape.blockSize * ropeShape.headNum * ropeShape.headDim;

        uint64_t dStride = shape.headDim;
        uint64_t dRopeStride = ropeShape.headDim;
        if (KvLayout == KVLAYOUT::BBH){
            offset += (uint64_t)(startPos.n2Idx * shape.headDim) + remainRowCnt * shape.headDim * shape.headNum + startPos.dIdx;
            keyRopeOffset += (uint64_t)(startPos.n2Idx * ropeShape.headDim) + remainRowCnt * ropeShape.headDim * ropeShape.headNum;
            dStride = shape.headDim * shape.headNum;
            dRopeStride = ropeShape.headDim * ropeShape.headNum;
        }
        else{
            offset += (uint64_t)(startPos.n2Idx * shape.headDim * shape.blockSize) + remainRowCnt * shape.headDim + startPos.dIdx;
            keyRopeOffset += (uint64_t)(startPos.n2Idx * ropeShape.headDim * ropeShape.blockSize) + remainRowCnt * ropeShape.headDim;
        }

        uint32_t dValue = shape.actHeadDim;
        uint32_t srcDValue = dStride;
        uint32_t dRopeValue = ropeShape.actHeadDim;
        uint32_t srcRopeDValue = dRopeStride;
        LocalTensor<L1Type> tmpNopeDstTensor = nopeTensor[copyFinishRowCnt * blockElementCnt];
        GlobalTensor<L1Type> tmpNopeSrcTensor = nopeGmTensor[offset];
        DataCopyGmNDToL1(tmpNopeDstTensor, tmpNopeSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);

        LocalTensor<L1Type> tmpRopeDstTensor = ropeTensor[copyFinishRowCnt * blockElementCnt];
        GlobalTensor<L1Type> tmpRopeSrcTensor = ropeGmTensor[keyRopeOffset];
        DataCopyGmNDToL1(tmpRopeDstTensor, tmpRopeSrcTensor, copyRowCnt, shape.copyRowNumAlign, dRopeValue, srcRopeDValue);

        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

template<typename L1Type>
__aicore__ inline void GmCopyInToL1PA(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                GlobalTensor<int32_t>& blockTableGm, KVLAYOUT KvLayout,
                                const PAShape &shape, const Position &startPos)
{
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch; // 块表的基偏移量
    uint32_t curS2Idx = startPos.s2Offset;
    uint32_t blockElementCnt = 32 / sizeof(L1Type); // 每个块的元素数量
    while(copyFinishRowCnt < shape.copyRowNum){
        uint64_t blockIdOffset = curS2Idx / shape.blockSize; // 获取block table上的索引
        uint64_t remainRowCnt = curS2Idx % shape.blockSize; // 获取在单个块上超出的行数
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上获取的编号
        //计算可以拷贝行数
        uint32_t copyRowCnt = shape.blockSize - remainRowCnt; // 一次只能处理一个Block
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum){
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt; // 一个block未拷满
        }
        uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim; // PA的偏移

        uint64_t dStride = shape.headDim;
        if (KvLayout == KVLAYOUT::BBH) {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim) + remainRowCnt * shape.headDim * shape.headNum + startPos.dIdx;
            dStride = shape.headDim * shape.headNum;
        } else {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim * shape.blockSize) + remainRowCnt * shape.headDim + startPos.dIdx;
        }

        uint32_t dValue = shape.actHeadDim;
        uint32_t srcDValue = dStride;
        LocalTensor<L1Type> tmpNopeDstTensor = l1Tensor[copyFinishRowCnt * blockElementCnt];
        GlobalTensor<L1Type> tmpNopeSrcTensor = gmTensor[offset];
        DataCopyGmNDToL1(tmpNopeDstTensor, tmpNopeSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);

        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}
#endif