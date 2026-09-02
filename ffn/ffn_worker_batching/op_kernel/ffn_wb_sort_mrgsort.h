/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ffn_wb_sort_mrgsort.h
 * \brief
 */
#ifndef OP_KERNEL_FFN_WB_SORT_MRGSORT_H
#define OP_KERNEL_FFN_WB_SORT_MRGSORT_H

#include "ffn_wb_common.h"
#include "kernel_operator.h"

namespace FfnWbBatching {
using namespace AscendC;

struct SortCustomMrgsortParam {
    int64_t perListElements = 0;
    int64_t sortNumStride = 0;
    int64_t oneLoopMaxElements = 0;
};

class SortCustomMrgsort {
public:
    __aicore__ inline SortCustomMrgsort(){};
    __aicore__ inline void Init(SortCustomMrgsortParam *param);
    __aicore__ inline void Process();
    __aicore__ inline void SetInput(GlobalTensor<float> &gmInput, GlobalTensor<int32_t> &gmSortNum,
                                    LocalTensor<float> &ubInput);
    __aicore__ inline void SetOutput(GlobalTensor<float> &gmOutput, LocalTensor<float> &ubOutput,
                                     LocalTensor<int32_t> &outSortNumLocal);

private:
    __aicore__ inline void CopyOutSortNum();
    __aicore__ inline void CopyIn();
    __aicore__ inline void UpdateMrgParam();
    __aicore__ inline void MrgsortCompute();
    __aicore__ inline void UpdateSortInfo();
    __aicore__ inline void CopyOut();
    __aicore__ inline void ClearCache();

private:
    SortCustomMrgsortParam *param = nullptr;

    GlobalTensor<float> gmInputs[4];
    GlobalTensor<float> gmOutput;
    GlobalTensor<int32_t> gmActualSortNum_;

    LocalTensor<float> ubInputs[4];
    LocalTensor<float> ubOutput;
    LocalTensor<int32_t> ubOutSortNum_;

    int64_t listNum{0};
    int64_t remainListNum{0};
    int64_t outOffset{0};
    int64_t offsets[4] = {0};
    int64_t listRemainElements[4] = {0};
    int64_t lengths[4] = {0};
    int64_t allRemainElements{0};
    int64_t curLoopSortedNum{0};

    uint16_t validBitTail{0};
    uint16_t elementCountListTail[4] = {0};
    uint32_t listSortedNums[4] = {0};
    LocalTensor<float> tmpUbInputs[4];
};

__aicore__ inline void SortCustomMrgsort::ClearCache()
{
    this->listNum = 0;
    this->allRemainElements = 0;
    this->outOffset = 0;
}

__aicore__ inline void SortCustomMrgsort::SetInput(GlobalTensor<float> &gmInput, GlobalTensor<int32_t> &gmSortNum,
                                                   LocalTensor<float> &ubInput)
{
    this->gmInputs[listNum] = gmInput;
    this->ubInputs[listNum] = ubInput;
    gmActualSortNum_ = gmSortNum;
    this->listNum += 1;
}

__aicore__ inline void SortCustomMrgsort::SetOutput(GlobalTensor<float> &gmOutput, LocalTensor<float> &ubOutput,
                                                    LocalTensor<int32_t> &outSortNumLocal)
{
    this->gmOutput = gmOutput;
    this->ubOutput = ubOutput;
    ubOutSortNum_ = outSortNumLocal;
}

__aicore__ inline void SortCustomMrgsort::UpdateMrgParam()
{
    if (this->remainListNum == MERGE_LIST_TWO) {
        elementCountListTail[MERGE_LIST_IDX_TWO] = 0;
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0011;
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0111;
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        validBitTail = 0b1111;
    } else {
        validBitTail = 0b0001;
    }
}

__aicore__ inline void SortCustomMrgsort::CopyIn()
{
    this->remainListNum = 0;
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        lengths[i] = Min(param->oneLoopMaxElements, listRemainElements[i]);
        if (lengths[i] > 0) {
            DataCopy(this->ubInputs[i], this->gmInputs[i][offsets[i]],
                     Align(GetSortLen<float>(lengths[i]), sizeof(float)));
            tmpUbInputs[j] = this->ubInputs[i];
            elementCountListTail[j] = lengths[i];
            this->remainListNum += 1;
            j++;
        }
    }
}

__aicore__ inline void SortCustomMrgsort::MrgsortCompute()
{
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    if (this->remainListNum == MERGE_LIST_TWO) {
        MrgSortSrcList sortListTail = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[0], tmpUbInputs[0]);
        MrgSort<float, true>(this->ubOutput, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        MrgSortSrcList sortListTail =
            MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[MERGE_LIST_IDX_TWO], tmpUbInputs[0]);
        MrgSort<float, true>(this->ubOutput, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        MrgSortSrcList sortListTail = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[MERGE_LIST_IDX_TWO],
                                                     tmpUbInputs[MERGE_LIST_IDX_THREE]);
        MrgSort<float, true>(this->ubOutput, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else {
        if (elementCountListTail[0] > 0) {
            DataCopy(this->ubOutput, this->tmpUbInputs[0],
                     Align(GetSortLen<float>(elementCountListTail[0]), sizeof(float)));
        }
        listSortedNums[0] = elementCountListTail[0];
    }
}

__aicore__ inline void SortCustomMrgsort::UpdateSortInfo()
{
    curLoopSortedNum = 0;
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        if (lengths[i] > 0) {
            // update remain size
            listRemainElements[i] -= listSortedNums[j];
            allRemainElements -= listSortedNums[j];
            // update offset
            offsets[i] += GetSortOffset<float>(listSortedNums[j]);
            // update current loop sorted nums
            curLoopSortedNum += listSortedNums[j];
            j += 1;
        }
    }
}

__aicore__ inline void SortCustomMrgsort::CopyOut()
{
    if (curLoopSortedNum > 0) {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = GetSortLen<float>(curLoopSortedNum) * sizeof(float);
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(this->gmOutput[outOffset], this->ubOutput, intriParams);
        outOffset += GetSortLen<float>(curLoopSortedNum);
    }
}

__aicore__ inline void SortCustomMrgsort::CopyOutSortNum()
{
    ubOutSortNum_.SetValue(0, allRemainElements);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(gmActualSortNum_, ubOutSortNum_, copyParams);
}

__aicore__ inline void SortCustomMrgsort::Init(SortCustomMrgsortParam *param)
{
    this->param = param;
    this->remainListNum = listNum;
    for (int64_t i = 0; i < listNum; i++) {
        offsets[i] = GetSortOffset<float>(param->perListElements * i);
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_ALL>(
            gmActualSortNum_[param->sortNumStride * i]);
        listRemainElements[i] = gmActualSortNum_.GetValue(param->sortNumStride * i);
        allRemainElements += listRemainElements[i];
    }
}

__aicore__ inline void SortCustomMrgsort::Process()
{
    CopyOutSortNum();
    for (; allRemainElements > 0;) {
        CopyIn();
        UpdateMrgParam();
        MrgsortCompute();
        UpdateSortInfo();
        CopyOut();
    }
    ClearCache();
}
} // namespace FfnWbBatching
#endif // OP_KERNEL_FFN_WB_SORT_MRGSORT_H
