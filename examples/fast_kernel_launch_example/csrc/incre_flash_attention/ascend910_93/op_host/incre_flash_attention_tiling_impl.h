/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file incre_flash_attention_tiling_impl.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_IMPL_TEST_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_IMPL_TEST_H_

#include <queue>
#include <map>
#include <string>
#include <ATen/Operators.h>
#include "incre_flash_attention_tiling_base.h"
#include "incre_flash_attention_tiling_struct.h"
#include "incre_flash_attention_tiling_context.h"
#include "../op_kernel/incre_flash_attention_tilingdata.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#ifdef ASCENDC_OP_TEST
#define IFA_EXTERN_C extern "C"
#else
#define IFA_EXTERN_C
#endif
namespace custom {
enum class graphStatus {
    GRAPH_FAILED = 0,
    GRAPH_SUCCESS = 1
};
} // namespace custom

namespace optiling {

class IFATiling {
    struct ValidityConfigFD {
        std::vector<int32_t> validBatchSizes;
        std::vector<int32_t> validQSeqSizes;
        int32_t numHeads;
        int32_t numKvHeads;
        int32_t headDim;
        int32_t headDimV;
        int32_t sparseMode;
        int64_t expectedActualSeqLength; // -1 表示范围 [4096, 5120]
    };

public:
    IFATiling() = default;
    ~IFATiling() = default;
    // bool IsCapable() {return true;}
    custom::graphStatus DoSubOpTiling(IFAContext& ifaContext);
    custom::graphStatus RunBigKernelTiling(IFAContext& context, IncreFlashAttentionTilingDataV2* tilingData,
                                           bool isWorkspace = false);
    bool NeedRollBack() const
    {
        return passToOldTiling_;
    }

    bool TilingDataBack() const
    {
        return !atbRunFlag_;
    }
    uint32_t GetAntiquantSeqLength() const;
    bool CheckCommonConditions(const ValidityConfigFD& config) const;
    bool CheckBatchAndQSeqSize(const std::vector<int32_t>& validBatchSizes,
                               const std::vector<int32_t>& validQSeqSizes) const;
    bool CheckHeadDimensions(int32_t numHeads, int32_t numKvHeads, int32_t headDim, int32_t headDimV) const;
    bool CheckQuantizationFlags(int32_t sparseMode) const;
    bool CheckActualSeqLengths(int64_t expectedActualSeqLength) const;
    bool IsBalanceSplitCore();
    void IsFdBalanceCase();
    bool IsValidFlag3B();
    bool IsValidFlag560B();
    bool IsValidFlag();

    IncreFlashAttentionTilingDataV2* ifaTilingData;

private:
    custom::graphStatus GetNpuInfo();
    custom::graphStatus PreCheck();
    custom::graphStatus PreProcess();
    custom::graphStatus ProcessBaseInputs();
    custom::graphStatus ProcessOptionalTensors();
    custom::graphStatus ProcessPseShift();
    custom::graphStatus CheckTndMaskShapeWithSparseMode();
    custom::graphStatus CheckMaskShapeWithQSeq() const;
    custom::graphStatus CheckAttenMaskShape();
    custom::graphStatus ProcessAttenMask();
    // custom::graphStatus CheckMlaQueryRopeDesc() const;
    // custom::graphStatus CheckMlaQueryRopeBsndLayout(const gert::Shape &qRopeShape, const gert::Shape &qShape);
    // custom::graphStatus CheckMlaQueryRopeBnsdLayout(const gert::Shape &qRopeShape, const gert::Shape &qShape);
    // custom::graphStatus CheckMlaQueryRopeNzLayout(const gert::Shape &qRopeShape, const gert::Shape &qShape);
    custom::graphStatus ProcessActualSeqLen();
    custom::graphStatus CheckActualSeqLens();
    custom::graphStatus ParseActualSeqLens();
    custom::graphStatus ProcessQuant1() const;
    custom::graphStatus ProcessQuant2();
    custom::graphStatus ProcessDequant1() const;
    custom::graphStatus ProcessDequant2() const;
    custom::graphStatus ProcessAntiQuant();
    custom::graphStatus CheckKeyAndValueAntiquantScaleOffset();
    custom::graphStatus CheckKeyAndValueAntiquantOffset(const uint32_t keyAntiquantModeKvSep,
                                                        const uint32_t valueAntiquantModeKvSep);
    custom::graphStatus CheckKvAntiquant4SplitMode() const;
    custom::graphStatus ProcessAntiQuantMode();
    custom::graphStatus ProcessQuant();
    custom::graphStatus CheckQkvQuantParams4FullQuant() const;
    custom::graphStatus ProcessBlockTable();
    custom::graphStatus ProcessKVPaddingSize();
    custom::graphStatus ProcessSharedPrefix();
    custom::graphStatus ProcessSharedPrefixLen();
    custom::graphStatus ProcessMlaRope();
    custom::graphStatus ProcessCvRatio();
    void SetupPerfMode();
    bool EnableAllVec() const;
    bool EnableGQA310P();
    bool EnableCubeViewMM() const;
    bool EnableCubeViewMMDD() const;
    bool EnableCubeViewMMFullLoad();
    bool CheckDataCopyNd2Nz() const;
    bool CheckConstraints() const;
    bool EnableC1V1() const;
    custom::graphStatus CheckPseShiftDataType() const;
    void UpdatePerfMode();
    void UpdateL2CacheOffFlag();
    void SetCoreNum();
    void SetDealBN2Num();
    uint32_t GetTotalQBlockNum() const;
    uint32_t GetTotalWorkspaceSize() const;
    uint32_t GetHeadSize() const;
    uint32_t CalcSeqStepKv() const;
    uint32_t CalcSeqStepQ() const;

    custom::graphStatus InitInOutMode();
    // custom::graphStatus CheckBaseInputsNull() const;
    custom::graphStatus InputAttrsPreProcess();
    custom::graphStatus QKVPreProcess();
    custom::graphStatus GetInOutLayoutAndProcessQInfo(const std::string layout, uint32_t& sOfQuery,
                                                      uint32_t& sOfHeadnum, const uint32_t kDimNum);
    custom::graphStatus GetInOutLayout4BSH(const std::string layout, uint32_t& sOfQuery, uint32_t& sOfHeadnum,
                                           const uint32_t kDimNum, bool prefixFlag);
    custom::graphStatus GetRopeAndGqaFlag(const uint32_t sOfQuery, const uint32_t kDimNum, const uint32_t sOfHeadnum,
                                          const std::string layout);
    custom::graphStatus QKVPreProcess4TND(const std::string layout);
    custom::graphStatus ProcessPageAttentionFlag();
    custom::graphStatus KvShapePostProcess();
    // custom::graphStatus CheckKVHeadNum(const gert::StorageShape *inputShape) const;
    // custom::graphStatus CheckKVShape(const size_t &size, const gert::StorageShape *keyTensorInList, const
    // gert::StorageShape *valueTensorInList) const; custom::graphStatus CheckQKOutShapeBsh(size_t dimOfQ, size_t
    // dimOfK, size_t dimOfOut,
    //     const gert::StorageShape *queryShape, const gert::StorageShape *keyShape) const;
    // custom::graphStatus CheckQKOutShapeTnd(size_t dimOfQ, size_t dimOfK, size_t dimOfOut,
    //     const gert::StorageShape *queryShape, const gert::StorageShape *keyShape) const;
    // custom::graphStatus CheckQKOutShapeBnsdOrBsnd(size_t dimOfQ, size_t dimOfK, size_t dimOfOut,
    //     const gert::StorageShape *queryShape, const gert::StorageShape *keyShape) const;
    // custom::graphStatus CheckQKOutShape() const;
    // custom::graphStatus CheckKeyShapeTensor(const gert::Shape &aShape) const;
    custom::graphStatus ZeroTensorProcess();
    custom::graphStatus SharedPrefixTiling();
    custom::graphStatus SharedPrefixCheckBasic();
    custom::graphStatus SharedPrefixCheckShapes(const at::IntArrayRef& keyShape,
                                                const at::IntArrayRef& valueShape) const;

    bool CheckUbSpace();
    custom::graphStatus CheckPABlockSize() const;
    custom::graphStatus SetL2CacheFlag();
    custom::graphStatus SetQuantFlag();

    // custom::graphStatus CheckQuant2Shape(const at::IntArrayRef &inputParaShape) const;
    custom::graphStatus ProcessQuant2Dtype();
    custom::graphStatus CheckKVAntiQuantMode();
    // custom::graphStatus CheckKVAntiQuantPerToken(const gert::Shape &inputParaShape) const;
    // custom::graphStatus CheckKVAntiQuantPerHead(const gert::Shape &inputParaShape) const;
    custom::graphStatus CheckKVAntiQuantParamsInPagedAttention() const;
    custom::graphStatus CheckKVAntiQuantParamsShapeInPagedAttention(const at::IntArrayRef& inputParaShape) const;
    // custom::graphStatus CheckKVAntiQuantPerChannel(const gert::Shape &inputParaShape) const;
    // custom::graphStatus CheckGqaAntiQuantPerChannel(const gert::Shape &inputParaShape) const;
    // custom::graphStatus CheckKVAntiQuantParaShapeLegal(const gert::Shape &inputParaShape);
    // custom::graphStatus CheckAntiQuantParam(const gert::Tensor *antiquantScaleTensor,
    //     const gert::Tensor *antiquantOffsetTensor, const gert::CompileTimeTensorDesc *antiquantScaleDesc,
    //     const gert::CompileTimeTensorDesc *antiquantOffsetDesc);
    // custom::graphStatus CheckAntiQuantParamKeyType(const gert::Tensor *antiquantOffsetTensor,
    //     const gert::CompileTimeTensorDesc *antiquantScaleDesc, const gert::CompileTimeTensorDesc
    //     *antiquantOffsetDesc) const;
    // custom::graphStatus CheckAntiQuantParamValueType(const gert::Tensor *antiquantOffsetTensor,
    //     const gert::CompileTimeTensorDesc *antiquantScaleDesc, const gert::CompileTimeTensorDesc
    //     *antiquantOffsetDesc) const;
    // custom::graphStatus CheckAntiquantOffsetType(const gert::Tensor *antiquantOffsetTensor,
    //     const gert::CompileTimeTensorDesc *antiquantOffsetDesc, ge::DataType antiquantScaleType) const;
    custom::graphStatus CheckSupportKVLeftPadding();
    // custom::graphStatus CheckInputFormatAndLimits();
    // custom::graphStatus CheckInputParameterFormat() const;
    // custom::graphStatus CheckInputAntiquantFormat() const;
    // custom::graphStatus ProcessCheckATBInputWithPage() const;
    // custom::graphStatus CheckKeyShapeInput() const;
    // custom::graphStatus CheckValueShapeInput() const;
    // custom::graphStatus ProcessCheckATBInput();
    // custom::graphStatus CheckInputQKVTypeMatch() const;
    // bool IsSupportFormat(const ge::Format format) const;
    bool IsZeroDimTensor(const at::IntArrayRef& shape) const;
    std::string GetTensorDimString(const at::IntArrayRef& shape) const;
    uint32_t GetTypeSize(at::ScalarType dtype) const;

    // custom::graphStatus CheckMlaKeyRope() const;
    // custom::graphStatus CheckMlaQueryRope();
    // custom::graphStatus CheckMlaMisc() const;
    custom::graphStatus CheckDefaultMisc(std::string scene) const;
    // custom::graphStatus CheckMlaAttrs() const;
    // custom::graphStatus CheckMlaKeyRopeDesc() const;
    // custom::graphStatus CheckMlaKeyRopeShapeMatch(const gert::Shape keyRopeShape, const gert::Shape keyShape) const;
    // custom::graphStatus CheckMlaKeyRopeBsndLayout(const gert::Shape keyRopeShape, const gert::Shape keyShape) const;
    // custom::graphStatus CheckMlaKeyRopeBnsdLayout(const gert::Shape keyRopeShape, const gert::Shape keyShape) const;
    // custom::graphStatus CheckMlaKeyRopeNzLayout(const gert::Shape keyRopeShape, const gert::Shape keyShape) const;

    bool CalcUbBmm();
    bool CalcUbSoftMax();
    bool CalcUbAttenMask();
    bool CalcUbQuant();
    bool CalcUbDeQuant();
    bool CalcUbAntiQuant();
    bool CalcUbPageAttention();
    bool CalcUbKvSplit();

    bool CheckIfRollBack() const;
    bool CanChangeToNew() const;
    void AdjustPABmm1Tiling(uint32_t& bmm1BaseN) const;
    void AdjustPABmm2Tiling() const;
    bool ShapeEqual(const at::IntArrayRef& aShape, const at::IntArrayRef& bShape) const;

    custom::graphStatus Split();
    void GetSeqTilingInfo(const at::IntArrayRef& actualSeqKv, const ActualSeqInfo& actualSeqInfo,
                          SeqTilingInfo& seqTilingInfo) const;
    void FillBalancedSplitCoreInfo(const TilingIndexes& tilingIdx, BalancedSplitTilingInfo& tilingInfo);
    void EndSplitForCurrentCore(const TilingIndexes& tilingIdx, const SeqTilingInfo& seqTilingInfo,
                                uint32_t& currKvSplitPart, BalancedSplitTilingInfo& tilingInfo);
    void SplitBalancedForEachHeadFd(uint32_t bIdx, const SeqTilingInfo& seqTilingInfo,
                                    BalancedSplitTilingInfo& tilingInfo, std::vector<int64_t>& gS1SplitNumOfFdHead,
                                    uint32_t s1);
    void SplitFDMLa(uint32_t tndFDCoreArrLen, std::vector<int64_t>& gS1SplitNumOfFdHead, uint32_t* s2SplitNumOfFdHead,
                    uint32_t aivCoreNum, SeqTilingInfo& seqTilingInfo);
    custom::graphStatus SplitBalanced();
    custom::graphStatus SplitUnbalanced();
    custom::graphStatus SplitBalancedFd();
    custom::graphStatus CalcInnerSize(uint32_t seqSize);
    custom::graphStatus SplitBN();
    custom::graphStatus ProcessGqaKvNz() const;
    custom::graphStatus CheckGqaTensor() const;
    custom::graphStatus CheckGqaIOTensor() const;
    custom::graphStatus CheckGqaAntiquantTensor() const;
    // custom::graphStatus CheckGqaAntiquantType() const;
    custom::graphStatus CheckGqaBlockTable() const;
    custom::graphStatus CheckGqaTensorEmpty() const;
    custom::graphStatus CheckGqaSeqSize() const;
    custom::graphStatus CheckGqaAttribute() const;
    custom::graphStatus CheckGqaDefault() const;

    std::vector<int64_t> InitSparseValidArray(const int64_t* actualLens) const;
    bool BalanceLoad(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                     std::vector<int64_t>& localValue, std::vector<int64_t>& sparseStartIdx) const;
    void InitLoadValue(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                       const std::vector<int64_t>& sparseStartIdx, std::vector<int64_t>& localValue) const;
    void SetSparseStartIdx(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                           uint32_t* sparseStartIdx, int64_t splitFactorSize) const;

    bool IsFlashDecode(uint32_t coreNum, IfaPerfMode perfMode);
    bool CheckCoreOkFlag(uint32_t coreNum, IfaPerfMode perfMode) const;
    bool DealSameSeqEachBatch() const;
    bool HasZeroSeqBatch() const;
    bool IsKvZeroBatchSplit(bool needUpdate, uint32_t lastValidBMoreIdx, uint32_t bSize,
                            const std::vector<uint32_t>& s1OuterNum, const std::vector<uint32_t>& s2OuterNum) const;
    IfaAmlaMode GetAmlaMode() const;

    custom::graphStatus SplitBN_V0();
    custom::graphStatus SplitBNS();

    bool CheckWorkSpace();
    // bool GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType) const;

    std::pair<uint32_t, uint32_t> GetPreLoadNumAndActCoreNum() const;
    void CalcWorkSpaceForBmmAll(const IfaWorkSpaceSizeParams& params, uint32_t preLoadNum, uint32_t actCoreNum);
    custom::graphStatus CalcWorkSpace();
    custom::graphStatus CalcNumBlocks();
    custom::graphStatus GetKvLayoutInfo(KvLayoutInfo& kvLayoutInfo) const;
    custom::graphStatus GetInputLayoutVal(uint8_t& layoutVal) const;
    custom::graphStatus GetInputQueryVal(uint8_t& inputQVal) const;
    custom::graphStatus GetInputKvVal(uint8_t& inputKvVal) const;
    custom::graphStatus GetOutputVal(uint8_t& outputVal) const;
    custom::graphStatus GenTilingKey() const;

    custom::graphStatus FillTiling();
    void FillTilingBaseParams();
    void FillTilingSplitKV() const;
    void FillTilingCoreParams() const;
    void FillTilingSingleCoreParams() const;
    void FillTilingSingleCoreTensorSize() const;
    void FillTilingSoftmaxFlashTiling();
    void FillTilingTranspose();
    void FillTilingOutputParams() const;

    custom::graphStatus CalcSysPrefixWorkSpace();
    custom::graphStatus FillSysPrefixTiling();
    custom::graphStatus CalcSysPrefixNumBlocks();
    custom::graphStatus SplitForLseCombine();

    void CalcFDWorkSpace(const uint32_t actCoreNum);
    void NormalCalcFDWorkSpace(const uint32_t actCoreNum);
    void MLACalcFDWorkSpace(const uint32_t actCoreNum);

    uint32_t CalcBalanceFDParamNums(const uint32_t actCoreNum) const;
    uint32_t CalcUnbalanceFDParamNums() const;

    // custom::graphStatus CheckQueryQuantParam4FullQuant(const gert::Shape dequantScaleQueryShape) const;
    // custom::graphStatus CheckKVQuantParam4FullQuant(const gert::Shape dequantScaleKVShape) const;
    void GetEstimatedLoad(int64_t& estimatedLoad) const;

private:
    bool passToOldTiling_ = false;
    uint32_t numHeads_ = 0;
    float scaleValue_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t qTokens_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t innerPrecise_ = 0;
    uint32_t nNumOfQInOneGroup_ = 1;
    uint32_t msdIterNum_ = 1;
    uint32_t antiquantMode_ = 0;
    uint32_t antiquantPerTensorFlag_ = 0;
    uint32_t antiquantPerHeadFlag_ = 0;
    uint32_t antiquantParamsInPagedAttentionFlag_ = 0;
    uint32_t antiquantNum_ = 2;
    uint32_t maskHeadStride_ = 0;
    uint32_t maskBatchStride_ = 0;
    uint32_t headSplit_ = 0;

    uint32_t headDimRope_ = 64;
    uint32_t headDim_ = 0;
    uint32_t headDimV_ = 0;
    uint32_t seqSize_ = 0;
    uint32_t batchSize_ = 0;
    IfaLayout inputLayout_ = IfaLayout::BSH_BSND;
    IfaLayout inputKvLayout_ = IfaLayout::BSH_BSND;
    IfaLayout outputLayout_ = IfaLayout::BSH_BSND;
    uint32_t sMax_ = 0;
    uint32_t tSeqSize_ = 1;      // TND格式T轴长度
    uint32_t qSeqSize_ = 1;      // 默认S1 = 1
    uint32_t blockTypeSize_ = 0; // 计算中间量大小
    uint32_t kvSplitPart_ = 1;

    uint32_t sMaxPrefix_ = 0;
    uint32_t maxActualPrefixLen_ = 0;

    at::ScalarType inputQType_ = at::ScalarType::Half;  // float16
    at::ScalarType inputKvType_ = at::ScalarType::Half; // float16
    at::ScalarType outputType_ = at::ScalarType::Half;  // float16

    size_t ubSize_ = 0;
    size_t l1Size_ = 0;
    size_t l0cSize_ = 0;
    size_t l0bSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    uint32_t cvRatio_ = 2; // 2表示CV1:2; 1表示CV 1:1
    IfaSocVersion socVersion_ = IfaSocVersion::SOC_ASCEND_910B;
    size_t libapiSize_ = 0;

    size_t mmResUbSize_ = 0;
    size_t bmm2ResUbSize_ = 0;
    size_t qPreSizeMla_ = 0;

    size_t softmaxFlashTmpSize_ = 0;
    size_t softmaxTmpSize_ = 0;
    size_t softMaxSize_ = 0;

    size_t selectWithByteMaskTmpMinSize_ = 0;

    bool pseShiftFlag_ = false;
    uint32_t pseShiftTypeSize_ = NUM_BYTES_FLOAT16;
    uint32_t pseShiftBatch_ = 0U;
    uint32_t pseShiftS1_ = 0U;

    bool attenMaskFlag_ = false;
    uint32_t attenMaskSize_ = 0;
    uint32_t attenMaskTypeSize_ = 0;

    bool antiQuantFlag_ = false;
    size_t antiquantUb_ = 0;
    bool kvAntiParamSplitFlag_ = false;

    bool pageAttentionFlag_ = false;
    uint32_t maxBlockNumPerBatch_ = 0;
    size_t kvPageResUbSize_ = 0;
    uint32_t totalBlockNum_ = 0;

    bool batchContinuousFlag_ = true;
    std::vector<int64_t> kvListSeqLens_;

    bool actualSeqLenFlag_ = false;
    bool kvPaddingSizeFlag_ = false;

    bool quantFlag_ = false;
    size_t quantUbSize_ = 0;

    bool isOutQuantPerChnOut_ = false;
    bool isOutQuantTypeBf16_ = false;

    uint32_t actualLenQDims_ = 0;
    uint32_t actualLenDims_ = 0;
    uint32_t maxActualseq_ = 0;
    bool isSameActualseq_ = true;
    bool isSameSeqAllKVTensor_ = true;
    bool hasZeroActualseq_ = false;
    bool hasZeroSeqKVTensor_ = false;
    uint32_t dealBN2Num_ = DEAL_BN2_NUM;

    // flash config
    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 0;
    uint32_t sInnerSizeTail_ = 0;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t headDimAlign_ = 0;
    uint32_t headDimVAlign_ = 0;

    bool isSplitBPolicy_ = false;
    bool splitKVFlag_ = false;
    uint32_t kvSplit_ = 0;
    bool splitKVFlagPrefix_ = false;
    uint32_t antiqSeqSize_ = 0;

    IfaPerfMode perfMode_ = IfaPerfMode::NORMAL;
    TilingInOutMode inOutMode_ = TilingInOutMode::FP16_FP16;
    IfaAmlaMode amlaMode_ = IfaAmlaMode::DISABLE_AMLA;
    size_t workspaceSize_ = 0;

    uint32_t taskRation_ = 0;
    uint32_t usedCoreNum_ = 0;

    uint32_t startIdxEachCore_[MAX_CORE_NUM] = {};
    IFAContext* ifaContext_ = nullptr;
    IncreFlashAttentionTilingData* tilingData_ = nullptr;
    IncreFlashAttentionTilingDataPrefix* tilingDataPrefix_ = nullptr;
    IncreFlashAttentionBaseParams* tilingDataBase_ = nullptr;
    IncreFlashAttentionSplitCoreParams* tilingDataCore_ = nullptr;
    bool isWorkspace_ = false;

    uint32_t formerCoreNum_ = 0;
    uint32_t blockSplitBn2Range_ = 0;
    uint32_t tailSplitedBatchRange_ = 0;
    uint32_t groupSplitSize_ = 0;
    uint32_t s1SplitSize_ = 0;
    uint32_t gOuter_ = 1;
    uint32_t s1Outer_ = 1;
    uint32_t sparseMode_ = 3;
    uint32_t gMax_ = 128;

    // sliding window
    bool slidingFlag_ = false;
    int32_t windowSize_ = -1; // defalut is negative without sliding

    uint32_t l2CacheOffFlag_ = 0;
    // softmaxLse
    bool softmaxLseFlag_ = false;

    bool ropeFlag_ = false;
    bool gqaMtpFlag_ = false;
    bool gqaKvNZFlag_ = false;

    bool sysPrefixFlag_ = false;
    bool isSysPrefixTiling_ = false;
    uint32_t batchSizeQ_ = 1;
    uint32_t actualLenDimsPrefix_ = 0;
    bool atbRunFlag_ = false;

    uint64_t prefixAttenOutOffset_ = 0;
    uint64_t userPromptAttenOutOffset_ = 0;
    uint64_t tmpLseOffset_ = 0;

    uint32_t formerCoreNumSp_ = 0;
    uint32_t blockSplitBn2RangeSp_ = 0;
    uint32_t tailSplitedBatchRangeSp_ = 0;
    uint32_t combinUsedCore_ = 0;

    // IncreFlashAttentionTilingAtbDataV2 ifaTilingAtbData;
    // IncreFlashAttentionTilingDataMla tilingDataMla_;

    bool balanceModeFlag_ = false;

    // Atb param
    uint32_t seqStepQ_ = 0;
    uint32_t seqStepKv_ = 0;
};

std::string DataTypeToSerialString(at::ScalarType type);

} // namespace optiling
#endif
