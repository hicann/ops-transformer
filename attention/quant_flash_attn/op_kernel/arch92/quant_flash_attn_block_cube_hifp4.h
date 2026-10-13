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
 * \file quant_flash_attn_block_cube_hifp4.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_CUBE_HIFP4_H_
#define QUANT_FLASH_ATTN_BLOCK_CUBE_HIFP4_H_
#include "../../../common/op_kernel/offset_calculator.h"
#include "matmul_hif4.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "memory_copy_arch92_quant_flash_attn.h"
#include "../../../common/op_kernel/memcopy/attn_copy_gm_to_l1.h"

#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace fa_hif4_matmul;
using namespace AttentionCommon;

namespace BaseApi {

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false>
class QuantFlashAttnBlockCubeHifp4 {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;  // 128
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType; // 1024
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;   // 128
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType; // 128
    static constexpr uint32_t BASE_S2_C1_SIZE = 128U;
    static constexpr uint32_t BASE_K_BMM2 = 256U;
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr uint32_t MXFP_MULTI_BASE_SIZE = 2U;
    static constexpr uint32_t HIF4_GROUP_SIZE = 64U;
    static constexpr uint32_t DB = 2U;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    // static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool PAGE_ATTENTION = false;
    static constexpr bool BMM2_TOUB = true;
    static constexpr bool USE_DN = useDn;
    static constexpr bool SPLITD = false;

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, BMM2_TOUB};
    static constexpr FixpipeConfig PFA_CFG_COLUMN_MAJOR_UB = {
        CO2Layout::COLUMN_MAJOR,
        true}; // COLUMN_MAJOR: 使能NZ2DN，输出数据格式为ND格式; true: 用于用户指定目的地址的位置是否是UB
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();
    static constexpr GmFormat Q_SCALE_FORMAT = GetQueryScaleGmFormat<layout, USE_DN>();
    static constexpr GmFormat K_SCALE_FORMAT = GetKeyScaleGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();
    static constexpr GmFormat V_SCALE_FORMAT = GetValueScaleGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();
    /* ============核内event id 分配============= */
    static constexpr int32_t EVENT_ID_Q_MTE1_MTE2_PING = 0;
    static constexpr int32_t EVENT_ID_Q_MTE1_MTE2_PONG = 1;
    static constexpr int32_t EVENT_ID_Q_MTE2_MTE1_PING = 0;
    static constexpr int32_t EVENT_ID_Q_MTE2_MTE1_PONG = 1;
    static constexpr int32_t EVENT_ID_INIT_MTE2_MTE1 = 7;

    static constexpr int32_t EVENT_ID_K_MTE1_MTE2_PING = 2;
    static constexpr int32_t EVENT_ID_K_MTE1_MTE2_PONG = 3;
    static constexpr int32_t EVENT_ID_K_MTE2_MTE1_PING = 2;
    static constexpr int32_t EVENT_ID_K_MTE2_MTE1_PONG = 3;

    static constexpr int32_t EVENT_ID_V_MTE1_MTE2_PING = 4;
    static constexpr int32_t EVENT_ID_V_MTE1_MTE2_PONG = 5;
    static constexpr int32_t EVENT_ID_V_MTE2_MTE1_PING = 4;
    static constexpr int32_t EVENT_ID_V_MTE2_MTE1_PONG = 5;

    static constexpr int32_t EVENT_ID_C1_M_MTE1_PING = 3;
    static constexpr int32_t EVENT_ID_C1_M_MTE1_PONG = 4;
    static constexpr int32_t EVENT_ID_C1_M_MTE1_Q = 7;

    static constexpr int32_t EVENT_ID_C1_MTE1_M_PING = 3;
    static constexpr int32_t EVENT_ID_C1_MTE1_M_PONG = 4;

    static constexpr int32_t EVENT_ID_C2_M_MTE1_PING = 5;
    static constexpr int32_t EVENT_ID_C2_M_MTE1_PONG = 6;
    static constexpr int32_t EVENT_ID_C2_MTE1_M_PING = 5;
    static constexpr int32_t EVENT_ID_C2_MTE_M1_PONG = 6;

    static constexpr int32_t EVENT_ID_C1_FIX_M_PING = 0;
    static constexpr int32_t EVENT_ID_C1_FIX_M_PONG = 1;
    static constexpr int32_t EVENT_ID_C1_M_FIX_PING = 0;
    static constexpr int32_t EVENT_ID_C1_M_FIX_PONG = 1;

    static constexpr int32_t EVENT_ID_C2_FIX_M = 2;
    static constexpr int32_t EVENT_ID_C2_M_FIX = 2;
    /* ============核间flag id 分配============= */
    static constexpr uint16_t FLAG_ID_V1_C1_BUFFER0 = 0;
    static constexpr uint16_t FLAG_ID_V1_C1_BUFFER1 = 1;
    static constexpr uint16_t FLAG_ID_V1_C1_BUFFER2 = 2;
    static constexpr uint16_t FLAG_ID_V1_C1_BUFFER3 = 3;
    static constexpr uint16_t FLAG_ID_C1_V1 = 4;

    static constexpr uint16_t FLAG_ID_C2_V1_PING = 5;
    static constexpr uint16_t FLAG_ID_C2_V1_PONG = 6;
    static constexpr uint16_t FLAG_ID_V1_C2_PING = 7;
    static constexpr uint16_t FLAG_ID_V1_C2_PONG = 8;

    static constexpr uint16_t FLAG_ID_V2_C2 = 9;
    static constexpr uint16_t FLAG_ID_C2_V2 = 10;
    // =================================L1 Buffer=================================
    static constexpr uint32_t BUFFER_CNT2 = 2;
    static constexpr uint32_t BUFFER_CNT4 = 4;
    static constexpr uint32_t BUFFER_CNT8 = 8;
    static constexpr uint32_t BUFFER_CNT16 = 16;
    static constexpr uint32_t Q_L1_BYTE_SIZE =
        mBaseSize * dBaseSize / 2; // 2: sizeof hif4 is 0.5 Byte, 128*128/2 = 8*1024
    static constexpr uint32_t K_L1_BYTE_SIZE =
        BASE_S2_C1_SIZE * dBaseSize / 2; // 2: size of hif4 is 0.5 Byte, 128*128/2 = 8*1024
    static constexpr uint32_t V_L1_BYTE_SIZE =
        s2BaseSize * dBaseSize / 2; // 2: sizeof hif4 is 0.5 Byte, 1024*128/2 = 64*1024
    static constexpr uint32_t QSCALE_L1_BYTE_SIZE =
        mBaseSize * dBaseSize / HIF4_GROUP_SIZE * sizeof(float); // 128*128/64*4 = 1*1024
    static constexpr uint32_t KSCALE_L1_BYTE_SIZE =
        BASE_S2_C1_SIZE * dBaseSize / HIF4_GROUP_SIZE * sizeof(float); // 128*128/64*4 = 1*1024
    static constexpr uint32_t VSCALE_L1_BYTE_SIZE =
        s2BaseSize * dBaseSize / HIF4_GROUP_SIZE * sizeof(float);           // 1024*128/64*4 = 8*1024
    static constexpr uint32_t P_L1_BYTE_SIZE = mBaseSize * BASE_S2_C1_SIZE; // 128*128*1=16*1024
    static constexpr uint32_t PSCALE_L1_BYTE_SIZE = mBaseSize * BASE_S2_C1_SIZE / MXFP_GROUP_SIZE; // 128*128/32*1=512

    // =================================L0 Buffer=================================
    static constexpr uint32_t Q_L0B_SIZE = mBaseSize * dBaseSize;           // 128 * 128=16k
    static constexpr uint32_t K_L0A_SIZE = BASE_S2_C1_SIZE * dBaseSize;     // 128 * 128=16k
    static constexpr uint32_t V_L0B_SIZE = BASE_K_BMM2 * (dBaseSize + 16);  // 256*(128+16)=36k
    static constexpr uint32_t P_L0A_SIZE = BASE_K_BMM2 * mBaseSize;         // 256*128=32k
    static constexpr uint32_t BMM1_L0C_SIZE = mBaseSize * BASE_S2_C1_SIZE;  // 128*128=16k
    static constexpr uint32_t BMM2_L0C_SIZE = mBaseSize * (dBaseSize + 16); // 128*(128+16)=18k

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using SCALE_T = hif4_scale;
    using MM_T = T;
    using mm2ResPos = typename std::conditional<BMM2_TOUB, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_DBUF_T = mm2ResPos;

    /* ============确定key scale的类型============= */

    static constexpr bool IS_TND_LAYOUT =
        (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD);
    static constexpr bool Q_NEEDS_WZH = (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND);
    static constexpr bool KV_NEEDS_WZH = (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND);

    static constexpr ActualSeqLensMode Q_PARSER_MODE = GetQActSeqMode<LAYOUT>();
    static constexpr ActualSeqLensMode KV_PARSER_MODE = GetKvActSeqMode<LAYOUT, PAGE_ATTENTION>();

    using QSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<Q_PARSER_MODE, int32_t, true>,
                                                     ActualSeqLensParser<Q_PARSER_MODE, int32_t>>::type;

    using KvSeqParserType = typename std::conditional<(!PAGE_ATTENTION && IS_TND_LAYOUT),
                                                      ActualSeqLensParser<KV_PARSER_MODE, int32_t, true>,
                                                      ActualSeqLensParser<KV_PARSER_MODE, int32_t>>::type;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_NEEDS_WZH>;
    using FaGmTensorQScale = FaGmTensor<SCALE_T, Q_SCALE_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKScale = FaGmTensor<SCALE_T, K_SCALE_FORMAT, int32_t, KV_NEEDS_WZH>;
    using FaGmTensorVScale = FaGmTensor<SCALE_T, V_SCALE_FORMAT, int32_t, KV_NEEDS_WZH>;

    using ConstInfoX = ConstInfo_t;
    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm;
    FaGmTensorKV keyGm;
    FaGmTensorKV valueGm;
    FaGmTensorQScale queryScaleGm;
    FaGmTensorKScale keyScaleGm;
    FaGmTensorVScale valueScaleGm;
    GlobalTensor<int32_t> blockTableGm;

    QSeqParserType* qSeqParserPtr = nullptr;
    KvSeqParserType* kvSeqParserPtr = nullptr;

    CopyQueryGmToL1<Q_T, Q_FORMAT, L1Format::NZ, InnerMLayout::S1_ONLY_LAYOUT> copyQueryGmToL1;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;
    CopyQueryScaleGmToL1<SCALE_T, Q_SCALE_FORMAT> copyQueryScaleGmToL1;
    CopyKeyScaleGmToL1<SCALE_T, K_SCALE_FORMAT, L1Format::NZ> copyKeyScaleGmToL1;
    CopyValueScaleGmToL1<SCALE_T, V_SCALE_FORMAT, L1Format::NZ, ScaleTrans::DN2NZ> copyValueScaleGmToL1;

    /* =====================静态LocalTensor成员====================*/
    // L1 静态Tensor (替代原bank0Buffer/bank1Buffer的GetWithOffset切片)
    // L1布局: bank0 256K + bank1 256K, 与vec侧共享物理L1地址
    LocalTensor<uint8_t> KTensor[BUFFER_CNT8]; // bank less conflict
    LocalTensor<uint8_t> VTensor[BUFFER_CNT2];
    LocalTensor<uint8_t> QTensor[BUFFER_CNT2];
    LocalTensor<uint8_t> QscaleTensor[BUFFER_CNT2];
    LocalTensor<uint8_t> KscaleTensor[BUFFER_CNT8];
    LocalTensor<uint8_t> VscaleTensor[BUFFER_CNT2];
    LocalTensor<uint16_t> RowSumFullOneVandScale;
    LocalTensor<uint8_t> PTensor[BUFFER_CNT16];     // P矩阵, 与vec侧共享L1
    LocalTensor<uint8_t> PscaleTensor[BUFFER_CNT2]; // Pscale, 与vec侧共享L1

    // L0 静态Tensor (替代原TPipe->InitBuffer + TBuf->Get)
    LocalTensor<INPUT_T> mmQL0BTensor;
    LocalTensor<INPUT_T> mmVL0BTensors[BUFFER_CNT2];
    LocalTensor<uint8_t> mmL0ATensor; // L0A: P矩阵 (bmm1的K, bmm2的P)
    LocalTensor<float> mm1L0CTensors[BUFFER_CNT2];
    LocalTensor<float> mm2L0CTensor;

    // UB 静态Tensor (cube侧fixpipe输出, 与vec侧共享物理UB地址)
    LocalTensor<half> mm1ResUB;     // bmm1结果(fixpipe写, vec读)
    LocalTensor<float> mm2ResUB;    // bmm2结果(fixpipe写, vec读)
    LocalTensor<float> rowSumNewUB; // rowSumNew(fixpipe写, vec读)

    __gm__ uint8_t* keyPtr = nullptr;
    __gm__ uint8_t* valuePtr = nullptr;

    const ConstInfoX& constInfo;

    /*============================================================================== */
    __aicore__ inline QuantFlashAttnBlockCubeHifp4(ConstInfoX& constInfo)
        : constInfo(constInfo){};

    __aicore__ inline void InitInput(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                     __gm__ uint8_t* blockTable, __gm__ uint8_t* dequantScaleQuery,
                                     __gm__ uint8_t* dequantScaleKey, __gm__ uint8_t* dequantScaleValue,
                                     QSeqParserType& qParser, KvSeqParserType& kvParser)
    {
        this->qSeqParserPtr = &qParser;
        this->kvSeqParserPtr = &kvParser;
        InitCubeInput(query, key, value, blockTable, dequantScaleQuery, dequantScaleKey, dequantScaleValue);
    }

    __aicore__ inline void InitTensors()
    {
        AllocEventID();
        static constexpr uint32_t BUFFER_NUM0 = 0;
        static constexpr uint32_t BUFFER_NUM1 = 1;
        static constexpr uint32_t BUFFER_NUM2 = 2;
        static constexpr uint32_t BUFFER_NUM3 = 3;
        static constexpr uint32_t BUFFER_NUM4 = 4;
        static constexpr uint32_t BUFFER_NUM5 = 5;
        static constexpr uint32_t BUFFER_NUM6 = 6;
        static constexpr uint32_t BUFFER_NUM7 = 7;
        static constexpr uint32_t BUFFER_NUM8 = 8;
        static constexpr uint32_t BUFFER_NUM9 = 9;
        static constexpr uint32_t BUFFER_NUM10 = 10;
        static constexpr uint32_t BUFFER_NUM11 = 11;
        static constexpr uint32_t BUFFER_NUM12 = 12;
        static constexpr uint32_t BUFFER_NUM13 = 13;
        static constexpr uint32_t BUFFER_NUM14 = 14;
        static constexpr uint32_t BUFFER_NUM15 = 15;
        static constexpr uint32_t KB_UNIT = 1024U;
        // L1 静态分配 - bank0: 256K, bank1: 256K
        // bank0 布局(与原GetWithOffset偏移完全一致,保持bank冲突规避):
        //   KTensor[0-3]:   0~32K      (8K * 4)
        //   VTensor[0]:     32K~96K    (64K)
        //   KscaleTensor[0-3]: 96K~100K (1K * 4)
        //   VscaleTensor[0]:100K~104K  (4K)
        //   QTensor[0]:     108K~116K  (8K)
        //   QscaleTensor[0]:116K~117K  (1K)
        //   PTensor[0-7]:   117K~245K  (16K * 8)
        //   PscaleTensor[0]:245K~245.5K (256*2*8=4K? 实为0.5K,预留4K)
        //   RowSumFullOneVandScale: 249K~250.14K
        // bank1 布局(对称):
        //   KTensor[4-7]:   0~32K
        //   VTensor[1]:     32K~96K
        //   KscaleTensor[4-7]: 96K~100K
        //   VscaleTensor[1]:100K~104K
        //   QTensor[1]:     108K~116K
        //   QscaleTensor[1]:116K~117K
        //   PTensor[8-15]:  117K~245K
        //   PscaleTensor[1]:245K~245.5K
        // 使用 LocalTensor(TPosition, addr, size) 静态分配
        // bank0
        KTensor[BUFFER_NUM0] = LocalTensor<uint8_t>(TPosition::A1, 0 * KB_UNIT, K_L1_BYTE_SIZE);  // K_L1_BYTE_SIZE:8k
        KTensor[BUFFER_NUM1] = LocalTensor<uint8_t>(TPosition::A1, 8 * KB_UNIT, K_L1_BYTE_SIZE);  // 8: buffer address
        KTensor[BUFFER_NUM2] = LocalTensor<uint8_t>(TPosition::A1, 16 * KB_UNIT, K_L1_BYTE_SIZE); // 16: buffer address
        KTensor[BUFFER_NUM3] = LocalTensor<uint8_t>(TPosition::A1, 24 * KB_UNIT, K_L1_BYTE_SIZE); // 24: buffer address
        VTensor[BUFFER_NUM0] =
            LocalTensor<uint8_t>(TPosition::A1, 32 * KB_UNIT, V_L1_BYTE_SIZE); // 32: buffer address, V_L1_BYTE_SIZE:64k
        KscaleTensor[BUFFER_NUM0] = LocalTensor<uint8_t>(
            TPosition::A1, 96 * KB_UNIT, KSCALE_L1_BYTE_SIZE); // 96: buffer address, KSCALE_L1_BYTE_SIZE:1k
        KscaleTensor[BUFFER_NUM1] =
            LocalTensor<uint8_t>(TPosition::A1, 97 * KB_UNIT, KSCALE_L1_BYTE_SIZE); // 97: buffer address
        KscaleTensor[BUFFER_NUM2] =
            LocalTensor<uint8_t>(TPosition::A1, 98 * KB_UNIT, KSCALE_L1_BYTE_SIZE); // 98: buffer address
        KscaleTensor[BUFFER_NUM3] =
            LocalTensor<uint8_t>(TPosition::A1, 99 * KB_UNIT, KSCALE_L1_BYTE_SIZE); // 99: buffer address
        VscaleTensor[BUFFER_NUM0] = LocalTensor<uint8_t>(
            TPosition::A1, 100 * KB_UNIT, VSCALE_L1_BYTE_SIZE); // 100: buffer address, VSCALE_L1_BYTE_SIZE:8k
        QTensor[BUFFER_NUM0] = LocalTensor<uint8_t>(TPosition::A1, 108 * KB_UNIT,
                                                    Q_L1_BYTE_SIZE); // 108: buffer address, Q_L1_BYTE_SIZE:8k
        QscaleTensor[BUFFER_NUM0] = LocalTensor<uint8_t>(
            TPosition::A1, 116 * KB_UNIT, QSCALE_L1_BYTE_SIZE); // 116: buffer address, QSCALE_L1_BYTE_SIZE:1k
        PTensor[BUFFER_NUM0] = LocalTensor<uint8_t>(TPosition::A1, 117 * KB_UNIT,
                                                    P_L1_BYTE_SIZE); // 117: buffer address, P_L1_BYTE_SIZE:16k
        PTensor[BUFFER_NUM1] =
            LocalTensor<uint8_t>(TPosition::A1, 121 * KB_UNIT, P_L1_BYTE_SIZE); // 121: buffer address
        PTensor[BUFFER_NUM2] =
            LocalTensor<uint8_t>(TPosition::A1, 149 * KB_UNIT, P_L1_BYTE_SIZE); // 149: buffer address
        PTensor[BUFFER_NUM3] =
            LocalTensor<uint8_t>(TPosition::A1, 153 * KB_UNIT, P_L1_BYTE_SIZE); // 153: buffer address
        PTensor[BUFFER_NUM4] =
            LocalTensor<uint8_t>(TPosition::A1, 181 * KB_UNIT, P_L1_BYTE_SIZE); // 181: buffer address
        PTensor[BUFFER_NUM5] =
            LocalTensor<uint8_t>(TPosition::A1, 185 * KB_UNIT, P_L1_BYTE_SIZE); // 185: buffer address
        PTensor[BUFFER_NUM6] =
            LocalTensor<uint8_t>(TPosition::A1, 213 * KB_UNIT, P_L1_BYTE_SIZE); // 213: buffer address
        PTensor[BUFFER_NUM7] =
            LocalTensor<uint8_t>(TPosition::A1, 217 * KB_UNIT, P_L1_BYTE_SIZE); // 217: buffer address
        PscaleTensor[BUFFER_NUM0] =
            LocalTensor<uint8_t>(TPosition::A1, 245 * KB_UNIT,
                                 PSCALE_L1_BYTE_SIZE * BUFFER_CNT8); // 245: buffer address, PSCALE_L1_BYTE_SIZE:512
        static constexpr uint32_t RowSumFullOneVandScaleSize = 16 * 128 / 2 + 16 * 128 / 64 * sizeof(float);
        RowSumFullOneVandScale =
            LocalTensor<uint16_t>(TPosition::A1, 249 * KB_UNIT, RowSumFullOneVandScaleSize); // 249: buffer address
        // bank1 (偏移256K,即bank1起始)
        constexpr uint32_t BANK1_OFFSET = 256 * KB_UNIT;
        KTensor[BUFFER_NUM4] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 0 * KB_UNIT, K_L1_BYTE_SIZE); // K_L1_BYTE_SIZE:8k
        KTensor[BUFFER_NUM5] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 8 * KB_UNIT, K_L1_BYTE_SIZE); // 8: buffer address
        KTensor[BUFFER_NUM6] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 16 * KB_UNIT, K_L1_BYTE_SIZE); // 16: buffer address
        KTensor[BUFFER_NUM7] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 24 * KB_UNIT, K_L1_BYTE_SIZE); // 24: buffer address
        VTensor[BUFFER_NUM1] = LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 32 * KB_UNIT,
                                                    V_L1_BYTE_SIZE); // 32: buffer address, V_L1_BYTE_SIZE:64k
        KscaleTensor[BUFFER_NUM4] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 96 * KB_UNIT,
                                 KSCALE_L1_BYTE_SIZE); // 96: buffer address, KSCALE_L1_BYTE_SIZE:1k
        KscaleTensor[BUFFER_NUM5] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 97 * KB_UNIT, KSCALE_L1_BYTE_SIZE); // 97: buffer address
        KscaleTensor[BUFFER_NUM6] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 98 * KB_UNIT, KSCALE_L1_BYTE_SIZE); // 98: buffer address
        KscaleTensor[BUFFER_NUM7] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 99 * KB_UNIT, KSCALE_L1_BYTE_SIZE); // 99: buffer address
        VscaleTensor[BUFFER_NUM1] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 100 * KB_UNIT,
                                 VSCALE_L1_BYTE_SIZE); // 100: buffer address, VSCALE_L1_BYTE_SIZE:8k
        QTensor[BUFFER_NUM1] = LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 108 * KB_UNIT,
                                                    Q_L1_BYTE_SIZE); // 108: buffer address, Q_L1_BYTE_SIZE:8k
        QscaleTensor[BUFFER_NUM1] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 116 * KB_UNIT,
                                 QSCALE_L1_BYTE_SIZE); // 116: buffer address, QSCALE_L1_BYTE_SIZE:1k
        PTensor[BUFFER_NUM8] = LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 117 * KB_UNIT,
                                                    P_L1_BYTE_SIZE); // 117: buffer address, P_L1_BYTE_SIZE:16k
        PTensor[BUFFER_NUM9] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 121 * KB_UNIT, P_L1_BYTE_SIZE); // 121: buffer address
        PTensor[BUFFER_NUM10] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 149 * KB_UNIT, P_L1_BYTE_SIZE); // 149: buffer address
        PTensor[BUFFER_NUM11] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 153 * KB_UNIT, P_L1_BYTE_SIZE); // 153: buffer address
        PTensor[BUFFER_NUM12] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 181 * KB_UNIT, P_L1_BYTE_SIZE); // 181: buffer address
        PTensor[BUFFER_NUM13] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 185 * KB_UNIT, P_L1_BYTE_SIZE); // 185: buffer address
        PTensor[BUFFER_NUM14] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 213 * KB_UNIT, P_L1_BYTE_SIZE); // 213: buffer address
        PTensor[BUFFER_NUM15] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 217 * KB_UNIT, P_L1_BYTE_SIZE); // 217: buffer address
        PscaleTensor[BUFFER_NUM1] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 245 * KB_UNIT,
                                 PSCALE_L1_BYTE_SIZE * BUFFER_CNT8); // 245: buffer address, PSCALE_L1_BYTE_SIZE:512

        // L0 静态分配 (替代TPipe->InitBuffer + TBuf->Get)
        // L0B: Q (8K), V (18K * 2)
        mmQL0BTensor = LocalTensor<INPUT_T>(TPosition::B2, 0, Q_L0B_SIZE); // Q_L0B_SIZE:16k
        mmVL0BTensors[BUFFER_NUM0] = LocalTensor<INPUT_T>(TPosition::B2, Q_L0B_SIZE / 2,
                                                          V_L0B_SIZE); // 2: sizeof hif4 is 0.5 Byte, V_L0B_SIZE:36k
        mmVL0BTensors[BUFFER_NUM1] = LocalTensor<INPUT_T>(TPosition::B2, (Q_L0B_SIZE + V_L0B_SIZE) / 2,
                                                          V_L0B_SIZE); // 2: sizeof hif4 is 0.5 Byte
        // L0A: P (64K) - 用uint8_t存储,实际按需ReinterpretCast
        // 注意: L0A与L0B物理独立,地址从0开始
        mmL0ATensor = LocalTensor<uint8_t>(TPosition::A2, 0, 64 * KB_UNIT); // 64: 整块L0A size，K和P复用
        // L0C: mm1 (64K * 2), mm2 (72K)
        mm1L0CTensors[BUFFER_NUM0] = LocalTensor<float>(TPosition::CO1, 0, BMM1_L0C_SIZE); // BMM1_L0C_SIZE:16k
        mm1L0CTensors[BUFFER_NUM1] = LocalTensor<float>(TPosition::CO1, BMM1_L0C_SIZE * sizeof(float), BMM1_L0C_SIZE);
        mm2L0CTensor = LocalTensor<float>(TPosition::CO1, (BMM1_L0C_SIZE + BMM1_L0C_SIZE) * sizeof(float),
                                          BMM2_L0C_SIZE); // BMM2_L0C_SIZE:18k

        // UB 静态分配 (cube侧fixpipe输出, 与vec侧共享物理UB地址)
        // UB布局与原kernel InitMMResBuf一致:
        //   mm1Res: 0~128K (128*128 * 4个buffer, half类型)
        //   mm2Res: 192K~320K (128*128, float类型)
        //   rowSumOld: 327K (128, float)
        //   rowSumNew: 328K (128*8, float)
        mm1ResUB = LocalTensor<half>(TPosition::VECCALC, 0, mBaseSize * BASE_S2_C1_SIZE * BUFFER_CNT4);
        mm2ResUB = LocalTensor<float>(TPosition::VECCALC, 192 * KB_UNIT, mBaseSize * dBaseSize); // 192: buffer address
        static constexpr uint32_t FIX_NSIZE = 8; // 8: fixpipe使能channelSplit功能，nSize必须为8的倍数
        rowSumNewUB =
            LocalTensor<float>(TPosition::VECCALC, 328 * KB_UNIT, mBaseSize * FIX_NSIZE); // 328: buffer address
    }

    __aicore__ inline void ReleaseTensors()
    {
        FreeEventID();
    }

    __aicore__ inline void InitConstTensor()
    {
        // AMLA code
        static constexpr uint32_t constOffset =
            (16 * BASE_K_BMM2 / 2) /
            2; // 16行为单位添加, splitS2 = 128, sizeof(hif4)=0.5B, 2因为AMLAFullOneVandScale为uint16，每次偏移2B
        InitConstValueParams<uint16_t> initConstParam;
        // V
        initConstParam.repeatTimes = 1;
        initConstParam.blockNum =
            16 * BASE_K_BMM2 / 2 / 32; // NZ行方向最小分形16行，c2处s2切128，hif4数据类型，每个数据块32B 16*128*0.5/32
        initConstParam.dstGap = 0;
        initConstParam.initValue = (uint16_t)0x4444;            // 0x4444: 1 for hif4
        InitConstValue(RowSumFullOneVandScale, initConstParam); // 在mte2上

        // Vscale
        initConstParam.repeatTimes = BASE_K_BMM2 / 64; // 64: 32B of hif4
        initConstParam.blockNum = 1;
        initConstParam.dstGap = 1;
        initConstParam.initValue = (uint16_t)0x00c0; // 0x00c0: 1 for e6m2+e1_8
        InitConstValue(RowSumFullOneVandScale[constOffset], initConstParam);
        initConstParam.repeatTimes = BASE_K_BMM2 / 64; // 64: 32B of hif4
        initConstParam.blockNum = 1;
        initConstParam.dstGap = 1;
        initConstParam.initValue = (uint16_t)0x0000;                              // 0x0000: 0 for e1_16
        InitConstValue(RowSumFullOneVandScale[constOffset + 16], initConstParam); // 跨过第一个分形，16个2B

        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID_INIT_MTE2_MTE1);  // MTE2结束
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID_INIT_MTE2_MTE1); // MTE1开始

        // 搬运rowsum计算所需的矩阵
        LocalTensor<INPUT_T> VTensorTmp = RowSumFullOneVandScale.template ReinterpretCast<INPUT_T>();
        LocalTensor<uint16_t> VscaleTensorTmp = RowSumFullOneVandScale[constOffset];

        LocalTensor<INPUT_T> tempmmVL0ATensorPing = mmVL0BTensors[0][dBaseSize * 64]; // 64: 32B of hif4
        LocalTensor<INPUT_T> tempmmVL0AScaleTensorPing = mmVL0BTensors[0][dBaseSize * BASE_K_BMM2];
        LoadDataRowsumToL0A<INPUT_T, uint16_t>(tempmmVL0ATensorPing, VTensorTmp, VscaleTensorTmp, BASE_K_BMM2,
                                               dBaseSize); // C2在S2方向切128
        LoadDataRowsumMxToL0A<INPUT_T, uint16_t>(tempmmVL0AScaleTensorPing, VTensorTmp, VscaleTensorTmp, BASE_K_BMM2);
        LocalTensor<INPUT_T> tempmmVL0ATensorPong = mmVL0BTensors[1][dBaseSize * 64]; // 64: 32B of hif4
        LocalTensor<INPUT_T> tempmmVL0AScaleTensorPong = mmVL0BTensors[1][dBaseSize * BASE_K_BMM2];
        LoadDataRowsumToL0A<INPUT_T, uint16_t>(tempmmVL0ATensorPong, VTensorTmp, VscaleTensorTmp, BASE_K_BMM2,
                                               dBaseSize);
        LoadDataRowsumMxToL0A<INPUT_T, uint16_t>(tempmmVL0AScaleTensorPong, VTensorTmp, VscaleTensorTmp, BASE_K_BMM2);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                         __gm__ uint8_t* blockTable, __gm__ uint8_t* dequantScaleQuery,
                                         __gm__ uint8_t* dequantScaleKey, __gm__ uint8_t* dequantScaleValue)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        }

        InitQBuffer(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize, constInfo.s1Size, constInfo.dSize,
                    queryGm, query);

        InitQScaleBuffer(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize, constInfo.s1Size,
                         constInfo.dSize / HIF4_GROUP_SIZE, queryScaleGm, dequantScaleQuery);

        keyPtr = key;
        valuePtr = value;
        InitKVBuffer(constInfo.bSize, constInfo.s2Size, constInfo.n2Size, constInfo.blockSize, constInfo.dSize, keyGm,
                     key);
        InitKVBuffer(constInfo.bSize, constInfo.s2Size, constInfo.n2Size, constInfo.blockSize, constInfo.dSizeV,
                     valueGm, value);
        InitKScaleBuffer(constInfo.bSize, constInfo.s2Size, constInfo.n2Size, constInfo.blockSize,
                         constInfo.dSize / HIF4_GROUP_SIZE, keyScaleGm, dequantScaleKey);
        InitVScaleBuffer(constInfo.bSize, constInfo.s2Size / HIF4_GROUP_SIZE, constInfo.n2Size, constInfo.blockSize,
                         constInfo.dSizeV, valueScaleGm, dequantScaleValue);
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, FaGmTensorQ& qGmTensor, __gm__ uint8_t* gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T*)gm);
        if constexpr (Q_NEEDS_WZH) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, *this->qSeqParserPtr);
        } else {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, *this->qSeqParserPtr);
        }
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                            uint32_t headDim, FaGmTensorQScale& qScaleGmTensor, __gm__ uint8_t* gm)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T*)gm);
        if constexpr (Q_NEEDS_WZH) {
            qScaleGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, *this->qSeqParserPtr);
        } else {
            qScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, *this->qSeqParserPtr);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV& kvGmTensor,
                                        __gm__ uint8_t* gm)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T*)gm);
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                             constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
                                             constInfo.maxBlockNumPerBatch);
        } else if constexpr (KV_NEEDS_WZH) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, *this->kvSeqParserPtr);
        } else {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            kvGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr);
        }
    }

    __aicore__ inline void InitKScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, uint32_t headDim,
                                            FaGmTensorKScale& kScaleGmTensor, __gm__ uint8_t* gm)
    {
        kScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T*)gm);
        if constexpr (GmLayoutParams<K_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                                 constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<K_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_K_SCALE_PA_NZ) {
            uint32_t bs0 = 32 / sizeof(KV_T);
            uint32_t kvCacheBlockSize1 = kvCacheBlockSize / bs0 * MXFP_MULTI_BASE_SIZE;
            kScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize1, headDim / MXFP_MULTI_BASE_SIZE, bs0,
                                                 blockTableGm, constInfo.maxBlockNumPerBatch);
        } else if constexpr (KV_NEEDS_WZH) {
            kScaleGmTensor.offsetCalculator.Init(n2Size, headDim, *this->kvSeqParserPtr);
        } else {
            kScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            kScaleGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr);
        }
    }

    __aicore__ inline void InitVScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, uint32_t headDim,
                                            FaGmTensorVScale& vScaleGmTensor, __gm__ uint8_t* gm)
    {
        vScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T*)gm);
        if constexpr (GmLayoutParams<V_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            vScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                                 constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<V_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            vScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
                                                 constInfo.maxBlockNumPerBatch);
        } else if constexpr (KV_NEEDS_WZH) {
            vScaleGmTensor.offsetCalculator.Init(n2Size, headDim, *this->kvSeqParserPtr);
        } else {
            vScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            vScaleGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // 生产者 wait 消费者
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(FLAG_ID_C2_V1_PING); // C2.MTE1 trigger V1.MTE3
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(FLAG_ID_C2_V1_PONG);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_Q_MTE1_MTE2_PING); // C1
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_Q_MTE1_MTE2_PONG); // C1
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_K_MTE1_MTE2_PING);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_K_MTE1_MTE2_PONG);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_C1_M_MTE1_PING);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_C1_M_MTE1_PONG);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_C1_M_MTE1_Q);
        SetFlag<HardEvent::FIX_M>(EVENT_ID_C1_FIX_M_PING);
        SetFlag<HardEvent::FIX_M>(EVENT_ID_C1_FIX_M_PONG);

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_V_MTE1_MTE2_PING); // C2
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_V_MTE1_MTE2_PONG);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_C2_M_MTE1_PING);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_C2_M_MTE1_PONG);
        SetFlag<HardEvent::FIX_M>(EVENT_ID_C2_FIX_M);
    }

    __aicore__ inline void FreeEventID()
    {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_ID_V1_C1_BUFFER0); // V1.calc trigger C1.Fix
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_ID_V1_C1_BUFFER1);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_ID_V1_C1_BUFFER2);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_ID_V1_C1_BUFFER3);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_ID_V2_C2);     // V2.calc trigger C2.Fix
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_Q_MTE1_MTE2_PING); // C1
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_Q_MTE1_MTE2_PONG); // C1
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_K_MTE1_MTE2_PING);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_K_MTE1_MTE2_PONG);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_C1_M_MTE1_PING);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_C1_M_MTE1_PONG);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_C1_M_MTE1_Q);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID_C1_FIX_M_PING);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID_C1_FIX_M_PONG);

        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_V_MTE1_MTE2_PING); // C2
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_V_MTE1_MTE2_PONG);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_C2_M_MTE1_PING);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_C2_M_MTE1_PONG);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID_C2_FIX_M);
    }

    // copy query with full s1g
    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T>& dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX& runInfo)
    {
        constexpr uint32_t blockNumDtype = 32 / sizeof(Q_T);
        uint32_t nopeDealSize = dRealSize;
        uint32_t dstStride = (runInfo.actMSize + 15) >> 4 << 4;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmCoordS1Only gmCoord{.bIdx = runInfo.bIdx,
                              .n2Idx = runInfo.realN2Idx,
                              .s1Idx = runInfo.gS1Idx,
                              .dIdx = dOffset,
                              .s1DealSize = runInfo.actMSize,
                              .dDealSize = nopeDealSize};
        copyQueryGmToL1(l1Tensor, queryGm, gmCoord);
    }

    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T>& dstTensor, RunInfoX& runInfo)
    {
        CopyQuerySlice(dstTensor, 0, constInfo.dSize, runInfo);
    }

    // copy query scale with full s1g
    __aicore__ inline void CopyQueryScaleSlice(const LocalTensor<SCALE_T>& dstTensor, uint32_t dOffset,
                                               uint32_t dRealSize, RunInfoX& runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 15) >> 4 << 4;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.realN2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = dOffset,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = dRealSize};
        copyQueryScaleGmToL1(l1Tensor, queryScaleGm, gmCoord);
    }

    __aicore__ inline void CopyQueryScaleTile(const LocalTensor<SCALE_T>& dstTensor, RunInfoX& runInfo)
    {
        CopyQueryScaleSlice(dstTensor, 0, constInfo.dSize / HIF4_GROUP_SIZE, runInfo);
    }

    __aicore__ inline void CopyKeyAndScaleTile(const LocalTensor<KV_T>& keyL1Tensor,
                                               const LocalTensor<SCALE_T>& scaleL1Tensor, RunInfoX& runInfo,
                                               uint32_t s2RealSize, uint32_t subLoop)
    {
        const uint32_t D_SCALE = dBaseSize / HIF4_GROUP_SIZE;
        constexpr uint32_t HALF_SIZE_DIVISOR = 2;

        // 判断是否尾块-->runInfo.actSingleLoopS2Size
        bool isS2Align = s2RealSize % BASE_S2_C1_SIZE == 0;
        uint32_t s2Offset = (subLoop == 0) ? runInfo.s2Idx : runInfo.s2Idx + s2BaseSize / HALF_SIZE_DIVISOR;
        auto& keyOffsetCalculator = this->keyGm.offsetCalculator;
        auto& scaleOffsetCalculator = this->keyScaleGm.offsetCalculator;
        // MTE2 for Key
        Nd2NzParams nd2nzPara;
        Dn2NzParams dn2nzParams;

        uint32_t paraNdNum = s2RealSize / BASE_S2_C1_SIZE;
        if (likely(paraNdNum >= 1)) {
            nd2nzPara.ndNum = paraNdNum;        // 4;
            nd2nzPara.nValue = BASE_S2_C1_SIZE; // nd矩阵的行数
            nd2nzPara.dValue = dBaseSize / HALF_SIZE_DIVISOR;
            nd2nzPara.srcDValue = dBaseSize / HALF_SIZE_DIVISOR;
            nd2nzPara.dstNzC0Stride = BASE_S2_C1_SIZE;
            nd2nzPara.dstNzNStride = 1;
            nd2nzPara.srcNdMatrixStride = nd2nzPara.nValue * nd2nzPara.srcDValue;
            nd2nzPara.dstNzMatrixStride = nd2nzPara.nValue * nd2nzPara.srcDValue;
            uint64_t offset4Key = keyOffsetCalculator.GetOffset(runInfo.bIdx, runInfo.n2Idx, s2Offset, 0);
            DataCopy(keyL1Tensor, this->keyGm.gmTensor[offset4Key], nd2nzPara);
            // MTE2 for KeyScale
            dn2nzParams.dnNum = paraNdNum; // DN矩阵的个数，搬运4个基本块
            dn2nzParams.nValue = D_SCALE * 2; // 2: fp32伪装成bf16，2代表一个fp32类型可以占用2个L0最小分型的y方向长度
            dn2nzParams.dValue = BASE_S2_C1_SIZE; // 单个DN矩阵的实际行数D, 单位：元素个数
            dn2nzParams.srcDValue = dn2nzParams.nValue; // 同一个DN矩阵中相邻行起始地址之间的偏移, 单位：元素个数。
            dn2nzParams.dstNzC0Stride =
                dn2nzParams.nValue; // 来自源操作数同一列的多行数据相邻行起始地址间的偏移，单位：C0_SIZE（32B）。
            dn2nzParams.dstNzNStride =
                1; // 转换为NZ矩阵后，DN中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
            dn2nzParams.srcDnMatrixStride =
                dn2nzParams.nValue * dn2nzParams.dValue; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
            dn2nzParams.dstNzMatrixStride =
                dn2nzParams.srcDnMatrixStride; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素数量。
            LocalTensor<bfloat16_t> l1TensorCast = scaleL1Tensor.template ReinterpretCast<bfloat16_t>();
            uint64_t offset4Scale = scaleOffsetCalculator.GetOffset(runInfo.bIdx, runInfo.n2Idx, s2Offset, 0);
            GlobalTensor<bfloat16_t> gmTensorCast;
            gmTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t*)(this->keyScaleGm.gmTensor[offset4Scale].GetPhyAddr())));
            DataCopy(l1TensorCast, gmTensorCast, dn2nzParams);
        }

        // MTE2 for tail
        if (!isS2Align) {
            // key tail
            uint32_t tailS2 = s2RealSize - paraNdNum * BASE_S2_C1_SIZE;
            uint32_t tailS2Align16 = (tailS2 + 15) >> 4 << 4;
            nd2nzPara.ndNum = 1;       // ND矩阵的个数
            nd2nzPara.nValue = tailS2; // 单个ND矩阵的实际行数，单位为元素个数 62
            nd2nzPara.dValue =
                dBaseSize / HALF_SIZE_DIVISOR; // 单个ND矩阵的实际列数，单位为元素个数 2 是 hifp4 伪装 fp8
            nd2nzPara.srcDValue =
                dBaseSize / HALF_SIZE_DIVISOR; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
            nd2nzPara.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
            nd2nzPara.dstNzC0Stride = tailS2Align16; // tailS2跨度对齐到16，保持脏数据
            uint32_t s2OffsetForTail = s2Offset + paraNdNum * BASE_S2_C1_SIZE;
            uint64_t offset4Key = keyOffsetCalculator.GetOffset(runInfo.bIdx, runInfo.n2Idx, s2OffsetForTail, 0);
            DataCopy(keyL1Tensor[paraNdNum * BASE_S2_C1_SIZE * dBaseSize], this->keyGm.gmTensor[offset4Key],
                     nd2nzPara); // v121新增mte2 prefetch特性
            // key scale tail
            dn2nzParams.dnNum = 1;            // DN矩阵的个数
            dn2nzParams.nValue = D_SCALE * 2; // 2: fp32伪装成bf16。单个DN矩阵的列数N, 单位为元素个数
            dn2nzParams.dValue = tailS2;      // 不对齐，补零到64
            dn2nzParams.srcDValue =
                dn2nzParams.nValue; // 同一个DN矩阵中相邻行起始地址之间的偏移,
                                    // 单位为元素个数。2代表一个fp32类型可以占用2个L0最小分型的y方向长度
            dn2nzParams.dstNzNStride =
                1; // 转换为NZ矩阵后，DN中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
            dn2nzParams.dstNzC0Stride =
                dn2nzParams
                    .nValue; // 来自源操作数同一列的多行数据相邻行起始地址间的偏移。2代表一个fp32类型可以占用2个L0最小分型的y方向长度
            LocalTensor<bfloat16_t> l1TensorTailCast =
                scaleL1Tensor[paraNdNum * BASE_S2_C1_SIZE * D_SCALE].template ReinterpretCast<bfloat16_t>();
            uint64_t offset4Scale = scaleOffsetCalculator.GetOffset(runInfo.bIdx, runInfo.n2Idx, s2OffsetForTail, 0);
            GlobalTensor<bfloat16_t> gmTensorTailCast;
            gmTensorTailCast.SetGlobalBuffer(
                ((__gm__ bfloat16_t*)(this->keyScaleGm.gmTensor[offset4Scale].GetPhyAddr())));
            DataCopy(l1TensorTailCast, gmTensorTailCast, dn2nzParams);
        }
    }

    __aicore__ inline void CopyValueAndScaleTile(const LocalTensor<KV_T>& valueL1Tensor,
                                                 const LocalTensor<SCALE_T>& scaleL1Tensor, RunInfoX& runInfo)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        bool isS2Align = s2RealSize % BASE_K_BMM2 == 0;
        uint32_t vScaleS2 = (this->constInfo.s2Size + HIF4_GROUP_SIZE - 1) / HIF4_GROUP_SIZE;
        Nd2NzParams gm2L1Nd2NzParams;
        Dn2NzParams dn2nzParams;

        constexpr uint32_t HALF_SIZE_DIVISOR = 2;
        uint32_t paraNdNum = s2RealSize / BASE_K_BMM2;
        if (likely(paraNdNum >= 1)) {
            gm2L1Nd2NzParams.ndNum = paraNdNum;    // ND矩阵的个数
            gm2L1Nd2NzParams.nValue = BASE_K_BMM2; // 单个ND矩阵的实际行数，单位为元素个数
            gm2L1Nd2NzParams.dValue =
                dBaseSize / HALF_SIZE_DIVISOR; // 单个ND矩阵的实际列数，单位为元素个数 2 是 hifp4 伪装 fp8
            gm2L1Nd2NzParams.srcDValue =
                dBaseSize / HALF_SIZE_DIVISOR; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
            gm2L1Nd2NzParams.dstNzC0Stride = BASE_K_BMM2; // NZ矩阵相邻Block起始地址之间的偏移， 单位为Block个数
            gm2L1Nd2NzParams.dstNzNStride =
                1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
            gm2L1Nd2NzParams.srcNdMatrixStride =
                gm2L1Nd2NzParams.nValue * gm2L1Nd2NzParams.dValue; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
            gm2L1Nd2NzParams.dstNzMatrixStride =
                gm2L1Nd2NzParams.nValue * gm2L1Nd2NzParams.dValue; // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
            auto& valueOffsetCalculator = this->valueGm.offsetCalculator;
            int64_t offset4Value =
                runInfo.bIdx * this->constInfo.n2Size * this->constInfo.s2Size * this->constInfo.dSize +
                runInfo.n2Idx * this->constInfo.s2Size * this->constInfo.dSize + runInfo.s2Idx * this->constInfo.dSize;
            DataCopy(valueL1Tensor, this->valueGm.gmTensor[offset4Value],
                     gm2L1Nd2NzParams); // v121新增mte2 prefetch特性

            dn2nzParams.dnNum = paraNdNum; // DN矩阵的个数
            dn2nzParams.nValue = BASE_K_BMM2 / HIF4_GROUP_SIZE *
                                 2; // 2: fp32伪装成bf16，2代表一个fp32类型可以占用2个L0最小分型的y方向长度
            dn2nzParams.dValue = dBaseSize;       // 单个DN矩阵的实际行数D, 单位为元素个数
            dn2nzParams.srcDValue = vScaleS2 * 2; // 同一个DN矩阵中相邻行起始地址之间的偏移,
                                                  // 单位为元素个数。2代表一个fp32类型可以占用2个L0最小分型的y方向长度
            dn2nzParams.dstNzC0Stride =
                dn2nzParams
                    .nValue; // 来自源操作数同一列的多行数据相邻行起始地址间的偏移。2代表一个fp32类型可以占用2个L0最小分型的y方向长度
            dn2nzParams.dstNzNStride =
                1; // 转换为NZ矩阵后，DN中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
            dn2nzParams.srcDnMatrixStride = dn2nzParams.nValue; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
            dn2nzParams.dstNzMatrixStride =
                dn2nzParams.nValue *
                dn2nzParams.dValue; // 两个NZ矩阵，起始地址之间的偏移,
                                    // 单位为元素数量。2代表一个fp32类型可以占用2个L0最小分型的y方向长度
            LocalTensor<bfloat16_t> l1TensorCast = scaleL1Tensor.template ReinterpretCast<bfloat16_t>();
            auto& scaleOffsetCalculator = this->valueScaleGm.offsetCalculator;
            int64_t offset4Scale = runInfo.bIdx * this->constInfo.n2Size * this->constInfo.dSize * vScaleS2 +
                                   runInfo.n2Idx * this->constInfo.dSize * vScaleS2 + runInfo.s2Idx / HIF4_GROUP_SIZE;
            GlobalTensor<bfloat16_t> gmTensorCast;
            gmTensorCast.SetGlobalBuffer(
                ((__gm__ bfloat16_t*)(this->valueScaleGm.gmTensor[offset4Scale].GetPhyAddr())));
            DataCopy(l1TensorCast, gmTensorCast, dn2nzParams);
        }

        if (!isS2Align) {
            uint32_t tailS2 = s2RealSize - paraNdNum * BASE_K_BMM2;
            uint32_t tailS2Align64 = (tailS2 + 63) >> 6 << 6; // 64个hif4对应一个quant scale
            InitConstValueParams<uint16_t> initConstParam;
            // S2方向上补零
            initConstParam.repeatTimes = 1;
            initConstParam.blockNum = tailS2Align64 * (dBaseSize / HALF_SIZE_DIVISOR) / 32; // 单位32B
            initConstParam.dstGap = 0;
            initConstParam.initValue = (uint16_t)0x0000;
            InitConstValue(valueL1Tensor[paraNdNum * BASE_K_BMM2 * dBaseSize].template ReinterpretCast<uint16_t>(),
                           initConstParam); // 在mte2上

            PipeBarrier<PIPE_MTE2>();
            gm2L1Nd2NzParams.ndNum = 1;       // ND矩阵的个数
            gm2L1Nd2NzParams.nValue = tailS2; // 单个ND矩阵的实际行数，单位为元素个数
            gm2L1Nd2NzParams.dValue =
                dBaseSize / HALF_SIZE_DIVISOR; // 单个ND矩阵的实际列数，单位为元素个数 2 是 hifp4 伪装 fp8
            gm2L1Nd2NzParams.srcDValue =
                dBaseSize / HALF_SIZE_DIVISOR; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
            gm2L1Nd2NzParams.dstNzC0Stride = tailS2Align64; // NZ矩阵相邻Block起始地址之间的偏移， 单位为Block个数
            gm2L1Nd2NzParams.dstNzNStride =
                1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
            uint32_t s2OffsetForTail = runInfo.s2Idx + paraNdNum * BASE_K_BMM2;
            int64_t offset4Value =
                runInfo.bIdx * this->constInfo.n2Size * this->constInfo.s2Size * this->constInfo.dSize +
                runInfo.n2Idx * this->constInfo.s2Size * this->constInfo.dSize +
                s2OffsetForTail * this->constInfo.dSize;
            DataCopy(valueL1Tensor[paraNdNum * BASE_K_BMM2 * dBaseSize], this->valueGm.gmTensor[offset4Value],
                     gm2L1Nd2NzParams, true); // v121新增mte2 prefetch特性

            dn2nzParams.dnNum = 1; // DN矩阵的个数
            dn2nzParams.nValue = tailS2Align64 / HIF4_GROUP_SIZE *
                                 2; // 2: fp32伪装成bf16，2代表一个fp32类型可以占用2个L0最小分型的y方向长度
            dn2nzParams.dValue = dBaseSize;       // 单个DN矩阵的实际行数D, 单位为元素个数
            dn2nzParams.srcDValue = vScaleS2 * 2; // 同一个DN矩阵中相邻行起始地址之间的偏移,
                                                  // 单位为元素个数。2代表一个fp32类型可以占用2个L0最小分型的y方向长度
            dn2nzParams.dstNzC0Stride =
                dn2nzParams
                    .nValue; // 来自源操作数同一列的多行数据相邻行起始地址间的偏移。2代表一个fp32类型可以占用2个L0最小分型的y方向长度
            dn2nzParams.dstNzNStride =
                1; // 转换为NZ矩阵后，DN中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
            LocalTensor<bfloat16_t> l1TensorTailCast =
                scaleL1Tensor[paraNdNum * dBaseSize * (BASE_K_BMM2 / HIF4_GROUP_SIZE)]
                    .template ReinterpretCast<bfloat16_t>();
            int64_t offset4Scale = runInfo.bIdx * this->constInfo.n2Size * this->constInfo.dSize * vScaleS2 +
                                   runInfo.n2Idx * this->constInfo.dSize * vScaleS2 + s2OffsetForTail / HIF4_GROUP_SIZE;
            GlobalTensor<bfloat16_t> gmTensorTailCast;
            gmTensorTailCast.SetGlobalBuffer(
                ((__gm__ bfloat16_t*)(this->valueScaleGm.gmTensor[offset4Scale].GetPhyAddr())));
            DataCopy(l1TensorTailCast, gmTensorTailCast, dn2nzParams);
        }
    }

    __aicore__ inline void IterateBmm1(RunInfoX& runInfo, uint32_t subLoop)
    {
        IterateBmm1Dn(runInfo, subLoop);
    }

    /* 针对S1Base=128, S2Base = 256, D = 128场景, L1全载/L0全载, K * Q^T, Q矩阵驻留. GS1>80, S=S1*S2 */
    __aicore__ inline void IterateBmm1Dn(RunInfoX& runInfo, uint32_t subLoop)
    {
        uint32_t c1v1Loop = CeilDiv(runInfo.actSingleLoopS2Size, BASE_S2_C1_SIZE);
        static constexpr uint32_t DATACOPY_BASE_4 = 4; // 每次搬运4个小基本块
        if (subLoop % DATACOPY_BASE_4 == 0) {
            uint32_t s2DataCopySize = DATACOPY_BASE_4 * BASE_S2_C1_SIZE;
            if (subLoop == 0 && c1v1Loop <= DATACOPY_BASE_4) {
                s2DataCopySize = runInfo.actSingleLoopS2Size;
            }
            if (subLoop == DATACOPY_BASE_4) {
                s2DataCopySize = runInfo.actSingleLoopS2Size - DATACOPY_BASE_4 * BASE_S2_C1_SIZE;
            }
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_K_MTE1_MTE2_PING + (subLoop / DATACOPY_BASE_4) % DB);
            CopyKeyAndScaleTile(KTensor[subLoop].template ReinterpretCast<KV_T>(),
                                KscaleTensor[subLoop].template ReinterpretCast<SCALE_T>(), runInfo, s2DataCopySize,
                                subLoop);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID_K_MTE2_MTE1_PING + (subLoop / DATACOPY_BASE_4) % DB);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID_K_MTE2_MTE1_PING + (subLoop / DATACOPY_BASE_4) % DB);
        }
        uint32_t s2CalcSize = BASE_S2_C1_SIZE;
        if (unlikely(subLoop == c1v1Loop - 1)) {
            s2CalcSize = runInfo.actSingleLoopS2Size - (c1v1Loop - 1) * BASE_S2_C1_SIZE;
        }
        LocalTensor<INPUT_T> mmKL0ATensor = mmL0ATensor.template ReinterpretCast<INPUT_T>();
        mmKL0ATensor = mmKL0ATensor[K_L0A_SIZE * (subLoop % DB)];
        if (unlikely(runInfo.isFirstS2Loop && subLoop == 0)) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_Q_MTE1_MTE2_PING + runInfo.mloop % DB);
            CopyQueryTile(QTensor[runInfo.mloop % DB].template ReinterpretCast<Q_T>(), runInfo);
            CopyQueryScaleTile(QscaleTensor[runInfo.mloop % DB].template ReinterpretCast<SCALE_T>(), runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID_Q_MTE2_MTE1_PING + runInfo.mloop % DB);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID_Q_MTE2_MTE1_PING + runInfo.mloop % DB);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID_C1_M_MTE1_Q); // souter之间的Q mte1等cube

            MatmulMxHiF4Bmm1First<INPUT_T, INPUT_T, float, bfloat16_t, bfloat16_t, BASE_S2_C1_SIZE, mBaseSize,
                                  dBaseSize>(KTensor[subLoop].template ReinterpretCast<INPUT_T>(),
                                             QTensor[runInfo.mloop % DB].template ReinterpretCast<INPUT_T>(),
                                             KscaleTensor[subLoop].template ReinterpretCast<bfloat16_t>(),
                                             QscaleTensor[runInfo.mloop % DB].template ReinterpretCast<bfloat16_t>(),
                                             mmKL0ATensor, mmQL0BTensor, mm1L0CTensors[subLoop % DB], subLoop,
                                             runInfo.actMSize, s2CalcSize);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_Q_MTE1_MTE2_PING + runInfo.mloop % DB);
        } else {
            MatmulMxHiF4Bmm1<INPUT_T, INPUT_T, float, bfloat16_t, bfloat16_t, BASE_S2_C1_SIZE, mBaseSize, dBaseSize>(
                KTensor[subLoop].template ReinterpretCast<INPUT_T>(),
                QTensor[runInfo.mloop % DB].template ReinterpretCast<INPUT_T>(),
                KscaleTensor[subLoop].template ReinterpretCast<bfloat16_t>(),
                QscaleTensor[runInfo.mloop % DB].template ReinterpretCast<bfloat16_t>(), mmKL0ATensor, mmQL0BTensor,
                mm1L0CTensors[subLoop % DB], subLoop, runInfo.actMSize, s2CalcSize);
        }
        if (subLoop % DATACOPY_BASE_4 == 0) {
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_K_MTE1_MTE2_PING + (subLoop / DATACOPY_BASE_4) % DB);
        }
        if (runInfo.isLastS2Loop && subLoop == c1v1Loop - 1) {
            SetFlag<HardEvent::M_MTE1>(EVENT_ID_C1_M_MTE1_Q);
        }
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_ID_V1_C1_BUFFER0 + subLoop % BUFFER_CNT4); // V1.calc trigger C1.Fix
        WaitFlag<HardEvent::M_FIX>(EVENT_ID_C1_M_FIX_PING + subLoop % DB);
        FixpipeMm1Dn(mm1ResUB[subLoop % BUFFER_CNT4 * mBaseSize], mm1L0CTensors[subLoop % DB], runInfo.actMSize,
                     s2CalcSize);
        SetFlag<HardEvent::FIX_M>(EVENT_ID_C1_FIX_M_PING + subLoop % DB);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(FLAG_ID_C1_V1); // C1.Fix trigger V1.calc
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<half>& dstTensor, const LocalTensor<float>& l0C,
                                        uint32_t s1RealSize, uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize =
            (s1RealSize + 15) >> 4 << 4; // 15: 使能NZ2ND功能, 4: 2^4=16, nSize*sizeof(bf16)必须为32的倍数
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (s2RealSize + 7) >> 3 << 3; // 7: 对齐到8element, 3: 2^3=8
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = (s2RealSize + 15) >> 4 << 4; // 15: 对齐到16, 4: 2^4=16
        const uint32_t UB_BANK_STRIDE = 1024; // 只写bank0，stride=4*8*32=1024B，len=8*32=256B，loopNum=128
        fixpipeParams.dstStride = UB_BANK_STRIDE / sizeof(half); // 单位element
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        fixpipeParams.dualDstCtl = 0; // 单目标模式
        fixpipeParams.unitFlag = 0;
        fixpipeParams.quantPre = QuantMode_t::QF322F16_PRE;
        float deScaleValue = 0.08837890625f; // 根号d=128分之一
        fixpipeParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&deScaleValue));
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<half, float, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2(RunInfoX& runInfo)
    {
        IterateBmm2Mxfp8Hif4(runInfo);
    }

    __aicore__ inline void IterateBmm2Mxfp8Hif4(RunInfoX& runInfo)
    {
        // MXFP_GROUP_SIZE
        // MTE2在加载时先假设一次性加载V.shape[BASE_S2_C2 = 1024, D]
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_V_MTE1_MTE2_PING + (runInfo.loop % DB));
        CopyValueAndScaleTile(VTensor[runInfo.loop % DB].template ReinterpretCast<KV_T>(),
                              VscaleTensor[runInfo.loop % DB].template ReinterpretCast<SCALE_T>(), runInfo);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID_V_MTE2_MTE1_PING + (runInfo.loop % DB));
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID_V_MTE2_MTE1_PING + (runInfo.loop % DB));
        WaitFlag<HardEvent::FIX_M>(EVENT_ID_C2_FIX_M);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(FLAG_ID_V1_C2_PING + (runInfo.loop % DB)); // V1.MTE3 trigger C2.MTE1

        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        uint32_t basekCnt = s2RealSize / BASE_K_BMM2;
        uint32_t pTensorIdx = 8 * (runInfo.loop % DB);
        uint32_t pScaleIdx = runInfo.loop % DB;
        static constexpr uint32_t V_L1_BASE_SIZE = BASE_K_BMM2 * dBaseSize; // 256*128=32k
        static constexpr uint32_t VSCALE_L1_BASE_SIZE =
            V_L1_BASE_SIZE / HIF4_GROUP_SIZE * 2;            // 2: fp32伪装成bf16。(128*256/64)*2=1k
        for (uint32_t kIdx = 0; kIdx < (basekCnt); kIdx++) { // 4次mmad
            LocalTensor<mx_fp8_e4m3_t> mmPL0ATensor = mmL0ATensor.template ReinterpretCast<mx_fp8_e4m3_t>();
            mmPL0ATensor = mmPL0ATensor[P_L0A_SIZE * ((kIdx + 1) % DB)];
            LocalTensor<INPUT_T> mmVL1Tensor = VTensor[runInfo.loop % DB].template ReinterpretCast<INPUT_T>();
            mmVL1Tensor = mmVL1Tensor[V_L1_BASE_SIZE * kIdx];
            LocalTensor<bfloat16_t> mmVscaleL1Tensor =
                VscaleTensor[runInfo.loop % DB].template ReinterpretCast<bfloat16_t>();
            mmVscaleL1Tensor = mmVscaleL1Tensor[VSCALE_L1_BASE_SIZE * kIdx];
            // MMAD
            MatmulMxfp8HiF4Bmm2<fp8_e4m3fn_t, INPUT_T, float, fp8_e8m0_t, bfloat16_t, mBaseSize, dBaseSize,
                                BASE_K_BMM2>(
                this->PTensor[pTensorIdx][mBaseSize * BASE_K_BMM2 * kIdx].template ReinterpretCast<fp8_e4m3fn_t>(),
                mmVL1Tensor,
                this->PscaleTensor[pScaleIdx][mBaseSize * BASE_K_BMM2 / MXFP_GROUP_SIZE * kIdx]
                    .template ReinterpretCast<fp8_e8m0_t>(),
                mmVscaleL1Tensor, mmPL0ATensor, mmVL0BTensors[kIdx % DB], mm2L0CTensor, kIdx);
        }
        if (s2RealSize % BASE_K_BMM2 != 0) {
            uint32_t tailS2 = s2RealSize - basekCnt * BASE_K_BMM2;
            uint32_t tailS2Align64 = (tailS2 + 63) >> 6 << 6; // 64个hif4对应一个quant scale
            LocalTensor<mx_fp8_e4m3_t> mmPL0ATensor = mmL0ATensor.template ReinterpretCast<mx_fp8_e4m3_t>();
            mmPL0ATensor = mmPL0ATensor[P_L0A_SIZE * ((basekCnt + 1) % DB)];
            LocalTensor<INPUT_T> mmVL1Tensor = VTensor[runInfo.loop % DB].template ReinterpretCast<INPUT_T>();
            mmVL1Tensor = mmVL1Tensor[V_L1_BASE_SIZE * basekCnt];
            LocalTensor<bfloat16_t> mmVscaleL1Tensor =
                VscaleTensor[runInfo.loop % DB].template ReinterpretCast<bfloat16_t>();
            mmVscaleL1Tensor = mmVscaleL1Tensor[VSCALE_L1_BASE_SIZE * basekCnt];
            MatmulMxfp8HiF4Bmm2Tail<fp8_e4m3fn_t, INPUT_T, float, fp8_e8m0_t, bfloat16_t, mBaseSize, dBaseSize,
                                    BASE_K_BMM2>(
                this->PTensor[pTensorIdx][mBaseSize * BASE_K_BMM2 * basekCnt].template ReinterpretCast<fp8_e4m3fn_t>(),
                mmVL1Tensor,
                this->PscaleTensor[pScaleIdx][mBaseSize * BASE_K_BMM2 / MXFP_GROUP_SIZE * basekCnt]
                    .template ReinterpretCast<fp8_e8m0_t>(),
                mmVscaleL1Tensor, mmPL0ATensor, this->mmVL0BTensors[basekCnt % DB], this->mm2L0CTensor,
                this->RowSumFullOneVandScale, basekCnt, tailS2Align64);
        }
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_V_MTE1_MTE2_PING + (runInfo.loop % DB));
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(FLAG_ID_C2_V1_PING + (runInfo.loop % DB)); // C2.MTE1 trigger V1.MTE3
        SetFlag<HardEvent::M_FIX>(EVENT_ID_C2_M_FIX);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID_C2_M_FIX);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_ID_V2_C2); // V2.calc trigger C2.Fix
        FixpipeMm2Mxfp8Hif4(this->mm2ResUB, this->rowSumNewUB, this->mm2L0CTensor, runInfo);
        SetFlag<HardEvent::FIX_M>(EVENT_ID_C2_FIX_M);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(FLAG_ID_C2_V2); // C2.Fix trigger V2.calc
    }

    __aicore__ inline void FixpipeMm2Mxfp8Hif4(const LocalTensor<float>& mm2ResTensor,
                                               const LocalTensor<float>& rowSumNewTensor, const LocalTensor<float>& l0C,
                                               RunInfoX& runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParamsND; // 使能NZ2ND，输出数据格式为ND格式
        fixpipeParamsND.nSize = dBaseSize;                       // 单位element stride=1024 len=128B
        fixpipeParamsND.mSize = mBaseSize;                       // 单位element
        fixpipeParamsND.srcStride = fixpipeParamsND.mSize;       // 单位C0_size
        fixpipeParamsND.dstStride = fixpipeParamsND.nSize;       // 单位element
        fixpipeParamsND.unitFlag = 0;
        Fixpipe<float, float, BMM2_FIXPIPE_CONFIG>(mm2ResTensor, l0C, fixpipeParamsND);

        FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fixpipeParamsDN;
        fixpipeParamsDN.nSize = 8; // 8: 使能channelSplit功能，nSize必须为8的倍数
        fixpipeParamsDN.mSize = mBaseSize;
        fixpipeParamsDN.srcStride = fixpipeParamsDN.mSize;
        fixpipeParamsDN.dstStride = fixpipeParamsDN.mSize;
        fixpipeParamsDN.params.dnNum = 1;
        fixpipeParamsDN.params.srcNzMatrixStride = 0;
        fixpipeParamsDN.params.dstDnMatrixStride = 0;
        fixpipeParamsDN.params.srcNzC0Stride = 1; // 解决：loop0_src_std should not be zero when nz2dn transform.
        fixpipeParamsDN.dualDstCtl = 0;           // 单目标模式
        fixpipeParamsDN.unitFlag = 0;
        Fixpipe<float, float, PFA_CFG_COLUMN_MAJOR_UB>(rowSumNewTensor, l0C[mBaseSize * dBaseSize], fixpipeParamsDN);
    }
}; // FAFullQuantMxBlockCube

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false>
class QuantFlashAttnBlockCubeHifp4Dummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool BMM2_TOUB = true;
    static constexpr bool USE_DN = useDn;

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using mm2ResPos = typename std::conditional<BMM2_TOUB, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;

    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockCubeHifp4Dummy(ConstInfoX& constInfo){};
};
} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_BLOCK_CUBE_HIFP4_H_
