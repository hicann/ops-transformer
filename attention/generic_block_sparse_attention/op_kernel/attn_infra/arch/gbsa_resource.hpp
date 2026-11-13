/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RESOURCE_HPP
#define RESOURCE_HPP

#include "../../attn_infra/gbsa_base_defs.hpp"
#include "../../attn_infra/arch/gbsa_local_tensor_buffer.hpp"

namespace NpuArch::Arch {

template <class ArchTag>
struct Resource {};

template <>
struct Resource<Arch::AtlasA5> {
public:
    LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::C2> btBuf;
    LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::VECCALC> ubBuf;
    LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::C2PIPE2GM> fpBuf;

    __aicore__ inline Resource()
    {
        AscendC::InitSocState();
    }

    __aicore__ inline ~Resource()
    {
        AscendC::InitSocState();
    }
};

template <>
struct Resource<Arch::AtlasA2> {
public:
    AscendC::TPipe pipe;

    LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::C2> btBuf;
    LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::VECCALC> ubBuf;

    __aicore__ inline Resource()
    {
        pipe.Destroy();
    }
};

template <AscendC::TPosition Position, uint32_t Bytes>
struct Int8StaticBuffer {
    template <class Element = half>
    __aicore__ inline AscendC::LocalTensor<Element> GetBufferByByte(uint32_t offset) const
    {
        return AscendC::LocalTensor<uint8_t>(Position, 0, Bytes)[offset].template ReinterpretCast<Element>();
    }
};
struct Int8StaticResource {
    Int8StaticBuffer<AscendC::TPosition::A1, AtlasA2::L1_SIZE> l1Buf;
    Int8StaticBuffer<AscendC::TPosition::A2, AtlasA2::L0A_SIZE> l0ABuf;
    Int8StaticBuffer<AscendC::TPosition::B2, AtlasA2::L0B_SIZE> l0BBuf;
    Int8StaticBuffer<AscendC::TPosition::C2, AtlasA2::BIAS_SIZE> btBuf;
    Int8StaticBuffer<AscendC::TPosition::CO1, AtlasA2::L0C_SIZE> l0CBuf;
    Int8StaticBuffer<AscendC::TPosition::VECCALC, AtlasA2::UB_SIZE> ubBuf;
    __aicore__ inline Int8StaticResource()
    {
        AscendC::InitSocState();
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

} // namespace NpuArch::Arch

#endif // INCLUDE_ARCH_RESOURCE_HPP
