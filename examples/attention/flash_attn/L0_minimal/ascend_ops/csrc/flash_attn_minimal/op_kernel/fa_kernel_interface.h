/*!
 * \file fa_kernel_interface.h
 * \brief Simplified FA kernel interface — no mask, no combine.
 */

#ifndef FA_KERNEL_INTERFACE_H
#define FA_KERNEL_INTERFACE_H

#include "kernel/fa_block_vec.h"
#include "kernel/fa_block_cube.h"

template <uint32_t M_BASE = 128, uint32_t N_BASE = 128, uint32_t D_SIZE = 128>
__global__ __aicore__ void FaKernel(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, uint32_t blockDim, float softmaxScale,
    uint32_t B, uint32_t N1, uint32_t N2, uint32_t S1, uint32_t S2)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    #ifdef __DAV_C310_CUBE__
        CubeFunc<M_BASE, N_BASE, D_SIZE>(query, key, value, workspace, nullptr, blockDim, B, N1, N2, S1, S2);
    #else
        VectorFunc<M_BASE, N_BASE, D_SIZE>(query, key, value, attentionOut, workspace, nullptr, blockDim, softmaxScale, B, N1, N2, S1, S2);
    #endif
}

#endif  // FA_KERNEL_INTERFACE_H
