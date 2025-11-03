#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "distribute_barrier_tiling_def.h"
#include "../../../op_kernel/moe_distribute_base.h"

extern "C" __global__ __aicore__ void distribute_barrier(GM_ADDR xRef, GM_ADDR timeOut, GM_ADDR elasticInfo,
                                                         GM_ADDR xRefOut, GM_ADDR workspaceGM, GM_ADDR tilingGM);

extern uint8_t* g_hcclContextReserved[2];

class distribute_barrier_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        size_t ctxSize = sizeof(HcclOpResParam);
        g_hcclContextReserved[0] = (uint8_t*)AscendC::GmAlloc(ctxSize);
        std::cout << "DistributeBarrierInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        AscendC::GmFree((void*)g_hcclContextReserved[0]);
        std::cout << "DistributeBarrierInfershape TearDown" << std::endl;
    }
};

TEST_F(distribute_barrier_test, distribute_barrier_test_10000) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(DistributeBarrierTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    auto winContext = (__gm__ HcclOpResParam *) AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
    winContext->localWindowsExp = (uint64_t)workspace;

    DistributeBarrierTilingData *tiling_data = reinterpret_cast<DistributeBarrierTilingData*>(tiling);
    tiling_data->distributeBarrierInfo.worldSize = 16;
    tiling_data->distributeBarrierInfo.rankId = 0;
    tiling_data->distributeBarrierInfo.aivNum = 48;

    uint8_t *xRef = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));
    uint8_t *xRefOut = (uint8_t *)AscendC::GmAlloc(1024 * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(100000);
    ICPU_RUN_KF(distribute_barrier, 48, xRef, nullptr, nullptr, xRefOut, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xRef);
    AscendC::GmFree((void*)xRefOut);
}