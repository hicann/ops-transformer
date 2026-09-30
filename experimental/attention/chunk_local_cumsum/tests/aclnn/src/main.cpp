/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file main.cpp
 */
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>

#include "acl/acl.h"
#include "common.h"
#include "math.h"
#include "chunk_local_cumsum.h"
#include "tensor.h"
#include "op_runner.h"

bool g_isDevice = false;
int deviceId = 7;

template <typename T>
bool TestTime(T& opRunner, aclrtStream& stream)
{
    std::vector<int64_t> seqIdxDict{0, 1}, curlenDict{567, 567};
    int32_t warmup = 512;
    uint32_t ret;
    for (int32_t i = 0; i < warmup; ++i) {
        // Bug(@xh):
        // 不加InitWorkspace不能二次运行，每次调用aclnnLayerNormCustom之前都需要调用InitWorkspace
        opRunner.Forward(stream);
    }

    auto start = std::chrono::high_resolution_clock::now();
    int repeat = 1024;
    for (int i = 0; i < repeat; ++i) {
        opRunner.Forward(stream);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("Time measured: %.6f ms.\n", elapsed.count() * 1e-6 / repeat);
    return true;
}

bool CreateStream(aclrtStream& stream)
{
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        ERROR_LOG("Create stream failed");
        return false;
    }
    INFO_LOG("Create stream success");
    return true;
}

bool DestroyStream(aclrtStream& stream)
{
    INFO_LOG("Destroystream()");
    (void)aclrtDestroyStream(stream);
    INFO_LOG("Destroystream()");
    return true;
}

void DestoryResource()
{
    INFO_LOG("DestroyResource()");
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destory resource failed");
    } else {
        INFO_LOG("Destory resource success");
    }
}

bool InitResource()
{
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        } else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    // acl.json is dump or profiling config file
    if (aclInit(NULL) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestoryResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

int AlignUp16(int a)
{
    return (a + 15) & (~15);
}
bool RunOp(const std::vector<int64_t>& cuSeqlen, int32_t head_num, int32_t chunk_size)
{
    auto print = [](std::vector<int64_t> dict) {
        for (auto el : dict) {
            std::cout << el << " ";
        }
        std::cout << "\n";
    };

    aclrtStream stream = nullptr;
    CreateStream(stream);

    int32_t tokenNum = cuSeqlen.back();
    int32_t seqNum = cuSeqlen.size() - 1;

    Tensor* input = new Tensor(tokenNum * head_num * sizeof(float));
    Tensor* cu_seqlens = new Tensor((seqNum + 1) * sizeof(int64_t));
    Tensor* output = new Tensor(tokenNum * head_num * sizeof(float));

    input->FromFile("../input/input.bin");
    cu_seqlens->FromFile("../input/input_cu_seqlens.bin");

    std::cout << "Read data complete!" << std::endl;

    size_t workspaceSize = 222222222;
    void* workspace = nullptr;
    if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
        ERROR_LOG("Malloc device memory failed");
        return false;
    }
    // aclrtSynchronizeStream(stream);

    auto start = std::chrono::high_resolution_clock::now();
    // Run op
    ChunkLocalCumsum(stream, workspace, *input, *cu_seqlens, *output, head_num, chunk_size, cuSeqlen);

    const char* tmp_err_msg = NULL;
    tmp_err_msg = aclGetRecentErrMsg();
    if (tmp_err_msg != NULL) {
        printf(" ERROR Message : %s \n ", tmp_err_msg);
    }

    aclrtSynchronizeStream(stream);
    INFO_LOG("Run op matmul success");
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("Time measured: %.6f ms.\n", elapsed.count() * 1e-6);

    // testTime
    {
        int32_t warmup = 50;
        for (int32_t i = 0; i < warmup; ++i) {
            ChunkLocalCumsum(stream, workspace, *input, *cu_seqlens, *output, head_num, chunk_size, cuSeqlen);
        }
        aclrtSynchronizeStream(stream);

        start = std::chrono::high_resolution_clock::now();
        int repeat = 100;
        for (int i = 0; i < repeat; ++i) {
            ChunkLocalCumsum(stream, workspace, *input, *cu_seqlens, *output, head_num, chunk_size, cuSeqlen);
        }
        aclrtSynchronizeStream(stream);

        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        printf("Time measured: %.6f ms.\n", elapsed.count() * 1e-6 / repeat);
    }

    output->ToFile("../output/output1.bin");
    DestroyStream(stream);

    return true;
}

std::vector<int64_t> Stringsplit(const std::string& str, const char split)
{
    std::vector<int64_t> res;

    if (str == "")
        return {};
    // 在字符串末尾也加入分隔符，方便截取最后一段
    std::string strs = str + split;
    size_t pos = strs.find(split);

    // 若找不到内容则字符串搜索函数返回 npos
    while (pos != strs.npos) {
        std::string temp = strs.substr(0, pos);
        res.push_back(stoi(temp));
        // 去掉已分割的字符串,在剩下的字符串中进行分割
        strs = strs.substr(pos + 1, strs.size());
        pos = strs.find(split);
    }
    return res;
}

int main(int argc, char** argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    std::string curLen = std::string(argv[1]);
    std::vector<int64_t> cuSeqlen = Stringsplit(curLen, ',');
    int32_t headNum = argc > 2 ? std::stoi(argv[2]) : 8;
    int32_t chunkSize = argc > 3 ? std::stoi(argv[3]) : 64;

    if (!RunOp(cuSeqlen, headNum, chunkSize)) {
        DestoryResource();
        return FAILED;
    }

    DestoryResource();

    return SUCCESS;
}
