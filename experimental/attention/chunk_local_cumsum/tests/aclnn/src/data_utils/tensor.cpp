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
 * @file tensor.cpp
 */
#include "tensor.h"

#include "common.h"

Tensor::Tensor(size_t capacity)
    : capacity_(capacity),
      size_(capacity),
      host_(nullptr),
      device_(nullptr)
{
    if (aclrtMalloc((void**)&device_, capacity, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
        // TODO(@xh): 有必要的话用宏函数封装一下
        ERROR_LOG("Malloc device memory failed");
    }

    if (aclrtMallocHost((void**)&host_, capacity) != ACL_SUCCESS) {
        ERROR_LOG("Malloc host memory failed");
    }
}

Tensor::~Tensor()
{
    (void)aclrtFree(device_);
    (void)aclrtFreeHost(host_);
}

bool Tensor::FromFile(const std::string& fileName)
{
    if (fileName != "NONE")
        ReadFile(fileName, size_, host_, capacity_);
    if (aclrtMemcpy(device_, size_, host_, size_, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        ERROR_LOG("Copy host to device failed");
    }
    return true;
}

bool Tensor::ToFile(const std::string& fileName)
{
    if (aclrtMemcpy(host_, size_, device_, size_, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        ERROR_LOG("Copy device to host failed %lu", size_);
        const char* tmp_err_msg = NULL;
        tmp_err_msg = aclGetRecentErrMsg();
        if (tmp_err_msg != NULL) {
            printf(" ERROR Message : %s \n ", tmp_err_msg);
        }
        return false;
    }
    if (fileName != "NONE")
        WriteFile(fileName, host_, size_);
    return true;
}
