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
 * \file test_nsa_compress_attention_infer_2.cpp
 * \brief
 */


#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_nsa_compress_attention_infer.h"

using namespace std;

#define CHECK_RET(cond, return_expr) \
	do {                               \
		if (!(cond)) {                   \
		return_expr;                   \
		}                                \
	} while (0)

#define LOG_PRINT(message, ...)     \
	do {                              \
		printf(message, ##__VA_ARGS__); \
	} while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
	int64_t shapeSize = 1;
	for (auto i : shape) {
		shapeSize *= i;
	}
	return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
	// 固定写法，AscendCL初始化
	auto ret = aclInit(nullptr);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
	ret = aclrtSetDevice(deviceId);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
	ret = aclrtCreateStream(stream);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
	return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
	auto size = GetShapeSize(shape) * sizeof(T);
	// 调用aclrtMalloc申请device侧内存
	auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
	// 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
	ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
	
	// 计算连续tensor的strides
	std::vector<int64_t> strides(shape.size(), 1);
	for (int64_t i = shape.size() - 2; i >= 0; i--) {
		strides[i] = shape[i + 1] * strides[i + 1];
	}

	// 调用aclCreateTensor接口创建aclTensor
	*tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
								shape.data(), shape.size(), *deviceAddr);
	return 0;
}

int main() {
	// 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
	// 根据自己的实际device填写deviceId
	int32_t deviceId = 0;
	aclrtStream stream;
	auto ret = Init(deviceId, &stream);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

	// 2. 构造输入与输出，需要根据API的接口自定义构造
	int32_t batchSize = 20;
	int32_t headDimsQK = 192;
	int32_t blockNum = 640;
	int32_t headDimsV = 128;
	int32_t sequenceLengthK = 4096;
	int32_t maxNumBlocksPerSeq = 32;
	// attr
    int64_t numHeads = 64;
    int64_t numKeyValueHeads = 4;
    int64_t selectBlockSize = 64;
    int64_t selectBlockCount = 16;
    int64_t compressBlockSize = 32;
    int64_t compressStride = 16;
    double scaleValue = 0.088388;
	string sLayerOut = "TND";
	char layOut[sLayerOut.length()];
	strcpy(layOut, sLayerOut.c_str());
    int64_t pageBlockSize = 128;
    int64_t sparseMod = 0;
	std::vector<int64_t> queryShape = {batchSize, numHeads, headDimsQK};
	std::vector<int64_t> keyShape = {blockNum, pageBlockSize, numKeyValueHeads * headDimsQK};
	std::vector<int64_t> valueShape = {blockNum, pageBlockSize, numKeyValueHeads * headDimsV};
	std::vector<int64_t> blockTableOptionalShape = {batchSize, maxNumBlocksPerSeq};
    std::vector<int64_t> outputShape = {batchSize, numHeads, headDimsV};
    std::vector<int64_t> topkIndicesShape = {batchSize, numKeyValueHeads, selectBlockCount};
	void *queryDeviceAddr = nullptr;
	void *keyDeviceAddr = nullptr;
	void *valueDeviceAddr = nullptr;
	void *blockTableOptionalDeviceAddr = nullptr;
	void *outputDeviceAddr = nullptr;
	void *topkIndicesDeviceAddr = nullptr;
	aclTensor *queryTensor = nullptr;
	aclTensor *keyTensor = nullptr;
	aclTensor *valueTensor = nullptr;
	aclTensor *blockTableOptionalTensor = nullptr;
	aclTensor *outputTensor = nullptr;
	aclTensor *topkIndicesTensor = nullptr;
	std::vector<op::fp16_t> queryHostData(batchSize * numHeads * headDimsQK, 1.0);
	std::vector<op::fp16_t> keyHostData(blockNum * pageBlockSize * numKeyValueHeads * headDimsQK, 1.0);
	std::vector<op::fp16_t> valueHostData(blockNum * pageBlockSize * numKeyValueHeads * headDimsV, 1.0);
	std::vector<int32_t> blockTableOptionalHostData(batchSize * maxNumBlocksPerSeq, 1);
	std::vector<op::fp16_t> outputHostData(batchSize * numHeads * headDimsV, 1.0);
	std::vector<int32_t> topkIndicesHostData(batchSize * numKeyValueHeads * selectBlockCount, 1);

	// 创建query aclTensor
	ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
	CHECK_RET(ret == ACL_SUCCESS, return ret);
	// 创建key aclTensor
	ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
	CHECK_RET(ret == ACL_SUCCESS, return ret);
	// 创建v aclTensor
	ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
	CHECK_RET(ret == ACL_SUCCESS, return ret);
	// 创建blockTableOptional aclTensor
	ret = CreateAclTensor(blockTableOptionalHostData, blockTableOptionalShape, &blockTableOptionalDeviceAddr, aclDataType::ACL_INT32, &blockTableOptionalTensor);
	CHECK_RET(ret == ACL_SUCCESS, return ret);
	// 创建output aclTensor
	ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &outputTensor);
	CHECK_RET(ret == ACL_SUCCESS, return ret);
	// 创建topkIndices aclTensor
	ret = CreateAclTensor(topkIndicesHostData, topkIndicesShape, &topkIndicesDeviceAddr, aclDataType::ACL_INT32, &topkIndicesTensor);
	CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> actualCmpKvSeqLenVector(batchSize, sequenceLengthK);
    auto actualCmpKvSeqLen = aclCreateIntArray(actualCmpKvSeqLenVector.data(), actualCmpKvSeqLenVector.size());
    
	// 3. 调用CANN算子库API
	uint64_t workspaceSize = 0;
	aclOpExecutor* executor;
	// 调用第一段接口
	ret = aclnnNsaCompressAttentionInferGetWorkspaceSize(queryTensor, keyTensor, valueTensor, nullptr, blockTableOptionalTensor, nullptr, actualCmpKvSeqLen,
        nullptr, nullptr,
        numHeads, numKeyValueHeads, selectBlockSize, selectBlockCount, compressBlockSize, compressStride,
        scaleValue, layOut, pageBlockSize, sparseMod, outputTensor, topkIndicesTensor, &workspaceSize, &executor);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttentionInferGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
	// 根据第一段接口计算出的workspaceSize申请device内存
	void* workspaceAddr = nullptr;
	if (workspaceSize > 0) {
		ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
		CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
	}
	// 调用第二段接口
	ret = aclnnNsaCompressAttentionInfer(workspaceAddr, workspaceSize, executor, stream);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttentionInfer failed. ERROR: %d\n", ret); return ret);

	// 4. （固定写法）同步等待任务执行结束
	ret = aclrtSynchronizeStream(stream);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

	// 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
	auto size = GetShapeSize(outputShape);
	std::vector<op::fp16_t> resultData(size, 0);
	ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr,
						size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy [attn] result from device to host failed. ERROR: %d\n", ret); return ret);
	uint64_t printNum = 10;
	for (int64_t i = 0; i < printNum; i++) {
		std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
	}
    auto topksize = GetShapeSize(topkIndicesShape);
	std::vector<op::fp16_t> topkresultData(topksize, 0);
	ret = aclrtMemcpy(topkresultData.data(), topkresultData.size() * sizeof(topkresultData[0]), topkIndicesDeviceAddr,
						topksize * sizeof(topkresultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy [top k] result from device to host failed. ERROR: %d\n", ret); return ret);
	for (int64_t i = 0; i < printNum; i++) {
		std::cout << "topk index: " << i << ": " << static_cast<int32_t>(topkresultData[i]) << std::endl;
	}

	// 6. 释放资源
	aclDestroyTensor(queryTensor);
	aclDestroyTensor(keyTensor);
	aclDestroyTensor(valueTensor);
	aclDestroyTensor(blockTableOptionalTensor);
	aclDestroyIntArray(actualCmpKvSeqLen);
	aclDestroyTensor(outputTensor);
	aclDestroyTensor(topkIndicesTensor);
	aclrtFree(queryDeviceAddr);
	aclrtFree(keyDeviceAddr);
	aclrtFree(valueDeviceAddr);
	aclrtFree(blockTableOptionalDeviceAddr);
	aclrtFree(outputDeviceAddr);
	aclrtFree(topkIndicesDeviceAddr);
	if (workspaceSize > 0) {
		aclrtFree(workspaceAddr);
	}
	aclrtDestroyStream(stream);
	aclrtResetDevice(deviceId);
	aclFinalize();
	return 0;
}
