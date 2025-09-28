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
 * \file mc2_gen_task_ops_utils.cpp
 * \brief
 */

#include "mc2_gen_task_ops_utils.h"

#include "graph/ascend_string.h"
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "mc2_log.h"

namespace {
constexpr int64_t INVALID_INT_VAL = -1;
const std::string SO_NAME = "libccl_kernel.so";
const std::string KERNEL_NAME_V1 = "RunAicpuKfcSrvLaunch";

// 对已有结构的重复定义，只在本文件插入 aicpu desc 的时候使用
struct HcclCommParamDescTemp {
    uint64_t version   : 4;
    uint64_t groupNum  : 4;
    uint64_t hasFfts   : 1;
    uint64_t tilingOff : 7;
    uint64_t isDyn     : 48;
};

} // namespace

namespace ops {
bool Mc2GenTaskOpsUtils::IsComputationOnly()
{
    const char *env = getenv("ASCEND_MC2_DEBUG_MODE");
    return (env != nullptr && std::atoi(env) == 1);
}

int64_t Mc2GenTaskOpsUtils::GetAttachStreamIdByContext(const gert::ExeResGenerationContext *context, size_t idx)
{
#ifndef ASCEND_OPSPROTO_UT
    const auto stream_infos = context->GetAttachedStreamInfos();
    if (idx >= stream_infos.size()) {
        OPS_LOG_E(context->GetNodeName(), "Invalid index %zu in streams count %zu.", idx, stream_infos.size());
        return INVALID_INT_VAL;
    }

    const int64_t stream_id = (stream_infos[0].is_valid ? stream_infos[0].stream_id : INVALID_INT_VAL);
#else
    const int64_t stream_id = 1;
#endif
    return stream_id;
}

ge::Status Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(gert::ExeResGenerationContext *context,
                                                         const ge::AscendString &name,
                                                         const ge::AscendString &reuse_key)
{
    GE_ASSERT_NOTNULL(context);
    gert::StreamInfo stream_info;
    std::vector<int64_t> stream_depend_value(0);
    stream_info.name = name;
    stream_info.reuse_key = reuse_key;
    stream_info.depend_value_input_indices = stream_depend_value;
    stream_info.required = true;

    std::vector<gert::StreamInfo> stream_infos;
    stream_infos.push_back(stream_info);
    const auto ret = context->SetAttachedStreamInfos(stream_infos);
    if (ret != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Failed to set attached stream infos.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2GenTaskOpsUtils::InsertHiddenInputsForAicoreTask(
    const gert::ExeResGenerationContext *context, ge::KernelLaunchInfo &aicore_task,
    size_t (*get_insert_idx)(const std::vector<ge::ArgDescInfo> &), size_t input_cnt)
{
    GE_ASSERT_NOTNULL(context);
    GE_ASSERT_NOTNULL(get_insert_idx);

    std::vector<ge::ArgDescInfo> argDescInfos; // ArgDescInfo

    auto argsFormatStr = aicore_task.GetArgsFormat();
    if (argsFormatStr == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get Args Format from aicore task.");
        return ge::GRAPH_FAILED;
    }
    argDescInfos = ge::ArgsFormatSerializer::Deserialize(argsFormatStr);
    size_t insert_idx = get_insert_idx(argDescInfos); // ffts在mixL2时仍有task, 在aicore时没有, 故这里还是要查找插入位置
    OPS_LOG_I(context->GetNodeName(), "Insertion position is %zu, insert inputCnt is %zu.", insert_idx, input_cnt);

    for (size_t i = 0; i < input_cnt; ++i, ++insert_idx) {
        argDescInfos.insert(argDescInfos.begin() + insert_idx,
                            ge::ArgDescInfo::CreateHiddenInput(ge::HiddenInputSubType::kHcom));
    }
    auto args_format_str = ge::ArgsFormatSerializer::Serialize(argDescInfos).GetString();
    if (aicore_task.SetArgsFormat(args_format_str) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Failed to set args format for aicore task.");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(context->GetNodeName(), "aicore ArgsFormat: %s", args_format_str);

    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2GenTaskOpsUtils::InsertHiddenInputForAicoreV1(const gert::ExeResGenerationContext *context,
                                                            std::vector<std::vector<uint8_t>> &tasks)
{
    const auto get_idx = [](const std::vector<ge::ArgDescInfo> &args_desc_infos) {
        size_t insert_idx = 0U;
        for (; insert_idx < args_desc_infos.size(); ++insert_idx) {
            if (args_desc_infos[insert_idx].GetType() == ge::ArgDescType::kIrInput) {
                break;
            }
        }
        return insert_idx;
    };

    ge::KernelLaunchInfo aicore_task = ge::KernelLaunchInfo::LoadFromData(context, tasks.back());
    if (InsertHiddenInputsForAicoreTask(context, aicore_task, get_idx) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Failed to insert hidden input for mix task.");
        return ge::GRAPH_FAILED;
    }
    tasks.back() = aicore_task.Serialize();

    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2GenTaskOpsUtils::CreateAicpuTaskV1(const gert::ExeResGenerationContext *context,
                                                 ge::KernelLaunchInfo &aicpu_task)
{
    size_t inputSize = context->GetComputeNodeInfo()->GetIrInputsNum();
    size_t outputSize = context->GetComputeNodeInfo()->GetIrOutputsNum();
    OPS_LOG_I(context->GetNodeName(), "IR inputNum: %zu, IR outputNum: %zu", inputSize, outputSize);
    const size_t index = 3;
    // desc 配置
    union {
        HcclCommParamDescTemp hcclCommParamDesc;
        uint64_t customValue;
    } desc;
    desc.hcclCommParamDesc.version = 1;                                // 保留参数，暂时保持为 1
    desc.hcclCommParamDesc.groupNum = 1;                               // group 只支持1
    desc.hcclCommParamDesc.hasFfts = 0;                                // aicpu 暂时用不到此参数
    desc.hcclCommParamDesc.tilingOff = index + inputSize + outputSize; // tiling 在args中的index
    desc.hcclCommParamDesc.isDyn = 0;                                  // 现在默认为 0

    // aicpu 参数顺序: {desc}{hcom}{INPUT0}...{INPUTN}{OUTPUT0}...{OUTPUTN}{WORKSPACE}{TILING}
    std::vector<ge::ArgDescInfo> aicpu_args_format;
    aicpu_args_format.emplace_back(ge::ArgDescInfo::CreateCustomValue(desc.customValue));
    aicpu_args_format.emplace_back(ge::ArgDescInfo::CreateHiddenInput(ge::HiddenInputSubType::kHcom));
    aicpu_args_format.emplace_back(ge::ArgDescInfo(ge::ArgDescType::kIrInput, 0));
    for (size_t i = 1; i < inputSize; i++) {
        aicpu_args_format.emplace_back(ge::ArgDescInfo(ge::ArgDescType::kIrInput, 0));
    }
    for (size_t j = 0; j < outputSize; j++) {
        aicpu_args_format.emplace_back(ge::ArgDescInfo(ge::ArgDescType::kIrOutput, j));
    }
    aicpu_args_format.emplace_back(ge::ArgDescInfo(ge::ArgDescType::kWorkspace));
    aicpu_args_format.emplace_back(ge::ArgDescInfo(ge::ArgDescType::kTiling));
    auto args_format_str = ge::ArgsFormatSerializer::Serialize(aicpu_args_format).GetString();
    aicpu_task.SetArgsFormat(args_format_str);
    OPS_LOG_I(context->GetNodeName(), "aicpu ArgsFormat: %s", args_format_str);
    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2GenTaskOpsUtils::CommonKFCMc2GenTask(const gert::ExeResGenerationContext *context,
                                                   std::vector<std::vector<uint8_t>> &tasks)
{
    GE_ASSERT_NOTNULL(context);
    int64_t aicore_idx = 0; // 新定义下aicoreTask的vector 成员变量数为1，有且仅有一个aicoreTask
    OPS_LOG_I(context->GetNodeName(), "Start to generate task for MC2, task size %lu, aicore index %ld.", tasks.size(),
              aicore_idx);

    // get main/attach stream ids
    const int64_t attach_stream_id = GetAttachStreamIdByContext(context);
    const int64_t stream_id = context->GetStreamId();

    /* wait aicpu record [aicore] wait */
    // wait task
    ge::KernelLaunchInfo aicpu_wait_for_aicore_task = ge::KernelLaunchInfo::CreateHcomWaitTask(context);
    aicpu_wait_for_aicore_task.SetStreamId(attach_stream_id);
    tasks.insert(tasks.begin() + aicore_idx, aicpu_wait_for_aicore_task.Serialize());
    ++aicore_idx;

    // aicpu task
    ge::KernelLaunchInfo aicpu_task =
        ge::KernelLaunchInfo::CreateAicpuKfcTask(context, SO_NAME.c_str(), KERNEL_NAME_V1.c_str());
    aicpu_task.SetStreamId(attach_stream_id);
    GE_ASSERT_SUCCESS(CreateAicpuTaskV1(context, aicpu_task)); // 之后换一个名称可能清晰一点，上面已经有create了
    tasks.insert(tasks.begin() + aicore_idx, aicpu_task.Serialize());
    ++aicore_idx;

    // record task
    ge::KernelLaunchInfo aicore_record_for_aicpu_task = ge::KernelLaunchInfo::CreateHcomRecordTask(context);
    aicore_record_for_aicpu_task.SetStreamId(stream_id);
    tasks.insert(tasks.begin() + aicore_idx, aicore_record_for_aicpu_task.Serialize());
    ++aicore_idx;

    return InsertHiddenInputForAicoreV1(context, tasks);
}
} // namespace ops
