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
 * \file aclnn_quant_grouped_matmul_inplace_add_param.cpp
 * \brief QGMMInplaceAdd Aclnn 参数信息.
 */

#include "aclnn_quant_grouped_matmul_inplace_add_param.h"
#include <utility>
#include "tests/utils/case.h"
#include "tests/utils/io.h"
#include "tests/utils/tests_utils_log.h"

using ops::adv::tests::utils::ReadFile;
using ops::adv::tests::utils::WriteFile;
using ops::adv::tests::utils::TensorIntf;
using ops::adv::tests::utils::AclnnTensorList;
using ops::adv::tests::utils::AclnnTensor;

using namespace ops::adv::tests::quant_grouped_matmul_inplace_add;

AclnnQGMMInplaceAddParam::AclnnQGMMInplaceAddParam(std::vector<Tensor> inputs, std::vector<int64_t> groupListData,
                                                   int32_t groupListType, int32_t groupSize)
    : Param({}, Tensor(), Tensor(), std::move(groupListData), groupListType, groupSize)
{
    for (auto &tensor : inputs) {
        mTensors[tensor.Name()] = tensor;
    }
}

AclnnQGMMInplaceAddParam::~AclnnQGMMInplaceAddParam()
{
    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (scale1 != nullptr && scale1 != aclnnScale1.GetAclTensor()) {
        aclDestroyTensor(scale1);
    }
}

bool AclnnQGMMInplaceAddParam::CreateX1AclTensor()
{
    auto shape = aclnnX1.ShapeView();
    std::vector<int64_t> viewShape = {shape[1], shape[0]};
    std::vector<int64_t> stride = {1, shape[1]};
    x1 = aclCreateTensor(viewShape.data(), viewShape.size(), aclnnX1.GetAclDataType(), stride.data(), 0,
                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), aclnnX1.GetDevData());
    return x1 != nullptr;
}

bool AclnnQGMMInplaceAddParam::CreateScale1AclTensor()
{
    auto shape = aclnnScale1.ShapeView();
    std::vector<int64_t> viewShape = {shape[1], shape[0], shape[2]};
    std::vector<int64_t> stride = {shape[2], shape[1] * shape[2], 1};
    scale1 = aclCreateTensor(viewShape.data(), viewShape.size(), aclnnScale1.GetAclDataType(), stride.data(), 0,
                             aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), aclnnScale1.GetDevData());
    return scale1 != nullptr;
}

bool AclnnQGMMInplaceAddParam::Init()
{
    aclnnX1 = ops::adv::tests::utils::AclnnTensor(mTensors["x1"]);
    aclnnX2 = ops::adv::tests::utils::AclnnTensor(mTensors["x2"]);
    aclnnY = ops::adv::tests::utils::AclnnTensor(mTensors["y"]);
    aclnnGroupList = ops::adv::tests::utils::AclnnTensor(mTensors["groupList"]);
    aclnnScale2 = ops::adv::tests::utils::AclnnTensor(mTensors["scale2"]);
    aclnnScale1 = ops::adv::tests::utils::AclnnTensor(mTensors["scale1"]);
    auto ret = InitGroupList();
    LOG_IF_EXPR(ret == false, LOG_ERR("InitGroupList failed"), return false);
    auto *cs = static_cast<ops::adv::tests::utils::Case *>(ops::adv::tests::utils::Case::GetCurrentCase());
    LOG_IF_EXPR(cs == nullptr, LOG_ERR("Can't get current case"), return false);
    for (auto *t :
         std::vector<TensorIntf *>{&aclnnX1, &aclnnX2, &aclnnScale2, &aclnnGroupList, &aclnnY, &aclnnScale1}) {
        t->FreeDevData();
        auto tFormat = t->GetFormat();
        if (t->GetExpDataSize() <= 0) {
            continue;
        }
        uint8_t *devData = nullptr;
        if (tFormat == ge::Format::FORMAT_FRACTAL_NZ) {
            devData = t->AllocDevDataNz(0, 0);
        } else {
            devData = t->AllocDevData(0, 0);
        }
        if (devData == nullptr) {
            return false;
        }
        std::string filePath = std::string(cs->GetRootPath()) + t->Name() + ".bin";
        if (ops::adv::tests::utils::FileExist(filePath)) {
            if (!t->LoadFileToDevData(filePath)) {
                return false;
            }
        }
    }
    LOG_IF_EXPR(CreateX1AclTensor() == false, LOG_ERR("Can't init x1 tensor"), return false);
    if (aclnnScale1.GetAclDataType() == ACL_FLOAT8_E8M0) {
        LOG_IF_EXPR(CreateScale1AclTensor() == false, LOG_ERR("Can't init scale1 tensor"), return false);
    } else {
        scale1 = aclnnScale1.GetAclTensor();
    }
    return true;
}

bool AclnnQGMMInplaceAddParam::InitGroupList()
{
    if (mGroupListData.size() == 0) {
        return true;
    }

    size_t dataSize = mGroupListData.size() * sizeof(int64_t);
    std::string fileName = "groupList.bin";
    if (!WriteFile(fileName, mGroupListData.data(), dataSize)) {
        LOG_ERR("Write groupList data to file[%s] failed", fileName.c_str());
        return false;
    }

    return true;
}
