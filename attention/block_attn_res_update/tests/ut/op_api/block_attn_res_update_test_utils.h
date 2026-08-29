/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_UPDATE_TEST_UTILS_H
#define BLOCK_ATTN_RES_UPDATE_TEST_UTILS_H

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "aclnn/aclnn_base.h"
#include "opdev/op_errno.h"
#include "opdev/platform.h"

namespace block_attn_res_update::test {

constexpr size_t PATH_BUFFER_SIZE = 4096U;

inline void SplitString(const std::string &input, const std::string &delimiter, std::vector<std::string> &output)
{
    const size_t delimiterLength = delimiter.size();
    size_t currentPosition = 0U;
    size_t nextPosition = input.find(delimiter, currentPosition);
    while (nextPosition != std::string::npos) {
        output.emplace_back(input.substr(currentPosition, nextPosition - currentPosition));
        currentPosition = nextPosition + delimiterLength;
        nextPosition = input.find(delimiter, currentPosition);
    }
    if (currentPosition <= input.size()) {
        output.emplace_back(input.substr(currentPosition));
    }
}

inline std::string Trim(std::string value)
{
    const auto isNotSpace = [](unsigned char character) { return !std::isspace(character); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), isNotSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), isNotSpace).base(), value.end());
    return value;
}

inline bool ParseBool(std::string value)
{
    value = Trim(value);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return value == "true" || value == "1" || value == "yes";
}

inline std::vector<int64_t> ParseDims(const std::string &value)
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }

    std::vector<std::string> tokens;
    SplitString(trimmed, ":", tokens);
    std::vector<int64_t> dims;
    dims.reserve(tokens.size());
    for (const auto &token : tokens) {
        dims.emplace_back(std::stoll(Trim(token)));
    }
    return dims;
}

struct AclTensorSpec {
    std::vector<int64_t> viewDims;
    std::vector<int64_t> stride;
    std::vector<int64_t> storageDims;
};

inline AclTensorSpec ParseAclTensorSpec(const std::string &value)
{
    std::vector<std::string> sections;
    SplitString(Trim(value), "#", sections);

    AclTensorSpec spec;
    if (!sections.empty()) {
        spec.viewDims = ParseDims(sections[0]);
    }
    if (sections.size() > 1U) {
        spec.stride = ParseDims(sections[1]);
    }
    if (sections.size() > 2U) {
        spec.storageDims = ParseDims(sections[2]);
    }
    return spec;
}

inline aclDataType ParseAclDtype(const std::string &dtype)
{
    static const std::map<std::string, aclDataType> dtypeMap = {
        {"FLOAT", ACL_FLOAT},
        {"BF16", ACL_BF16},
    };
    const auto iterator = dtypeMap.find(Trim(dtype));
    return iterator == dtypeMap.end() ? ACL_DT_UNDEFINED : iterator->second;
}

inline aclFormat ParseAclFormat(const std::string &format)
{
    static const std::map<std::string, aclFormat> formatMap = {
        {"ND", ACL_FORMAT_ND},
        {"FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ},
    };
    const auto iterator = formatMap.find(Trim(format));
    return iterator == formatMap.end() ? ACL_FORMAT_UNDEFINED : iterator->second;
}

inline aclnnStatus ParseAclnnStatus(const std::string &status)
{
    static const std::map<std::string, aclnnStatus> statusMap = {
        {"ACLNN_SUCCESS", ACLNN_SUCCESS},
        {"ACLNN_ERR_PARAM_INVALID", ACLNN_ERR_PARAM_INVALID},
        {"ACLNN_ERR_PARAM_NULLPTR", ACLNN_ERR_PARAM_NULLPTR},
        {"ACLNN_ERR_RUNTIME_ERROR", ACLNN_ERR_RUNTIME_ERROR},
        {"ACLNN_ERR_INNER_NULLPTR", ACLNN_ERR_INNER_NULLPTR},
        {"ACLNN_ERR_INNER", ACLNN_ERR_INNER},
    };
    const std::string trimmed = Trim(status);
    const auto iterator = statusMap.find(trimmed);
    return iterator == statusMap.end() ? static_cast<aclnnStatus>(std::stoll(trimmed)) : iterator->second;
}

inline void SetPlatformSocVersion(const std::string &socVersion)
{
    static const std::map<std::string, op::SocVersion> socVersionMap = {
        {"Ascend910B", op::SocVersion::ASCEND910B},
        {"Ascend950", op::SocVersion::ASCEND950},
    };
    const auto iterator = socVersionMap.find(Trim(socVersion));
    if (iterator == socVersionMap.end()) {
        throw std::invalid_argument("Unsupported socVersion: " + socVersion);
    }
    op::SetPlatformSocVersion(iterator->second);
}

inline std::string JoinPath(const std::string &directory, const std::string &fileName)
{
    if (directory.empty()) {
        return fileName;
    }
    return directory.back() == '/' ? directory + fileName : directory + "/" + fileName;
}

inline std::string GetDirectory(std::string path)
{
    std::replace(path.begin(), path.end(), '\\', '/');
    const size_t position = path.find_last_of('/');
    return position == std::string::npos ? "." : path.substr(0U, position);
}

inline bool FileExists(const std::string &path)
{
    std::ifstream file(path);
    return file.good();
}

inline std::string ResolveCsvPath(const std::string &csvFileName, const std::string &repositoryRelativeDirectory,
                                  const char *currentFile)
{
    std::vector<std::string> candidates = {
        JoinPath(GetDirectory(currentFile), csvFileName),
        JoinPath(repositoryRelativeDirectory, csvFileName),
        csvFileName,
    };

    const char *codePath = std::getenv("CODE_PATH");
    if (codePath != nullptr) {
        candidates.emplace_back(JoinPath(JoinPath(codePath, repositoryRelativeDirectory), csvFileName));
    }

    char currentWorkingDirectory[PATH_BUFFER_SIZE] = {0};
    if (getcwd(currentWorkingDirectory, sizeof(currentWorkingDirectory)) != nullptr) {
        candidates.emplace_back(JoinPath(JoinPath(currentWorkingDirectory, repositoryRelativeDirectory), csvFileName));
    }

    for (const auto &candidate : candidates) {
        if (FileExists(candidate)) {
            return candidate;
        }
    }
    return candidates.front();
}

inline std::string MakeSafeParamName(std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char character) {
        return std::isalnum(character) ? static_cast<char>(character) : '_';
    });
    return name;
}

inline std::string BuildCsvParseErrorMessage(const std::string &csvPath, size_t lineNumber, const std::string &caseName,
                                             const std::exception &error)
{
    std::ostringstream message;
    message << "Failed to parse CSV case. file=" << csvPath << ", line=" << lineNumber;
    if (!caseName.empty()) {
        message << ", case=" << caseName;
    }
    message << ", error=" << error.what();
    return message.str();
}

} // namespace block_attn_res_update::test

#endif // BLOCK_ATTN_RES_UPDATE_TEST_UTILS_H
