# Security Statement

## Running User Recommendation

For security reasons, do not use administrator accounts such as root to execute any commands. Follow the principle of least privilege.

## File Permission Control

- Set the system umask value to 0027 or higher on the host machine (including the host machine) and in containers. This ensures that newly created directories have a maximum default permission of 750 and newly created files have a maximum default permission of 640.
- Implement permission control and other security measures for personal privacy data, commercial assets, source files, and various files saved during operator development. For example, for permission management of the project installation directory and input public data files, refer to [A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario](#a-file-folder-permission-control-recommended-maximum-values-for-each-scenario).
- During operator execution, compiled operator files may be cached and stored in the `kernel_meta_*` folder under the running directory to accelerate subsequent operator calls. Implement permission control on the generated files as needed.
- Implement proper permission control during installation and usage. Refer to [A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario](#a-file-folder-permission-control-recommended-maximum-values-for-each-scenario) for file permission reference settings.

## Build Security Statement

When compiling and installing this project from source code, you need to compile it yourself. The compilation process generates intermediate files. After compilation, implement permission control on the intermediate files to ensure file security.

## Runtime Security Statement

- Write operator invocation scripts based on the runtime environment resource conditions. If the operator invocation script does not match the resource conditions, such as when the space used for generating input data or benchmark calculation results exceeds the memory capacity limit, or when the script saves data locally exceeding the disk space size, errors may occur and cause the process to exit unexpectedly.
- When an operator runs abnormally, the process exits and prints error information. Locate the specific error cause based on the error prompt, including setting operator synchronous execution, viewing log files, and other methods.
- When operators are invoked through [PyTorch](https://gitee.com/ascend/pytorch), runtime errors may occur due to version mismatch. For details, refer to the [PyTorch Security Statement](https://gitee.com/ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E).

## Public Network Address Statement

The public network addresses included in this project code are as follows:

| Type | Open Source Code Address | File Name | Public Network IP Address/Public URL Address/Domain Name/Email Address/Compressed File Address | Usage Description |
| :------------: |:------------------------------------------------------------------------------------------:|:----------------------------------------------------------| :---------------------------------------------------------- |:-----------------------------------------|
| Dependency | Not applicable | cmake/third_party/makeself-fetch.cmake | [https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz)| Download makeself source code from gitcode, used as a compilation dependency |
| Dependency | Not applicable | cmake/third_party/protobuf.cmake | [https://github.com/protocolbuffers/protobuf/archive/v25.1.tar.gz](https://github.com/protocolbuffers/protobuf/archive/v25.1.tar.gz) | Download protobuf source code from github, used as a compilation dependency |
| Dependency | Not applicable | cmake/third_party/json.cmake | [https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) | Download JSON source code from gitcode, used as a compilation dependency |
| Dependency | Not applicable | cmake/third_party/gtest.cmake |[https://github.com/google/googletest/archive/release-1.8.0.tar.gz](https://github.com/google/googletest/archive/release-1.8.0.tar.gz)| Download googletest source code from github, used as a compilation dependency |
| Dependency | Not applicable | cmake/third_party/secure_c.cmake |[https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.10.tar.gz](https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.10.tar.gz)| Download libboundscheck source code from gitee, used as a compilation dependency |

## Vulnerability Mechanism Description

[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario

| Type | Linux Permission Reference Maximum Value |
| -------------- | ---------------  |
| User home directory | 750 (rwxr-x---) |
| Program files (including script files, library files, and so on) | 550 (r-xr-x---) |
| Program file directory | 550 (r-xr-x---) |
| Configuration file | 640 (rw-r-----) |
| Configuration file directory | 750 (rwxr-x---) |
| Log files (completed recording or archived) | 440 (r--r-----) |
| Log files (currently recording) | 640 (rw-r-----) |
| Log file directory | 750 (rwxr-x---) |
| Debug file | 640 (rw-r-----) |
| Debug file directory | 750 (rwxr-x---) |
| Temporary file directory | 750 (rwxr-x---) |
| Maintenance upgrade file directory | 770 (rwxrwx---) |
| Business data file | 640 (rw-r-----) |
| Business data file directory | 750 (rwxr-x---) |
| Key component, private key, certificate, ciphertext file directory | 700 (rwx------) |
| Key component, private key, certificate, encrypted ciphertext | 600 (rw-------) |
| Encryption/decryption interface, encryption/decryption script | 500 (r-x------) |
