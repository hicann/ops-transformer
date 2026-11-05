# Security Statement

## Recommendations for the Running User

For security reasons, do not use administrator accounts such as root to execute any command. Follow the principle of least privilege.

## File Permission Control

- It is recommended that you set the umask value of the running system to 0027 or higher on the host (including the host machine) and in containers. This ensures that the default maximum permission is 750 for new folders and 640 for new files.
- It is recommended that you take security measures such as permission control for sensitive content, including personal privacy data, commercial assets, source files, and various files saved during operator development. For example, this involves permission control over the installation directory of this project and over input public data files. For recommended permission settings, refer to [A-Recommended Maximum File (Folder) Permissions for Different Scenarios](#a-recommended-maximum-file-folder-permissions-for-different-scenarios).
- During operator running, operator compilation files may be cached and stored in the `kernel_meta_*` folder in the running directory to speed up subsequent operator invocations. You can perform permission control on the generated files as needed.
- You need to perform proper permission control during installation and use. For the settings, it is recommended that you refer to the file permission reference in [A-Recommended Maximum File (Folder) Permissions for Different Scenarios](#a-recommended-maximum-file-folder-permissions-for-different-scenarios).

## Build Security Statement

When compiling and installing this project from source code, you need to compile it yourself. The compilation process generates some intermediate files. After the compilation is complete, it is recommended that you perform proper permission control on the intermediate files to ensure file security.

## Running Security Statement

- It is recommended that you write operator invocation scripts based on the resource status of the running environment. If an operator invocation script does not match the resource status, for example, the space used for generating input data or benchmark computation results exceeds the memory capacity limit, or the data saved locally by the script exceeds the disk space, errors may occur and the process may exit unexpectedly.
- When an operator runs abnormally, it exits the process and prints error information. It is recommended that you locate the specific error cause based on the error message, using methods such as setting the operator to run synchronously and viewing log files.
- When operators are invoked through [PyTorch](https://gitee.com/ascend/pytorch), version mismatches may cause running errors. For details, refer to the [PyTorch security statement](https://gitee.com/ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E).

## Public Network Address Statement

The public network addresses contained in the code of this project are declared as follows:

|      Type      |                                           Open-source code address                                           |                            File name                             |             Public IP address/Public URL/Domain name/Email address/Archive file address             |                   Purpose description                    |
| :------------: |:------------------------------------------------------------------------------------------:|:----------------------------------------------------------| :---------------------------------------------------------- |:-----------------------------------------|
|  Dependency  | N/A  | cmake/third_party/makeself-fetch.cmake | https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/makeself/makeself-release-2.5.0.tar.gz | Downloads the makeself source code from obs as a compilation dependency |
|  Dependency  | N/A  | cmake/third_party/protobuf.cmake | https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/protobuf/protobuf-25.1.tar.gz | Downloads the protobuf source code from obs as a compilation dependency |
|  Dependency  | N/A  | cmake/third_party/json.cmake | https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/json/json-3.12.0.tar.gz | Downloads the JSON source code from obs as a compilation dependency |
|  Dependency  | N/A  | cmake/third_party/gtest.cmake | https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/googletest/googletest-1.14.0.tar.gz | Downloads the googletest source code from obs as a compilation dependency |
|  Dependency  | N/A  | cmake/third_party/secure_c.cmake | https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.10.tar.gz | Downloads the libboundscheck source code from gitee as a compilation dependency |

---

## Vulnerability Mechanism Description

[Vulnerability management](https://atomgit.com/cann/community/blob/master/security/security.md)

## Appendix

### A-Recommended Maximum File (Folder) Permissions for Different Scenarios

| Type           | Reference maximum Linux permission |
| -------------- | ---------------  |
| User home directory                        |   750 (rwxr-x---)            |
| Program files (including script files, library files, and so on)       |   550 (r-xr-x---)             |
| Program file directory                      |   550 (r-xr-x---)            |
| Configuration file                          |  640 (rw-r-----)             |
| Configuration file directory                      |   750 (rwxr-x---)            |
| Log file (recording completed or archived)        |  440 (r--r-----)             |
| Log file (being recorded)                |    640 (rw-r-----)           |
| Log file directory                      |   750 (rwxr-x---)            |
| Debug file                         |  640 (rw-r-----)         |
| Debug file directory                     |   750 (rwxr-x---)  |
| Temporary file directory                      |   750 (rwxr-x---)   |
| Maintenance and upgrade file directory                  |   770 (rwxrwx---)    |
| Service data file                      |   640 (rw-r-----)    |
| Service data file directory                  |   750 (rwxr-x---)      |
| Key component, private key, certificate, and ciphertext file directory    |  700 (rwx------)      |
| Key components, private keys, certificates, and encrypted ciphertext        | 600 (rw-------)      |
| Encryption/decryption interfaces and encryption/decryption scripts            |   500 (r-x------)        |
