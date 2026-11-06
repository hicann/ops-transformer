# CHANGELOG

> This document records the important changes of each version. Versions are listed in reverse chronological order.

## 8.5.0-beta.1

Release date: 2025-12-30

The first Beta version 8.5.0-beta.1 of the ops-transformer operators is now available. This version introduces multiple new features, bug fixes, and performance improvements, and is still in the testing stage.
We sincerely welcome community feedback to further improve the stability and functional completeness of ops-transformer.
For usage instructions, refer to the [official documentation](https://atomgit.com/cann/ops-transformer/blob/master/README.md).

### 🔗 Version Address

[CANN 8.5.0-beta 1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/)

```
The version directory is described as follows:
├── aarch64                  # CPU is of the ARM type
│   ├── ops                  # ops operator package directory, used to archive operator sub-packages
│   ├── ...
├── x86_64                   # CPU is of the X86 type
│   ├── ops                  # ops operator package directory, used to archive operator sub-packages
│   ├── ...
```

### 📌 Version Compatibility

**Compatibility between the ops-transformer sub-package and related components and CANN versions**

| **CANN sub-package version**          | **Compatible CANN version** |
|:----------------------------------|---------------------|
| cann-ops-transformer 8.5.0-beta.1 | CANN 8.5.0-beta.1   |
| cann-ops-math 8.5.0-beta.1        | CANN 8.5.0-beta.1   |
| cann-ops-nn 8.5.0-beta.1          | CANN 8.5.0-beta.1   |
| cann-ops-cv 8.5.0-beta.1          | CANN 8.5.0-beta.1   |
| cann-hccl 8.5.0-beta.1            | CANN 8.5.0-beta.1   |
| cann-hixl 8.5.0-beta.1            | CANN 8.5.0-beta.1   |

### 🚀 Key Features

- [Engineering capability] Support for transformer-type ONNX operator plugins. ([#539](https://atomgit.com/cann/ops-transformer/pull/539))
- [Operator implementation] Some inference operators now support KirinX90. ([#609](https://atomgit.com/cann/ops-transformer/pull/609))
- [Documentation optimization] Added QUICK_START, the offline compilation mode, and improved the development guides for aicore/graph modes. ([#612](https://atomgit.com/cann/ops-transformer/pull/612), [#629](https://atomgit.com/cann/ops-transformer/pull/629), [#342](https://atomgit.com/cann/ops-transformer/pull/342))
- [Documentation optimization] Optimized the new operator contribution process in the contribution guide. ([#384](https://atomgit.com/cann/ops-transformer/pull/384))

### 🐛 Bug Fixes

- Fixed the issue of supporting a unified torch.dist.group for the mc2 communicator. ([Issue47](https://atomgit.com/cann/ops-transformer/issues/47))
- Fixed the execution and invocation issue of the add_example sample operator. ([Issue174](https://atomgit.com/cann/ops-transformer/issues/174))
- Fixed the issue that the install_deps.sh script does not support the openEuler system. ([Issue255](https://atomgit.com/cann/ops-transformer/issues/255))
