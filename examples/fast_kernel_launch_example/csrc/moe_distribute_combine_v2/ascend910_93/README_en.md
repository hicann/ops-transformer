# moe_distribute_combine_v2

English | [简体中文](./README.md)

## Overview

For information such as environment deployment and installation steps, refer to [README_en.md](../../../README_en.md).

## Overview

For the operator design and invocation flow, refer to [README_en.md](../../moe_distribute_dispatch_v2/ascend910_93/README_en.md)

## Constraints

To fully enable the feature, the following version requirements must be met:
- gcc 9.4.0+
- python 3.9+
- torch>=2.7.1
- cann 9.0.0+

The platform supported by this operator is ascend910_93, while the default build environment is ascend910b. To build in the ascend910_93 build environment, run the following command first:
```sh
export NPU_SOC_VERSION=ascend910_93
```
