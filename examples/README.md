## 简介

> 说明：

> - 关于AI Core的介绍请参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“概念原理和术语 > 硬件架构与数据处理原理”。

本项目提供了AI Core算子的开发和调用样例，请开发者根据实际情况参考对应实现。

## 目录说明

```
├── example
│   ├── add_example                # AddExample算子的目录名，一般小写下划线形式
│   │   ├── CMakeLists.txt         # 算子编译配置文件，保留原文件即可
│   │   ├── examples               # 算子使用示例
│   │   ├── op_graph               # 算子构图相关目录
│   │   ├── op_host                # 算子信息库、Tiling、InferShape相关实现
│   │   ├── op_kernel              # 算子kernel目录
│   │   └── op_kernel_aicpu        # 算子kernel_aicpu目录
│   ├── attention                  # attention类算子示例
│   │   └── flash_attn
│   │       └── L0_minimal         # FlashAttention极简样例，仅op_kernel，无Host Tiling
│   ├── mc2                        # 通算融合类算子示例
│   │   ├── all_gather_add         # AllGatherAdd算子的目录名，一般小写下划线形式
│   │   │   └── ...
│   ├── CMakeLists.txt             # 算子编译配置文件，保留原文件即可
│   └── README.md                  # 算子说明文档

```

## 算子开发样例

|样例目录| 	样例介绍	           |算子开发|算子调用 |
|---|------------------|---|---|
| add_example | 	实现两个张量相加功能的算子。	 | 算子端到端开发过程参见[AI Core算子开发指南](../docs/zh/develop/aicore_develop_guide.md)。 |调用样例参见[README](add_example/README.md)|
| mc2/all_gather_add | 	先进行AllGather集合通信，再执行逐元素相加。	 | 算子端到端开发过程参见[AI Core算子开发指南](./mc2/all_gather_add/docs/AllGatherAdd算子设计实现介绍.md)。 |调用样例参见[README](mc2/all_gather_add/README.md)|
| attention/flash_attn/L0_minimal | 	FlashAttention极简实现，无mask、无combine、无独立Host Tiling。	 | 算子实现介绍参见[FA_IMPL.md](attention/flash_attn/L0_minimal/ascend_ops/csrc/flash_attn_minimal/FA_IMPL.md)。 |调用样例参见[README](attention/flash_attn/L0_minimal/README.md)|
