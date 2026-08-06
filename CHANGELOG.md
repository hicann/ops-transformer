# CHANGELOG

> 本文档记录各版本的重要变更，版本按时间倒序排列。

## 8.5.0-beta.1

发布日期：2025-12-30

ops-transformer算子首个Beta版本8.5.0-beta.1 现已发布。本版本引入了多项新增特性、问题修复及性能改进，目前仍处于测试阶段。
我们诚挚欢迎社区反馈，以进一步提升ops-transformer的稳定性和功能完备性。
使用方式请参阅[官方文档](https://gitcode.com/cann/ops-transformer/blob/master/README.md)。

### 🔗 版本地址

[CANN 8.5.0-beta 1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/)

```
版本目录说明如下：
├── aarch64                  # CPU为ARM类型
│   ├── ops                  # ops算子包目录，用于归档算子子包
│   ├── ...
├── x86_64                   # CPU为X86类型
│   ├── ops                  # ops算子包目录，用于归档算子子包
│   ├── ...
```

### 📌 版本配套

**ops-transformer子包及相关组件与CANN版本配套关系**

| **CANN子包版本**                      | **配套CANN版本**        |
|:----------------------------------|---------------------|
| cann-ops-transformer 8.5.0-beta.1 | CANN 8.5.0-beta.1   |
| cann-ops-math 8.5.0-beta.1        | CANN 8.5.0-beta.1   |
| cann-ops-nn 8.5.0-beta.1          | CANN 8.5.0-beta.1   |
| cann-ops-cv 8.5.0-beta.1          | CANN 8.5.0-beta.1   |
| cann-hccl 8.5.0-beta.1            | CANN 8.5.0-beta.1   |
| cann-hixl 8.5.0-beta.1            | CANN 8.5.0-beta.1   |

### 🚀 关键特性

- 【工程能力】transformer类onnx算子插件支持。([#539](https://gitcode.com/cann/ops-transformer/pull/539))
- 【算子实现】部分推理算子新增对KirinX90支持。([#609](https://gitcode.com/cann/ops-transformer/pull/609))
- 【资料优化】增加QUICK_START，离线编译模式，aicore/graph模式下开发指南完善。([#612](https://gitcode.com/cann/ops-transformer/pull/612)、[#629](https://gitcode.com/cann/ops-transformer/pull/629)、[#342](https://gitcode.com/cann/ops-transformer/pull/342))
- 【资料优化】优化贡献指南中新算子贡献流程。([#384](https://gitcode.com/cann/ops-transformer/pull/384))

### 🐛 问题修复

- mc2通信域支持统一的torch.dist.group问题。([Issue47](https://gitcode.com/cann/ops-transformer/issues/47))
- add_example样例算子执行调用问题修复。([Issue174](https://gitcode.com/cann/ops-transformer/issues/174))
- 修复install_deps.sh脚本不支持openEuler系统的问题。([Issue255](https://gitcode.com/cann/ops-transformer/issues/255))
