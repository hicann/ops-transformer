# ops-transformer

English | [简体中文](./README.md)

## 🔥Latest News

- [2026/02] Newly supported operators [mhc_post](experimental/mhc/mhc_post), [mhc_pre](experimental/mhc/mhc_pre), [mhc_res](experimental/mhc/mhc_res).
- [2026/01] Newly supported [grouped_matmul<<<>>> invocation sample](examples/fast_kernel_launch_example/csrc/grouped_matmul), convenient for user customization.
- [2026/01] Newly supported operators [fused_floyd_attention](attention/fused_floyd_attention), [fused_floyd_attention_grad](attention/fused_floyd_attention_grad), [matmul_allto_all](mc2/matmul_allto_all).
- [2025/12] Added [QuickStart](QUICKSTART_en.md), guiding beginners with zero foundation to get started with operator project deployment (supporting Docker environments), operator development, and contribution processes.
- [2025/12] Optimized guide-type documents, focusing on the [Operator Development Guide](docs/en/develop/aicore_develop_guide.md), clarifying minimum deliverables and key sample code, and providing migration guidance for operators from the [Ascend/samples](https://gitee.com/ascend/samples/tree/master) repository to this project.
- [2025/12] Supported transformer-class ONNX operator plugins, including [NPUFlashAttention](attention/flash_attention_score/framework), [NPUMultiHeadAttention](common/src/framework), [NPUMoeComputeExpertTokens](moe/moe_compute_expert_tokens/framework), and so on.
- [2025/12] Newly supported operators [kv_rms_norm_rope_cache](posembedding/kv_rms_norm_rope_cache), [attention_update](attention/attention_update), [attention_worker_scheduler](attention/attention_worker_scheduler), [gather_pa_kv_cache](attention/gather_pa_kv_cache), [kv_quant_sparse_flash_attention](attention/kv_quant_sparse_flash_attention), [lightning_indexer_grad](attention/lightning_indexer_grad), [mla_preprocess](attention/mla_preprocess), [mla_preprocess_v2](attention/mla_preprocess_v2), [grouped_matmul_swiglu_quant_v2](gmm/grouped_matmul_swiglu_quant_v2), [attention_to_ffn](mc2/attention_to_ffn), [ffn_to_attention](mc2/ffn_to_attention).
- [2025/12] Open-source operators support Ascend 950PR/Ascend 950DT/KirinX90, and can be developed and debugged through the [CANN Simulator](docs/en/debug/cann_sim.md) simulation tool.
- [2025/11] Newly supported operators [kv_quant_sparse_flash_attention](attention/kv_quant_sparse_flash_attention), [lightning_indexer](attention/lightning_indexer), [quant_lightning_indexer](attention/quant_lightning_indexer), [sparse_flash_attention](attention/sparse_flash_attention).
- [2025/11] Newly supported sample operators [rope_matrix](experimental/posembedding/rope_matrix) and [all_gather_add](examples/mc2/all_gather_add).
- [2025/11] Added operator development project template [NpuOpsTransformerExt](experimental/npu_ops_transformer_ext), seamlessly integrating PyTorch tensor operations, supporting automatic differentiation and GPU/NPU unified interfaces.
- [2025/10] Added [experimental](experimental) directory, improved [Contribution Guide](CONTRIBUTING_en.md), supporting developers to debug and contribute custom operators.
- [2025/09] ops-transformer project first launched, open-source operators support Atlas A2/A3 series products.

## 🚀Overview

ops-transformer is an advanced operator library provided by the [CANN](https://hiascend.com/software/cann) (Compute Architecture for Neural Networks) operator library for transformer-class large model computation, including attention-class, moe-class, and other operators. The operator library architecture diagram is as follows:

<img src="docs/en/figures/architecture.png" alt="Architecture Diagram" width="700px" height="320px">

## 📝Version Compatibility

This project source code is released along with CANN software versions. For the correspondence between CANN software versions and this project tags, refer to the corresponding version description in the [release repository](https://gitcode.com/cann/release-management).
Note that to ensure your source code customization development proceeds smoothly, select the compatible CANN version and Gitcode tag source code. Using the master branch may pose version mismatch risks.

## ⚡️Quick Start

If you want to **understand and quickly experience the project from zero to one**, visit the following documents. You can first learn about the project operator information, and then try operator invocation, development, contribution, and so on.

1. [Operator List](docs/en/op_list.md): Full operator information of the project for quick querying.
2. [QuickStart](QUICKSTART_en.md): Provides a simplified quick start guide **based on WebIDE or Docker environments**, including environment setup, compilation and deployment, operator invocation/development/debugging, contribution, and so on.

    > **Note**: Whether using WebIDE or Docker environments, the latest commercially released CANN software package is provided by default, which is currently CANN 8.5.0. If you want to manually install the CANN package or experience the latest capabilities of the master branch, refer to the steps in [Learning Tutorial](#learning-tutorial) to complete environment setup, compilation and execution, operator development, and other operations.

## 📖Learning Tutorial

If you have completed the **Quick Start** learning, have a certain understanding of this project, and want to **deeply understand and experience the project**, visit the following documents.

These documents provide diverse scenarios and more comprehensive operation guidance, convenient for you to apply to various AI business scenarios.

1. [Environment Deployment](docs/en/context/quick_install.md): **Basic environment setup** guide, providing CANN software package installation methods for multiple scenarios, such as Docker deployment, manual installation, and so on.
2. [Operator Invocation](docs/en/invocation/quick_op_invocation.md): Operator **source code compilation and execution** guide, providing multiple methods for compiling operator packages and running operators (including executing operator samples and UT).
3. [Operator Development](docs/en/develop/aicore_develop_guide.md): Guide for **developing new operators** based on the project, learning from scratch to create operator projects and implement Tiling and Kernel core deliverables.
4. [Operator Debugging and Tuning](docs/en/debug/op_debug_prof.md): Provides common operator **debugging and tuning** methods, such as DumpTensor, msProf, Simulator, and so on.

In addition to the above guides, other documents are also provided, such as [Operator Invocation Methods](docs/en/invocation/op_invocation.md), terminology concepts, build parameter introduction, and so on. For full documentation, visit [docs](docs/README_en.md).

## 🔍Directory Structure

Key directories are as follows. For detailed directory introduction, refer to [Project Directory](./docs/en/context/dir_structure.md).

```text
├── cmake                          # Project engineering compilation directory
├── common                         # Project common header files and common source code
├── attention                      # attention-class operators
│   ├── flash_attention_score      # All deliverables of the flash_attention_score operator
│   │   ├── CMakeLists.txt         # Operator compilation configuration file
│   │   ├── docs                   # Operator description documents
│   │   ├── examples               # Operator usage samples
│   │   ├── op_host                # Operator information library, Tiling, InferShape-related implementation directory
│   │   ├── op_kernel              # Operator Kernel directory
│   │   └── README.md              # Operator description document
│   ├── ...
│   └── CMakeLists.txt             # Operator compilation configuration file
├── docs                           # Project document introduction
├── examples                       # End-to-end operator development and invocation samples
├── experimental                   # User custom operator storage directory
├── ...
├── moe                            # moe-class operators
├── posembedding                   # posembedding-class operators
├── scripts                        # Script directory, including custom operator, Kernel build-related configuration files
├── tests                          # Test engineering directory
├── CMakeLists.txt
├── README.md
├── build.sh                       # Project engineering compilation script
├── install_deps.sh                # Install dependency package script
└── requirements.txt               # Third-party dependency packages required by this project
```

## 💬Related Information

- [Contribution Guide](CONTRIBUTING_en.md)
- [Security Statement](SECURITY_en.md)
- [License](LICENSE)
- [Affiliated SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-transformer)

## 🤝Contact Us

This project features and documentation are continuously updated and improved. We recommend you follow the latest version.

- **Issue Feedback**: Submit issues through GitCode [Issues](https://gitcode.com/cann/ops-transformer/issues).
- **Community Interaction**: Participate in discussions through GitCode [Discussions](https://gitcode.com/cann/ops-transformer/discussions).
- **Technical Column**: Obtain technical articles through GitCode [Wiki](https://gitcode.com/cann/ops-transformer/wiki), such as serialized tutorials, best practices, and so on.

    |Technical Topic|Sample|
    |----|----|
    |Operator Performance Optimization Practice|[FA Operator Performance Optimization Practice and Effect Analysis](https://gitcode.com/cann/ops-transformer/wiki/FA%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E5%AE%9E%E8%B7%B5%E5%92%8C%E6%95%88%E6%9E%9C%E5%88%86%E6%9E%90.md)|
