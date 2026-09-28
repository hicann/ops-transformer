# 独立实验算子 Torch wheel 打包

`build_torch_vendor.py` 使用仓库的原始 `torch_extension/setup.py`，在临时目录中完成打包。
它只收集选中的 `experimental/<category>/<op>/torch_extension`，避免与主库同名算子混用。

流程：复制 Torch 框架和选中的接口文件 → 仅在临时副本将 `OpBuilder` 的
`arg_check` 导入改为相对导入 → 构建并检查 wheel → 复制 wheel 到输出目录 →
清理临时目录 → 按需安装到指定目录。

仓库中的公共 `torch_extension/cann_ops_transformer/op_builder/builder.py` 保持原样。
`build/`、`dist/`、egg-info 和 vendor 符号链接均在临时目录生成，不会出现在源仓库中。
生成的 vendor wheel 自带相对导入修复，无需安装默认 `cann_ops_transformer` 包。

在配置好 Python、torch、torch_npu、setuptools、wheel 的远端环境，从仓库根目录执行：

```bash
python3 experimental/tools/build_torch_vendor.py \
  --ops lightning_indexer,lightning_indexer_grad_kl_loss \
  --vendor codex_lig \
  --output-dir /workspace/lightning_grad_wheels \
  --install-dir /workspace/lightning_grad_python
```

省略 `--install-dir` 则只生成 wheel；`--output-dir` 是 wheel 的持久保存位置。
安装使用 `pip --no-index --no-deps --upgrade --target`，不下载或替换 Torch 依赖。
如果需要选择临时目录，可设置 Python 标准的 `TMPDIR` 环境变量。

三个便捷入口保留各自的默认 vendor，均调用此脚本：

```bash
bash experimental/attention/lightning_indexer/tools/build_torch.sh \
  /workspace/lightning_indexer_python codex_li /workspace/lightning_indexer_wheels

bash experimental/attention/lightning_indexer_grad_kl_loss/tools/build_torch.sh \
  /workspace/lightning_grad_python codex_lig /workspace/lightning_grad_wheels

bash experimental/gmm/tools/build_torch.sh \
  /workspace/group_matmul_python codex_gmm /workspace/group_matmul_wheels
```

便捷入口参数为 `PYTHON_SITE [VENDOR] [WHEEL_DIR]`；第三个参数省略时，wheel 保存到
`PYTHON_SITE` 父目录下的 `wheels/`。梯度入口同时打包前向 LightningIndexer，供 golden 使用。
C++ 桥接仍由 `OpBuilder.load()` 在首次调用时编译或从已有 JIT 缓存加载。

直接在源仓库运行 `TORCH_EXTENSION_OPS=... python3 setup.py bdist_wheel` 不会应用
这一临时修复；独立 vendor 包应使用上述入口。默认包名的整包构建保持原方式。

## 验证记录（2026-09-24）

在 `root@hlsc-data-k8s-gpu-h910b-node2016.mt:8419`、CANN 8.5.0、Python 3.11.13、
Torch 2.9.0 / torch_npu 2.9.0 环境验证：

| vendor | 打包内容 | wheel 构建、安装及 C++ 桥接加载 |
| --- | --- | --- |
| codex_li | LightningIndexer | 通过 |
| codex_lig | LightningIndexer、普通/skip-padding 梯度接口 | 通过 |
| codex_gmm | gmm、local_exp_gmm、local_exp_gmm_with_zero | 通过 |

三个环境均未安装默认 `cann_ops_transformer` 包。校验结果确认：

- 源码公共 `builder.py` 保留原绝对导入，只有 wheel 中的副本使用相对导入。
- 打包、安装、加载前后，源码目录文件内容、目录及符号链接清单一致。
- 临时目录构建完成后已清理，wheel 保存在各任务的 `wheels/`，安装目录仍为 `python_site/`。
- 本次仅验证打包和扩展加载，未重新执行 NPU 计算用例。

远端验证日志为以下各任务目录中的 `validate-vendor-packaging.log`：

- `/workspace/codex-lightning-port-20260924.9Atcdy`
- `/workspace/codex-lightning-grad-port-20260924.nWrQZP`
- `/workspace/codex-group-matmul-port-20260924.eEqY45`
