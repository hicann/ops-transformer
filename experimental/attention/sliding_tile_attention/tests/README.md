# SlidingTileAttention 测试

`ut/op_host` 覆盖 InferShape：BNSD 无文本、含文本预留槽、BFLOAT16 三种正常场景下输出形状与 query 一致，
以及 query 不是四维 BNSD 时推导失败。运行：

```bash
bash build.sh --ophost_test --experimental --soc=ascend910b \
    --ops=sliding_tile_attention
```

端到端 ACLNN 用例位于 `../examples/test_aclnn_sliding_tile_attention.cpp`，构造确定性输入并校验两段式接口的
调用流程、输出形状与数值。正式验收矩阵（官方用例精度、性能与内存自验证）和性能基线由验收交付件保存，
不复制进开源仓目录。
