# ApplyRotaryPosEmbTensorMoveFusionPass

## 融合模式

删除ApplyRotaryPosEmb算子query/key输入前冗余的TensorMove节点（TensorMove为恒等拷贝），将其上游生产者直接连接到ApplyRotaryPosEmb输入，如下图所示。

![](../../../docs/zh/figures/ApplyRotaryPosEmbTensorMoveFusionPass_1.png)

融合时TensorMove节点的输入/输出控制边会一并转移到ApplyRotaryPosEmb节点，保证执行时序不变；由于TensorMove是恒等拷贝，ApplyRotaryPosEmb的输入描述无需刷新。


## 使用约束

- 无平台门控，所有支持ApplyRotaryPosEmb图模式的平台上均可生效。
- 仅处理ApplyRotaryPosEmb的query（输入0）和key（输入1）两路输入前的TensorMove节点，cos/sin输入不做处理。
- TensorMove的输出只能被该ApplyRotaryPosEmb节点消费（单消费者），否则不融合。
- TensorMove上游生产者的对应输出只能被该TensorMove节点消费（单消费者），否则不融合。
- query和key两路独立判定，单路满足条件即融合单路。
