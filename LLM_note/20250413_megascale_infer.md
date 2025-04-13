学习资料：[字节 MegaScale-Infer：分离式专家并行 MoE 服务系统](https://mp.weixin.qq.com/s/yIIocrCbEW6fiz6wrhwFOA)

MegaScale-Infer：一个高效且经济的 MoE 模型服务系统。该系统将各模型层中的 Attention 模块与 Expert 模块解耦，实现独立扩展、定制化并行策略及异构部署。MegaScale-Infer 创新性地采用 **Ping-Pong 流水线并行技术**，将请求 Batch 划分为 Micro-Batch 并在 Attention 与 Expert 模块间动态调度以完成 Inference。结合各模块专属的模型并行策略，该系统可以有效隐藏通信开销并最大化 GPU 利用率。

为最小化数据传输开销（如 Token Dispatch），MegaScale-Infer 实现了高性能 M2N 通信库，消除了不必要的 GPU-CPU 数据拷贝、Group 初始化开销及 GPU 同步等待开销。

假设激活专家与总专家的比例是 1/16：
- Dense 模型可能在 Batch Size 为 156（A100：312T FLOPS / 2TB/s = 156）即可以达到 Compute Bound。
- MoE 模型需要 Batch Size 为 156*16=2496 才能达到 Compute Bound，并且由于负载不均，可能会更加严重。

MegaScale-Infer 想要做的事情就是：
- Attention：单位 Cost 下的 GPU 利用率更高。
- FFN：更小的 Batch Size 即可以达到 Compute Bound。

采用常见的 PD 分离方案，也就是 Prefill 和 Decoding 有单独的集群。本文的重点是 Decoding 阶段，进一步对 Decoding 集群进行切分，包含 Attention 节点和 Expert 节点：
- Attention 节点：
    每个节点包含全部 Attention 参数，采用 TP（Tensor Parallelism）。

- Expert 节点：
    - 每个节点多个 GPU 存储一个 Expert，采用 TP。
    - 所有专家节点一起组成一个 EP（Exert Parallelism）组。

节点内 TP 可以充分利用高带宽的 NVLink 通信。


