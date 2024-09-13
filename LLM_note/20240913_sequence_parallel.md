学习资料：[大规模分布式 AI 模型训练系列——序列并行](https://mp.weixin.qq.com/s/4tB1UCHdYOG9pOq7wxiNKA)

# colossal AI 的 SP

作者将输入序列分成多个 Chunk，并将每个 Chunk 输入到相应的设备。为了计算注意力输出，将环形通信与自注意力计算相结合，提出了 RingSelf-Attention（RSA）。实验表明，与 TP 相比，扩展到 64 个 NVIDIA P100 GPU 时，提出的方法可以实现 13.7x 的最大 Batch Size，3x 的序列长度。

PS：本文的方案还是针对 Bert 这种 Encoder Only 的模型。针对 Decoder Only 的 GPT 模型，由于 Attention Score 是个下三角矩阵，需要进一步考虑负载均衡问题。

与 PP 和 TP 的区别，其核心是每个设备分到一部分 Token，而且每一层也都需要通信：

## Ring Self-Attention

作者将其按照上述两个矩阵乘法分为两个 Step，每个 Step 实质上都是矩阵乘法的分块计算。假设序列长度为 L，设备数为 N：

Q * KT：每个设备上 L/N 个 Token 的 Query 和 Key，通过 Ring 的方式传递 Key。N-1 次传输，N 次计算后得到完整的 Attention Score，每个设备上只有 1/N 的 Attention Score。

Attention Score * V：每个设备上 L/N 个 Token 的 Value，通过 Ring 的方式传递 Value。N-1 次传输，N 次计算后得到完整的 Attention Output，每个设备上只有 1/N 的 Attention Output。

核心就是分块矩阵乘，每次计算一个 Block，通过移动 Key 和 Value 可以实现计算不同的 Block。

##  FFN 切分

每个设备上都有完整的 FFN 输入 X 和输出 Z，当序列比较长时，输入和输出占据的显存空间反而可能比权重 A 和 B 更大。此外，FFN 中 Token 之间是没有交叉的，也就是每个 Token 可以单独计算，基于这个特性，作者提出了 FFN 的序列并行方案（每个设备上需要存储完整的 A 和 B）。如下图 Table 1 所示为 Tensor Parallelism 和 Sequence Parallelism 的显存占用：

![image1](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTiamJSmb3nicxYt7FOoTav1fX80VhH64lF0nbO8Diae6pGPaePECowszvD54y8WsNbEzTUB4aibia5ia18A/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

可以看出，当满足以下条件，也就是 BL > 32H 时，序列并行会更加节约显存：

![image2](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTiamJSmb3nicxYt7FOoTav1fX1HffRkZ4ehbN96SWVAbXg66FgVxsiayXuGKUftCt0WicXA3wwicOFybDA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)


# Megatron SP

使用 TP 切分方案，每个设备每层的 Activation 占据的显存空间如下所示，其中 s 表示序列长度，b 表示 Batch Size，h 表示 Hidden Size，t 表示设备数，a 表示 Head 个数（假设激活都占用 2 个字节，Mask 占用 1 个字节）：

![image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTiamJSmb3nicxYt7FOoTav1fXLGoHgLqjtGSCw4zIzKbiazWJTdAwNntoRfDLArMlFAaD88OAHqmOLww/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

由于 LayerNorm 和 Dropout 的计算量很小，因此这一部分并没有切分，在每个设备上都会独立、重复的计算，所以对应的 Activation 也并没有切分。上述公式中的 10sbh 对应着未切分的 Activation，包括 LayerNorm 和 Dropout 的输入各 2sbh（各 2 层，共 8sbh），以及 2 个 Dropout 的 Mask 各 sbh，共 10sbh。然而，实际上这一部分 Token 之间并没有交叉，可以按 Token 分到不同的 Device 独立计算，作者称这种方式为 Sequence Parallelism，如下图所示，对应的通信操作也从 f 变为 g。

![image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTiamJSmb3nicxYt7FOoTav1fXnMmGx2cTTTIAsictjeDYcZq25VHpzLXJM3wR1F5tbMwAFoTOabLTaEQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

按上述 Sequence Parallelism 方式切分后，10sbh 的 Activation 也可以明显降低，每个设备只需要 1/t，最终每个设备每层需要的 Activation 如下所示：

![image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTiamJSmb3nicxYt7FOoTav1fX4YWEGwamj9Zq1nd6E18uRfH9f4YepsFe96DM06FVFIibFw2HYwD1uzg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)




