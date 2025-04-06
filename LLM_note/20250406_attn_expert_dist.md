学习资料：[zartbot谈谈字节的Attention/Expert分离](https://mp.weixin.qq.com/s?src=11&timestamp=1744634541&ver=5930&signature=qKqnOMtthGVyYtDsbmLm0j4eyqTHhGrRcEPe2H*pMK7o-4WNtF38hELOk-QRNfQTAqchqKdc846swjKyU3varpYfKb*lgd--qDIxHNxv6GipuPLo5mSKOlC4NXxiHGDH&new=1)

摘录内容：

其实本质的问题是, 加大BatchSize后,如果按照DeepEP的方式来看, 显存容量和一些低算力卡(H20)在Attn计算上太慢带来约束, 高算力卡(H800)在小的batchsize下Expert的GroupGEMM计算利用率又太低,显存80GB又比较难拉高batchsize,退而求其次只能选择大规模EP(144/320)并行.

主要难点还是在通信上, 字节把同构的All2All通信变成**M:N的Mesh通信**,实际上还有很多问题没处理干净.
直接的叙事应该是引用**Kingman公式**, 然后想办法在网络上和计算上降低变异系数.

- [ ] M:N的Mesh通信
- [ ] Kingman公式 

![kingman公式](https://mmbiz.qpic.cn/sz_mmbiz_png/9v5mpBibQrkgDhWPrVo3Tfd1S5ib18WY4Fbo7via2DbVHW6eJ59d3JK340hfKwtIibkW5XEFNafQHoAt1aUibicsnxEg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

来自 [wikipedia](https://en.wikipedia.org/wiki/Kingman%27s_formula)


## XBAR, nvlink, nvswitch

[扩展阅读-大模型训练的通信问题与目前的解决方案图解](https://zhuanlan.zhihu.com/p/660599727)

XBAR 是在一张gpu内部交换，nvlink是gpu之间交换数据，nvswitch 是不同机器的gpu之间交换，nvlink可以通过xbar 可以在不同gpu的不同模块之间交换


1. XBAR（Crossbar Switch）

- 定义：XBAR 是一种内部的交叉开关（Crossbar Switch）架构，通常用于 GPU 内部或多个 GPU 之间的数据交换。
- 作用：
    - 内部通信：在单个 GPU 内部，XBAR 用于连接不同的功能模块（如计算单元、内存控制器、I/O 接口等），使得数据可以在这些模块之间高效传输。
    - 多 GPU 通信：在 NVLink 环境下，XBAR 作为 GPU 内部的一个关键组件，负责将来自 NVLink 的数据包分发到 GPU 内部的各个模块，或者将内部模块的数据通过 NVLink 发送到其他 GPU。
- 特点：
    - 高效的非阻塞数据传输。
    - 基于 SRAM 缓冲技术，确保数据传输的连续性和效率。

2. NVLink
- 定义：NVLink 是 NVIDIA 开发的一种高速点对点通信技术，用于 GPU 之间或 GPU 与 CPU 之间的直接互联。
- 作用：
    - 点对点通信：允许两个 GPU 或 GPU 与 CPU 之间直接传输数据，无需经过传统的 PCIe 总线，从而大幅提高通信带宽和降低延迟。
    - 统一内存池：支持多个 GPU 共享内存地址空间，形成一个逻辑上的统一内存池，简化多 GPU 编程。
- 特点：
    - 高带宽（例如，第四代 NVLink 可达 112 Gbps 每通道）。
    - 支持缓存一致性，允许 GPU 直接访问其他 GPU 或 CPU 的内存。
    - 低延迟，适合需要频繁通信的场景（如深度学习训练）。
3. NVSwitch
- 定义：NVSwitch 是一种基于 NVLink 的硬件交换机，用于支持大规模 GPU 集群的全互连通信。
- 作用：
    - 大规模互连：在多个 GPU 之间提供全互联的高速交换能力，使得任意两个 GPU 之间都可以直接通信，而无需经过复杂的多跳路由。
    - 扩展性：解决了 NVLink 点对点连接在大规模系统中的扩展性问题，支持多达数百个 GPU 的全互连。
- 特点：
    - 极高的总带宽（例如，第三代 NVSwitch 总双向带宽可达 25.6 Tb/s）。
    - 低延迟，确保数据传输的高效性。
    - 支持复杂的通信模式（如 All-Reduce 等），并可通过硬件加速这些操作。
三者的层级关系
- XBAR 是 GPU 内部的一个组件，用于连接 GPU 内部的不同模块。
- NVLink 是 GPU 之间的点对点通信链路，通过 XBAR 连接到 GPU 内部。
- NVSwitch 是一个硬件交换机，用于连接多个 GPU 的 NVLink 接口，实现大规模 GPU 集群的全互连。
