Cuda shared memory按照4字节一个bank，总共32个bank（128字节）来组织，其store和load操作在一定情况下存在bank conflict的情况,
bank conflict会导致warp被stall，冲突较多会对整个pipeline的耗时会有较大的影响。
当发生bank conflict时，warp需要额外的一个cycle来重新提交shared memory的访问指令到LSU单元，该指令需要在MIO中排队，这种排队会导致访问延迟增加，此时warp可能处于等待数据返回的状态，warp state标识为Stall Short Scoreboard。如果MIO队列满，此时warp先需要等待MIO队列处于非空的状态，此时warp state标识为Stall MIO Throttle。

解决bank conflict的主要有下面几种：

- padding
- 转置存储（例如矩阵乘法的优化措施）。
- swizzling机制

warp是SM的基本执行单元，一个block内相邻的32个线程划分为一个warp。
padding的缺点有：
- 可能降低SM的occupancy。由于每个SM的可使用的shared memory有限，如果每个block使用的共享内存增加，则SM内最大可并发的block数目减少，导致资源不能被充分利用，一些计算资源被闲置。
- 地址访问对齐问题。需要仔细考虑padding的大小来避免地址不对齐的问题，比如访问shared memory时可能是向量化的访问，比如int4访问，也就是每次访问4个int，即16字节，那每次访问的地址必须是16字节对齐的。