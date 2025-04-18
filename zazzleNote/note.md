# Lecture 1

## vllm 现状

所有的代码文件，除了 V1-folder 其他的代码文件都属于 V0，vLLM 现在正在从 V0 到 V1 迁移
因为一些历史因素

## vLLM 的模块组成

1. EntryPoint (两种方式, LLM, API server)
2. **Engine**
3. Core.Scheduler => step
4. KV Cache Manager(Page Attention)     Toread: LMCache
5. Worker
6. Model executor (Model runner)
7. Modelling
8. Attention backend

### EntryPoint

#### LLM Class(offline)
存储位置：`vllm/entrypoints/llm.py`

#### API server(online)
存储位置：`vllm/entrypoints/openai/api_server.py`
调用 FastAPI 把 request 发送到后面的 engine

### engine
真正在干活的 => llm_engine.py
让 vLLM 拥有异步性 => async_llm_engine.py

一个模型经过一次 inference 的过程是 step
大预言模型一个 request 可能会 inference 多次 eg. 一个提问要返回 1k 个 token
一个 token 就是一次 inference，而每个 step 放什么 request => Scheduler

### Scheduler
core.scheduler.py

### KV Cache Manager
core.block_manager.py

**Prefix Caching**:
    what if prefix doesn't match ? => **CacheBlend**
    what if prefix cache on another machine ? => KV cache sharing across nodes.

> Deepseek(MLA Multi-Layer-Attention optimization)    压缩版的 KV Cache 然后在 Attention 复原

### Worker
针对到各种后端的具体执行，执行 Scheduler 的命令，包含各种 XPU 的后端去适配
不涉及纯硬件的话 => worker_base.py & worker.py
初始化 Model executor 的一系列变量和环境 eg. 分布式环境

### Model executor (Model runner)
Worker 的底层，真正执行 Worker 的命令，和硬件层对接


### Modeling
model_executor.models.llama.py  这个重点看  因为之前 llama 最主流所以设计的最规范

Modeling 就是 Huggingface 上各种乱七八糟的模型写成 vLLM 能理解的，能优化的标准化模型
> Tip: 有很多提 PR 的机会

其中 llama.py 的 forward() 函数重点中的重点 line_265


### Attention Backend
真正实现 attention 算子的地方  重点看这个: **attention.backends.flash_attn.py**
在这个代码文件中  可以看到给 prefill 和 decode 配不同的 kernel Q: Prefill 阶段和 Decode 阶段是什么意思

在 prefill 的过程中 调用 flash_attn_varlen_func()  不需要从 GPU 中读取任何 data 来完成 prefill
而当需要从 GPU 中读取数据的时候 => flash_attn_with_kvcache (Page Memory)


# Lecture 2


## Distruibuted Inference

### Distributed

1. Why distributed inference:

即使是现在最先进的 H100 也只有 80G，而模型文件动辄 40B+ => 80G+ 这样子，所以完全放不下
现在分布式更多是为了榨干硬件的所有资源 eg. prefill 计算密集任务 和 Decode 这种内存密集任务按阶段分开执行

2. Type of distributed inference:

TP / PP/ EP

### TP / Tensor Parallel

#### Design & Algorithm

在运算的过程中把一个 operator 的输入拆成很多分来执行，最后执行 All-reduce 把结果合在一起
Q: 这个是不是就是多头注意力的原理 ?
eg. vllm 把一个 model,weights 切分为 number of TP 的数量，但是 vllm 会骗 worker 说你拿到的是一个完整的 model
Q: 还是不理解这里说的切分

#### Communication

一般并行通信的接口文件 vllm/distributed/parallel_state.py
用于 _TP 通信的数据接口 get_tp_group()
init_model_parallel_group() => Q: ??

### PP / Pipeline Parallel

#### Design & Algorithm

对 device 之间连接性的需求大大降低，但是代价就是在并行的时候并不会提升 latency 因为本质上变成线性计算了
eg. TP 是 4 个 GPU 一起处理一个 request，但 PP 是流水线的方式，所以一般被仍在便宜硬件上进行服务

vllm 的设计是希望每个 worker 负责一个 layer 的子集 => vllm/model_executor/models/llama.py
每个 worker 只会执行 self.start_layer -> self.end_layer 之间指定的内容  其他的内容不会 load

#### Communication

worker 在执行完后需要 communication: between workers, IntermediateTensor
这个 communication 对应的数据结构是 get_pp_group()
    Q: 为什么会有 is_first_rank() & is_last_rank() 的判断

### EP / Expert parallel & data parallel (advanced)

目前以 deepseek 为 leader

#### Design & Algorithm

Mistral / Mixtral / Deepseek model: MoE
一般的模型，所有的权重都参与计算，但是在 deepseek 中，是以 expert 为单位去做计算，一次只有一小部分 expert 参与运算

每次参与权重计算的 expert 小子集(5个) 是根据 request **动态**决定的
以 deepseek-R1 671B 模型为例，每个 request 只会启动 30+B 的 layer
所以，很自然的想法是把不同的 experts 放在不同的 GPU 上，从而实现 expert 的并行（把大模型的权重拆分成以 expert 为单位

但是设想一个场景：如果 32 个 requests 打包进来，而每个 request 碰巧都是不同 expert 来负责的，怎么办

##### 解决设计

1. shuffle
前提，只有在得到 attention layer 的 output 之后才能知道这个 request 对应的是哪个 expert
eg. 0 号 machine 收到了 request 但是这个 request 对应的 expert 在 5 号机器上，那么就要迁移过去

2. forward
在特定的机器上完成 linear layer 的计算
实际上在 MOE 模型中，存在一些一直处于 activate 状态的 expert，那么这些 expert 的负载注定非常高
可以 duplicated shared expert for the experts has high load，在一组 shared expert 中内部 balance 一下

3. shuffle back
可能出现那种所有 request 都在询问某个 expert，那么会导致某个 GPU 的某段时间负载特别高
这堆 requests 在当前 GPU 计算完成后，后续的 attention 还要分摊执行（这里实际上是进行了负载均衡 

Deepseek 有专门为这种特殊的通信方式开发了 DPEP Q: DPEP 要查一下

### DP / Data parallel (Deepseek)

TP is for attention while EP is for linear layers

Q: 不是很理解这里的意思  EP 和 DP, TP 的关系是什么呢，难道是说可以组合起来用吗
TP < max_attention_head     max TP << ep needed

所以 TP 要 duplicate DP 的数量

针对多头注意力计算，可能也就 16 num_heads  但是 ep 可能好几百份

tp * dp == ep   # Q: ???

整个线程池的 process 的数量是恒定的 eg. EP=320 => 需要 320 个 processes
从 linear layer 的角度看，计算任务是 320 个 process，而从 attention 的角度看，用不满这 320 个
无法使 attention 并行 => 使 request 并行    Q: 1:25:00 这里完全没听懂

### 硬件高速通信

1. NV Link:
direct communication between GPUs, 直接在物理层面的连接（跨机器还要 NV Link Switch

2. rack: 
可以理解为一个数据中心，一个竖着的柜子，这个柜子的每一层放着一个四卡或者八卡机器，好几个 server

3. Infinity Band:
硬件 直接插在各种计算节点之间

4. RDMA: Remote Direct Memory Access
RDMA NIC
Software Solution
Key advantage: bypass operating system / zero copy
RoCE

zero copy: 运行时产生的中间数据  需要 copy 一下

### vLLM Communication Library

vllm.distributed.device_communicators
    `PyNccl`: Nvidia 发布的在 NV 系硬件上通信的软件
    `shared memory`: OS 进行不同进程的数据共享
    `custom allreduce`: A kernel just for All reduce operation
        before all-reduce:
            0 machine [0]
            1 machine [1]
        after all-reduce
            0 machine [0, 1]
            1 machine [0, 1]
        Q: all-reduce & all-gather 的区别
    `torch.distributed`: wide support to a list of communication library

### Algorithm-side

支持 TP 和 PP 的并行 => vllm.model_executor.models.llama.py
TP 主要还是在 attention parallel => tp_size
head_num // tp_size => num_KV_heads     Q: 这个计算怎么来的

### PD Disaggregation