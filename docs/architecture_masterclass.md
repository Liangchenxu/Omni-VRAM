# Omni-VRAM 底层架构与面试白皮书

> 版本：v2.9.0 ｜ 适用代码：`vram_core/wddm_bridge.py`、`wddm_bypass/`、`WDDM-Bypass/csrc/`
> 本文所有结论都能回指到仓库里的具体文件行号；行号以 v2.9.0 提交为准（第 12 章为 v2.9.0 新增）。
> 文档本身不参与构建（纯 Markdown，位于 `docs/`），不影响 `setup.py` / `pyproject.toml` 打包。

---

## 0. 阅读指南：先说清楚“边界”

这套系统只做一件事，也只承诺一件事：

> **把 N 次 WDDM 提交压缩成 1 次 `cudaGraphLaunch`。**

它 **不是** 驱动补丁，**不是** 内核调度器替换，**也不会** 让 GEMM 变快。
在 Windows 上，用户态进程能控制的只有“提交次数”，控制不了 WDDM 的批量策略、GPU 时间片和分页行为。
凡是把这个项目描述成“绕过显卡驱动”的说法都是错的——本文用词一律是 **submission bypass（提交旁路）**。

三条硬边界，后面所有设计取舍都从这里推导：

| 边界 | 含义 | 代码证据 |
|---|---|---|
| 地址固定 | 捕获时记录的是裸地址，replay 不会重定位 | `WDDM-Bypass/csrc/include/graph_manager.h:192`（`GraphSlot`） |
| 形状固定 | 图只对捕获时的 grid/block/shape 有效 | `wddm_bypass/bucketing.py:53`（`ShapeKey`） |
| 拓扑固定 | `cudaGraphExecUpdate` 只能改参数/同类节点，不能加节点 | `WDDM-Bypass/csrc/graph_engine.cu:514`（`hot_update_slot`） |

### 路径映射（重要）

仓库里存在两个同名 `csrc` 语义：**源码在 `WDDM-Bypass/csrc/`，Python 包在根目录 `wddm_bypass/`**。
`WDDM-Bypass/` 只保留原生源码与构建脚本，`wddm_bypass/` 是可 `import` 的包（v2.6.0 合体结果）。

```
Omni-VRAM/
├─ vram_core/wddm_bridge.py        # 显存池 <-> 图引擎 的唯一接缝
├─ wddm_bypass/
│  ├─ __init__.py                  # Engine 门面：capture / replay / buffer / stats
│  ├─ bucketing.py                 # ShapeKey + BucketPolicy + ShapeRouter（纯 Python）
│  ├─ integration.py               # v2.8.0：解码层无感接管（DecodeHook / HookedDecoder）
│  └─ _wddm_bypass.cp310-win_amd64.pyd
└─ WDDM-Bypass/
   ├─ csrc/graph_engine.cu         # 引擎实现：捕获、实例化、热更新、静态缓冲
   ├─ csrc/bypass_core.cpp         # pybind11 绑定与异常转换
   ├─ csrc/torch_interop.cpp       # torch 互操作（data_ptr / ExternalStream / 当前流）
   └─ csrc/include/graph_manager.h # RAII 句柄、GraphSlot、CUDA_CHECK
```

---

## 1. 数据流全景图：从 `with capture_context:` 到 `cudaGraphExecUpdate`

### 1.1 一张图看完

```
[ Python 用户代码 ]
   with engine.capture(ShapeKey(1, 128, 768)) as gid:      # wddm_bypass/__init__.py:189
      ops.launch_scale(engine.buffer_ptr("hidden"), 98304, 0.5, engine.capture_stream())
                │
                │ ① begin_capture(key, bucket_str, hot_update)
                ▼
[ Engine 门面 ]  wddm_bypass/__init__.py:150
   graph_id = router.graph_id(key)                # 纯 Python，决定“槽位号”
   _raw.begin_capture(graph_id, str(bucket), hot_update)
                │
                │ ② pybind11 → C++
                ▼
[ C++ 引擎 ]  WDDM-Bypass/csrc/graph_engine.cu:381
   lock(mutex_)                                   # 单锁串行化
   activate_device()                              # cudaGetDevice / cudaSetDevice
   ensure_capture_stream()                        # 私有 non-blocking 流
   cudaStreamSynchronize(capture_stream_)         # 捕获前必须空闲
   cudaStreamBeginCapture(capture_stream_, ThreadLocal)
                │
                │ ③ 用户态 kernel launch（唯一合法目标流）
                ▼
   ops::launch_scale(dst_ptr, count, alpha, capture_stream_)   # graph_engine.cu:100
                │
                │ ④ end_capture()
                ▼
   cudaStreamEndCapture → staging_graph            # graph_engine.cu:432
   cudaGraphGetNodes(..., &recorded_nodes)         # 空图检测 :457
   ┌────────────── has_exec? ──────────────┐
   │ 否                                     │ 是 且 hot_update
   ▼                                        ▼
cudaGraphInstantiate(&exec, graph, 0ULL)    cudaGraphExecUpdate(exec, staging, &result)
   :503                                     :514（失败 → destroy + instantiate + rebuilds++）
                │
                │ ⑤ replay(key)
                ▼
   stream_from_token(0) → 当前流 / torch 流      # graph_engine.cu:57
   cudaGraphLaunch(exec, stream)                  # 一次提交，N 个节点
```

### 1.2 指针流转：三道关口

**关口一：注册期（地址唯一确定）**

```python
# vram_core/wddm_bridge.py:290
bound = bridge.bind_tensor("hidden", hidden_tensor)
# → engine.register_buffer("hidden", nbytes) → graph_engine.cu:694 → cudaMalloc
```

`register_static_buffer` 内部先 `cudaMalloc` 再替换 map 表项（`graph_engine.cu:694` 起）。
**结论：缓冲区一旦注册，其地址就是这张图的“身份”之一**；再次注册同名缓冲区会释放旧分配，
旧图里的节点指针立即悬空。v2.8.0 的 `DecodeHook.register()` 因此先查
`engine.buffer_bytes(name)`，只在“不存在或更小”时才分配（`wddm_bypass/integration.py`
的 `_engine_buffer_bytes`），这是踩过坑之后的硬约束。

**关口二：捕获期（只能发到捕获流）**

`cudaStreamBeginCapture(stream, ThreadLocal)` 之后，**只有发往该 stream 的 launch 会被记录**。
发到默认流的 launch 会真实执行一次，但不会进入图——重新播放时它就“消失”了。
引擎因此在 `end_capture` 里做了两层拦截：

```cpp
// graph_engine.cu:445 捕获结果为空
if (captured == nullptr)  throw EngineError("the capture contained no GPU work ...");
// graph_engine.cu:457 节点数为 0
if (recorded_nodes == 0)  throw EngineError("... launch the kernels on capture_stream_ptr() ...");
```

**关口三：回放期（流令牌）**

```cpp
// graph_engine.cu:57
cudaStream_t stream_from_token(std::uintptr_t token) noexcept {
    if (token != 0) return reinterpret_cast<cudaStream_t>(token);
#ifdef WDDM_BYPASS_WITH_TORCH
    return c10::cuda::getCurrentCUDAStream().stream();   // 跟随 torch 当前流
#else
    return static_cast<cudaStream_t>(nullptr);
#endif
}
```

`stream_token == 0` 是“跟随调用者当前流”的约定，不是“默认流”的硬编码。
在 PyTorch 场景下，`torch_interop.cpp:50` 的 `current_stream_token()` 把
`torch.cuda.Stream` 的 `cuda_stream` 整数交给 C++，两边共享同一套令牌语义，
避免“图跑在 A 流、张量在 B 流”的隐性竞争。

### 1.3 `vram_core` 侧的接缝：显存池怎么变成图

```python
# vram_core/wddm_bridge.py:312
def bind_paged_kv_cache(self, cache, name=DEFAULT_POOL_BUFFER):
    nbytes = int(cache.pool_bytes)
    shape_key = self.shape_key_for_cache(cache)   # ShapeKey(num_blocks, heads*block, head_dim)
    self._reserve(nbytes)                         # 预算 / VRAMOptimizer 闸门
    address = self._engine.register_buffer(name, nbytes)
    ...
    return BoundBuffer(name, nbytes, address, shape_key, "PagedKVCacheManager")
```

之后三步：

```python
bridge.capture(bound.shape_key)          # wddm_bridge.py:391
bridge.upload(name, tensor)              # :358 → cudaMemcpyAsync(cudaMemcpyDefault)
bridge.replay_for_cache(cache)           # :403 → 查 key → replay
```

`shape_key_for_cache` 把 KV 池的物理几何（块数、块长、head 维度）编码成 `ShapeKey`，
于是“池够大”就等价于“图可用”——同一个池服务所有能塞进池的序列长度，
这就是“一个物理池 + 多形状桶”的来源。

### 1.4 数据面 / 提交面 / 控制面的分离

| 平面 | 走什么 | 谁负责 | 代码 |
|---|---|---|---|
| 数据面 | `cudaMemcpyAsync(..., cudaMemcpyDefault, stream)` | `copy_to_static_input` / `copy_from_static` | `graph_engine.cu:786` / `:823` |
| 提交面 | `cudaGraphLaunch` | `replay` | `graph_manager.h:294` |
| 控制面 | `begin/end_capture`、`cudaGraphExecUpdate` | `WDDMBypassEngine` | `graph_engine.cu:381 / 420 / 514` |

显式分离的直接好处：**拷贝可以在图外做**。图里只有 kernel 节点，
所以“输入变了”不需要重建图，只要在 replay 前把新数据 memcpy 进同一块静态缓冲。
这就是 decode 循环（每步 token 都变、几何不变）能吃到收益的根本原因。

### 1.5 v2.8.0 新增：解码层的无感挂载路径

```
HuggingFace 侧                             Omni-VRAM 侧
────────────────────────────────────────────────────────────────────
model.forward(input_ids, ...)  ──►  HookedDecoder.forward({"x": hidden})
                                        │ ① feed: copy_into("x", tensor.data_ptr())
                                        ▼
                                    DecodeHook.execute(body, plan)
                                        │ ② replay: cudaGraphLaunch ×1
                                        │ ③ fetch: read_buffer("out", tensor.data_ptr())
                                        ▼
{"logits": ndarray / Tensor}   ◄──  outputs
```

`wddm_bypass/integration.py` 的三条不变量（与引擎一一对应）：

1. **固定几何**：`DecodePlan.key` 就是 `ShapeKey`；未捕获的几何按 `max_graphs` 预算走 eager 并计入
   `stats()["eager_fallbacks"]`，绝不静默重捕获。
2. **地址复用**：`register()` 只分配一次；同名更大尺寸才允许扩容，且必须在捕获前完成。
3. **数据显式搬运**：`feed` / `fetch` 在图外执行，捕获体里只有 launch，
   所以输入每步都能变而图不变。

真实 torch 张量走同一套路径：`data_pointer()` 接受 `data_ptr()` 整数，
`torch_stream(engine)` 把引擎的捕获流包成 `torch.cuda.ExternalStream`，
保证 torch 算子真的被录进图里（录到别的流上等于没录）。

---

## 2. 核心痛点攻坚记录

### 2.1 痛点 A：WDDM 提交队列饱和，eager 模式延迟越跑越差

**现象**：同一步计算，单次 launch 约 2.6 µs；连续提交 64 个 launch 后，单次提交涨到约 5.9 µs。
不是 GPU 变慢，是 **CPU 侧提交变慢**：WDDM 在用户态/内核态之间攒批，队列越深，每次
`cudaLaunchKernel` 的固定开销越高，并且这种恶化是非线性的。

**为什么难**：这是驱动行为，用户态没有 API 可以调队列深度、批大小或提交策略。
唯一能改的变量是 **提交次数**。

**解法**：把 64 次提交变成 1 次 `cudaGraphLaunch`。
参考机器（RTX 3060，源码 `WDDM-Bypass/README.md:191-230`）的实测：

| Buffer | Kernels/step | eager 单步 | graph 单步 | 收益 | eager 提交 | graph 提交 | 提交收益 |
|---|---|---|---|---|---|---|---|
| 16 KiB | 8 | 83.3 µs | 52.9 µs | 1.57× | 20.8 µs | 8.9 µs | **2.34×** |
| 16 KiB | 32 | 151.2 µs | 88.7 µs | 1.70× | 182.1 µs | 15.1 µs | **12.1×** |
| 16 KiB | 64 | 406.0 µs | 131.8 µs | 3.08× | 380.6 µs | 21.9 µs | **17.3×** |
| 256 KiB | 8 | 107.4 µs | 62.5 µs | 1.72× | 45.0 µs | 14.4 µs | 3.14× |
| 4 MiB | 8 | 318.8 µs | 288.9 µs | **1.10×** | 28.8 µs | 9.7 µs | 2.98× |

**怎么读这张表（面试时最容易被追问的点）**：

- 小 kernel 场景下，提交占总时间 94%，所以图收益接近理论上限（17×）。
- 4 MiB 那行只有 1.10×：GPU 已经变成瓶颈，图只能消掉提交间隙。
  **报告这一行，比报告 17× 更能证明测量是诚实的。**
- 图提交成本随节点数几乎不变（8→64 节点：8.9→21.9 µs），eager 则线性增长。
  这就是“代码路径不依赖模型有多少 kernel”的含义。

### 2.2 痛点 B：CUDA Graph 天生不支持动态 shape

**根因**：`cudaGraph_t` 记录的是**具体 grid/block/参数值**。形状一变，节点参数就得变；
朴素做法是“重新捕获 + 重新实例化”，而 `cudaGraphInstantiate` 是昂贵的一半：

| 操作 | 参考机中位数 |
|---|---|
| 冷捕获 + `cudaGraphInstantiate`（新 id） | 78.7 µs |
| `cudaGraphExecUpdate`（同拓扑合并） | 70.3 µs |
| `reset` + 捕获 + `cudaGraphInstantiate`（重建） | 77.9 µs |

**解法分两层**：

**第一层：多槽位 Shape Bucketing（把“无限形状”压成“有限桶”）**

```python
# wddm_bypass/bucketing.py
ShapeKey(batch, sequence, hidden, dtype)  →  BucketPolicy.bucket()  →  ShapeRouter.graph_id()
```

三种 regime：

| 策略 | 行为 | 适用 |
|---|---|---|
| `BucketPolicy()`（`step == 0`） | 每维向上取到 2 的幂 | 默认：一个图服务一整个 octave |
| `step > 0` | 向上取到硬件 tile（如 64/128） | 对齐良好的 matmul |
| `BucketPolicy.exact()`（`step == -1`） | 不合并，一形状一图 | 调试 / 极端确定性 |

图 id 是 **first-seen 稠密整数**，不是哈希：同样的形状序列在任何进程里都得到同样的 id，
日志可读、测试可断言、跨进程可比。这是“确定性”这一设计目标的直接体现。

**第二层：零重建热更新（`cudaGraphExecUpdate`）**

```python
with engine.capture(key, hot_update=True):   # 录进 staging graph
    ...                                      # 拓扑可以与线上不同
```

`end_capture` 的三条分支（`graph_engine.cu:469-487`）：

1. 没有 executable → 提升 staging 为正式图 + `cudaGraphInstantiate`；
2. 有 executable 且 `hot_update` → `hot_update_slot()` → `cudaGraphExecUpdate`，
   成功记 `updates++`；
3. 显式重捕获（非 hot_update）→ 销毁旧 exec 并重建，记 `rebuilds++`。

关键细节：**捕获永远写进 staging slot**（`graph_engine.cu:401-409`），
所以捕获中途抛异常时，旧的可执行图依然完整可用。这不是“顺手写的优雅”，
而是“捕获是把 stream 切进 capture mode 的全局状态操作”的必然要求。

`cudaGraphExecUpdate` 失败（拓扑差别过大，例如多了一个 kernel）→ 自动降级为
销毁 + 重新实例化，并 `rebuilds++`。**降级可见**是这里的设计重点：
如果某个工作负载每步都在 rebuild，日志会直接告诉你“它的拓扑和它被路由到的槽位不匹配”，
而不是悄悄变慢。

### 2.3 痛点 C：捕获期不允许分配（最容易踩的坑）

`cudaMalloc` 在 capture 区间内会返回 `cudaErrorStreamCaptureUnsupported` 一类错误。
项目用“**静态缓冲 + 预算闸门**”把分配彻底移出捕获区间：

```python
# vram_core/wddm_bridge.py:236 _reserve()
if self._budget_bytes is not None and self._committed_bytes + nbytes > self._budget_bytes:
    raise InsufficientVRAMError(...)
if callable(can_allocate) and not can_allocate(required_mb):
    raise InsufficientVRAMError(...)
```

顺序很重要：**先过闸门，再 `cudaMalloc`**。失败发生在驱动看到分配请求之前，
所以“预算不足”是可恢复的业务异常，而不是一次显存碎片化。

### 2.4 痛点 D：失败之后引擎处于什么状态

`begin_capture` 有两条熔断（`graph_engine.cu:386-396`）：

```cpp
if (capture_broken_)
    throw EngineError("the capture stream is unusable after a failed end_capture(); call reset_all() ...");
if (active_capture_id_ >= 0)
    throw EngineError("graph N is already capturing; CUDA captures cannot nest");
```

- `capture_broken_`：捕获流进入不可恢复状态后，后续操作全部拒绝，直到 `reset_all()`；
  防止“半坏状态”被继续使用。
- `active_capture_id_`：CUDA 捕获不可嵌套，用一个显式状态位代替“让驱动去报错”。

`end_capture` 里所有簿记（`slot.capturing`、`active_capture_id_`）都在可能抛异常之前复位
（`graph_engine.cu:434-436`），保证“抛异常后引擎仍然自洽”。

### 2.5 攻坚结果：一句话总结

| 矛盾 | 手段 | 代价 | 结果 |
|---|---|---|---|
| 提交次数 ↔ 延迟 | 单图 1 次提交 | 需静态缓冲 | 64 kernel 提交 17.3× |
| 动态 shape ↔ 图固定 | 多槽位 bucketing | 桶内浪费计算 | 一个图服务一个 octave |
| 形状变化 ↔ 重实例化 | `cudaGraphExecUpdate` | 拓扑变化会降级 | 100 次热更新 0 rebuild |
| 捕获 ↔ 分配 | 静态缓冲 + 预算闸门 | 显存提前占位 | 捕获区零分配 |
| 失败 ↔ 可用性 | staging + 熔断位 | 一次 reset | 旧图始终可回放 |

---

## 3. 面试夺命 15 问与硬核解答

> 定位：资深 CUDA / 系统架构岗。问题按“能不能把话说死在代码上”排序，
> 答案里出现的每个判断都能在仓库里找到出处。

### Q1. 你说 CUDA Graph“绕过”了 WDDM，到底绕过了什么？绕不过什么？

**绕过**：单步的提交次数。eager 时 64 个 kernel = 64 次
`cudaLaunchKernel`，每次都要走用户态 → 内核态 → WDDM 提交路径；
capture 之后是 1 次 `cudaGraphLaunch`，驱动内部展开（`WDDM-Bypass/csrc/include/graph_manager.h:294`
的 `replay` 声明）。

**绕不过**：WDDM 的驻留管理、分页、GPU 时间片、显存不足时的逐出。
**依然要过**：`cudaMemcpyAsync` 的 H2D/D2H（`graph_engine.cu:786`）。

面试里必须主动说清“提交面 vs 数据面 vs 执行面”，否则会被判定为“只会背 README”。

### Q2. 为什么用 `cudaStreamCaptureModeThreadLocal`，不用 Global？

`graph_engine.cu:413`：

```cpp
cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeThreadLocal);
```

`Global` 会把**进程内所有** host 线程的 CUDA 调用都视为非法（例如后台线程的
`cudaMalloc` 会直接让捕获失败），而本引擎只能约束自己这条流。
`ThreadLocal` 把约束收窄到当前线程，代价是：**其他线程若往捕获流上发操作，依然会被录进去**——
所以引擎用 `active_capture_id_` + 单锁把“谁能进捕获区”收紧，而不是靠 capture mode 兜底。

### Q3. 捕获期为什么不能 `cudaMalloc`？你们怎么保证零分配？

capture 记录的是“将要执行的图”，任何**新的**资源申请都无法被正确重放
（重放时分配已发生，地址会变），驱动因此直接报
`cudaErrorStreamCaptureUnsupported` 一类的错误。

保证手段有三层：

1. `register_buffer(name, bytes)` 在捕获**之前**完成分配（`graph_engine.cu:694`）；
2. `vram_core` 的 `_reserve()` 先过预算/`VRAMOptimizer` 闸门再分配（`wddm_bridge.py:236`）；
3. 捕获体只允许“指针 + 标量”的 launch，`DecodeBuffers` 不提供任何分配入口
   （`wddm_bypass/integration.py` 的 `DecodeBuffers`）。

### Q4. `cudaGraphExecUpdate` 什么时候会失败？失败之后为什么不能静默降级？

它只能做“**原地参数替换**”：节点数、节点类型、依赖拓扑必须兼容。
你在图里多加一个 kernel（比如换了个 kernel 融合策略），更新就会失败，
因为 executable graph 的节点表是固定的。

代码选择是**显式降级 + 计数**（`graph_engine.cu:514` 的 `hot_update_slot` →
失败路径销毁 exec 并 `cudaGraphInstantiate`，`rebuilds++`）：

```cpp
// graph_manager.h:297
// Merges the staging graph into the live executable with
// cudaGraphExecUpdate; falls back to destroy + instantiate when the
// topology changed too much for an in-place update.
```

静默降级的后果是“每步多花 70 µs 而没人知道”。把 rebuild 暴露成计数器，
等于让工作负载自己汇报“我的拓扑和槽位不匹配”，这是可观测性设计而不是保险丝。

### Q5. 捕获成功就一定可用吗？空图检测为什么必要？

不一定。`cudaStreamEndCapture` 成功但节点数为 0，说明**一个 launch 都没录进去**——
最常见原因是 launch 发到了默认流而不是 `capture_stream_ptr()`。
这种图实例化之后，每次 replay 都是**静默的 no-op**，比崩溃更难查。

引擎因此做两道拦截（`graph_engine.cu:445` / `:457`）：

```cpp
if (captured == nullptr)
    throw EngineError("the capture contained no GPU work ...");
if (recorded_nodes == 0)
    throw EngineError("... launch the kernels on capture_stream_ptr() ...");
```

**一句话**：捕获的失败模式是“安静地少做事”，所以必须主动把它变成异常。

### Q6. 多槽位路由为什么用 first-seen 稠密 id 而不是 shape 的哈希？

因为**可复现性**比“免碰撞”更重要。哈希值不可读、跨进程不稳定、日志里无法肉眼比对；
而 first-seen 整数保证了“同一串形状序列 → 同一串 id”（`wddm_bypass/bucketing.py:186`
`ShapeRouter` 的文档字符串直接把这个性质写成契约）。

代价是 id 依赖调用顺序：`ShapeKey(1,128,768)` 在进程 A 可能是 0，在进程 B 可能是 3。
所以 id **只能用于同进程内的槽位寻址和日志对比**，不能持久化、不能跨进程当键。
跨进程稳定的键是 `ShapeKey` 本身（或 `str(bucket)`，`graph_manager.h:284` 的 `shape_key`）。

### Q7. 幂次桶（`step == 0`）的隐藏代价是什么？什么时候该换成 `exact()`？

`BucketPolicy()` 把每一维向上取到 2 的幂：`seq=200 → 256`，浪费 22% 的算力；
`seq=1000 → 1024`，浪费 2.4%。**桶越"空"，浪费越大**，且浪费发生在 GPU 上（真实算力消耗），
不是可以忽略的簿记成本。

选择规则：

| 场景 | 策略 | 理由 |
|---|---|---|
| decode（seq=1，恒定） | 默认 | 只命中 1 个桶，无浪费 |
| 变长推理（seq 20~600） | 默认 | 用 12% 的算力换“图命中率≈100%” |
| 已知 tile 对齐（seq 是 64 的倍数附近） | `BucketPolicy(step=64)` | 桶更贴合，浪费更小 |
| 逐形状精确基准测试 | `BucketPolicy.exact()` | 测量不含桶浪费，但图数量爆炸 |

面试加分点：**桶浪费和提交开销是同一张预算表上的两个科目**，谁大就优化谁；
这也是为什么 `wddm_bypass/integration.py` 提供 `max_graphs`，而不是鼓励无限建桶。

### Q8. 为什么热更新要走 staging graph，而不是直接对 live graph 捕获？

因为捕获是“把 stream 切进 capture mode”的**全局状态操作**，一旦开始，
你就无法保证结束前不抛异常；如果直接覆盖 live graph，抛异常就等于同时失去
“旧的可执行图”和“新的捕获结果”。

`graph_engine.cu:401-409` 的做法：

```cpp
// Capture always goes to the staging slot, so a capture that throws at any
// point leaves the previous executable untouched and replayable.
GraphSlot& slot = slots_[graph_id];
slot.staging.reset();
slot.capturing_into_staging = hot_update;
```

于是不变量是：**live 的 `graph` / `exec` 只在 `end_capture` 的成功路径上被替换**。
代价是每个槽位多一个 `cudaGraph_t` 的生命周期，收益是“任何时刻 replay 都安全”。

### Q9. `GraphHandle` / `GraphExecHandle` / `StaticBuffer` 的 RAII 怎么保证不泄漏？

- `GraphHandle` / `GraphExecHandle`：禁止拷贝、只允许移动，析构调 `cudaGraphDestroy` /
  `cudaGraphExecDestroy`（`graph_manager.h:71` 起）；
- `StaticBuffer` 用 `std::unique_ptr<void, DeviceMemoryDeleter>`，deleter 里 `cudaFree`
  （`graph_manager.h:162`）；
- `WDDMBypassEngine` 的析构函数负责销毁所有图、exec 和缓冲（`graph_manager.h` 里的
  `~WDDMBypassEngine()` 声明注释：“Destroys every graph, executable and buffer, then the
  capture stream”）。

**关键点**：析构函数路径是 `noexcept` 语义下的 best-effort，不抛异常（CUDA 析构失败只能忽略）。
所以真正的错误处理发生在**构造/操作**路径（`CUDA_CHECK`），析构路径只负责“不泄漏”。
这也是为什么 `GraphHandle::reset()` 用 `noexcept`：如果析构能抛，栈展开会变成 `std::terminate`。

### Q10. 为什么整个引擎一把 `mutex_`？Python 侧还有 GIL，会不会双保险变双倍开销？

`mutex_` 保护的是**引擎内部状态**（slots、buffers、计数器、`active_capture_id_`），
与 Python 的 GIL 无关：`Engine` 的入口可能在 `Py_BEGIN_ALLOW_THREADS` 之外被调用，
而且 `WDDMBypassEngine` 可以被 C++ 侧或未来其他语言绑定直接使用。

粒度选择的理由：

- 单个 CUDA 上下文里，捕获/实例化/`cudaGraphExecUpdate` 本来就是全局串行的，
  细分锁只会增加“状态不一致”的可能；
- 每槽位一把锁会让 `reset_all()` 需要全部加锁，反而更容易死锁；
- **replay 持锁是唯一有争议的点**：`cudaGraphLaunch` 本身很快（µs 级），
  锁竞争只在高并发提交时出现；当前设计是“单生产者”的推理循环，取舍成立。
  如果未来要支持多流并发提交，正确做法是把 `slots_` 的查找和 `cudaGraphLaunch` 拆开，
  用 `shared_mutex` 让 replay 走读锁。

### Q11. pybind11 里注册异常，为什么必须把 `CudaError` 排在 `EngineError` 之前？

因为异常转换器是**按注册顺序倒序匹配**（后注册的先试），而 `CudaError` 继承 `EngineError`
（`graph_manager.h:39`）。如果先注册基类，`CudaError` 会被当成 `EngineError` 抛给 Python，
丢掉 `.status`（原始 `cudaError_t`）——调用方就无法区分
`cudaErrorMemoryAllocation`（可恢复：降预算重试）和
`cudaErrorIllegalAddress`（不可恢复：必须重启上下文）。

`bypass_core.cpp` 的注释写得很明确：“Registered after the EngineError handler, so it is tried first.”
同一个函数里还有一处同样关键的操作：`CudaError` 用 `PyErr_NewExceptionWithDoc` 手工建类型，
而不是 `py::register_exception`，目的就是能把 `.status` 挂到异常对象上。

**顺带一个坑**：`throw_cuda_error`（`graph_engine.cu:26`）在抛之前调用 `cudaGetLastError()`
清掉 sticky 错误位。不清的话，下一次 CUDA 调用会读到上一次的残留错误，
错误归因直接变成“张冠李戴”。

### Q12. 指针越界保护有哪几层？`device_pointer` 为什么拒绝 NumPy 地址？

四层，从上到下：

| 层 | 保护 | 代码 |
|---|---|---|
| Python | 类型/尺寸检查、预算闸门 | `wddm_bridge.py:236`、`integration.py` 的 `_check_size` |
| 桥接 | host 地址拒绝进入 device 语义 | `wddm_bridge.py:151-164` |
| 引擎 | 拷贝长度 ≤ 缓冲区大小 | `graph_engine.cu:772`（`copy_to_static_input`）、`:811`（`copy_from_static`） |
| 驱动 | CUDA 自身返回值 + `CUDA_CHECK` 上下文 | `graph_manager.h:63` |

`device_pointer` 拒绝 NumPy 的原因写在同一段注释里：`__array_interface__` 给的是**主机地址**，
把它交给 `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, ...)` 得到的是崩溃而不是错误码
（`wddm_bridge.py:161-164`）。**“能拿到整数”不等于“这个整数是设备地址”**，
所以这里选择“明确拒绝”而不是“猜一个方向”。

v2.8.0 的 `data_pointer()` 走的是另一条路：它允许 host 指针，因为引擎统一用
`cudaMemcpyDefault`（统一寻址下由指针自身决定方向，`graph_engine.cu:786` 的注释），
但**同一函数绝不用于 kernel 参数**——kernel 只接受 `DecodeBuffers.ptr(name)`。

### Q13. `stream_token = 0` 到底是什么语义？跨流依赖缺失会怎样？

`0` = “跟随调用者当前流”，不是“默认流”（`graph_engine.cu:57`）：

- 编译了 `WDDM_BYPASS_WITH_TORCH` → `c10::cuda::getCurrentCUDAStream().stream()`；
- 否则 → `nullptr`（legacy default stream）。

跨流依赖缺失的典型事故：你在 stream A 上 H2D 拷贝输入，在 stream B（图的回放流）上 replay，
两者没有事件同步 → 图可能读到旧数据。表现是“偶发数值错误”，且在快机器上更难复现。

**正确姿势**（引擎已经给了工具，但同步责任在调用方）：

```python
stream = bridge.engine.capture_stream()
bridge.upload(name, tensor)                      # 当前流
bridge.replay(key, stream_token=stream)          # 捕获流
# 两条流之间若存在数据依赖，必须插 event/wait，或统一用同一条流
```

`torch_stream(engine)` 的存在就是为了减少这类错误：它把引擎的捕获流**变成** torch 的当前流，
让“录制”和“回放”天然在同一个流上。

### Q14. `reset(key)` 为什么保留 shape → id 映射？这算不算泄漏？

保留映射是有意的（`wddm_bypass/__init__.py:351` 的文档字符串）：
“The shape -> id mapping survives, so the next capture for the same key reuses the same id
instead of leaking a new one.”

如果 reset 也清掉路由，那么“reset → 重新捕获”会拿到**新 id**，
槽位表随运行时间单调增长，最终把 `max_graphs` 预算吃光——这才是真泄漏。
保留映射意味着：id 是“形状的身份”，`reset` 只是把该身份的**资源**释放，
身份本身仍可复用。

显存侧则完全由 `StaticBuffer` 的 RAII 负责：`reset` 不释放缓冲区（缓冲区可能被其它几何共享），
`release_buffer(name)` 才释放；`engine.close()` 释放全部。

### Q15. 你怎么证明这些加速不是测量噪声？

四条硬规则，写在基准脚本与 README 里：

1. **中位数而非均值**：偶发调度抖动会把均值拉偏，中位数稳定。
2. **先 warm-up 再计时**：排除 CUDA 上下文初始化、cuBLAS handle、allocator 增长。
3. **区分“提交时间”和“单步时间”**：提交时间只包住 enqueue，末尾一次 `synchronize()`；
   单步时间是端到端。两者混在一起就会把“图让 GPU 更忙”误读成“图更慢”。
4. **报告不利结果**：4 MiB 那行只有 1.10×，照样写进 README
   （`WDDM-Bypass/README.md:191-230`）。只报 17.3× 的基准是不可信的。

**一句话**：能被证伪的基准才是基准；`tests/test_transformer_integration.py` 里的
GPU 用例沿用了同一套方法（3 轮取最优、一次性 synchronize、断言“至少 1.5× 才算过”）。

---

## 4. 附录：怎么复现

```powershell
# 1) 全量引擎 + 桥接测试（GPU 用例需要已构建的扩展）
python -m pytest tests/test_engine_suite.py -q --tb=short

# 2) v2.8.0 解码层挂载测试（纯 Python 部分不需要 GPU）
python -m pytest tests/test_transformer_integration.py -q --tb=short

# 3) 重建扩展（需要 vcvars64 + softmax 环境）
cd WDDM-Bypass
python setup.py build_ext --inplace
```

v2.8.0 挂载 API 速查（`wddm_bypass/integration.py`）：

| 名称 | 作用 |
|---|---|
| `DecodePlan.build(key, shapes, inputs=, outputs=)` | 固定几何 + 命名缓冲 + 输入输出声明 |
| `DecodeHook(engine, plan=, warmup=, max_graphs=)` | 捕获/回放/降级/计数 |
| `hook.execute(body, feed=, fetch=)` | 一次 decode step：feed → (capture \| replay) → fetch |
| `accelerate_decode(hook)` | 装饰器：首调用捕获，后续只 replay |
| `HookedDecoder(body, plan).forward({...})` | HF 形状的 `forward`，返回 numpy 输出 |
| `torch_stream(engine)` | 把捕获流包成 `torch.cuda.ExternalStream` |

**边界复述**：这套东西只压缩提交次数。GEMM、attention 的算力、
显存带宽都不会因为“用了图”而变快；把项目讲成“比 cuBLAS 更快”就是撒谎。






---

## 12. v2.9.0：4 MiB 带宽墙、外部流捕获与真实 `nn.Module` 接管

### 12.1 先把边界钉死：图不变快，融合才变快

第 0 节的三条硬边界在 v2.9.0 里一条都没被推翻：**CUDA Graph 只压缩提交次数，不改变
kernel 本身的算力和访存**。v2.9.0 的提速来自另一条完全不同的路径——**把三个 kernel 合成一个、
并把访存升到 128-bit**。这两件事必须分开讲：把"用了图所以变快"和"少搬显存所以变快"混为一谈，
和把项目说成"绕过驱动"一样是错的。

### 12.2 HBM 流量数学模型

未融合（`scale` → `add` → `relu`，三个 kernel）：

```text
Kernel 1  scale:  tmp1 = a * s          1 Read (a)        + 1 Write (tmp1) = 8  B/elem
Kernel 2  add:    tmp2 = tmp1 + b       2 Reads (tmp1,b)  + 1 Write (tmp2) = 12 B/elem
Kernel 3  relu:   out  = max(0, tmp2)   1 Read (tmp2)     + 1 Write (out)  = 8  B/elem
-------------------------------------------------------------------------------
Total_HBM_Traffic_Unfused = 28 B/elem
N = 1,048,576 float32 (4 MiB):  1,048,576 * 28 B = 28 MiB per call
Arithmetic_Intensity = 3 FLOPs / 28 B = 0.107 FLOP/Byte    <- 完全受限于显存带宽
```

寄存器融合 + `float4`（128-bit）向量化：

```text
单 kernel：out = max(0.0f, a * scale + b)，中间结果只活在寄存器 (RF) 里
1 Read (a) + 1 Read (b) + 1 Write (out) = 12 B/elem      (原地写回则为 8 B/elem)
-------------------------------------------------------------------------------
Total_HBM_Traffic_Fused = 12 B/elem  ->  12 MiB per call
Bandwidth_Reduction_Ratio = 28 / 12 = 2.33x
Arithmetic_Intensity = 3 FLOPs / 12 B = 0.250 FLOP/Byte
```

4 MiB 恰好是一个尴尬尺寸：数据小到提交开销仍占一定比例，又大到已经撞上带宽墙。
所以 v2.8.0 在那个尺寸上只剩 **1.10x**——提交次数砍掉了，流量一点没少。

### 12.3 实现要点（可回指源码）

| 要点 | 做法 | 位置 |
|---|---|---|
| 融合内核 | `out4[i]` 四个分量各做 `fmaxf(0, fmaf(a, s, b))`，一次 LDG.128 + 一次 STG.128 | `WDDM-Bypass/csrc/graph_engine.cu:147` |
| 对齐判定 | 三个指针按位或后 `& 0xF`；不齐则回落标量路径，算术完全等价 | `graph_engine.cu:157` |
| 尾部元素 | `vector_count = n >> 2`，剩下 1~3 个走 `tail_start + idx` 的标量循环 | `graph_engine.cu:180` |
| 模块级发射器 | `ops::launch_fused_affine_relu(dst, a, b, scale, count, stream_token)` | `graph_engine.cu:249` |
| 引擎级发射器 | 按名字解析三个静态缓冲、校验 `count * 4 <= bytes`，`count<=0` 直接拒绝 | `graph_engine.cu:961` |
| 捕获内默认真流 | 捕获中 `stream_token=0` 表示"正在录制的那条流"，不写 token 也能录进图 | `graph_engine.cu:1002` |
| 计数 | `stats()["fused_launches"]` | `graph_engine.cu:1060` |
| Python 门面 | `Engine.launch_fused_affine_relu()` / `DecodeBuffers.fused_affine_relu()` | `wddm_bypass/__init__.py:351`、`wddm_bypass/integration.py:375` |

### 12.4 实测（RTX 3060 / compute capability 8.6 / 1,048,576 float32 = 4 MiB）

计时方式：每轮 40 次迭代、5 轮取中位数，**每轮前后 `torch.cuda.synchronize()`**。
不用 CUDA event——记录在默认流上的 event 不会被另一条流上的工作排序，那样量到的只是提交时间
（早期版本就踩过这个坑，读到 0.00005 ms 这种明显不可能的数字）。

| 路径 | 每元素 HBM 流量 | 中位耗时 | 相对 fused |
|---|---|---|---|
| eager：`relu(add(mul(a,s), b))` 三次独立发射 | 28 B | 0.1004 ms | 1.78x 慢 |
| 三次 kernel 已捕获成图（只砍提交次数） | 28 B | 0.1049 ms | 1.86x 慢 |
| **融合 1 个 kernel（图回放）** | **12 B** | **0.0565 ms** | 1.00x |

两行对照很关键：把三 kernel 捕获成图，在 4 MiB 上几乎没有收益（0.1049 vs 0.1004 ms，
提交开销已经被数据量摊薄）；换成融合 kernel 后直接掉到 0.0565 ms。
**收益来自流量，不来自图。**

### 12.5 外部 PyTorch CUDA Stream 的非拥有式捕获

PyTorch 的算子（`torch.mm`、`nn.Linear`、`F.silu`）发射在
`torch.cuda.current_stream().cuda_stream` 上。v2.8.0 的引擎只会捕获自己创建的内部流，
于是"真实模型的 kernel"永远进不了图。v2.9.0 的改动：

```cpp
cudaStream_t WDDMBypassEngine::resolve_stream(std::uintptr_t external_stream_ptr) const noexcept {
    if (external_stream_ptr != 0) {
        return reinterpret_cast<cudaStream_t>(external_stream_ptr);  // 借用，不拥有
    }
    return capture_stream_;                                          // 引擎自己的流
}
```

- `begin_capture(graph_id, shape_key, hot_update, stream_ptr)`：`stream_ptr != 0` 时用
  **`cudaStreamCaptureModeRelaxed`** 录制（`graph_engine.cu:511`）。
  这是关键点：录制期间 aten / cuBLAS 会查询流与分配器状态，严格模式
  （`Global` / `ThreadLocal`）会把这类查询判成 `cudaErrorStreamCaptureUnsupported`，Relaxed 放行。
- 内部流路径仍用 `ThreadLocal`，`stream_ptr=0` 的行为与 v2.8.0 完全一致，
  所以原有引擎用例不需要改一行（这也是默认参数的唯一目的）。
- `end_capture()` 关闭的是 `active_capture_stream_`（`graph_engine.cu:537`），
  而不是"当前内部流"——否则外部流永远停在捕获态。
- 析构函数只 `cudaStreamDestroy(capture_stream_)`；外部流归 PyTorch 所有，引擎连引用都不碰。

### 12.6 为什么必须先预热（最容易踩的一条）

```python
entry["static_in"].copy_(x, non_blocking=True)
self.engine.begin_capture(entry["slot"], stream_ptr=stream_ptr)
output = self.module(entry["static_in"])      # cuBLAS GEMM + ATen kernel
entry["static_out"].copy_(output, non_blocking=True)
self.engine.end_capture(entry["slot"])
```

首次调用会做三件"分配类"的事：cuBLAS 建 workspace、torch caching allocator 为中间张量建块、
kernel 选择 launch 配置。**捕获期间分配正是捕获失败的经典原因**，所以
`TorchGraphModule` 先在同一私有流上跑 `warmup_steps` 次 eager（默认 3），
把这些一次性开销挪到捕获之外。

### 12.7 `TorchGraphModule` 三段式

| 阶段 | 触发条件 | 做了什么 | stats 计数 |
|---|---|---|---|
| Warm-up | 该形状调用次数 < `warmup_steps` | eager 跑在私有流上，分配 `static_out` 场 | `Warmups` |
| Capture | 预热结束且未捕获 | 绑定外部流 → `begin_capture` → 前向 → 拷出 → `end_capture` | `Captures` |
| Replay | 已捕获 | 拷入 + 一次 `cudaGraphLaunch` 跑完整步 | `Replays` |
| Eager 回退 | 非 CUDA 张量 / 槽位预算耗尽 | 直接调用原模块 | `EagerFallbacks` |

回放返回的是**静态输出场本身**，下一次回放会覆盖它；需要留存就 `clone()`。
这是刻意的：回放路径上不允许再分配，否则"零重建"就名不副实。
入口在 `wddm_bypass/integration.py:1121`，槽位键在 `integration.py:1224`。

### 12.8 一个真实踩坑：`BucketPolicy` 会把槽位编号并桶

第一版把槽位编号塞进 `ShapeKey` 的整数维度（`ShapeKey(1, 1, 320)`），结果两个不同形状
只产生了 1 个图。原因在 `wddm_bypass/bucketing.py:161`：`BucketPolicy()` 的三个 step 默认是 0，
`_round()` 走 `power_of_two_at_least()`——320 和 321 都归到 512，落进同一个桶，共用一个 graph id。
测试里表现为 `Captures == 2` 但 `engine.graphs() == [0]`，很容易误判成"第二次捕获没生效"。

修正：唯一性放进 **`dtype`** 字段（`ShapeKey(1, 1, 1, "slot-320")`），因为 `bucket()` 会把
`dtype` 原样带过（`bucketing.py:175`），不参与取整。

### 12.9 Windows DLL 自愈

直接跑 `python.exe` 导入链接 cudart 的扩展时，有两个坑：

1. 依赖缺失会弹出**模态**对话框，进程卡住等用户点确定。`SetErrorMode(0x8003)`
   = `SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX`
   把它变成可捕获异常。
2. Python 3.8+ 用受限搜索标志加载扩展，`PATH` 不再生效；必须
   `os.add_dll_directory()` 注册 Conda 的 `Library\bin`、`DLLs`、`Scripts` 与 CUDA 的 `bin`。

`wddm_bypass/backend.py:42` 的 `heal_windows_dll_environment()` 一次做这两件事，
`load_backend()` 顶部调用，`vram_core/__init__.py` 导入时也调一次。
幂等：内部记 `_DLL_HEALED` / `_HEALED_DIRS`，第二次调用返回空列表且不重复注册。

### 12.10 复述边界（v2.9.0 版）

- **图**：只压提交次数。4 MiB 数据量下，三 kernel 捕获成图几乎没有收益（0.1049 vs 0.1004 ms）。
- **融合**：压 HBM 流量，28 → 12 B/elem，实测 1.78x（对比 eager 三发射）、1.86x（对比三 kernel 图）。
- **外部流捕获**：让真实 cuBLAS / ATen kernel 能被录进图，**它本身不产生任何加速**；
  加速仍来自"这些 kernel 原本要提交 N 次"。
- 一句话：v2.9.0 的 1.78x 是**算子融合 + 向量化访存**的功劳；
  CUDA Graph 的功劳仍然是那 17.3x 的提交开销削减，两者不能互相冒充。


