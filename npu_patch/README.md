# npu_patch（MoonEP Ascend NPU 后端）

`npu_patch` 是 MoonEP 仓库内为 **Ascend（NPU）** 提供的后端模块，由
[`ascend-moonep`](../../../ascend-moonep) 迁移而来，保持 Python `Buffer` API
（shmem 对称内存，`dispatch` / `prefetch_weight` / `combine` / `reduce_grad`）
与调用面兼容。C++ runtime 负责 CANN SHMEM 生命周期、对称内存与 runtime
workspace 管理；AscendC kernel 经预编译的 `ascend_moonep_kernels`（native
CMake）暴露 C ABI launcher，再由 PyTorch C++ extension `npu_patch._C` 调用。

当前构建目标仅面向 **Ascend950 / A5**：UDRMA/UDMA（URMA）通信依赖 A5 的
对称内存 RMA 能力（SIMT），故不支持 A2/A3 等其它芯片。

## 目录结构

```text
npu_patch/
├── 3rdparty/
│   ├── shmem/                    # CANN SHMEM submodule
│   └── catlass/                  # CATLASS submodule
├── __init__.py                   # 导出 Buffer / MoonEPCommPlan / ShmemRuntime
├── api.py                        # MoonEP Buffer API（shmem 对称内存）
├── buffer_c.py                   # 对称分配/包装 + 框架侧 VMM 助手（create_vmm_physical 等）
├── planning.py                   # MoonEPCommPlan / 计划输出分配
├── shmem_runtime.py              # ShmemRuntime（shmem Python API 生命周期）
├── kernel_op.py / ops.py         # KernelOp 基类与各算子对象（AscendC 迁移接缝）
├── launchers.py                  # moonep launch_* 兼容 shim（launch_planning / launch_dispatch/…）
├── _common.py                    # 2MiB 对齐 / sym put-get / 跨 rank 屏障
├── _device.py                    # 把 ASCEND_RT_VISIBLE_DEVICES 映射成可见索引 0..N-1
├── native/                       # CMakeLists、AscendC/Bisheng kernel、C++ host csrc
│   ├── csrc/ops/                 # 每个 Python 侧算子的参数校验、输出分配、tiling 构造
│   ├── kernels/                  # AscendC kernel launcher 与 kernel（含 *_udma 变体）
│   └── include/ascend_moonep/kernels/
├── scripts/
│   ├── build_base.sh             # 唯一构建实现（CANN 发现 / shmem / kernel 库 / pip install）
│   ├── build_a5.sh               # Ascend950/A5 薄封装
│   └── run_test.sh               # 测试驱动（含 BUILD=1 重建）
├── benchmarks/                   # MoonEP benchmark 集（--engine {ubmem,urma}）
└── lib/                          # 随包分发的 kernel / shmem .so（构建产物）
```

## 获取代码

```bash
git clone --recursive <repository-url>
cd MoonEP
```

已有仓库时初始化 submodule：

```bash
git submodule update --init --recursive
```

当前 submodule：

```text
npu_patch/3rdparty/shmem    https://gitcode.com/cann/shmem.git
npu_patch/3rdparty/catlass  https://gitcode.com/cann/catlass.git
```

## 构建

先加载 CANN 环境（未设置 `ASCEND_HOME_PATH` 时构建脚本会自动发现）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

一键构建（Ascend950 / A5）：

```bash
bash npu_patch/scripts/build_a5.sh [CANN_SET_ENV]
```

或显式指定 SoC：

```bash
bash npu_patch/scripts/build_base.sh -soc_type Ascend950 [CANN_SET_ENV]
```

`build_base.sh` 是构建逻辑的**唯一实现**，依次：

1. （如配置CANN_SET_ENV）`source` CANN `set_env.sh`；
2. `git submodule update --init --recursive`（cann/shmem、catlass）；
3. 为当前 SoC 构建 cann/shmem 到 `npu_patch/3rdparty/shmem/install`；
4. `cmake -S npu_patch/native -B <build dir>`，用 Bisheng 编译 kernel，
   生成 `ascend_moonep_kernels`；
5. 将 kernel 与 shmem `.so` 拷入 `npu_patch/lib/`；
6. `pip install -e .`，编译并安装 `npu_patch._C` PyTorch extension。

> kernel `.so` 与 host 扩展 `npu_patch._C` 是**链接**关系，必须整套跑完，
> 不能只跑其中一步。可用环境变量 `FORCE_SHMEM=1`（重建 shmem）、
> `SHMEM_HOME_PATH`（复用已构建 install）、`BUILD_DIR`（指定 cmake 构建目录）
> 调整。

## 运行测试

统一入口是 `npu_patch/scripts/run_test.sh`（仓库根目录执行即可），按引擎拆分
**ubmem（MTE/AscendC，默认）** 与 **urma（URMA/UDMA，仅 A5/Ascend950）**：

```bash
# ubmem / MTE 引擎（默认）
bash npu_patch/scripts/run_test.sh
# urma 引擎
bash npu_patch/scripts/run_test.sh --use_urma
# 生产 8 卡形状（S=4096, K=16, H=7168, E=896；要求 RANK_N=8）
bash npu_patch/scripts/run_test.sh --use_urma --prod
```

`run_test.sh` 覆盖三类测试：

1. **Buffer 公共 API（`test_npu/buffer_test`）**：`test_buffer_*` pytest 文件按
   `--use_urma` 用 `pytest -k` 筛选 parametrize 出的 `urma` / `mte` 变体；脚本式
   C / cpython 接口测试（`test_buffer_c_api` / `test_buffer_cpython_api`）引擎
   不可切、原样运行；分组 e2e（`test_buffer_grouped_e2e`）单独一节、不带 `-k`。
2. **urma 算子（`test_npu/urma`，仅 `--use_urma` 且仅 A5）**：`test_*_udma` +
   各 UDMA 脚本式算子测试（`test_moonep_combine_udma_with_plan` /
   `test_moonep_dispatch_udma_with_plan` / `test_dispatch_combine_udma_joint`）。
   去重域（rank / server）与 kernel 族都在这里覆盖：
   - dispatch / joint / combine 各跑 `--dedup-mode rank` 与 `server` 两遍（server 需要
     `RANK_N % 8 == 0` 且 `K >= 2`）；
   - server dispatch 带 `--plan-source kernel --output-mode both`：真正执行
     `planning_server_dedup` 内核（含融合 combine plan），并同时校验 zero-copy 与
     copy 输出；
   - combine 的 rank 分支默认保持 legacy MTE 内核，另有一条小 shape
     `--combine-engine udma` 用例覆盖冻结的 rank 级 UDMA combine 内核；
   - `test_e2e_urma.py` 参数化 `dedup_mode=rank|server`（server 在非 A5 / R%8!=0 时
     skip 而不是 FAIL），并新增与去重无关的 `K * hidden` 身份校验。
3. **mte ubmem 算子（`test_npu/mte`，默认）**：`test_*` + `test_planning` /
   `test_moonep_combine` / `test_moonep_dispatch_with_plan` /
   `test_dispatch_combine_joint`；`--prod` 时同样切生产形状。

> 8 卡单机只有 1 个 server，跨 server 分支需要 R>8（更大单机或按 README 的多机
> 命令分别启动三个算子测试）。

多进程下所有 rank 使用相同的 shmem bootstrap：单机走 UID bootstrap
（`ShmemRuntime.init_with_buffer`），多机走 ip_port/DEFAULT bootstrap（见
「多机运行（buffer-backed + ip_port bootstrap）」）。

## Python API

`moonep` 顶层已按后端替换，统一
`from moonep import ...` 即可拿到Ascend实现（当 NPU 后端启用时，`moonep.Buffer` /
`moonep.MoonEPCommPlan` / `moonep.ShmemRuntime` 即 `npu_patch` 的同名对象，并保留 `moonep.init_shmem`，见 `MoonEP/moonep/__init__.py`）。

对外可用API：

- **MoonEP 风格 Buffer API**（推荐，Python + shmem 对称内存）：`Buffer` /
  `MoonEPCommPlan` / `ShmemRuntime`，`dispatch` / `prefetch_weight` / `combine` /
  `reduce_grad` 用法见下一节。

```python
# Buffer / Plan / shmem / init 一律从 moonep 顶层入口拿（NPU 上已替换为 npu_patch 实现）
from moonep import Buffer, MoonEPCommPlan, ShmemRuntime, init_shmem
# 框架侧对称张量助手统一走 moonep.buffer，下列都是 NPU 专有名字
# （GPU 进程里不存在，也无需使用）：
from moonep.buffer import (
    create_full_weight,                  # 创建并填充 full weight（[E+B,H,H']）
    create_full_grad_and_reduce_buffer,  # reduce_grad 的 (full_grad, reduce_buffer)
    remote_home_rows,                    # 按 pe 点读 home chunk（免 bind，init 后可用）
    read_reduce_slot,                    # 按 pe 点读 reduce slot chunk（免 bind）
    expert_pool_row,                     # 全局 expert 的 [H, H'] 源行（init 后免 bind）
)
# 说明：GPU 的 create_nvl_dist_tensor / create_nvl_single_owner_tensor /
# pad_dim0_for_alignment 在昇腾侧没有对应实现——NPU 不需要 NVLink 映射，
# 跨 rank 张量由 shmem 对称堆 + 上面的构造器/点读助手覆盖。
# 说明：buffer_c.create_vmm_physical / launch_* 等是低层内部实现，应用侧无需
# 直接 import npu_patch —— 与 tests/test_e2e_npu_gpu.py 的写法一致。
```

## MoonEP Buffer API（shmem 对称内存）

`moonep.Buffer`（NPU 上即 `npu_patch` 的实现，见前节）是 `moonep/api.py`
Buffer 类的昇腾移植：在 `Buffer` 构造时
一次性分配 shmem 对称通信内存（`hidden_buf` / 各 meta 独立缓冲 / plan 广播区，
各变量单独分配 shmem 地址，无合并 `meta_buf`、无 `xxx_off`）并实例化一组
`KernelOp` 算子对象，`dispatch` / `prefetch_weight` / `combine` / `reduce_grad`
调用时执行 `op.compute`。跨 rank 访问走对称寻址：host 侧 `put/get`（v1），
kernel 侧 `aclshmem_ptr`（AscendC 迁移接缝）。

Ascend多卡传输每个对称块按实际 shape 精确分配
（各 meta 变量单独分配 shmem 地址），buffer-backed 对称堆的 descriptor
总大小按 ACL 查询到的 VMM 粒度对齐。默认使用 2 MiB 页；页大小按**两个独立的作用域**
解析，训练框架参数与 MoonEP dispatch/combine 参数可以分别指定：

| 作用域 | 覆盖对象 | 进程级环境变量 | API 覆盖 |
|---|---|---|---|
| `moonep` | Buffer 的通信块：`hidden_buf` / server `combine_workspace` / comm arena | `ASCEND_MOONEP_VMM_PAGE_SIZE` | `Buffer(vmm_page_size=...)` |
| `framework` | 训练框架参数池：`create_full_weight` / `create_full_grad_and_reduce_buffer` 的 home/slot 块 | `ASCEND_MOONEP_FRAMEWORK_VMM_PAGE_SIZE`（未设置时回落到 `ASCEND_MOONEP_VMM_PAGE_SIZE`） | 上述 builder 的 `vmm_page_size=...` |

解析优先级统一为：显式参数 > 该作用域的环境变量 > 默认页（`framework` 作用域先查自己的
环境变量，再回落 `ASCEND_MOONEP_VMM_PAGE_SIZE`；都没有则为 `2m`）。
`create_vmm_physical(shape, dtype, device, page_size=...)` 上仍可**按 allocation** 选择页大小：

- `page_size=None` / `"default"`：沿用作用域默认值（见上面的作用域表；`moonep` 作用域还受
  去重域影响，见下文），payload 数据块与 comm arena 一致；
- `"1g"`：1 GiB 大页（ACL 默认）。ACL 按 1 GiB 粒度向上取整，对数据面大块
  （`hidden_buf` / server 的 `combine_workspace`）收益最大；Buffer 的 comm arena
  同样随该策略，整个 arena 会取整成一个 1 GiB 对称堆块；
- `"2m"`：2 MiB 大页（进程默认）。ACL 会按粒度向上取整，因此 KiB~MiB 级的 metadata / sync /
  workspace chunk 用 2 MiB 页，可避免每块白占 1 GiB 对称堆。`Buffer` 把小尺寸通信块
  （weights / barrier / combine_sync / dispatch_sync / planning workspace）打包进
  一个 `create_vmm_arena(...)` 分配；算子不再各自持有
  op workspace（planning 借用 arena 内的一视图）。

`moonep` 作用域的默认页还取决于去重域：`dedup_mode="server"` 时默认 **1g**，rank 去重时默认
**2m**。server 默认开 1g 会产生一条 warning，说明 1g 大页已开启、性能提升，同时给出该 Buffer
的显存占用总额（按页粒度取整后的 VMM 总量，单位 GiB），并提示可通过 `vmm_page_size="2m"`
关闭（或导出 `ASCEND_MOONEP_VMM_PAGE_SIZE=2m` 进程级关闭）。显式传入 `vmm_page_size` 或导出
环境变量时不再告警。当前页策略可用 `Buffer.vmm_page_size` 读取。

不同页大小可混用（每个 descriptor 携带自己的物理属性），但 descriptor 顺序
必须满足“大页在前、小页在后”：ACLSHMEM 会按**各 buffer 自己的粒度**校验累计
`segment_offset`，1 GiB 块之后接 2 MiB 块时前缀才是两者共同对齐的。
`symmetric_descriptors()` 已按粒度降序稳定排序；若手工传 `descriptors`，需自行保证该顺序。
只导出 `ASCEND_MOONEP_VMM_PAGE_SIZE` 时两个作用域都走同一页大小，等同于改动前的“进程级统一页”行为；
`ASCEND_MOONEP_FRAMEWORK_VMM_PAGE_SIZE` 用于把训练框架参数单独切到另一个页大小。
两者都只支持 `1g` / `2m`（默认 `2m`，其他值会在分配前报错），所有 rank 须使用相同设置。
`[E+B]` 投影仍要求 `epn * H * Hp * element_size` 是实际粒度的整数倍；
URMA E2E 测试显式选用 2 MiB 页以覆盖其小尺寸投影。对称块不额外创建 shmem Buffer 对象：tensor
本身即对称块，下游算子通过 `tensor.data_ptr()` 获取对称基址（host 侧
put/get/signal/free 需要 Buffer 时现场构造）。

### 使用流程

当前只有 buffer-backed 初始化（`ShmemRuntime.init_with_buffer` /
`init_with_ip_port`）：**先构造 Buffer 与框架侧全部对称 VMM chunk（登记进全局
descriptor 注册表），再做一次对称堆 init，之后 Buffer 在首次
dispatch/combine/prefetch/reduce_grad 时把 chunk 翻译成内核要的对称地址**。
顺序不可颠倒：init 的 descriptor 集在 init 时冻结，之后创建的 VMM chunk 不会
进入当前对称堆。

```python
import torch.distributed as dist
from moonep import Buffer, ShmemRuntime

dist.init_process_group(backend="hccl")
R = dist.get_world_size()
E, S, H, K = 64, 4096, 256, 16
epn = E // R
B = epn                    # prefetch / 归约槽位数，默认等于 epn

# 1. 构造 Buffer（此时仅登记 VMM descriptor，shmem 尚未 init）
use_udma = True            # False = MTE（AscendC）；True = URMA/UDMA（仅 A5）
buffer = Buffer(S, H, K, E, R, B=B, num_sms=32, group=dist.group.WORLD,
                explicitly_destroy=True, use_udma=use_udma)

# 2. 构造框架侧 full weight / full grad / reduce buffer（见下节，
#    同样必须在 init 之前登记 descriptor）。

# 3. 一次性 buffer-backed 对称堆 init（单机 UID bootstrap；多机见下）。
#    enable_udma 必须与 buffer 的 use_udma 一致；URMA 参考 qp_count=2。
ShmemRuntime.init_with_buffer(group=dist.group.WORLD,
                              enable_udma=use_udma, qp_count=2)

# 4. 首次 dispatch/combine/prefetch/reduce_grad 时 Buffer 自动翻译对称地址。

# 5. 释放顺序：先 finalize 再 destroy（VMM 逆操作）。
ShmemRuntime.finalize()
buffer.destroy()
```

### ubmem（MTE）与 urma（URMA/UDMA）路径：初始化与调用

planning / dispatch / combine / prefetch / reduce_grad 各有两条 kernel 路径，
通过 `Buffer` 构造时的 `use_udma`（及每算子 `*_use_udma` 覆盖）选择：

- **ubmem / MTE 路径**（`use_udma=False`，默认）：AscendC kernel 走 MTE2/MTE3
  搬运，无 UDRMA。无论传不传 `use_udma`，非 A5 芯片都会自动回落 MTE。
- **urma / UDMA 路径**（`use_udma=True`，仅 Ascend950/A5）：dispatch / combine /
  prefetch / reduce_grad 走 UDRMA RDMA 直写；planning 默认保持 MTE
  （`planning_use_udma=False`）。

`Buffer` 构造时会设置进程级环境变量（`ASCEND_MOONEP_USE_UDMA` 与每算子
`ASCEND_MOONEP_{OP}_USE_UDMA`），kernel launcher / 算子据此选择 native 或 UDMA
变体；无需手动 export，但可用环境变量临时覆盖。

```python
# urma：dispatch/combine/prefetch/reduce_grad 走 URMA，planning 留 MTE
buffer = Buffer(S, H, K, E, R, B=B, num_sms=32, group=dist.group.WORLD,
                explicitly_destroy=True, use_udma=True, planning_use_udma=False)
# ubmem/mte 全路径：全部算子走 AscendC
buffer = Buffer(S, H, K, E, R, B=B, num_sms=32, group=dist.group.WORLD,
                explicitly_destroy=True, use_udma=False)
```

初始化时 `enable_udma` 必须与 `use_udma` 一致（URMA 对称堆带 UDRMA/QP 配置），
URMA 用 `qp_count` 指定每 rank 的 Dispatch UDMA QP 数（`0` 表示未配置，回落到
`ASCEND_MOONEP_QP_COUNT`，再回落 2）。服务端去重（`Buffer(dedup_mode="server")` 已发布
`ASCEND_MOONEP_SERVER_DEDUP=1`）时该值恒为 2：传其它值会被覆盖并给出 warning。

```python
# urma：enable_udma=True + qp_count
ShmemRuntime.init_with_buffer(group=dist.group.WORLD, enable_udma=True, qp_count=2)
# ubmem/mte：enable_udma=False
ShmemRuntime.init_with_buffer(group=dist.group.WORLD, enable_udma=False, qp_count=0)
```

测试侧统一入口是 `test_npu/kernel_test_utils.py::init_shmem`（自动选单机 UID /
多机 ip_port bootstrap，`enable_udma` 透传）；调用仍走公共 Buffer 方法
`buffer.dispatch(...)` / `buffer.prefetch_weight(...)` / `buffer.combine(...)` /
`buffer.reduce_grad(...)`。端到端参考：`test_npu/mte/test_e2e.py`（ubmem）与
`test_npu/urma/test_e2e_urma.py`（urma）。

### 框架层要创建的对象：Full Weight / Full Grad / Reduce Buffer

调度侧（dispatch/combine）的通信 chunk 与 plan 由 `Buffer` 内部管理；框架层
需要为 **prefetch_weight（权重侧）** 和 **reduce_grad（梯度侧）** 准备三组对称
张量。下列构造器是 **NPU 专有**名字（仅由 `moonep.buffer` 在 NPU 后端
re-export，GPU 进程中不存在——GPU 侧等价的稠密 `[E+B]` / `[R,B,H,H']`
张量由框架/测试直接以 `torch.empty` 构造）；在 NPU 上它们会登记进全局
descriptor 注册表并在首次算子调用时翻译成对称基址，
应用侧无需直接操作低层 `npu_patch.buffer_c`：

- **Full Weight（pool）**：`buffer.prefetch_weight` 的 `full_gate_weight` /
  `full_up_weight` / `full_down_weight` **就是该 full_weight**——由
  `create_full_weight` 一步创建并填充的逻辑 `[E+B, H, H']` bf16 权重池（NPU 上即
  本 rank home/slot chunk 的 `[E+B]` 投影，附 `_npu_home_chunk` / `_npu_slot_chunk` /
  `_projection`）：从全局 expert 表把本 rank 拥有的权重写入 home 行、slot 清零，
  `source_offset` 为逐投影偏置（gate/up/down 取 0/1/2）：

  ```python
  B = E // R                            # prefetch / 归约槽位数，默认等于 epn
  full_weight = create_full_weight(E, B, H, Hp, rank, R, remote_expert,
                                   source_offset=source_offset,
                                   dtype=torch.bfloat16,
                                   device=torch.npu.current_device())
  ```

- **Full Grad + Reduce Buffer**（`buffer.reduce_grad` 的 `full_gate_grad` /
  `full_up_grad` / `full_down_grad` 与 `gate_reduce_buffer` / `up_reduce_buffer` /
  `down_reduce_buffer`）：用 `create_full_grad_and_reduce_buffer` 一并创建，返回
  `(full_grad, reduce_buffer)`；`full_grad` 为 `[E+B, H, H']` **fp32**（附
  `_npu_home_chunk` / `_npu_slot_chunk` / `_projection`），Ascend reduce_grad 只读
  本 rank slot chunk 的对称基址。`grad_table` / `reduce_table` 分别是 rank 不变的
  `[E, H, H']` 梯度参考表与 `[R, B, H, H']` slot 参考表，本 rank 只拷贝自己的
  home / slot 切片：

  ```python
  full_grad, reduce_buffer = create_full_grad_and_reduce_buffer(
      E, B, H, Hp, rank, R, grad_table, reduce_table,
      dtype=torch.float32, device=torch.npu.current_device())
  ```

  示例见 `tests/test_e2e_npu_gpu.py`：权重侧用 `make_prefetch_args`，梯度侧用
  `make_grad_reduce_args`。

要点：

- 两个 builder 都接受 `vmm_page_size=`，用于**单独**指定训练框架参数的 VMM 大页大小
  （`framework` 作用域，进程级默认 `ASCEND_MOONEP_FRAMEWORK_VMM_PAGE_SIZE`，未设置时回落
  `ASCEND_MOONEP_VMM_PAGE_SIZE`），与 MoonEP `Buffer` 的 `vmm_page_size` 互不影响；同一投影的
  home/slot 块使用同一策略。1g 页会让 ACL 按 1 GiB 粒度取整，小投影需确认
  `epn * H * Hp * element_size` 是粒度的整数倍（见上节 `[E+B]` 约束）。
- 三组对称张量（full_weight / home-slot 投影）必须在 `moonep.init_shmem`
  （`ShmemRuntime.init_with_buffer` / `init_with_ip_port`）**之前**创建——init 的
  descriptor 集在 init 时冻结，之后的 VMM chunk 不会进入当前对称堆。
- **init 之后**跨 rank 读取**免 bind**：`remote_home_rows` /
  `read_reduce_slot` / `expert_pool_row` 按需解析对端对称地址
  （`buffer_c.remote_view`），返回 live 零拷贝视图。
- NPU 上 full_weight / full_grad 的投影视图基址**未登记**
  buffer-backed init，不能直接作对称基址用；算子只读 home/slot chunk 的对称基址，
  不读投影视图本体（投影段别名同一份物理内存）。
- `B`（prefetch / 归约槽位数）默认等于 `epn`；`Buffer` 内部的组合 `[E+B]` slot
  布局假定 `B == epn`（benchmark runner 构造时会断言），而 pool 投影路径对 `B`
  无此约束。
- 完整参考：`tests/test_e2e_npu_gpu.py` 的 `make_prefetch_args` /
  `make_grad_reduce_args`；benchmark 见「基准测试」一节。

### Buffer dispatch / combine / grad reduce

```python
# hidden_sh [S,H] bf16, route_weights_sk [S,K] fp32,
# topk_experts_sk [S,K] int32, tokens_per_expert [E] int32
hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
    hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert,
)
assert isinstance(plan, MoonEPCommPlan)

# GEMM 就地在 hidden_nvsh 上执行后：
out_sh, out_sk, _ = buffer.combine(
    plan=plan, hidden_nvsh=hidden_nvsh, route_weights_nvs=route_weights_nvs,
)

# 复用 plan（跳过 planning，只重放 dispatch；返回同一个 plan 对象）：
hidden_nvsh2, w2, cu2, plan2 = buffer.dispatch(hidden_sh, plan=plan)

# 权重侧 prefetch（先构造好三份 full weight，见上节）：
buffer.prefetch_weight(
    plan=plan,
    full_gate_weight=full_gate_weight,
    full_up_weight=full_up_weight,
    full_down_weight=full_down_weight,
)

# 任意投影集合的通用入口（两个参数都可省）：
#   projections   —— 非空投影序列，FC1/FC2 可同批（M/N 逐投影推导）
#   async_finish  —— True 时整体发到 comm_stream 并返回 torch.npu.Event
#   rank_barriers —— True（默认）由 API 负责跨 rank 的 entry/exit 会合；
#                    False 时两道会合都归调用方（见下节契约）
buffer.prefetch_projections(
    plan=plan,
    projections=(full_gate_weight, full_up_weight, full_down_weight),
    async_finish=True,
    rank_barriers=True,
)

# 梯度侧 reduce（先构造好三份 full grad + reduce buffer）：
buffer.reduce_grad(
    plan=plan,
    full_gate_grad=full_gate_grad, full_up_grad=full_up_grad,
    full_down_grad=full_down_grad,
    gate_reduce_buffer=gate_reduce_buffer, up_reduce_buffer=up_reduce_buffer,
    down_reduce_buffer=down_reduce_buffer,
)
```

- `zero_copy=True`：dispatch 返回本地对称分片的视图（`hidden_buf_local` /
  `weights_buf_local`，远程 rank 已直接写入）；combine 的 `hidden_nvsh` 必须
  别名该视图（`data_ptr` 相同），否则断言失败。
- `async_finish=True`：返回 `torch.npu.Event`。当前实现是兼容 shim——仍在
  current stream 上执行，随后设备同步 + 跨 rank 重排，避免连续 async 操作共享
  shmem 栅栏代际而死锁（跨 rank kernel 内含设备侧全 rank 栅栏）。
- 断言级参考见 `test_npu/mte/test_e2e.py` 与 `test_npu/urma/test_e2e_urma.py`
  （含 plan 复用、zero_copy、async、prefetch 投影读回、grad reduce 参考校验）。

依赖 `shmem` Python 包（cann-shmem `src/python`）。Buffer 路径与 `_C` 必须复用
同一份 SHMEM runtime，不能在同一进程中再初始化 C++ `ShmemInstance`。

### 多机运行（buffer-backed + ip_port bootstrap）

单机默认走 UID bootstrap（`ShmemRuntime.init_with_buffer`：rank0 生成 UID 经
`torch.distributed` 广播）。多机（多节点）改用 ip_port / DEFAULT bootstrap：
全体 rank 传同一个 `tcp://<rank0_ip>:<port>`，rank0 建 config store server、
其余连 client——对称空间可跨节点。

```python
from moonep import Buffer, ShmemRuntime

# 构造完所有 Buffer / 对称 pool（moonep.buffer，登记 VMM descriptor）后：
ShmemRuntime.init_with_ip_port(
    group=dist.group.WORLD,
    ip_port="tcp://192.168.1.10:29501",   # 缺省时从 MASTER_ADDR/MASTER_PORT 推导（端口 +1）
    descriptors=None,                      # None = 进程内已登记的 VMM 分配
)
# 或等价：ShmemRuntime.init_with_buffer(ip_port="tcp://192.168.1.10:29501")
```

### 基准测试（npu_patch/benchmarks）

`npu_patch/benchmarks` 提供 Ascend benchmark 集：`bench_comm`、
`bench_grad_reduce`、`bench_prefetch`、`bench_vs`、`bench_vs_all2all`、
`bench_barrier`、`bench_dispatch_maxvio`、`bench_grad_reduce_maxvio`，共享 `_bench_common.py`
助手与 `run_bench.sh`（torchrun wrapper）。每个 benchmark 都带
`--engine {ubmem,urma}`（默认 `ubmem` = MTE/UB 原生路径；`urma` = UDMA/URMA
路径，仅 A5/Ascend950），作用到 Buffer、原始 `launch_*` 入口与 SHMEM 堆 init。
`bench_vs_all2all` 是 dispatch/combine 的对手实验：以 `bench_vs.py` 为模板，
把 MoonEP 与直接基于 `torch.distributed` `all_to_all_single` 的“无 plan、无
prefetch”实现放在同一组 routing/计时下对比（`--libs moonep all2all`，也可
`--libs all2all` 单独验证该集合通信，此时不加载 Buffer / shmem 堆）。

`bench_vs.py --verify` 在计时前逐 MaxVio 点做正确性自检：`--verify-stages`
可选 `plan,dispatch,grad_reduce,prefetch,combine`（默认全开），
`--verify-trace` 打印逐 rank 标记。其中 `plan` 用向量化 torch 计划参考
（`benchmarks/_verify_reference.py`，含 MaxVio 平衡策略、prefetch 选表、
`cu_seqlens` 段布局、`dst`/dup 编码与去重语义）逐字段校验 planning，
`dispatch` 按 plan 的 `dst` 重建每个 slot 的期望 route-weight 与 hidden
行指纹，覆盖 fwd（hidden + weights）与 bwd（saved plan、仅 hidden）两条
路径；其余 stage 分别校验 combine、prefetch 槽位与 grad_reduce。

`bench_vs.py --vmm-page-size {default,1g,2m}` 控制该 benchmark 自己注册的 5 块
per-rank VMM（3 块权重 + `--verify` 时的 2 块 fp32 grad）。默认 `default`
（=2 MiB 页）；`--vmm-page-size 1g` 可切到 1 GiB 页对比带宽，但
`--verify-grad-shape model` 叠加 `--dedup-mode server` 的 multi-GiB combine
workspace 时容易超过 40 GiB 上限（Buffer 内部的 metadata/sync/workspace 已显式
2 MiB 页）。

`bench_vs.py --verify-grad-shape {small,model}`（默认 `small`）控制 grad_reduce
自检使用的矩阵形状。grad_reduce kernel 只把 `hidden*projection` 当作展平的一维
`matrix` 处理，`small`（128×256，3 个 12288 元素的 kernel tile 迭代）同样覆盖
plan 查表、跨 rank 累加、已消费 slot 清零与空闲 slot 不动这些路径，但两块 fp32
chunk 从生产形状的 ~12 GiB 降到 32 MiB；确实需要复现与生产形状相关的问题时传
`model` 使用 `--hidden/--hp`。

先构建，再运行：

```bash
bash npu_patch/scripts/build_a5.sh

# 单机 8 卡，默认脚本 + 默认引擎（ubmem）
bash npu_patch/benchmarks/run_bench.sh
# 指定脚本
bash npu_patch/benchmarks/run_bench.sh bench_prefetch.py
# urma 引擎（仅 A5/Ascend950）
ENGINE=urma bash npu_patch/benchmarks/run_bench.sh bench_comm.py
```

`bench_vs.py` / `bench_vs_all2all.py` / `bench_comm.py` / `bench_dispatch_maxvio.py` /
`bench_combine_udma_maxvio.py` 支持 `--dedup-mode {rank,server}`，作用在构造
`Buffer` 时（与 API 一致，去重域在构造期固定）。`server` 需要 `--engine urma`、
`R % 8 == 0` 且 `K >= 2`，并会分配 server 模式的对称 PUT combine workspace
（ring ABI：有界的输入环 + 稳定结果区，S=8192、K=16、H=3584 时约 2~3 GiB/rank，
随 R 略增；rank 模式为 0，具体字节数见 `combine_put_workspace_bytes()`）。EP=8 只有一个 server，
跨 server 的分支要在 R>8 才覆盖。`--verify` 的参考模型按所选去重域校验 `dst` 与重复表
（server 含 8*NvS 全地址与 R+2 计数/前缀）。CSV 新增 `dedup` 列，两次运行可分开对比。

`bench_comm.py` 的 `--dedup-mode` 让五相表（planning / d_f / d_b / c_f / c_b）覆盖
server 域；server 的 combine 直接消费 planning 融合产出的 execution plan，缺 plan 时
会在计时前 fail-fast。`bench_combine_udma_maxvio.py` 默认 `--dedup-mode server`，
传 `--dedup-mode rank` 可测冻结的 rank 级 UDMA combine 内核（无 execution plan、
无 workspace），两条曲线可直接对比。

多机启动（`NODES>1` 各节点各执行一次，`MASTER_ADDR` 指向 rank0）；多机 shmem
ip_port 缺省从 `MASTER_ADDR` / `MASTER_PORT` 自动推导（端口 +1），需要时可显式
指定 `SHMEM_IP_PORT="tcp://<ip>:<port>"`：

```bash
# node 0
NODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.10 bash npu_patch/benchmarks/run_bench.sh bench_vs.py
# node 1
NODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.10 bash npu_patch/benchmarks/run_bench.sh bench_vs.py

# MoonEP vs torch.distributed all_to_all（同样支持 NODES/NODE_RANK 多机）
bash npu_patch/benchmarks/run_bench.sh bench_vs_all2all.py
```

## UDMA Dispatch 精度与性能测试

完成前面的构建后，在仓库根目录执行。下面以 **8 节点 × 8 卡**为例，每个节点各运行一次，
共用 `MASTER_ADDR`、`MASTER_PORT`，`NODE_RANK` 分别设置为 0～7。各节点需先加载 CANN 和 Python 环境。

```bash
export MASTER_ADDR="<node-0-address>"
export MASTER_PORT=29500
export NODE_RANK=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SHMEM_IP_PORT="tcp://${MASTER_ADDR}:$((MASTER_PORT + 1))"
```

### 去重域（构造期配置）

去重域是 `Buffer` 的构造期属性，同一 EP 组内所有 rank 必须一致；Buffer 的方法本身不再接受
`dedup_mode` 参数。`dispatch` / `combine` 的入参与 GPU 版本完全一致；`prefetch_weight` /
`reduce_grad` 仍使用 Ascend 的 `full_*` 投影入参（GPU 版本是 `local_*` + prefetch buffer）。

```python
from moonep import Buffer

# rank 级去重（默认；等价本地 a5_64p_step01 的代码路径）
buffer = Buffer(S=8192, H=3584, K=16, E=1024, num_ep_ranks=R)

# server 级去重：连续 8 卡为一个去重域，需要 UDMA 后端
buffer = Buffer(S=8192, H=3584, K=16, E=1024, num_ep_ranks=R,
                dedup_mode="server", ranks_per_server=8)
```

- `dedup_mode` 取 `rank`（默认）或 `server`；默认行为就是本地 rank 级去重。
- `ranks_per_server` 默认 8，当前内核 ABI 只编译了 8，传其它值会在构造期报错。
- server 模式要求 `num_ep_ranks` 能被 8 整除，且设备支持 UDMA（Ascend 950）；
  Dispatch/Combine 各自的 UDMA 开关仍在调用点检查（combine-only 场景可不启用 dispatch UDMA）。
- Buffer 构造期会把去重域发布成进程环境变量 `ASCEND_MOONEP_DEDUP_MODE`（`rank` / `server`）
  与 `ASCEND_MOONEP_SERVER_DEDUP`（`1` / `0`）：shmem 堆 init 发生在所有 Buffer 构造之后、
  手里没有 Buffer 句柄，只能靠这对标志判断去重域（例如服务端去重的 QP 约束）。启动器自己
  导出的同名变量效果一致，可用于在 Python 之外配置去重域。
- **server 去重只支持 `qp_count=2`**：`ShmemRuntime.init_with_buffer` /
  `init_with_ip_port` 若收到其它 QP 数（含 `ASCEND_MOONEP_QP_COUNT` 配置的值），会强制改成
  2 并给出 warning。server 内核按 `active_qp_count=2` 做 chunk→QP 映射，其它值会破坏
  dispatch/combine 的 QP 同步，因此这里是硬覆盖而不是报错；rank 去重不受影响。

### rank / server 的 kernel 划分

去重域直接决定使用哪一套 kernel，两套实现物理隔离、互不共享 ABI：

| 去重域 | Planning | Dispatch | Combine | kernel 库 |
|---|---|---|---|---|
| `rank`（默认） | `kernels/planning.cpp`（2802271 冻结版） | `kernels/moonep_dispatch_udma.cpp` | `kernels/moonep_combine_udma.cpp` | `libascend_moonep_kernels.so` |
| `server` | `kernels/planning_server_dedup.cpp` | `kernels/moonep_dispatch_udma_server_dedup.cpp` | `kernels/moonep_combine_udma_server_dedup.cpp` | `libascend_moonep_kernels_server_dedup.so` |

- 选择由 Python 侧按 `plan.dedup_mode` 传给算子（`server_dedup` 形参），C++ 入口据此选符号，
  不再靠 `dup_counts` 形状猜测。
- **维护契约**：server 三份与 upstream `dev_server_dedup` 的同名实现保持同步，只改导出符号
  （`*_server_dedup_*`；planning 另加一次退出去重会合）；rank 三份冻结在 2802271。
  同步 upstream 后用 `git diff upstream/dev_server_dedup:npu_patch/native/kernels/<base>.cpp \
  npu_patch/native/kernels/<base>_server_dedup.cpp` 校验差异应只有改名。
- rank 模式不分配 server 的对称 PUT workspace：`combine_workspace` 长度为 0，老 combine 在算子内部
  自持 rank-local gather workspace；因此 rank 模式不会占用数 GiB 的对称内存。
- rank 模式不产生 combine 执行计划：`plan.combine_execution_plan` 恒为 `None`，
  `Buffer.prepare_combine_plan()` 在 rank 模式下会直接报错。
- 两套 kernel 需要同时构建、同时安装（`build_a5.sh` 会产出两个 `.so`，`setup.py` 会校验并链接两个库）。

### 精度验证

```bash
python3 test_npu/urma/test_moonep_dispatch_udma_with_plan.py \
    --mode kernel --bootstrap ipport --R 64 \
    --nproc-per-node 8 --node-rank "$NODE_RANK" \
    --shmem-ip-port "$SHMEM_IP_PORT" \
    --S 8192 --K 16 --H 3584 --E 1024 \
    --plan-source kernel --dedup-mode server \
    --output-mode both --max-checks 0
```

- `--plan-source golden`：CPU 生成 plan，验证 Dispatch；`kernel`：验证 Planning + Dispatch。
  `run_test.sh --use_urma` 的 server 用例现在固定带 `--plan-source kernel`，因此
  `planning_server_dedup`（含融合 combine plan）进入一键精度链路。
- `--dedup-mode rank|server`：分别按目标卡或连续八卡 server 去重；该选项只作用于测试/基准
  内部构造 `Buffer` 时传入的 `dedup_mode`，server 内需支持 MTE 转发。
- 两组选项组合可验证 `golden/rank`、`golden/server`、`kernel/rank`、`kernel/server` 四种路径。

`test_moonep_combine_udma_with_plan.py` 也接受 `--dedup-mode`：rank 模式按 rank 表走旧
combine kernel，server 模式复用 dispatch 测试的 server 表、先 `prepare_combine_plan`
再 combine（server 的 CPU 参考在 server 全域上做 prologue 折叠）。`--combine-engine
{auto,udma,mte}` 选择 kernel 族：`auto`（默认）= server 用 UDMA、rank 用 legacy MTE；
`udma` 额外覆盖冻结的 rank 级 UDMA combine 内核；`mte` 仅 rank 可用。
`run_test.sh --use_urma` 现在对 dispatch / joint / combine 三个算子测试各跑 rank 与 server
两遍，并用 `K=16` 覆盖 server 的满 top-k；另有一条小 shape 的
`--combine-engine udma` rank 用例。
- `--output-mode both --max-checks 0` 全量检查 copy 和 zero-copy 输出；所有 rank 应打印 `MoonEPDispatch PASS`，
  所有节点应打印 `NODE <index> LOCAL RANKS PASS` 并正常退出。

### 性能验证

精度任务在所有节点结束后，再执行：

```bash
torchrun --nnodes 8 --nproc-per-node 8 --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    npu_patch/benchmarks/bench_dispatch_maxvio.py \
    --engine urma --no-dist --dedup-mode server --ip-port "$SHMEM_IP_PORT" \
    --tokens 8192 --topk 16 --hidden 3584 --experts 1024 \
    --maxvios 0.2,1,10,20 --kernel-timing --out maxvio_server.csv
```

改为 `--dedup-mode rank --out maxvio_rank.csv` 可测 rank 去重。每档先运行 Planning，计时仅覆盖 Dispatch；
`--kernel-timing` 额外输出模块耗时。rank 0 打印结果表并保存 CSV，模块耗时保留在其日志中。
`--no-dist` 使用 CPU TCPStore 汇总统计，需允许其他节点连接 rank 0 的 `MASTER_PORT + 2` 端口。

需要端到端（planning + dispatch + prefetch + combine）口径时用 `bench_vs.py`；加
`--dedup-mode server` 即可在同样的计时下测 server 去重，CSV 的 `dedup` 列区分两次运行：

```bash
torchrun --nproc_per_node=8 npu_patch/benchmarks/bench_vs.py \
    --engine urma --dedup-mode rank   --maxvios 1 --out bvs_rank.csv
torchrun --nproc_per_node=8 npu_patch/benchmarks/bench_vs.py \
    --engine urma --dedup-mode server --maxvios 1 --out bvs_server.csv
```

`bench_vs_all2all.py` 同样支持 `--dedup-mode`（只作用于 MoonEP 一侧，all2all 基线没有
去重）。两种模式都用 `--no-cudagraph` 时口径一致（server 模式的跨 rank 会合通常不可
capture，脚本会自动回落到 eager）。server 去重的收益主要看 `c_f`/`c_b` 与 `d_f`，代价
看 `plan_us`（server 模式包含 in-kernel combine plan 生成）与 workspace 显存；`d_b`
在 Ascend 上仍会重建 epilogue（`build_dedup_map` 未接入 kernel），两种模式一致。

需要 planning / d_f / d_b / c_f / c_b 五相表与带宽时用 `bench_comm.py`，它同样支持两种
去重域（CSV 末尾带 `dedup` / `engine` 列，一次跑一个域，便于并排比较）：

```bash
# mte rank（默认引擎）
bash npu_patch/benchmarks/run_bench.sh bench_comm.py
# urma rank
ENGINE=urma bash npu_patch/benchmarks/run_bench.sh bench_comm.py
# urma server（生产 8 卡形状；server workspace 约 2~3 GiB/rank）
ENGINE=urma BENCH=bench_comm.py BENCH_ARGS="--dedup-mode server --prod" \
    bash npu_patch/benchmarks/run_bench.sh
```

combine 单独的 MaxVio 曲线用 `bench_combine_udma_maxvio.py`（默认 server，可切 rank）：

```bash
# server（PUT kernel + execution plan）
bash npu_patch/benchmarks/run_bench.sh bench_combine_udma_maxvio.py "--check --plan-mode cyclic"
# rank（冻结的 rank 级 UDMA 内核，无 execution plan / workspace）
bash npu_patch/benchmarks/run_bench.sh bench_combine_udma_maxvio.py \
    "--dedup-mode rank --check --plan-mode cyclic"
```
