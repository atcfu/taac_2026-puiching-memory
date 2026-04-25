---
icon: lucide/server
---

# 比赛线上服务器设备信息

本文根据一次真实线上任务启动日志整理，采样时间为 2026-04-25 21:47:29 +0800。
这是一份单次观测结果，适合用于估计线上训练容器的大致硬件与运行时约束，但不应假设所有作业实例都完全一致。

## 观测摘要

| 项目         | 观测结果                                                 |
| ------------ | -------------------------------------------------------- |
| 任务入口     | 平台最终执行 `bash /home/taiji/dl/runtime/script/run.sh` |
| 代码根目录   | `/home/taiji/dl/runtime/script`                          |
| 请求 profile | `cpu`                                                    |
| 请求 Python  | `3.13`                                                   |
| 主机名       | `ts-95d1b2ad9d8ff20b019dc4e17c02014f-launcher`           |
| 操作系统     | Ubuntu 22.04.5 LTS                                       |
| 内核         | `5.4.241-1-tlinux4-0017.7`                               |
| CPU          | 双路 AMD EPYC 9K84 96-Core Processor                     |
| 逻辑核数     | 384                                                      |
| NUMA         | 2 个 NUMA 节点                                           |
| 内存         | 2.2 TiB                                                  |
| Swap         | 0 B                                                      |
| GPU 可见性   | 1 个切片 GPU，可见为 `0.2x NVIDIA H20`                   |
| GPU 显存     | 19574 MiB                                                |
| 驱动版本     | NVIDIA Driver 535.247.01                                 |
| CUDA 运行时  | CUDA 12.6                                                |
| `nvcc`       | 12.6.77                                                  |
| 网络工具     | 容器内未提供 `ip` 命令                                   |
| uv           | 启动时不存在，随后由脚本自动安装                         |

## 计算资源

### CPU

- 架构：`x86_64`
- 处理器型号：AMD EPYC 9K84 96-Core Processor
- Socket 数：2
- 每核线程数：2
- 总逻辑核数：384
- NUMA 分布：
  - node0: `0-95,192-287`
  - node1: `96-191,288-383`
- 指令集能力中包含 AVX2、AVX-512、BF16 等扩展

这说明线上容器的 CPU 侧资源非常宽裕，数据预处理、DataLoader 并发和较重的 host 侧特征整理不会太容易先撞到 CPU 上限。

### 内存

日志中的 `free -h` 摘要为：

| 指标       | 数值    |
| ---------- | ------- |
| 总内存     | 2.2 TiB |
| 已使用     | 967 GiB |
| 空闲       | 908 GiB |
| Buff/Cache | 388 GiB |
| 可用       | 1.2 TiB |
| Swap       | 0 B     |

可见宿主机总内存很大，但当前节点在采样时已有较高占用，因此仍然不应把线上环境当成“无限内存”。尤其是没有 swap，超内存后更容易被直接杀进程。

## 存储与文件系统

### 容器内可见磁盘

| 设备      | 容量   | 备注      |
| --------- | ------ | --------- |
| `sda`     | 447.1G | 系统盘    |
| `nvme0n1` | 5.8T   | 本地 NVMe |
| `nvme1n1` | 5.8T   | 本地 NVMe |

### 当前容器挂载

日志显示根文件系统为 overlay，容量约 12T，采样时使用约 487G，占用约 5%。

这意味着：

- 线上运行目录本身不是一个很紧的小容器层
- 训练中间产物、解压后的 bundle 和 checkpoint 在磁盘层面通常不是首要瓶颈
- 但是否允许长期持久化、是否有作业级清理策略，仍应以平台规则为准

## GPU 与 CUDA 环境

### GPU 设备

| 项目                     | 观测结果                                   |
| ------------------------ | ------------------------------------------ |
| `NVIDIA_VISIBLE_DEVICES` | `GPU-27b235bd-2915-4f24-33db-bfb4bc9d2c41` |
| `CUDA_VISIBLE_DEVICES`   | 未设置                                     |
| `nvidia-smi -L`          | `GPU 0: 0.2x NVIDIA H20`                   |
| 显存                     | `0 MiB / 19574 MiB`                        |
| 功耗上限                 | 500 W                                      |
| MIG 模式                 | Disabled                                   |

虽然设备名字里带有 H20，但从 `0.2x NVIDIA H20` 和约 19.6 GiB 可见显存判断，这更像是平台切分后的 GPU 配额，而不是整卡独占。

对训练配置的直接影响是：

- 不要按完整 H20 显存去估 batch size
- 大模型实验应优先启用 AMP，并谨慎设置 embedding 维度、序列长度和负样本规模
- 即使物理机 GPU 很强，当前作业实际上拿到的是受限切片资源

### CUDA 工具链

| 项目          | 版本       |
| ------------- | ---------- |
| NVIDIA Driver | 535.247.01 |
| CUDA Runtime  | 12.6       |
| `nvcc`        | 12.6.77    |

这与仓库现有 `cuda126` profile 是对齐的；如果线上环境坚持使用 CPU profile，则需要由入口脚本或平台参数显式切换。

## 运行时行为观察

从这次日志还能看到几条很重要的环境事实：

1. 平台传入的 `requested_profile=cpu` 与“容器内确实能看到 GPU”可以同时成立，因此 profile 选择不能仅靠是否检测到 GPU 来推断。
2. 启动阶段 `uv not found`，随后脚本尝试从官方安装脚本补装 uv，说明线上镜像不应假设已经内置 uv。
3. 日志里 `ip not found`，表示排查网络问题时不能依赖 `ip addr` 这类工具必然存在。
4. 采样时机器负载很高，`load average` 约为 `55.15, 59.29, 57.08`，说明共享节点上的瞬时抖动是可能存在的。

## 对当前仓库的启示

结合本仓库现有线上 bundle 入口，可以得到几条比较稳妥的结论：

- `run.sh` 继续保留“缺少 uv 时自动安装”的逻辑是必要的。
- 文档和脚本里不应默认线上一定拿到完整 GPU，而应按小显存切片场景设计保守默认值。
- 若线上平台默认传 `cpu` profile，但任务实际希望走 CUDA 依赖，应显式设置 `TAAC_CUDA_PROFILE=cuda126` 或由平台侧配置覆盖。
- 日志采集脚本继续打印 CPU、内存、磁盘、GPU、`nvcc` 和 torch/CUDA 信息是有价值的，因为这些信息足以快速判断线上失败是否来自环境错配。

## 原始日志中的关键片段

下面这些字段是本页整理时最重要的来源：

```text
requested_profile=cpu
requested_python=3.13
Ubuntu 22.04.5 LTS
CPU(s): 384
Mem: 2.2Ti 967Gi 908Gi 52Gi 388Gi 1.2Ti
GPU 0: 0.2xNVIDIA H20
Driver Version: 535.247.01    CUDA Version: 12.6
nvcc: release 12.6, V12.6.77
uv not found; installing uv from https://astral.sh/uv/install.sh
```

如果后续拿到了新的线上日志，建议继续追加到本文，区分“稳定事实”和“单次任务观测差异”。