---
icon: lucide/server
---

# 比赛线上服务器设备信息

本文根据两次真实线上任务启动日志整理，采样时间分别为 2026-04-25 21:47:29 +0800 与 2026-04-26 00:27:26 +0800。
第二次日志使用了增强版网络探测脚本，因此本文现在更侧重那些在重复观测中都成立、并且对训练 bundle 设计有直接影响的环境约束。
其中 CPU、内存、磁盘和 GPU 配额仍然属于“高概率稳定事实”，而节点负载、瞬时剩余内存和具体 hostname 仍然可能随作业而变化。

## 观测摘要

| 项目                     | 观测结果                                                                                              |
| ------------------------ | ----------------------------------------------------------------------------------------------------- |
| 任务入口                 | 平台最终执行 bash /home/taiji/dl/runtime/script/run.sh                                                |
| 探测脚本记录的 repo_root | /home/taiji/dl/runtime                                                                                |
| 请求 profile             | cpu                                                                                                   |
| 请求 Python              | 3.13                                                                                                  |
| 操作系统                 | Ubuntu 22.04.5 LTS                                                                                    |
| 内核                     | 5.4.241-1-tlinux4-0017.7                                                                              |
| CPU                      | 双路 AMD EPYC 9K84 96-Core Processor                                                                  |
| 逻辑核数                 | 384                                                                                                   |
| NUMA                     | 2 个 NUMA 节点                                                                                        |
| 内存                     | 2.2 TiB，Swap 为 0 B                                                                                  |
| GPU 可见性               | 1 个切片 GPU，可见为 0.2x NVIDIA H20                                                                  |
| GPU 显存                 | 19574 MiB                                                                                             |
| 驱动版本                 | NVIDIA Driver 535.247.01                                                                              |
| CUDA 运行时              | CUDA 12.6                                                                                             |
| nvcc                     | 12.6.77                                                                                               |
| 网络工具                 | 容器内未提供 ip 命令                                                                                  |
| 代理环境                 | 仅设置小写 http_proxy 与 https_proxy，值为 http://21.100.120.217:3128                                 |
| 继承代理的公网 HTTPS     | 对 example、GitHub、Python、PyPI、Astral、PyTorch 索引的探测全部失败，失败类型为 proxy_tunnel_failure |
| 禁用代理后的公网访问     | 对同一批域名的探测全部失败，失败类型为 dns_failure                                                    |
| uv 与 pip                | 任务启动时 uv 不存在；在线 uv bootstrap 与在线 pip 下载都不能作为可靠前提                             |

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

两次日志中的 free -h 摘要略有波动，但量级一致：

| 指标       | 观测范围           |
| ---------- | ------------------ |
| 总内存     | 2.2 TiB            |
| 已使用     | 708 GiB 到 977 GiB |
| 空闲       | 908 GiB 到 1.3 TiB |
| Buff/Cache | 255 GiB 到 438 GiB |
| 可用       | 1.2 TiB 到 1.5 TiB |
| Swap       | 0 B                |

可见宿主机总内存很大，但当前节点在采样时的已使用内存波动也很明显，因此仍然不应把线上环境当成“无限内存”。尤其是没有 swap，超内存后更容易被直接杀进程。

## 存储与文件系统

### 容器内可见磁盘

| 设备      | 容量   | 备注      |
| --------- | ------ | --------- |
| `sda`     | 447.1G | 系统盘    |
| `nvme0n1` | 5.8T   | 本地 NVMe |
| `nvme1n1` | 5.8T   | 本地 NVMe |

### 当前容器挂载

日志显示根文件系统为 overlay，容量约 12T，两次采样时分别使用约 380G 与 487G，占用约 4% 到 5%。

这意味着：

- 线上运行目录本身不是一个很紧的小容器层
- 训练中间产物、解压后的 bundle 和 checkpoint 在磁盘层面通常不是首要瓶颈
- 但是否允许长期持久化、是否有作业级清理策略，仍应以平台规则为准

## 代理与出网约束

第二次日志对代理环境和公网目标做了显式探测，这部分信息比“能否看到 GPU”更直接影响线上训练成败。

### 代理环境

探测脚本记录到的代理变量为：

- HTTP_PROXY, HTTPS_PROXY, ALL_PROXY, NO_PROXY 均未设置
- http_proxy 与 https_proxy 被平台注入为 http://21.100.120.217:3128
- all_proxy 与 no_proxy 未设置

这意味着依赖 wget、curl、pip、requests 之类遵循小写代理变量的工具，默认都会走平台代理。

### 站点连通性矩阵

增强版探测脚本分别在 inherited 与 no_proxy 两种模式下测试了以下公网目标：

- https://example.com
- https://github.com
- https://www.python.org
- https://pypi.org/simple
- https://astral.sh/uv/install.sh
- https://download.pytorch.org/whl/cpu

结果如下：

| 模式      | 结果                                      | 解释                                                                                                            |
| --------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| inherited | 全部失败，失败类型为 proxy_tunnel_failure | 容器能连到 21.100.120.217:3128，但代理返回 Proxy tunneling failed: Service Unavailable，无法建立公网 HTTPS 隧道 |
| no_proxy  | 全部失败，失败类型为 dns_failure          | 去掉代理后，容器本身连最基本的公网 DNS 解析都过不去，报 Temporary failure in name resolution                    |

这比“是否有外网”更精确的结论是：

- 线上容器不能依赖平台代理访问公网 HTTPS
- 线上容器也不能依赖禁用代理后的直连公网 DNS/HTTPS
- 对当前 bundle 设计而言，应当把“在线安装 uv、在线拉 PyPI、在线拉 PyTorch 轮子”视为不可用路径

### uv 与 pip 下载探测

日志同时验证了两条常见依赖安装路径：

1. uv bootstrap
  结果是 uv 启动时不存在，而访问 https://astral.sh/uv/install.sh 的 inherited 模式探测失败类型为 proxy_tunnel_failure，因此不能把“缺少 uv 时自动补装”当成线上可靠方案。

2. pip download
  探测脚本使用 pip download 访问 https://pypi.org/simple，并测试 sampleproject==4.0.0。
  inherited 与 no_proxy 两种模式都返回 from versions: none / No matching distribution found。
  单看这一行像“包不存在”，但结合同一批日志里对 pypi.org/simple 的站点探测已经失败，更合理的解释是：当前环境根本拿不到可用的索引内容，因此不能依赖在线 pip 安装新库。

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

这与仓库现有 cuda126 profile 是对齐的；如果线上环境坚持使用 CPU profile，则需要由入口脚本或平台参数显式切换。

## 运行时行为观察

综合两次日志，可以确认几条对线上训练最关键的环境事实：

1. 平台传入的 requested_profile=cpu 与“容器内确实能看到 GPU”可以同时成立，因此 profile 选择不能仅靠是否检测到 GPU 来推断。
2. 启动阶段 uv not found 是真实现象，但在当前平台网络策略下，自动在线补装 uv 的路径并不可靠。
3. 日志里 ip not found，表示排查网络问题时不能依赖 ip addr 这类工具必然存在。
4. 平台只设置小写 http_proxy 与 https_proxy，而不是大写代理变量；排查联网失败时必须把大小写都看一遍。
5. 继承代理时公网 HTTPS 隧道全部失败，禁用代理时公网 DNS 全部失败，因此线上 bundle 必须按“无公网依赖”设计。
6. 采样时机器负载很高，load average 大约在 47 到 66 之间，说明共享节点上的瞬时抖动是可能存在的。

## 对当前仓库的启示

结合本仓库现有线上 bundle 入口，可以得到几条比较稳妥的结论：

- run.sh 保留环境探测和失败诊断逻辑仍然有价值，但不能再把“缺少 uv 时自动在线安装”当成线上主路径。
- 线上 bundle 应尽量做到自包含：要么把 uv 和依赖预打包进去，要么切到平台内网镜像源，而不是依赖公网 PyPI、Astral 或 PyTorch 索引。
- 文档和脚本里不应默认线上一定拿到完整 GPU，而应按小显存切片场景设计保守默认值。
- 若线上平台默认传 cpu profile，但任务实际希望走 CUDA 依赖，应显式设置 TAAC_CUDA_PROFILE=cuda126 或由平台侧配置覆盖；同时 CUDA 依赖的准备应在线下完成。
- 日志采集脚本继续打印 CPU、内存、磁盘、GPU、nvcc、代理环境、站点矩阵和 pip 下载探测是有价值的，因为这些信息足以快速判断线上失败是环境错配、代理隧道失败，还是公网 DNS 不可用。

## 原始日志中的关键片段

下面这些字段最能概括当前线上网络约束：

```text
http_proxy=http://21.100.120.217:3128
https_proxy=http://21.100.120.217:3128
uv_download_failure_class=proxy_tunnel_failure
site_pypi_inherited_failure_class=proxy_tunnel_failure
site_pypi_no_proxy_failure_class=dns_failure
site_astral_no_proxy_probe_detail=... Temporary failure in name resolution ...
pip_download_inherited_probe_detail=ERROR: Could not find a version that satisfies the requirement sampleproject==4.0.0 (from versions: none)
pip_download_no_proxy_probe_detail=ERROR: Could not find a version that satisfies the requirement sampleproject==4.0.0 (from versions: none)
```

如果后续平台修复了代理、补齐了直连 DNS，或者提供了内网包源，建议继续追加到本文，区分“稳定事实”和“平台策略变更”。