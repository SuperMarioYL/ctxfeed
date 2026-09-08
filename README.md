[English](README.en.md) | **简体中文**

<picture>
  <source media="(max-width: 640px) and (prefers-color-scheme: dark)" srcset="assets/presentation/hero-mobile-dark.svg">
  <source media="(max-width: 640px)" srcset="assets/presentation/hero-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="assets/presentation/hero-dark.svg">
  <img src="assets/presentation/hero-light.svg" width="1000" alt="按 token 预算组织仓库文件，跟踪内容变化，并通过 CLI 或 MCP 提供项目上下文。">
</picture>

**按 token 预算组织仓库文件，跟踪内容变化，并通过 CLI 或 MCP 提供项目上下文。**

`v0.7.0` · `Python 3.12+` · [MIT](LICENSE)

[Website](https://ctxfeed.lei6393.com) · [Demo record](docs/demo-results.json)

## 为什么使用

把很多文件交给模型之前，先要知道哪些文件会被纳入、顺序是什么、相对上次修改了哪些内容。ctxfeed 用 ShardPlan 表达这些选择，保留稳定前缀、正文和变化集合，方便检查一次查询实际组织了什么。

## 架构

<picture>
  <source media="(max-width: 640px) and (prefers-color-scheme: dark)" srcset="assets/presentation/architecture-mobile-dark.svg">
  <source media="(max-width: 640px)" srcset="assets/presentation/architecture-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="assets/presentation/architecture-dark.svg">
  <img src="assets/presentation/architecture-light.svg" width="1000" alt="ingest 扫描并计数文件，shard 按预算和稳定顺序构建计划，CachePlan 将文件哈希写到仓库自己的 SQLite 缓存。CLI 和 MCP 共用该路径，模型适配器在查询时发送组装后的上下文。delta 是本地变化账本，不代表仅发送变化内容，也不是服务端 prefix cache 遥测。">
</picture>

ingest 扫描并计数文件，shard 按预算和稳定顺序构建计划，CachePlan 将文件哈希写到仓库自己的 SQLite 缓存。CLI 和 MCP 共用该路径，模型适配器在查询时发送组装后的上下文。delta 是本地变化账本，不代表仅发送变化内容，也不是服务端 prefix cache 遥测。

源码入口：[src/ctxfeed/cli.py](src/ctxfeed/cli.py) · [src/ctxfeed/cache_plan.py](src/ctxfeed/cache_plan.py) · [src/ctxfeed/ingest.py](src/ctxfeed/ingest.py) · [src/ctxfeed/shard.py](src/ctxfeed/shard.py) · [src/ctxfeed/mcp_server.py](src/ctxfeed/mcp_server.py) · [src/ctxfeed/models/glm.py](src/ctxfeed/models/glm.py) · [src/ctxfeed/models/deepseek.py](src/ctxfeed/models/deepseek.py)

## 安装

需要 Python 3.12+ 与 uv。演示只构建本地计划，无需 API key；首次 tokenizer 使用可能需要缓存其编码文件。

```bash
git clone https://github.com/SuperMarioYL/ctxfeed.git
cd ctxfeed
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
```

## 快速开始

完整脚本创建两份临时文件，连续构建三次计划。结果中的 cached_fraction 是本地文件哈希复用比例，不是模型缓存命中或实测费用。

```bash
.venv/bin/python examples/presentation-demo.py
```

完整输入与执行步骤见上方命令及 [Demo 记录](docs/demo-results.json)。

## 使用

```bash
.venv/bin/ctxfeed init --repo .
.venv/bin/ctxfeed add ./src/ctxfeed/cli.py
.venv/bin/ctxfeed cost --repo .
.venv/bin/ctxfeed mcp --repo .
```
init 在没有对应模型密钥时使用 dry-run；设置密钥后该命令可能调用服务。MCP 客户端应将 command 配置为本仓库 `.venv/bin/ctxfeed` 的绝对路径，args 为 `["mcp", "--repo", "/absolute/repo"]`。

## 实际 Demo

<picture>
  <source media="(max-width: 640px) and (prefers-color-scheme: dark)" srcset="assets/presentation/process-mobile-dark.svg">
  <source media="(max-width: 640px)" srcset="assets/presentation/process-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="assets/presentation/process-dark.svg">
  <img src="assets/presentation/process-light.svg" width="1000" alt="完整脚本创建两份临时文件，连续构建三次计划。结果中的 cached_fraction 是本地文件哈希复用比例，不是模型缓存命中或实测费用。">
</picture>

### 查看三次计划

首次、未修改、修改一次的变化集合分别是 2、0、1。

```text
$ .venv/bin/python examples/presentation-demo.py
[
  {
    "stage": "first",
    "files": 2,
    "delta_hashes": 2,
    "cached_fraction": 0.0,
    "stable_prefix": [
      "README.md"
    ]
  },
  {
    "stage": "unchanged",
    "files": 2,
    "delta_hashes": 0,
    "cached_fraction": 1.0,
    "stable_prefix": [
      "README.md"
    ]
  },
  {
    "stage": "edited",
    "files": 2,
    "delta_hashes": 1,
    "cached_fraction": 0.5,
    "stable_prefix": [
      "README.md"
    ]
  }
]
```

## 能力与接入

<picture>
  <source media="(max-width: 640px) and (prefers-color-scheme: dark)" srcset="assets/presentation/integrations-mobile-dark.svg">
  <source media="(max-width: 640px)" srcset="assets/presentation/integrations-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="assets/presentation/integrations-dark.svg">
  <img src="assets/presentation/integrations-light.svg" width="1000" alt="CLI 提供 init、add、cost、mcp。MCP 的 list_files 展示文件与缓存标记，query_repo 提交问题，cost_delta 按仓库内的静态费率估算。GLM 与 DeepSeek 是可选适配器，不会自动完成模型供应商间的运行时故障切换。">
</picture>

CLI 提供 init、add、cost、mcp。MCP 的 list_files 展示文件与缓存标记，query_repo 提交问题，cost_delta 按仓库内的静态费率估算。GLM 与 DeepSeek 是可选适配器，不会自动完成模型供应商间的运行时故障切换。



## 配置

`--model` 优先于 `CTXFEED_MODEL`，可选 glm 或 deepseek。GLM 读取 `ZHIPU_API_KEY`/`GLM_API_KEY`，DeepSeek 读取 `DEEPSEEK_API_KEY`。IngestConfig 管理 window、headroom、max_file_bytes 和 cache_db；默认缓存为 `.ctxfeed/cache.db`。供应商模型名、窗口和费率是适配器中的配置假设，应按实际账户核对。

## 路线图与范围

已实现预算计划、增量账本、两个适配器与 MCP。后续扩展应以真实仓库问答质量、服务端缓存遥测和供应商配置验证为依据。

- 文件可能因忽略规则、大小或预算被跳过；不能保证任意仓库完整进入上下文。
- dry-run 答案是内置模拟结果，cost 是静态估算；本示例没有调用模型。

![Terminal recording](assets/demo.gif) · [Recording script](docs/demo.tape)

## 许可证

[MIT](LICENSE)
