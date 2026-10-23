# 在 SWE-bench 上评测 LightCoder

本目录是评测 / 消融实验台。它在 [SWE-bench](https://www.swebench.com) 题目上运行
LightCoder,并用**官方** harness 打分,所以 resolve 分数是可信的,而不是自造的。

**流水线(一句话):**

```
按 base_commit 还原仓库  →  LightCoder 针对 issue 单轮作答  →  git diff = 补丁
    →  predictions.jsonl  →  官方 swebench Docker harness(打补丁 + 跑隐藏测试)
    →  resolved / 未解决
```

---

## 各类文件在哪

三种产物,别混为一谈:

| 产物 | 位置 | 说明 |
|---|---|---|
| **题目数据**(issue 文本 + 隐藏的 FAIL_TO_PASS / PASS_TO_PASS 测试规格) | `~/.cache/huggingface/datasets/princeton-nlp___swe-bench_lite/` | 首次 `load_dataset` 时自动从 HuggingFace 下载(约 4 MB) |
| **每题的仓库代码** | `results/_repos_batch/<run_id>/<instance_id>/testbed/` | **不**从 GitHub clone —— 运行时用 `docker cp` 从镜像里抽出来(临时产物;44 题约 2.6 GB) |
| **每题的运行环境**(真实依赖) | Docker 镜像 `swebench/sweb.eval.x86_64.*` | 存在 Docker 自己的存储里;`docker images \| grep swebench`。每个约 1–2.5 GB。打分和 `verify` 都用它。 |
| **子集定义**(跑哪些题) | `eval/subsets/*.json` | 固定的 id 列表,保证每个配置跑的是**同一批**题 |

---

## 前置条件

### 1. Python 环境(带 `eval` 可选依赖)

评测 harness 在 agent 核心之外还需要两个较重的依赖:`datasets` 和 `swebench`。

```bash
# uv(本仓库用 uv 管理;这里已经有一个 .venv)
uv sync --extra eval

# 或者用普通 pip 装进 venv
pip install -e ".[eval]"
```

下面所有命令都用 `.venv/bin/python`(等价于 `uv run python`)。

### 2. Docker 正在运行

harness 和 `verify` 都需要 Docker。本机跑的是 **rootless** Docker。

```bash
docker version                    # client 和 server 都应打印出来
docker info >/dev/null && echo OK # 守护进程可达
# 如果 rootless 守护进程没起:  systemctl --user start docker
```

### 3. 模型端点

LightCoder 对接任意 OpenAI 兼容端点(vLLM、DeepSeek、OpenAI)。在 `~/.lightcoder/models.json`
里**按模型名**登记,端点和密钥写在一起:

```json
{
  "default": "Qwen3_5-27B",
  "models": {
    "Qwen3_5-27B": {
      "model": "/path/or/name/of/your/model",
      "base_url": "http://YOUR_VLLM_HOST:8000/v1",
      "api_key": "dummy",
      "max_tokens": 16384, "timeout": 180
    },
    "deepseek-v4-pro": {
      "model": "deepseek-v4-pro",
      "base_url": "https://api.deepseek.com/v1",
      "api_key": "sk-...",
      "max_tokens": 16384, "timeout": 180
    }
  }
}
```

`model` 是真正发给 API 的 id(vLLM 用启动时 `--model` 的值注册,所以本地模型常常是一串路径),
**每个条目都显式写出来**,配置才自明;省略时会退回用键名,但不建议依赖。文件含密钥,权限应为 **600**。

**只有登记过的模型可选**——未登记的名字没有端点和密钥,接受它只会悄悄打到不确定的地方,
所以直接报错。选择方式:

```bash
lightcoder --list-models          # 看有哪些
```

- **交互式**:REPL 里 `/model` 弹出编号选择器(输数字或名字,**Esc 取消**),`/model <名字>` 直接切;
- **非交互(`-p`)**:在 `settings.json` 里写 `"model": "<名字>"`——**项目级 `.lightcoder/settings.json`
  可以给每个仓库钉死一个模型**;临时覆盖用环境变量。

**优先级**:shell 环境变量 > `settings.json` 指定的模型 > 项目 `.env`(可选,仓库级密钥)。
跑评测时通常直接用 shell 变量覆盖,例如:

```bash
LIGHTCODER_BASE_URL=... LIGHTCODER_MODEL=... .venv/bin/python -m eval.batch ...
```

### 4. `crane` —— 仅当 Docker 守护进程无外网时才需要

在"shell 有代理但守护进程没有"的机器上,`docker pull` 会失败。`crane` 通过
`HTTPS_PROXY` 拉镜像并写成可 `docker load` 的 tar 包;`eval/pull_images.py` 就是用它。
本机已装好(`~/.local/bin/crane`)。

```bash
# 若缺失则安装(Go 用户态二进制,不需要 sudo):
go install github.com/google/go-containerregistry/cmd/crane@latest   # -> ~/go/bin/crane
# 并导出代理让 crane 能访问 docker.io:
export HTTPS_PROXY=http://127.0.0.1:27890
```

如果你有 sudo,更干净的替代方案是给守护进程配一次代理
(`/etc/systemd/system/docker.service.d/http-proxy.conf` + `systemctl restart docker`)——
这样官方 harness 能原生拉镜像,完全不需要 `pull_images.py`。

---

## 第 1 步 —— 预载题目镜像

打分和 `verify` 都复用本地镜像(不重新构建),所以先把它们拉下来。`m5_44` 子集的镜像通常
已经缓存好,已存在时这一步会直接跳过。

```bash
# 按 id 列表拉整个子集:
.venv/bin/python -m eval.pull_images --ids $(python -c \
  "import json;print(' '.join(json.load(open('eval/subsets/m5_44.json'))))")

# 或按排序后的 id 拉 SWE-bench Lite 的前 N 个:
.venv/bin/python -m eval.pull_images --dataset lite --n 50

# 单个题目:
.venv/bin/python -m eval.pull_images --ids pallets__flask-4045
```

查看已有哪些镜像:

```bash
docker images --format '{{.Repository}}' | grep -c swebench
```

---

## 第 2 步 —— 先跑单题(derisk)

最快地端到端看到整条流水线解出一题:

```bash
.venv/bin/python -m eval.run_single --instance pydata__xarray-4094 --extras verify
# 去掉 --extras verify 就是普通 baseline agent
```

产物在 `results/derisk_pydata__xarray-4094/`,运行结束会打印:

```
[run_single] === VERDICT ===
  resolved=True  (1/1)
```

常用参数:`--skip-harness`(只生成补丁,不跑 Docker 打分)、`--max-rounds N`、`--workdir DIR`。

---

## 第 3 步 —— 跑整个子集(baseline vs 某个优化)

```bash
.venv/bin/python -m eval.batch \
  --subset eval/subsets/m5_44.json \
  --run-id my_baseline \
  --gen-workers 3 --harness-workers 4 --max-rounds 50
```

同一条命令**加上 `--extras verify`** 就是执行式验证那一组 —— 这个开关本身就是 A/B:

```bash
.venv/bin/python -m eval.batch \
  --subset eval/subsets/m5_44.json --extras verify \
  --run-id my_verify --gen-workers 3 --harness-workers 4 --max-rounds 50
```

**batch 的三个阶段:**

1. **prepare(准备)** —— 用 `docker cp` 把每个仓库从镜像抽到 `results/_repos_batch/<run_id>/`
   (串行,这样并行阶段完全不碰 Docker,不会拖垮守护进程)。
2. **generate(生成)** —— 用 `--gen-workers` 个进程并行地在每题上跑 agent(工具是基于 cwd 的,
   所以用进程隔离)。产出补丁。**可续跑**:重跑会跳过 `records.jsonl` 里已生成的题(非空、无报错)。
3. **evaluate(评测)** —— 把所有预测一次性交给官方 harness,`--harness-workers` 个 Docker
   评测并行跑。

关键参数:`--skip-harness`、`--n N`(跑前 N 个,替代子集文件)、以及下面的消融开关。

> **关于 `stdout` 缓冲:** Python 对重定向到文件的 `stdout` 是块缓冲,所以你 `tee`/重定向到日志时
> `[k/44]` 进度行会滞后。而 `httpx` 请求行(走 `stderr`)是实时刷新的 —— 盯那些行,或者用
> `docker ps | grep sweb` 看当前在跑哪些题。

---

## 第 4 步 —— 读结果

一次运行的所有东西都在 `results/<run_id>/`:

| 文件 | 内容 |
|---|---|
| `summary.json` | 核心结果:`{"resolved": N, "total": M, "resolved_ids": [...], ...}` |
| `predictions.jsonl` | 每题一条 `{instance_id, model_name_or_path, model_patch}`(harness 的输入) |
| `records.jsonl` | 每题的完整记录:补丁 **+ 完整 trajectory + meta**(工具调用数、耗时、空补丁标志、报错、`verify` 标志) |
| `*.<run_id>.json` | 官方 harness 的原始报告 |
| `logs/`(在 harness 运行目录下) | 打分时每题的测试日志 |

```bash
cat results/my_verify/summary.json
# 每题的补丁 / trajectory:
python -c "import json;[print(r['instance_id'], r['meta']['n_tool_calls'], r['meta']['empty_patch']) \
  for r in map(json.loads, open('results/my_verify/records.jsonl'))]"
```

---

## `verify` 模式(执行式自验证)

`--extras verify` 让 agent **真去运行它的修复,而不只是看一眼**。不同于 baseline
(在没有依赖的宿主 checkout 上改代码,其提示词甚至明说*别跑测试*),verify 的 runner 会:

- 把 `run_bash` 路由进这道题**自己的 swebench 容器**(和打分用的是同一个镜像),把宿主仓库
  bind-mount 到 `/testbed`,并 `--network none`。镜像里的 editable 安装解析到的就是这份被挂载
  (且被改过)的源码,于是 `edit_file`(宿主)→ `python`(容器内,带真实依赖)是即时生效的;
- 换上一套 **复现 → 修 → 重跑** 的任务提示词;
- 把 agent 的复现临时文件从补丁里剔除(`extract_patch(include_new=False)`),这样某个漏下的
  `test_*.py` 不会在打分时破坏测试收集。

agent 是对着**它自己根据 issue 写的复现**(以及已有测试)来验证的 —— **绝不**碰隐藏的
FAIL_TO_PASS 打分测试。这正是 `docs/M1_NOTES.md` §3–4 里预想的那个"把 agent 的工具放进
实例容器里跑"的 proper fix。

交互式(非评测)使用时,同一套机制通过
`lightcoder --extras verify --sandbox docker --sandbox-image <带你依赖的镜像>` 暴露。

---

## 消融开关 & extras

每个旋钮都是一个 batch 参数(对应 `AgentConfig`),所以一次 sweep 就是把这条命令按每个配置各跑
一遍。run id 默认取描述该配置的标签(`baseline`、`+verify`、`no_compression`、`prompt_react` …)。

| 参数 | 作用 |
|---|---|
| `--no-parallel` | 一轮里的多个工具调用改为串行 |
| `--no-compression` | 关闭分层上下文压缩 |
| `--no-subagent` | 隐藏 `delegate` 子 agent 工具 |
| `--prompt-variant {default,minimal,verbose,react}` | 切换系统提示词风格 |
| `--extras verify` | 执行式验证(见上) |
| `--extras todo` | 常驻目标 + 运行 TODO,每轮重新注入(抗偏移) |
| `--max-rounds N` | 每题最多的 LLM⇄工具 轮数 |

---

## 子集

| 文件 | 数量 | 说明 |
|---|---|---|
| `m5_44.json` | 44 | 主对比子集(hermetic 仓库;剔除了 `psf/requests` —— 它的测试需要联网,离线会卡死) |
| `m5_50.json` | 50 | m5_44 的超集(含那 6 个需要联网的题) |
| `m2_small20.json` | 20 | 快速冒烟子集 |
| `m2_ready.json` | — | 早期 M2 的镜像就绪列表 |

子集是固定 id 列表,保证同一批题跑遍每个配置 —— 差异只归因于 agent,而非抽样。

---

## 排错

- **`image not present locally`** —— 对这些 id 跑第 1 步(`pull_images`),或给守护进程配代理。
- **某题卡住约几分钟然后超时** —— 它的测试需要联网(如 `psf/requests` 会访问 `httpbin.org`)。
  用 hermetic 子集;`m5_44` 已经排除了这些。
- **`docker pull` 超时但 `curl` 正常** —— 守护进程/代理不匹配;用 `crane`(前置第 4 条)。
- **进度看起来卡在 0** —— 基本都是 `stdout` 缓冲(见第 3 步的说明),不是卡死。用
  `docker ps | grep sweb` 和实时的 `httpx` 日志行确认。
- **两个 run 同时共用一个模型端点** —— 会抢吞吐、也会把计时搅乱。串行跑,或指向不同端点。
- **磁盘** —— 镜像占大头(每个约 1–2.5 GB);抽出来的仓库在 `results/_repos_batch/` 下每次
  44 题约 2.6 GB(run 之间可安全删除)。
