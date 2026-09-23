# GPU Fair-Queuing Scheduler (FastAPI + SQLite)

多租户 GPU 作业**加权 Deficit Round-Robin（WDRR）公平排队 API**，内置**等待时间老化补偿**防止小作业长期饥饿，并在任何时刻**不越过集群 GPU 容量约束**。

- **语言/框架**：Python 3.11 + FastAPI + Uvicorn
- **存储**：SQLite（WAL 模式，零外部依赖，数据持久化到挂载卷）
- **可复现**：固定租户访问顺序（租户 id 升序）、租户内 FIFO、顺序作业 id、无随机数；相同输入必然得到相同选择顺序
- **可审计**：每笔 `AGING / QUANTUM / DISPATCH` 余额变动都写入 `ledger`

---

## 一条命令启动

需要本机已安装 Docker 与 Docker Compose 插件（Docker Desktop / `docker-compose-plugin`）。
镜像构建只需 `requirements.txt`，SQLite 数据库文件保存在命名卷 `scheduler-data`
（容器内 `/data/scheduler.db`），重建容器不丢数据：

```bash
docker compose up --build -d        # 构建镜像并后台启动
docker compose logs -f scheduler    # 跟踪服务日志（Ctrl-C 仅退出查看）
docker compose ps                   # 查看容器与健康检查状态
```

服务监听 **`http://localhost:8000`**（容器内 8000，可在 `docker-compose.yml` 改宿主机端口）。

停止 / 清理：

```bash
docker compose down          # 停止，数据保留在卷 scheduler-data
docker compose down -v       # 停止并删除数据
```

健康检查：

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

交互式 API 文档（Swagger UI）：<http://localhost:8000/docs>，OpenAPI：<http://localhost:8000/openapi.json>。

---

## 调度模型

### 成本单位

作业成本 = 资源占用：

```
cost_gpu_min = gpus × duration_min        # GPU-分钟
```

每个租户持有同一单位的 **deficit 余额（balance）**。

### 加权 DRR（每次 `POST /scheduler/batch`）

1. 取所有含 QUEUED 作业的租户，按**租户 id 升序**固定排列。
2. 每个 DRR 轮次，每位“队首作业**同时**能放入当前空闲 GPU、且未超过本租户并发
   GPU 上限（见下节）”的租户获得 `BASE_QUANTUM_GPU_MIN × weight` 的 QUANTUM 信用。
3. 按固定顺序访问租户：当 `余额 ≥ 队首作业成本` **且** `空闲 GPU ≥ 作业 GPU 数`
   **且** `租户 RUNNING 占用 GPU + 作业 GPU 数 ≤ 租户上限` 时，派发该作业并扣减余额；否则继续。
4. **容量永远不越界**：放不下的作业——无论是受**集群容量**还是受**租户上限**
   阻塞（已有运行作业占满上限，或队首作业自身超过上限）——本轮不获得任何
   QUANTUM——等待释放由老化补偿，也避免重复调用接口“刷”余额。各租户的资格彼此
   独立：被上限阻塞的租户被跳过，**其他符合条件的租户本轮照常调度**。
5. “放得下但暂时买不起”的租户可在**同一批次内连续多轮累积**直到买得起；
   批次结束时未促成任何派发的尾部授权自动回滚，保证空转调用不改变余额。

### 租户并发 GPU 上限（可选 `max_gpus`）

每个租户可设置一个**并发 GPU 占用上限**（整数，正）；`null`（或旧租户未设置）
表示**不限额**，只受集群总容量约束：

- 通过 `POST /tenants` **设置 / 更新 / 清除**：
  - 请求体带 `"max_gpus": 4` → 设置为 4；
  - 请求体显式带 `"max_gpus": null` → 清除上限（恢复不限额）；
  - **完全省略** `max_gpus` 字段 → 保留该租户既有上限不变（只更新 `weight`）；
    新租户省略时默认为不限额。
- 提交作业（`POST /jobs`，含自动注册租户）**不会**改动既有上限；自动注册的
  新租户以及升级前已存在的旧租户一律默认为 `null`（不限额）。
- 调度时队首作业必须**同时**满足集群空闲容量与租户上限才会被派发；受上限阻塞
  的租户本轮**不获得 QUANTUM**（但与容量阻塞一样可正常获得老化补偿，不会饥饿），
  其余租户不受影响。
- 作业 **`complete` 或 `cancel`（RUNNING 中取消）立即释放**其占用的额度。
- 若队首作业自身 GPU 数大于租户上限，它会保持 QUEUED（提交时不拒绝，因为上限
  可随时调整）；调高或清除上限后下一批次即可调度。
- 上限**持久化在 SQLite** 的 `tenants.max_gpus` 列；对没有该列的旧数据库，启动时
  自动执行兼容迁移（`ALTER TABLE ADD COLUMN`），旧租户该列为 `NULL`（不限额）。
- 租户对象（`GET /tenants`、`GET /scheduler/state`、批次结果等）含
  `max_gpus`（上限，`null` 为不限）与 `running_gpus`（当前 RUNNING 作业占用 GPU
  数，即上限已用量）。

### 老化补偿（防饥饿）

- 自租户**最早排队作业的到达时刻**起，每等待 1 分钟按
  `AGING_RATE_GPU_MIN_PER_MIN` 发放 GPU-分钟信用（可用
  `AGING_CAP_GPU_MIN` 限制单批次上限）。
- 效果：小作业即便排在高权重大租户后面，等待越久信用越高，最终必然能负担，
  从而对等待时间形成上界。
- 队列排空后老化时钟重置，新作业从自己的到达时刻重新计时。
- 同一时刻重复调度不会产生信用（不可刷）。

### 生命周期

```
QUEUED ──(batch 选中)──▶ RUNNING ──complete──▶ COMPLETED
   │                        │
   └──────cancel────────────┴──────cancel────▶ CANCELLED
```

RUNNING 作业占用 GPU，直到 `complete`/`cancel` 才释放给后续批次——释放的既包括
**集群容量**，也包括**租户并发 GPU 上限**（`max_gpus`）的占用额度。

### 作业依赖（`depends_on`，DAG）

提交作业时可传入前置作业 id 列表；依赖是**集群级**的，允许**跨租户**依赖：

- 前置作业**全部 `COMPLETED`** 后，依赖作业才可被批次调度。
- 提交时校验：前置作业不存在 → `400 unknown depends_on`；形成环路（含自依赖）
  → `400 dependency cycle`，作业不会创建。
- 批次中只有「队首作业的前置已全部完成」的租户参与 DRR；队首被依赖阻塞时
  **不获得 QUANTUM、也不产生老化信用**（仅推进老化计时点），因此空等依赖
  无法“刷”出余额；严格 FIFO，被阻塞队首后面的本租户作业同样等待。
- 前置作业被 **`CANCELLED`** 后永远不可能 COMPLETED：依赖作业**保持 QUEUED**，
  通过 `GET /jobs/{id}`、`GET /jobs` 与 `GET /queue` 的 `blocked_by` 字段可查询
  阻塞原因（前置作业 id 及其当前状态）。
- 作业对象包含：
  - `depends_on`：全部前置作业 id（去重、按 id 排序）；
  - `blocked_by`：尚未 COMPLETED 的前置作业 `[{id, status}]`，仅对 QUEUED 作业有意义，空数组表示立即可调度。

---

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `CLUSTER_GPUS` | `8` | 集群总 GPU 数，任何批次不会超过 |
| `BASE_QUANTUM_GPU_MIN` | `60` | 每单位权重每轮的量子（GPU-分钟） |
| `AGING_RATE_GPU_MIN_PER_MIN` | `1` | 每等待 1 分钟补偿的 GPU-分钟 |
| `AGING_CAP_GPU_MIN` | 空（无上限） | 单批次老化补偿上限 |
| `DB_PATH` | `/data/scheduler.db` | SQLite 文件路径 |

提交作业时若 `gpus > CLUSTER_GPUS`（永远无法运行）会被拒绝（400）。

> 注意：租户上限 `max_gpus` **不是**环境变量，而是通过接口逐租户设置并持久化在
> SQLite 中；环境变量只配置上面这些全局参数。

---

## API 用法

基础地址：`http://localhost:8000`

### 0. 管理租户（权重与并发 GPU 上限）`POST /tenants`

```bash
# 创建租户：权重 2，并发 GPU 上限 4
curl -X POST http://localhost:8000/tenants \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-a","weight":2.0,"max_gpus":4}'

# 更新权重但保留原上限（省略 max_gpus 字段）
curl -X POST http://localhost:8000/tenants \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-a","weight":1.5}'

# 清除上限（显式 null => 恢复不限额）
curl -X POST http://localhost:8000/tenants \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-a","weight":1.5,"max_gpus":null}'
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `tenant_id` | 是 | 非空租户标识 |
| `weight` | 是 | 租户权重，必须 > 0 |
| `max_gpus` | 否 | 并发 GPU 上限，正整数；显式 `null` 清除上限；**省略**时保留既有上限（新租户为不限额 `null`）。0/负数 → 422 |

响应为租户对象，除权重/余额外还包含：

- `max_gpus`：并发 GPU 上限（`null` = 不限额）；
- `running_gpus`：该租户当前 RUNNING 作业合计占用的 GPU 数（上限已用量）；
- `queued_jobs` / `running_jobs`：队列中 / 运行中的作业数。

### 1. 提交作业 `POST /jobs`

提交作业时租户会被自动注册（也可用 `POST /tenants` 单独管理权重与 `max_gpus`）。
同一租户重复提交可更新其权重，但**不会**改动该租户已设置的 `max_gpus` 上限
（自动注册的新租户上限为 `null`，即不限额）。

```bash
curl -X POST http://localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-a","gpus":2,"duration_min":30,"weight":1}'
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `tenant_id` | 是 | 非空租户标识 |
| `gpus` | 是 | 所需 GPU 数（正整数） |
| `duration_min` | 是 | 预计时长（分钟，正整数） |
| `weight` | 否 | 租户权重，默认 `1.0`，必须 > 0 |
| `depends_on` | 否 | 前置作业 id 列表（默认 `[]`），**跨租户允许**；未知 id 或形成环路返回 400，自动去重 |

响应：作业对象（顺序 id 如 `J000001`，`status: QUEUED`），含 `depends_on` 与
`blocked_by` 字段（见上文「作业依赖」一节）。

依赖提交示例（`J000002` 等 `J000001` 完成后才可调度）：

```bash
curl -X POST http://localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-b","gpus":1,"duration_min":30,"depends_on":["J000001"]}'
# 201 {"id":"J000002","status":"QUEUED",
#      "depends_on":["J000001"],
#      "blocked_by":[{"id":"J000001","status":"QUEUED"}], ...}
```

### 2. 调度下一批 `POST /scheduler/batch`

执行一次 WDRR + 老化批次，返回**选择顺序**、每租户**余额**、容量使用与完整事件流水：

```bash
curl -X POST http://localhost:8000/scheduler/batch
```

```jsonc
{
  "batch_id": 1,
  "capacity_gpus": 8,
  "used_gpus_before": 0,
  "available_gpus_at_start": 8,
  "used_gpus_after": 8,
  "free_gpus_after": 0,
  "rounds": 3,
  "jobs_selected": 2,
  "selection_order": ["J000001", "J000002"],
  "selected_jobs": [ { "id": "J000001", "run_index": 1, ... } ],
  "tenants": [
    {"id":"team-a","weight":1.0,"balance_gpu_min":0.0,"max_gpus":null,"running_gpus":1,"queued_jobs":0,"running_jobs":1},
    {"id":"team-b","weight":2.0,"balance_gpu_min":60.0,"max_gpus":null,"running_gpus":1,"queued_jobs":0,"running_jobs":1}
  ],
  "events": [
    {"kind":"AGING",  "tenant_id":"team-a", "amount":0.01, ...},
    {"kind":"QUANTUM","tenant_id":"team-a", "amount":60.0, "detail":"round 1"},
    {"kind":"DISPATCH","tenant_id":"team-a","amount":-60.0,"job_id":"J000001"}
  ]
}
```

### 3. 完成 / 取消作业

```bash
curl -X POST http://localhost:8000/jobs/J000001/complete   # RUNNING -> COMPLETED，释放 GPU
curl -X POST http://localhost:8000/jobs/J000002/cancel     # QUEUED/RUNNING -> CANCELLED
```

- 只有 RUNNING 可 complete（否则 409）；已结束作业重复取消 409；不存在 404。

### 4. 查询队列 / 作业 / 租户 / 流水

```bash
curl http://localhost:8000/queue                    # 每租户 FIFO 队列、位置、权重、余额
curl "http://localhost:8000/jobs?status=QUEUED"     # 可按 status / tenant_id 过滤
curl http://localhost:8000/jobs/J000001
curl http://localhost:8000/tenants                  # 全部租户 + 余额
curl http://localhost:8000/tenants/team-a
curl http://localhost:8000/scheduler/state          # 容量、空闲 GPU、调参、租户余额
curl "http://localhost:8000/ledger?tenant_id=team-a&limit=200"  # AGING/QUANTUM/DISPATCH 审计
```

`GET /queue` 示例（`J000005` 等待 `J000003` 完成，队列中直接标出阻塞原因）：

```jsonc
{
  "cluster_gpus": 8, "used_gpus": 6, "free_gpus": 2,
  "queued_jobs": 1, "running_jobs": 1,
  "queues": [{
    "tenant_id": "team-c", "weight": 1.0, "balance_gpu_min": 12.5,
    "max_gpus": null, "running_gpus": 0,
    "oldest_queued_at": "2026-09-22T10:00:00+00:00",
    "jobs": [{
      "id": "J000005", "position": 1, "gpus": 4, "status": "QUEUED",
      "depends_on": ["J000003"],
      "blocked_by": [{"id": "J000003", "status": "RUNNING"}], ...
    }]
  }]
}
```

### 5. 按时间窗汇总 GPU 用量 `GET /usage/gpu`

按 UTC 时间窗 `[start, end)`（**左闭右开**）汇总每租户与全局的实际 GPU 用量：

- 用量来自作业的**实际 RUNNING 区间**，而非提交时的 `duration_min` 估算：
  - **已结束作业**（`COMPLETED`，或运行中被 `CANCELLED`）：`scheduled_at → completed_at`；
  - **仍在运行的作业**：`scheduled_at` 起算，**截断到查询截止时刻 `end`**（不假设作业在未来仍持续）。
- 与查询区间按左闭右开求交：作业恰好在 `end` 才启动、或恰好在 `start` 结束，用量均为 **0**。
- **QUEUED 作业不计入**（从未占用 GPU）；排队期间被取消的作业同样不计。
- 用量 = `gpus × 交集分钟数`；`overlapping_jobs` 为 RUNNING 区间与时间窗有非空交集的作业数（不是 GPU 数）。
- 返回所有已知租户（按 id 排序），**无重叠作业的租户数值为零**；时间窗内无任何作业时，全局用量也为零。
- 查询参数：

| 参数 | 必填 | 说明 |
|---|---|---|
| `start` | 是 | 起始时刻（**UTC**，ISO-8601，如 `2026-09-23T00:00:00Z`），含 |
| `end` | 是 | 截止时刻（**UTC**，ISO-8601），不含；**必须晚于 `start`，否则 400** |
| `tenant_id` | 否 | 只统计单个租户（未知租户返回 404）；缺省统计全部租户 |

```bash
curl -G "http://localhost:8000/usage/gpu" \
  --data-urlencode "start=2026-09-23T00:00:00Z" \
  --data-urlencode "end=2026-09-23T01:00:00Z"
# 只看某租户：
curl -G "http://localhost:8000/usage/gpu" \
  --data-urlencode "start=2026-09-23T00:00:00Z" \
  --data-urlencode "end=2026-09-23T01:00:00Z" \
  --data-urlencode "tenant_id=team-a"
```

```jsonc
{
  "start": "2026-09-23T00:00:00+00:00",
  "end": "2026-09-23T01:00:00+00:00",
  "tenant_id": null,
  "global": { "gpu_minutes": 180.0, "overlapping_jobs": 2 },
  "tenants": [
    {"tenant_id": "team-a", "gpu_minutes": 60.0,  "overlapping_jobs": 1},
    {"tenant_id": "team-b", "gpu_minutes": 120.0, "overlapping_jobs": 1}
  ]
}
```

### 端点一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/jobs` | 提交作业（自动注册/更新租户权重，支持 `depends_on` 依赖） |
| GET | `/jobs` | 列作业（可选 `status`、`tenant_id`） |
| GET | `/jobs/{id}` | 作业详情 |
| POST | `/jobs/{id}/cancel` | 取消作业 |
| POST | `/jobs/{id}/complete` | 完成运行中作业并释放 GPU |
| POST | `/tenants` | 创建租户或更新权重；可选 `max_gpus` 设置/更新/清除并发 GPU 上限 |
| GET | `/tenants` / `/tenants/{id}` | 租户与余额 |
| POST | `/scheduler/batch` | **调度下一批**，返回选择顺序与余额 |
| GET | `/scheduler/state` | 集群/调参/租户快照 |
| GET | `/queue` | 每租户 FIFO 队列与位置 |
| GET | `/usage/gpu` | **按 UTC 时间窗汇总**每租户/全局 GPU-分钟与重叠作业数（可选 `tenant_id`） |
| GET | `/ledger` | 信用/扣减审计流水 |
| GET | `/health` | 健康检查 |

---

## 端到端示例

```bash
# alice: 2 GPU × 30 分钟 = 60 成本，权重 1；bob: 6 GPU × 10 = 60 成本，权重 2
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"alice","gpus":2,"duration_min":30,"weight":1}'
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"bob","gpus":6,"duration_min":10,"weight":2}'
# carol 需要 4 GPU，此刻放不下
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"carol","gpus":4,"duration_min":10}'

curl -s -X POST localhost:8000/scheduler/batch        # => [J000001, J000002]，用满 8 GPU
curl -s -X POST localhost:8000/scheduler/batch        # => []（容量不足，carol 等待并老化）
curl -s -X POST localhost:8000/jobs/J000001/complete  # 释放 2，仍不足 4
curl -s -X POST localhost:8000/jobs/J000002/complete  # 再释放 6
curl -s -X POST localhost:8000/scheduler/batch        # => [J000003]，carol 获得等待老化补偿
curl -s localhost:8000/queue

# 作业依赖（跨租户）：dave 的作业等 carol 的 J000003 完成后才会被调度
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"dave","gpus":2,"duration_min":15,"depends_on":["J000003"]}'
curl -s -X POST localhost:8000/scheduler/batch        # => []（J000003 仍 RUNNING，dave 不消耗量子）
curl -s localhost:8000/jobs/J000004                   # blocked_by 标明 J000003/RUNNING
curl -s -X POST localhost:8000/jobs/J000003/complete

# 租户并发 GPU 上限：team-c 上限 2，其 2-GPU 作业可派发，后续作业被本租户上限挡住，
# 但其他租户不受影响
curl -s -X POST localhost:8000/tenants -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-c","weight":1,"max_gpus":2}'
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-c","gpus":2,"duration_min":10}'   # 占满上限
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-c","gpus":1,"duration_min":10}'   # 超上限，排队
curl -s -X POST localhost:8000/scheduler/batch             # 只派发第一个
curl -s localhost:8000/tenants/team-c                      # max_gpus=2, running_gpus=2
# 完成后释放额度，下一批次派发第二个；也可再次 POST /tenants 调高或清除上限
curl -s -X POST localhost:8000/scheduler/batch        # => [J000004]

# 按 UTC 时间窗汇总实际 GPU 用量（已结束作业取 scheduled_at→completed_at，
# 仍 RUNNING 的作业截断到 end；QUEUED 不计；左闭右开）
curl -s -G localhost:8000/usage/gpu \
  --data-urlencode "start=2026-09-23T00:00:00Z" \
  --data-urlencode "end=2026-09-23T02:00:00Z"
# {"global":{"gpu_minutes":...,"overlapping_jobs":...},"tenants":[...]}
```

---

## 本地开发（不使用 Docker）

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt

# 运行全部测试（50 个：服务层算法/依赖/用量汇总/租户 GPU 上限 + HTTP 端到端）
.venv/bin/python -m pytest -q

# 无需第三方库的核心算法验证脚本（仅用标准库 sqlite3）
python3 scripts/verify_core.py

# 本地启动（默认 SQLite 写入 ./data/scheduler.db）
CLUSTER_GPUS=8 .venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

## 项目结构

```
app/
  config.py       # 环境变量配置
  db.py           # SQLite 连接、schema（tenants/jobs/batches/ledger/job_dependencies）
  scheduler.py    # 公平排队核心（WDRR + 老化，框架无关）
  main.py         # FastAPI 路由与请求模型
scripts/
  verify_core.py  # 零依赖核心算法验证
tests/
  test_scheduler.py  # 算法测试（注入时间，确定性断言）
  test_api.py        # HTTP 端到端测试
Dockerfile
docker-compose.yml
```
