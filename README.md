# 随访风险模型 · 三折分组验证 API

面向临床统计复验的 FastAPI 服务：上传随访 CSV，按 **受试者分组** 做三折逻辑回归验证，
返回 **按原行序** 的折外阳性概率和每折 AUC，并提供并行批次上限、状态查询、取消与完整报告下载。

## 快速启动

```bash
docker compose up --build
```

- Swagger UI：<http://localhost:8000/docs>
- OpenAPI：<http://localhost:8000/openapi.json>
- 健康检查（含名额）：<http://localhost:8000/api/v1/health>

本地不用 Docker 时：

```bash
python3 -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> 名额与任务都保存在进程内存中，**必须单 worker 运行**。Compose 已固定为 1 个 worker。

## 输入 CSV

第一行为表头，UTF-8 编码：

| 列 | 要求 |
| --- | --- |
| `subject_id` | 受试者号，非空；同一受试者的所有随访行同折 |
| `outcome` | 每行必须为 `0` 或 `1` |
| 其他列 | 全部视为数值特征；单元格可留空（`""`/`NA`/`NaN` 等），按当折训练行中位数填补 |

约束：至少 3 行、至少 3 个不同受试者、整体结局须同时含 0 和 1；行数 ≤ 50,000；
特征数 ≤ 198；文件 ≤ 20 MiB。违规返回 `400` 与机器可读的 `error.code`。

## 方法学

- 划分：`StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)`，
  受试者号作为 group，保证同一受试者不跨折；报告逐折记录 `group_leakage=false`。
- 每折流水线：`SimpleImputer(median)` → `StandardScaler` → `LogisticRegression(lbfgs, max_iter=1000)`。
  **填补器和标准化器只用该折训练行 `fit`**，再 `transform` 验证行；报告给出每折的训练统计量供核对。
- 阳性概率：模型 `predict_proba` 中类别 `1` 的概率；折外结果按上传 CSV 的原始行序返回。
- 每折 AUC：用该折验证行的折外阳性概率计算。
  - 某个 **训练折** 结局单一 → 批次按 `TRAIN_FOLD_SINGLE_CLASS` 失败，不产生任何概率或报告；
  - 某个 **验证折** 结局单一 → 该折 `auc: null`、`auc_status: not_defined_single_class`，
    并附 `auc_reason`；其余折正常完成。
- 另附 `supplementary_pooled_oof_auc` 便于整体参考，它不是验收口径。

## 并发、取消与终态

| 规则 | 实现 |
| --- | --- |
| 全站最多 2 批同时训练 | 进程内 2 槽；第 3 批立即 `429 CAPACITY_EXCEEDED`（带 `Retry-After`），不排队 |
| 取消需等计算真正停止 | `POST .../cancel` 阻塞到该批 worker 退出计算并释放槽位后才返回 `200` |
| 另一批照常运行 | 槽位与事件按任务隔离，取消 A 不影响 B |
| 取消与完成并发只保留一种终态 | 终态在单把锁内发布；取消请求先到则丢弃迟到结果，完成先到则取消返回 `409` |
| 取消后不发布迟到结果 | worker 发布前在锁内复查 `cancel_requested`，命中即清空结果并写 `cancelled` |
| 失败不供下载 | 仅 `completed` 的任务有报告；失败/取消/运行中下载返回 `409` |
| 任务只存内存 | 无数据库、无落盘，重启即清空；`read_only` 容器根文件系统 |

取消中的任务仍占用名额，直到 `cancel/cancel_confirmed_at` 返回。状态值：
`running` → `cancelling` → `cancelled`，或 `completed` / `failed`。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/v1/health` | 健康与名额 |
| `POST` | `/api/v1/validations` | 上传 CSV（multipart：`file`，可选 `seed`、`demo_delay_seconds`），返回 `202` + `job_id` |
| `GET` | `/api/v1/validations` | 列出内存中任务与名额 |
| `GET` | `/api/v1/validations/{job_id}` | 状态、进度、结果摘要或失败原因 |
| `POST` | `/api/v1/validations/{job_id}/cancel` | 请求取消并等待计算停止 |
| `GET` | `/api/v1/validations/{job_id}/report` | 完整 JSON 报告（仅成功任务） |
| `GET` | `/api/v1/validations/{job_id}/oof.csv` | 折外概率 CSV（仅成功任务） |

`demo_delay_seconds`（0–20）是每折的合成等待，用于演示并行与取消；真实分析请保持 0。

```bash
# 提交
curl -F file=@data/followup_demo.csv -F demo_delay_seconds=0 \
     http://localhost:8000/api/v1/validations

# 查询 / 取消 / 下载
curl http://localhost:8000/api/v1/validations/<job_id>
curl -X POST http://localhost:8000/api/v1/validations/<job_id>/cancel
curl -OJ http://localhost:8000/api/v1/validations/<job_id>/report
```

## 虚构演示数据

`data/` 下均为脚本生成的合成数据（无真实患者信息）：

| 文件 | 用途 |
| --- | --- |
| `followup_demo.csv` | 常规演示，三折 AUC 均可得 |
| `followup_validation_single.csv` | 第 2 折验证集全为 0，AUC 留空但批次成功 |
| `followup_train_single.csv` | 第 1 折训练集全为 1，整批失败（无报告） |
| `followup_slow_demo.csv` | 配合 `demo_delay_seconds=8` 观察并行与取消 |

重新生成：`python3 scripts/generate_demo_data.py`（脚本会用与服务相同的划分器校验每种情形）。

## 演示并行验证与取消

```bash
docker compose up --build -d
python3 scripts/demo_parallel_cancel.py
```

脚本会依次展示：提交 A/B 两批 → 第 3 批 `429` → 取消 A（计时并确认槽位释放）→
B 不受影响并完成 → 释放的名额可再提交 → 验证折单一结局 AUC 为空 → 训练折单一结局整批失败。

## 测试

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest tests -q
```

覆盖 CSV 校验、行序与同折约束、逐折预处理拟合范围、两类单一结局、容量上限、
取消等待与释放、取消/完成的两种确定性竞态、失败与取消不可下载、Swagger 可用性。

## 代码结构

```
app/validation.py   CSV 解析与校验 → FollowupDataset
app/analysis.py     分组划分、逐折拟合/预测、AUC 与报告主体（无 I/O）
app/manager.py      内存任务表、2 槽容量、取消屏障、单终态发布
app/main.py         FastAPI 路由、错误契约、报告/折外 CSV 下载
app/schemas.py      OpenAPI 响应模型
scripts/            演示数据生成、并行/取消演示
tests/              pytest 套件
```

## 已知边界

- 单进程内存实现：多 worker 或多副本会各自持有名额，需外部协调（本服务固定单 worker）。
- 训练为线程内 CPU 计算；取消是协作式检查点（每折、填补/拟合/预测之间），不能在单个
  scikit-learn 调用内部中断，因此取消确认最多等待当前最小计算步完成。
