# NLP-Agent-01 家庭记账助手 — 优化文档

> 工单编号：人工智能NLP-Agent数字人项目-记账本任务
> 来源：北京八维信息集团 · 八维文化与产业研究院
> 版本：v1.1（优化建议）
> 日期：2026-06-28

---

## 一、现状分析与优化方向概览

### 1.1 当前架构评价

| 维度 | 现状 | 评价 |
|------|------|------|
| 业务逻辑准确性 | 规则引擎保证，数据校验完善 | ✅ 可靠 |
| AI 自主性 | 强规则 + 弱 AI，LLM 只负责润色 | ⚠️ 有提升空间 |
| 代码组织 | 多模式并存，存在遗留代码 | ⚠️ 需要重构 |
| 扩展性 | 工具系统与 Agent 分离较好 | ✅ 可扩展 |
| 用户体验 | 交互简洁，多轮确认完善 | ✅ 良好 |

### 1.2 优化方向总览

```
优化方向
├── 一、代码架构优化      — 清理遗留代码，统一 Agent 模式
├── 二、AI 能力增强       — 让 LLM 承担更多判断，减少硬编码规则
├── 三、功能体验优化      — 批量操作、统计分析、报表导出
├── 四、数据能力增强      — 数据分析、趋势图、预算提醒
├── 五、稳定性与运维      — 错误处理、监控、测试
└── 六、优先实施路线图    — 按优先级排序的实施计划
```

---

## 二、代码架构优化

### 2.1 清理遗留代码

**问题**：`agent/` 目录下存在三套 Agent 实现，共存但只有一套被使用：

```
agent/
├── run.py              ✅ 被 main.py 调用（当前生产路径）
├── guardrails.py       ✅ 被 run.py 调用（核心规则引擎）
├── llm_agent.py        ❌ 遗留代码，AGENT_MODE=llm 时未完全实现
├── langchain_agent.py  ❌ 遗留代码，注释标注"假langchain"
├── prompts.py          ✅ 被使用
└── memory.py           ⚠️ 存在但未被当前流程调用
```

**优化建议：**

1. **短期**：删除 `llm_agent.py` 和 `langchain_agent.py`，避免维护负担
2. **中期**：`memory.py` 的 `SessionMemory` 接入主流程，支持真正的多轮对话记忆
3. **模式清理**：移除 `config.py` 中未使用的 `llm` 和 `rule` 模式，只保留 `langchain`

```python
# config.py 优化后
SUPPORTED_AGENT_MODES = {"langchain"}  # 只保留一个，减少心智负担
```

### 2.2 合并重复的条件分支

**问题**：`run.py` 中存在重复的 `if` 分支：

```python
# run.py 当前代码
if guardrail_reply and guardrail_reply != DEFAULT_WELCOME:
    return guardrail_reply
if guardrail_reply == DEFAULT_WELCOME:   # ← 与上面逻辑重叠
    return guardrail_reply
return "处理完成，无返回内容。"
```

**优化建议**：合并为一个条件：

```python
# run.py 优化后
if guardrail_reply:
    return guardrail_reply
return "处理完成，无返回内容。"
```

### 2.3 引入结构化日志框架

**当前问题**：使用 `logging.basicConfig` 全局配置，日志格式不统一。

**优化建议**：统一使用 `tools/logging_utils.py` 的日志工具，避免 `main.py` 直接配 `basicConfig`：

```python
# 新增 log_config.py
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# main.py 中
from log_config import setup_logging
setup_logging(LOG_LEVEL)
```

---

## 三、AI 能力增强

### 3.1 将规则逻辑迁移到 LLM（渐进式）

**现状**：关键字段提取（成员、日期、金额）全部依赖正则表达式，LLM 只负责最后一步润色。

**问题**：

- 正则无法处理模糊表达，如"前天"、"上周五"需要硬编码
- 新增字段或修改规则需要改代码
- 无法理解复杂语义，如"和妈妈一起看电影花了50"

**优化建议（渐进式）**：分三步将判断权交给 LLM：

#### 步骤一：LLM 提取 + 代码校验（推荐立即实施）

在 `guardrails.py` 中增加一条 LLM 提取路径，让 LLM 尝试从文本中提取结构化信息，代码再做校验和补全：

```python
def _extract_slots_with_llm(text: str) -> dict | None:
    """让 LLM 提取槽位信息，失败时降级到正则"""
    prompt = f"""从以下用户输入中提取记账信息，返回 JSON：
    输入：{text}
    返回格式：{{"date": "YYYY-MM-DD", "member": "爸爸/妈妈/女儿", "amount": 数字, "category": "事项"}}
    如果无法提取某字段，返回 null"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        return json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        return None  # 降级到正则
```

#### 步骤二：LLM 主导意图识别（中期）

将 `_detect_intent()` 从关键词匹配改为 LLM 判断，同时保留代码兜底：

```python
def _detect_intent_llm(text: str) -> str:
    """LLM 判断用户意图"""
    prompt = f"""判断用户意图，可选意图：ADD_EXPENSE/ADD_INCOME/DELETE/QUERY
    输入：{text}
    直接返回一个意图词，不要其他内容"""
    ...
```

#### 步骤三：完全信任 LLM（长期目标）

将整个 GuardrailAgent 重写为纯 LLM 驱动的状态机，只保留关键的业务约束（如金额不能为负、成员只能是家庭成员）。

### 3.2 增强 System Prompt

**当前 `prompt.md` 痛点**：是纯文本文件，没有版本管理，难以动态注入上下文。

**优化建议**：

1. **结构化提示词**：用 LangChain 的 `PromptTemplate` 模板化，支持变量注入：
   ```python
   from langchain_core.prompts import ChatPromptTemplate

   prompt = ChatPromptTemplate.from_messages([
       ("system", SYSTEM_PROMPT_TEMPLATE),
       ("placeholder", "{chat_history}"),
       ("human", "{input}"),
       ("placeholder", "{agent_scratchpad}"),
   ])
   ```

2. **动态注入上下文**：让 LLM 知道最近几条记录、当前状态：
   ```python
   context = f"最近记录：{recent_records}\n当前状态：{pending_action}"
   ```

3. **Few-shot 示例**：在 prompt 中加入 2-3 个对话示例，帮助 LLM 理解期望的回复风格。

### 3.3 接入真正的对话记忆

**当前问题**：`SessionMemory` 类存在但未被调用，每轮对话都是独立上下文。

**优化建议**：

```python
# agent/run.py 改造
from agent.memory import SessionMemory

memory = SessionMemory()

def run(user_input: str) -> str:
    guardrail_reply = _guardrail_agent.reply(user_input)
    if guardrail_reply and guardrail_reply != DEFAULT_WELCOME:
        # 确认类回复不需要走 LLM
        return guardrail_reply

    # 注入历史上下文
    chat_history = memory.snapshot()
    messages = [HumanMessage(content=user_input)]

    result = graph.invoke({"messages": messages})
    last_message = result["messages"][-1]

    # 记录到记忆
    memory.add_user(user_input)
    memory.add_assistant(last_message.content)

    return last_message.content
```

---

## 四、功能体验优化

### 4.1 批量记账

**现状**：每条记录需要单独输入、单独确认。

**优化建议**：支持一次性输入多条记录：

```
您：今天中午吃了碗面花了15，晚上买菜花了32，明天给妈妈发了500红包
小账：好的，我帮您记录以下 3 笔：
  [1] 2026-06-28 爸爸 支出 15.0元（中午吃面）
  [2] 2026-06-28 爸爸 支出 32.0元（晚上买菜）
  [3] 2026-06-28 爸爸 支出 500.0元（给妈妈发红包）
  确认无误吗？
您：确认
小账：3 笔记录全部保存 ✅
```

**实现思路**：

- `guardrails.py` 的 `_split_records()` 已支持分割多条记录
- 只需在 `_format_insert_confirmation()` 中改为批量格式输出
- 在 `_save_records()` 中做批量插入（当前已是循环批量插入）

### 4.2 月度/年度统计报告

**优化建议**：新增统计类回复模板：

```python
# 示例回复格式
回复 = """
📊 2026年6月 家庭收支报告

💰 收入：8,500元
  ├─ 爸爸工资：6,000元
  └─ 妈妈奖金：2,500元

💸 支出：3,280元
  ├─ 爸爸：1,500元
  ├─ 妈妈：1,200元
  └─ 女儿：580元

📈 结余：5,220元

🏆 本月支出 TOP3：
  [1] 餐饮：850元（占比26%）
  [2] 交通：620元（占比19%）
  [3] 购物：500元（占比15%）
"""
```

### 4.3 预算提醒功能

**优化建议**：支持设定月度预算，超出时主动提醒：

```python
# 用户配置预算
您：给这个月设个餐饮预算1000
小账：好的，已设置本月餐饮预算：1000元

# 超预算提醒
您：晚上请朋友吃饭花了400
小账：⚠️ 提醒：本月餐饮支出已达 980元，接近 1000元预算。
  本次记录：+400元
  记录已保存 ✅
```

### 4.4 支持更多家庭成员

**现状**：硬编码成员列表 `爸爸、妈妈、女儿`。

**优化建议**：

```python
# 支持动态添加/删除家庭成员
您：家里还有奶奶，也要一起记账
小账：好的，已添加家庭成员"奶奶"~

# 成员别名扩展
您：我妈的退休金到账了
# 识别为：妈妈 + 收入
```

---

## 五、数据能力增强

### 5.1 数据导出

**优化建议**：支持导出为 Excel/CSV：

```python
# 新增工具
def export_records(
    date_from: str = None,
    date_to: str = None,
    format: str = "csv"  # csv / excel
) -> str:
    """导出记录为文件，返回文件路径"""
    ...
```

### 5.2 数据可视化

**优化建议**：生成简单的 ASCII 图表：

```python
您：看看这个月支出趋势
小账：2026年6月 每日支出趋势

0  ┤
50 ┤         ╭──
100┤      ╭──╯
150┤  ╭───╯
200┤──╯
    └─────────────────────
      1  5  10  15  20  25  30
```

### 5.3 智能分析

**优化建议**：基于历史数据做简单分析：

- "这个月和上个月比怎么样" → 对比分析
- "爸爸花钱最多的是什么" → 成员画像
- "哪些地方可以省钱" → 建议生成

---

## 六、稳定性与运维

### 6.1 引入单元测试

**现状**：无测试覆盖，改动风险高。

**优化建议**：至少覆盖以下核心函数：

```python
# tests/test_guardrails.py
def test_detect_intent_expense():
    assert _detect_intent("中午吃饭花了15") == "ADD_EXPENSE"

def test_detect_intent_income():
    assert _detect_intent("工资到账了8000") == "ADD_INCOME"

def test_extract_member():
    assert _extract_member("爸爸今天买菜花了20") == "爸爸"

def test_split_multiple_records():
    records = _split_records("中午吃面15，晚上买菜32")
    assert len(records) == 2

# tests/test_db_tools.py
def test_insert_and_query():
    result = insert_record("2026-06-28", "爸爸", "支出", "吃饭", 15.0)
    assert result["success"]
    record_id = result["id"]
    verified = verify_record(record_id)
    assert verified["exists"]
    delete_record(record_id)
```

**测试框架**：pytest + pytest-cov

```bash
pip install pytest pytest-cov
pytest tests/ --cov=agent --cov=tools --cov=db
```

### 6.2 错误处理增强

**当前问题**：`main.py` 中 `except Exception` 全捕获，不够精细。

**优化建议**：分类处理：

```python
try:
    response = run(user_input)
except ValueError as e:
    logger.error("参数错误: %s", e)
    print("输入格式有误，请检查后重试~")
except ConnectionError as e:
    logger.error("API 连接失败: %s", e)
    print("网络有点问题，请稍后重试~")
except Exception as e:
    logger.exception("未知错误")
    print(f"遇到了一点问题：{e}，请稍后重试~")
```

### 6.3 异步化改造

**现状**：所有操作同步阻塞，用户等待时间长。

**优化建议**：对于查询类操作和批量插入，支持异步：

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def run_async(user_input: str) -> str:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, run, user_input)
    return result
```

### 6.4 健康检查接口

**优化建议**：为后续接入 Web/API 预留健康检查：

```python
def health_check() -> dict:
    """健康检查，返回系统状态"""
    return {
        "status": "ok",
        "llm": "connected" if test_llm_connection() else "error",
        "db": "ok" if test_db_connection() else "error",
        "version": "1.0.0",
    }
```

---

## 七、优先实施路线图

### 第一阶段：快速修复（1-2 天）

| 优先级 | 任务 | 收益 |
|--------|------|------|
| P0 | 修复已发现的 bug（遗留代码、重复分支）| 代码健康度 |
| P0 | 删除 `llm_agent.py` 和 `langchain_agent.py` | 减少维护负担 |
| P1 | 接入 `SessionMemory` 支持对话历史 | 体验提升 |
| P1 | 批量记账格式优化 | 功能完善 |

### 第二阶段：AI 能力提升（3-5 天）

| 优先级 | 任务 | 收益 |
|--------|------|------|
| P0 | LLM 提取槽位信息（LLM + 正则双重校验）| 减少硬编码 |
| P1 | Few-shot 示例注入 prompt | 回复质量 |
| P1 | 动态上下文注入（最近记录、当前状态）| 多轮理解 |
| P2 | LLM 主导意图识别替代正则 | 灵活性 |

### 第三阶段：功能增强（5-7 天）

| 优先级 | 任务 | 收益 |
|--------|------|------|
| P0 | 月度统计报告 | 核心功能 |
| P1 | 预算提醒功能 | 用户粘性 |
| P1 | 数据导出（CSV）| 数据可移植 |
| P2 | ASCII 可视化图表 | 体验提升 |

### 第四阶段：工程化（3-5 天）

| 优先级 | 任务 | 收益 |
|--------|------|------|
| P0 | 单元测试覆盖核心函数 | 代码质量 |
| P1 | 分类错误处理 | 稳定性 |
| P1 | 异步化改造 | 响应速度 |
| P2 | Web API 接口（可选）| 扩展性 |

---

## 八、总结

| 维度 | 当前状态 | 目标状态 |
|------|---------|---------|
| **架构** | 多模式并存，遗留代码多 | 单一模式，代码清晰 |
| **AI 能力** | 强规则 + 弱 AI | 规则兜底 + LLM 主导 |
| **对话记忆** | 无 | 真正的多轮上下文 |
| **数据能力** | 基础 CRUD | 统计、分析、可视化 |
| **稳定性** | 无测试 | 核心函数测试覆盖 |

**核心原则**：在保证业务准确性的前提下，逐步将判断权从代码规则迁移到 LLM，同时补齐测试和工程化短板。
