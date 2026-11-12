# 学习陪伴功能 · 第一期设计文档

**版本**：v1.0  
**日期**：2026-11-11  
**作者**：Chris & Sol  
**目标**：从"番茄钟工具"进化为"学习资产积累系统"

---

## 一、设计理念（基于 Naval 财富框架）

### 1.1 核心定位

第0期实现了基础计时和提醒，本质是"出卖时间"的工具——用户按下开始，机器人计时，时间到了提醒，**没有沉淀任何可复用的资产**。

第一期要跨越的鸿沟：**从"计时工具"变成"学习资产管理器"**。

- **专长识别**：每个人的专注节奏、高效时段、易分心场景都不同，机器人要通过数据发现用户的独特学习专长（哪种节奏下效率最高）
- **杠杆放大**：把发现的规律自动应用到后续学习中，无需用户每次手动调整
- **长期复利**：沉淀学习历史，让用户看到"今天的1小时"如何累积成"一年的365小时"，可视化复利效果
- **资产判定**：用户结束使用后，留下的不是空白，而是一份可复用的"个人学习画像"

### 1.2 反模式警示（Naval §9 对照）

- ❌ **忙碌无沉淀**：只记录时长，不分析质量和模式
- ❌ **地位游戏**：搞排行榜、攀比学习时长（零和博弈）
- ❌ **短视改口径**：今天按25分钟，明天改40分钟，后天又改回来，没有数据支撑的瞎调
- ❌ **救火不留 runbook**：用户说"我刚才分心了"，机器人记下来但不总结规律、不给下次建议

### 1.3 第一期的"资产交付物"

完成第一期后，每个用户拥有：

1. **个人学习画像**：最佳专注时长、高效时段、易分心场景、恢复速度
2. **历史趋势报告**：周/月学习时长、完成率、中断模式，可导出留存
3. **自适应提醒策略**：机器人根据历史数据自动调整提醒节奏，无需手动配置

---

## 二、功能架构

### 2.1 数据层扩展

**新增表：`study_analytics`（学习分析快照）**

```sql
CREATE TABLE study_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    period_start TEXT NOT NULL,  -- 统计周期起点（ISO 8601）
    period_end TEXT NOT NULL,    -- 统计周期终点
    period_type TEXT NOT NULL,   -- 'daily' | 'weekly' | 'monthly'
    
    -- 基础指标
    total_study_minutes INTEGER NOT NULL,
    total_break_minutes INTEGER NOT NULL,
    total_sessions INTEGER NOT NULL,
    completed_sessions INTEGER NOT NULL,  -- 正常结束的会话数
    interrupted_sessions INTEGER NOT NULL, -- 被中断的会话数
    
    -- 质量指标
    avg_focus_minutes REAL,      -- 平均专注时长
    max_focus_minutes INTEGER,   -- 单次最长专注
    extend_count INTEGER,        -- 延长次数（说明状态好）
    early_break_count INTEGER,   -- 提前休息次数（说明状态差）
    
    -- 模式识别
    best_time_slot TEXT,         -- 最高效时段（如 '09:00-11:00'）
    worst_time_slot TEXT,        -- 最低效时段
    distraction_triggers TEXT,   -- JSON 数组，常见分心触发词
    recovery_speed_seconds INTEGER, -- 休息后恢复专注的平均时长
    
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(person_id, period_start, period_type)
);
```

**扩展表：`study_events` 新增字段**

```sql
-- 已有字段保持不变，新增：
ALTER TABLE study_events ADD COLUMN quality_score INTEGER; -- 0-10，用户自评或自动判定
ALTER TABLE study_events ADD COLUMN context_tags TEXT;     -- JSON 数组，如 ["晚上", "嘈杂", "deadline前"]
```

### 2.2 核心模块

#### 模块 A：学习画像生成器（`study_profiler.py`）

**职责**：分析历史数据，生成个人学习特征

**关键函数**：

```python
def build_profile(person_id: int, days: int = 30) -> dict:
    """
    分析最近N天数据，返回学习画像：
    {
        "optimal_focus_minutes": 45,  # 最佳专注时长
        "best_hours": [9, 10, 14],     # 高效时段（24小时制）
        "avg_recovery_minutes": 7,     # 平均恢复时间
        "distraction_patterns": ["晚上", "手机通知"],
        "completion_rate": 0.82,       # 会话完成率
        "trend": "improving"           # improving | stable | declining
    }
    """

def suggest_focus_duration(person_id: int) -> int:
    """根据历史完成率建议下次专注时长（分钟）"""

def predict_effective_slot(person_id: int, target_date: str) -> list:
    """预测指定日期的高效时段，返回小时列表"""
```

#### 模块 B：自适应提醒引擎（扩展 `study.py`）

**职责**：根据画像动态调整提醒策略

**当前逻辑**：

```python
# 第0期：固定40分钟提醒休息
if elapsed >= 40 * 60:
    remind("该休息了")
```

**第一期逻辑**：

```python
# 根据用户画像调整
profile = study_profiler.build_profile(person_id)
threshold = profile["optimal_focus_minutes"] * 60

# 如果用户历史经常延长，提前5分钟柔性提醒
if elapsed >= (threshold - 5 * 60) and profile["extend_count"] > 5:
    remind("还有5分钟，要继续还是休息？")
elif elapsed >= threshold:
    remind("时间到，休息一下")

# 休息结束提醒也自适应
recovery_time = profile["avg_recovery_minutes"] * 60
if break_elapsed >= recovery_time:
    remind("休息够了，回来继续")
```

#### 模块 C：学习报告生成器（`study_report.py`）

**职责**：定期生成可视化报告

**输出格式**：飞书文档（通过 `lark-doc` skill）

**报告模板**：

```markdown
# 学习周报（2026-11-04 ~ 2026-11-10）

## 本周成绩单

- 📚 总学习时长：**12小时35分钟**（较上周 +15%）
- ✅ 完成会话：**18次**（完成率 90%）
- 🔥 最长专注：**62分钟**（周三晚上）
- ⚡ 平均专注：**42分钟**

## 发现的规律

- ✨ **你的黄金时段**：上午 9-11 点，完成率 95%，建议把重要学习放这里
- ⚠️ **易分心时段**：晚上 8 点后，完成率仅 65%，建议降低难度或缩短时长
- 💪 **进步信号**：本周延长学习 5 次，说明状态在变好

## 下周建议

1. 把专注时长从 40 分钟调整到 45 分钟（你已经适应了）
2. 晚上时段改用 25 分钟短冲刺
3. 继续保持上午的学习习惯

---
_数据来源：阿伴学习陪伴系统_
```

### 2.3 用户交互扩展

#### 新增对话能力

**场景1：询问学习状态**

```
用户："我最近学习效率怎么样？"
机器人：
"本周你学了 8 小时，完成率 85%，比上周提升了 10%。
你的黄金时段是上午 9-11 点，这个时间段完成率 95%。
晚上容易分心，建议缩短时长或降低难度。"
```

**场景2：主动建议**

```
机器人（上午9点）：
"早上好！现在是你的高效时段，要开始学习吗？
根据你的习惯，建议专注 45 分钟。"
```

**场景3：自评质量**

```
用户（结束学习后）："今天学得不错"
机器人："太好了！记下来了，这次是 45 分钟，下次继续这个节奏。"
（后台：quality_score = 8, context_tags = ["状态好"]）
```

#### 新增 Agent 工具

```python
schema('study_review', '查看学习报告。可指定周期：today|week|month', {
    'period': {'type': 'string', 'enum': ['today', 'week', 'month']}
})

schema('study_suggest', '根据历史数据建议学习时长和时段')

schema('study_rate', '用户自评本次学习质量', {
    'score': {'type': 'integer', 'minimum': 1, 'maximum': 10},
    'tags': {'type': 'array', 'items': {'type': 'string'}}
})
```

---

## 三、实施路线

### 3.1 数据迁移

```sql
-- 为现有数据补充默认值
UPDATE study_events SET quality_score = 5, context_tags = '[]' WHERE quality_score IS NULL;
```

### 3.2 开发顺序（按依赖关系）

1. **数据层**：创建 `study_analytics` 表，扩展 `study_events` 字段
2. **画像生成器**：实现 `study_profiler.py` 核心算法
3. **自适应提醒**：改造 `study.py` 的提醒逻辑
4. **报告生成**：实现 `study_report.py`，集成飞书文档输出
5. **Agent 集成**：注册新工具，更新 `harness.patch.yml` 的 persona
6. **前端适配**：Web UI 增加"查看报告"按钮

### 3.3 测试策略

**单元测试**：

- `test_profiler.py`：喂入模拟数据，验证画像生成准确性
- `test_adaptive_reminder.py`：验证提醒时机根据画像调整

**集成测试**：

- 连续使用7天，检查周报是否正确生成
- 模拟"新用户"（无历史数据）→ 使用默认策略 → 积累数据 → 切换到自适应策略

**验收标准**：

- [ ] 用户能看到个人学习画像（最佳时长、高效时段）
- [ ] 提醒时间根据历史数据自动调整（不是固定40分钟）
- [ ] 每周自动生成学习报告并推送飞书
- [ ] 报告中至少包含1条可执行建议

---

## 四、资产判定（Naval §7 对照）

| 交付物 | 是否资产 | 理由 |
|--------|---------|------|
| 个人学习画像 | ✅ 是 | 可复用，换设备后导入继续生效 |
| 历史趋势报告 | ✅ 是 | 可导出留存，长期对比 |
| 自适应提醒策略 | ✅ 是 | 沉淀在代码逻辑中，自动为用户服务 |
| 单次学习时长记录 | ❌ 否 | 单独看无意义，必须聚合才有价值 |

**复利效应体现**：

- 第1周：收集数据，使用默认策略
- 第2周：生成初步画像，提醒时间开始调整
- 第4周：画像成熟，完成率提升10%
- 第8周：积累足够数据，可预测最佳学习时段
- 第24周：半年报告显示学习效率提升30%，用户看到复利曲线

---

## 五、风险与对策

### 5.1 数据冷启动

**问题**：新用户没有历史数据，无法生成画像

**对策**：

- 前3天使用"通用最佳实践"（25分钟番茄钟）
- 第4天起开始应用初步数据
- 第8天起切换到完全自适应

### 5.2 过拟合用户习惯

**问题**：用户某周因特殊情况（考试周、出差）数据异常，画像被污染

**对策**：

- 画像计算只用"最近30天"，滚动窗口
- 检测异常值（如某天学习12小时），自动标记为"非典型"
- 提供"重置画像"按钮，用户可手动清除异常数据

### 5.3 隐私敏感度

**问题**：学习数据涉及个人习惯，用户可能担心隐私

**对策**：

- 所有数据仅本地存储，不上传云端
- 报告生成时脱敏处理（不记录具体学习内容）
- 提供"删除所有学习数据"功能

---

## 六、成功指标

第一期上线后，3个月内达到：

1. **用户留存**：连续使用7天以上的用户占比 > 60%
2. **完成率提升**：自适应提醒上线后，会话完成率从 70% 提升到 85%
3. **报告阅读率**：周报推送后24小时内打开率 > 50%
4. **用户反馈**：至少1位用户主动说"提醒时间变合理了"

---

_下一步：启动第二期设计（深度洞察与社交杠杆）_
