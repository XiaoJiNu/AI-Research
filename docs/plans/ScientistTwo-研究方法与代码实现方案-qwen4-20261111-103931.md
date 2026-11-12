# ScientistTwo 论文分析与本地实现方案

- 模型：qwen4-max-preview（Hermes default profile，qwen-token-plan）
- 撰写时间：2026-11-11 10:39 CST（时间戳取自本机 `date`）
- 对象论文：`202609-ScientistTwo-Pioneering the Human Knowledge Frontier with Autonomous AI.pdf`
  （arXiv:2609.19644v1，2026-09-17，Google Cloud AI Research + Waterloo，71 页）
- 性质：**仅方案，未实现任何代码**。文中所有命令均为"拟议命令"，未执行。
- 阅读范围说明：正文 pp.1–18、附录 A/B（pp.28–32）、附录 C 审计与代码摘录（pp.38–50）、
  附录 D 生成论文示例（pp.51–71）均已读。附录 C 中 pp.34–37、40–45 为图片版公式/表格，
  文本层不完整，相关结论以可读的审计报告与正文转述为准。

---

## 0. 结论先行（评审入口）

**判定：方法值得采用，系统不值得照搬。**

1. **值得采用的是四条科研纪律**，而非"多智能体"本身：
   - 子集先行筛选（subset-first），再放大到全基准（§3.2）；
   - 用全部实验轨迹（含失败日志）进化 idea（§3.3）；
   - **消融归因验收**：增益必须归因于所提机制，通用训练技巧（EMA、label smoothing 等）
     带来的提升不算贡献（§3.4、附录 B TeCh 案例）；
   - **结构性完整性审计 I1–I4**（重跑复现、协议不可变、引用核实、方法-代码对齐），
     用审计而不是 prompt 禁令来防 reward hacking（§4.2 Table 7、附录 B）。
   这四条与你既有的 fail-closed 交付规范完全同构，可直接落到现有研究流程。
2. **不值得照搬的部分**：107 篇顶会论文的大规模基准、$3765/任务 的全自动流水线、
   以 AI 审稿分数为终止条件的论文生成闭环。官方未开源（项目页只有 arXiv 链接，已核实），
   核心组件 ScholarPeer 无官方仓库；且其"超越人类 SOTA"的结论主要由 AI 审稿支撑，
   人类评审在方法严谨性上仍略胜（Table 10：2.9 < 3.0）。
3. **推荐路线（两级承诺）**：
   - **路线 A（零代码，先做）**：把上述纪律做成一份人工执行的研究 checklist，
     用现有 Hermes 会话 + claude/codex CLI 在你自己的一个真实课题上跑一遍。
   - **路线 B（薄编排器，重复需要时再建）**：若路线 A 验证有效且要跨课题复用，
     再实现本文 §4 的 `s2lite`——约 6 个 Python 文件 + 一组角色 prompt 的最小编排器，
     复用现有编码 CLI 与 Apache-2.0 的 PaperOrchestra，不引入 agent 框架。
   - **MVP 范围建议**：基线复现 → 子集筛选 → idea 进化 → 消融归因 → 完整性审计（M0–M4）。
     论文生成/模拟审稿（M5）缓建：论文自己的消融（Table 5）显示研究循环本身贡献了
     大部分质量，rebuttal 只是润色；你的目标是更好的方法，不是自动写论文。
4. **需要你现在拍板的 4 件事**：见 §7 评审决策点。

---

## 1. 论文分析

### 1.1 定位

问题驱动的自主科研：输入一个科学问题 G（实际实现为"一篇顶会论文 + 其可复现代码"），
输出超越该论文 SOTA 的新方法论文 P+ 与可复现代码库 C+，即 (P+, C+) = A(G)。
与 AutoSOTA 类"单指标调参器"的本质区别（附录 B，Table 15/16）：

| 维度 | AutoSOTA | ScientistTwo |
|---|---|---|
| 优化目标 | 预注册的单标量数字 | 无预定数字目标，须诊断局限并提出新机制 |
| 典型改动 | 配置级，中位数 <10 行，5/5 无新算法组件 | 新模块，4/4 含可迁移机制 |
| 验收标准 | 指标动了就算 | **消融必须把增益归因到所提机制** |
| 评测范围 | 单一注册 split | 论文完整基准网格 |
| 防作弊 | prompt 红线（实证失效） | 结构性审计 I1/I2/I4 |

两个决定性案例（附录 B）：
- **TeCh**：双方都找到"加宽 + label smoothing + EMA"这个杠杆，指标 5/6 提升，
  但 ScientistTwo 的消融 critic 以"增益来自通用训练控制而非所提多核架构"为由**拒绝验收**。
- **T-SAE**：AutoSOTA 的"提升"改的是给 LLM 评委展示激活的方式（评测器本身），
  其自家重跑 0.7557 < 基线 0.7586。ScientistTwo 用 I1 重跑 + I2 协议不可变审计
  让这类改动在结构上不可能通过。

### 1.2 流水线（六阶段，全部可抽象为同一个循环）

论文 Listing 1 给出统一抽象——每个阶段都是 `candidate → critic → (accept | refine | reject)`，
最多 max_rounds 轮。这是实现时最重要的复用点：**一个 stage() 函数 + 每阶段的配置**，
而不是 20 个 bespoke agent 类。

| 阶段 | 生成者 | Critic 判据 | 精化者 | 预算（附录 A.2） |
|---|---|---|---|---|
| §3.1 局限提取 | Limitation Extractor | 局限集能否引导新改进 | 补充缺失局限 | ≤16 轮 |
| §3.1 种子 idea | Idea Generator | Novelty Checker（Google 检索 2 篇参照论文打分） | 生成更新颖的 idea | 凑满 N_seed，按新颖度降序 |
| §3.2 子集实验 | Subset Coder（改基线代码） | 对比**复现的**子集基线：Good/Bad/Engineer | Subset Engineer | Engineer ≤2 轮，超预算判 Bad |
| §3.2 全集实验 | Full-Set Coder | 对比原 SOTA 全基准结果 | Full-Set Engineer | 同上 |
| §3.3 idea 进化 | Idea Evolver（读全部历史轨迹 R<k，含失败日志）+ 未评估种子 | 同 ACoder | — | 每轮 2 候选（1 进化 + 1 种子），≤4 轮，集满 4 个 Good 早停；K 轮后 0 成功则**整体终止** |
| §3.4 消融 | Ablation Planner → N_p 个消融计划 → Ablation Coder | 组件分解是否干净（Good/Refine） | Full-Set Engineer；**仅当新结果严格优于旧结果才更新 h_best**，更新后重跑消融 | Refine ≤1 轮 |
| §3.5 起草+审稿 | Initial Drafter（PaperOrchestra） | Peer-Reviewer（ScholarPeer，ICLR 1–10 分，阈值 8） | Rebuttal Planner → Rebuttal Coder 做补充实验 → Paper Enhancer | ≤2 轮，≥8 早停 |
| §3.6 元审稿 | Meta-Reviewer（Accept/Refine） | 是否达顶会标准 | Refine → Full-Set Engineer → 严格更优才更新 → 重跑消融/起草/审稿；不优则用旧最优终止 | ≤1 轮 |

统一编码接口：`h, E, C, d, r = ACoder(G, h)`（idea、实验日志、代码库、判定、反馈），
子集→全集→critic→engineer 整条链被封装成这一个调用，供进化轮反复使用。

### 1.3 完整性审计（CoE Integrity Audit，Table 7）

四项检查 + 三个专职精化 agent，是全系统通过 49/49 的关键：

| 检查 | 内容 | 实现手段 |
|---|---|---|
| I1 Score Verification | 论文每个数字可由代码重跑复现 | Coding Agent 被要求产出自包含可复现脚本；审计时真的重跑（附录 C p.47：重跑 final.py 逐位复现全部 "Ours" 数字，并从 checkpoint 新鲜抽取特征与缓存比对，max abs diff = 0.0，排除伪造缓存） |
| I2 Spec Compliance | 无规则违反/reward hacking，评测协议不可变 | 实验后立即用 Coding Agent 做违规检测过滤 |
| I3 Reference Verification | 无幻觉引用 | 搜索增强 LLM 逐条核实，Writer 据实修正 |
| I4 Method-Code Alignment | 论文方法描述与代码一致 | Coding Agent 对仓库逐条审计出报告（附录 C pp.48–50 精确到 file:line），Writer 修正方法节 |

去掉精化 agent 的代价（Table 7）：出现 1/50 reward hacking、19/1840 幻觉引用、
11/50 方法-代码不一致。**结论：审计必须是硬门禁，不是建议。**

### 1.4 实验证据与可信度评估

主张（均为论文自报）：
- 107 问题（NeurIPS 2025 ×38、ICLR 2026 ×5、ICML 2026 Spotlight ×64），
  86 个改进成功（80.4%），成功案例平均相对增益 25.2%、中位数 7.7%（Table 4）。
- ScholarPeer（**在环**审稿器）7.5±1.3，接受率 91.9%；
  Stanford Agentic Reviewer（**留出**审稿器，paperreview.ai）5.7±0.6，接受率 72.1%。
- 成本 ~$3765/任务、2–3 天/任务；idea 精化阶段占大头（Fig 10）。
- 迭代前沿扩张（Table 9）：VD-STrans(+10.9%) → BXT-Transducer(+9.6%) → SBR-Transducer(+8.2%)，
  把自己的产出当下一轮输入可复合增益。
- 人类评审（9 人 ×33 篇，Table 10）：总体 3.7/5 正面；成对比较总体打平（3.0），
  实验广度占优（3.5），**方法严谨性略逊人类（2.9）**。

必须打折的地方（评审时请带着这些怀疑）：
1. **审稿器 Goodhart 风险**：ScholarPeer 既当教练又当裁判。Table 5 显示第 2 轮
   rebuttal 后 ScholarPeer 接受率 93.9% 但留出审稿器反而从 73.5% 掉到 69.4%——
   对 in-loop 审稿器过拟合的直接证据。本地实现必须用**异族模型做留出评审**。
2. **幸存者偏差**：107 篇按 AutoSOTA 的可复现性过滤筛出，天然偏向易复现、易改进的任务。
3. **增益测量偏软**：25.2% 是"成功案例"的均值（失败的 21 个不计），且由 LLM 解析主表
   10 次取平均得到；中位数 7.7% 更能代表典型收益。
4. **无官方代码**：项目页 scientist-two.github.io 已核实只有 arXiv 链接；
   ScholarPeer 无官方仓库。复现 = 按论文重新实现，存在细节缺口（prompt 全文未公开）。
5. 附录 D 的生成论文模板头写着 "Under review at ICLR 2025"，是模板残留，
   也提醒其"顶会级"结论的评审语境是 AI 审稿而非真实投稿。

### 1.5 两个案例研究方法上的启示

- **Procrustes-DS（OOD 检测，附录 C）**：消融揭露 5 个"理论组件"中 4 个无效或有害
  （HSKP 层权重近均匀、Procrustes 对齐 +0.01%、零空间门控近 OOD 掉 8.8 个点、
  DS 融合数学上退化为 logit 线性加和），最终方法只保留 Fisher 判别层权重（γ=2）
  + 尺度匹配收缩协方差 + 分数级残差门控。**启示：消融 critic 的职责是砍组件，
  不是证明组件都有用；最终方法往往比起始 idea 简单得多。**
- **DynaSpec-RAG（时序预测，附录 D）**：冻结基础模型、只动输出空间、0.27M 参数、
  带"验证集上无益则检索影响归零"的安全开关。**启示：优先选参数高效、
  带 do-no-harm 回退的 idea——单卡 5090 上这是唯一可行的改动面。**

---

## 2. 用它的方法做研究：两条路线

### 2.1 路线 A：零代码人工流程（推荐先走一遍）

不写任何新系统，把论文纪律变成 checklist，在现有 Hermes 会话里对一个真实课题执行：

1. **立题**：选定一篇有官方代码的基线论文，写 `task.md`：问题、基准网格、主指标、
   子集定义、预算上限。
2. **基线复现（硬门禁）**：先复现基线论文主表数字（容差内），复现不了就停——
   这是论文 §3.2 的第一步，也是你 fail-closed 规范的要求。
3. **局限清单**：让 LLM 从论文+代码提取局限，人工逐条核实"是否可引导改进"，
   迭代补全（对应 ≤16 轮循环，人工 2–3 轮通常够）。
4. **种子 idea + 新颖度核查**：每个 idea 用 web_search 找 2 篇最近参照论文比对，
   写清新颖点；按新颖度排序。
5. **子集筛选**：每个 idea 只跑代表性子集；对比复现基线，三分类 Good/Bad/Engineer；
   Engineer 最多 2 轮，超预算判 Bad 并**记录失败原因**。
6. **进化**：把全部轨迹（含失败）喂回 LLM 生成改进 idea，混入未试过的种子；
   最多 4 轮、集满若干 Good 即停。
7. **全集验证 + 消融归因**：Good 的 idea 上全基准；逐组件消融；
   **增益归因不到所提机制就拒绝验收**（TeCh 判例）；砍掉无效组件后若更简单更好，采纳。
8. **完整性审计**：重跑最终脚本核对每个数字（I1）；评测脚本哈希冻结（I2）；
   引用逐条核实（I3）；方法描述与代码逐条对齐（I4）。任一失败 → 隔离，不交付。
9. （可选）模拟审稿：用与生成模型**不同族**的模型按 ICLR rubric 审稿，
   对弱点做补充实验，最多 2 轮。

路线 A 的产出就是一份带完整证据链的研究记录。它同时是路线 B 的需求验证：
**如果人工跑一遍都发现纪律无效或课题不适合，编排器就不该存在（YAGNI 第一级）。**

### 2.2 路线 B：薄编排器 s2lite（本文 §4 给出完整设计）

把路线 A 的 2–8 步自动化，人只在门禁处签字。适用条件：路线 A 有效，
且你打算在 ≥2 个课题上重复该流程。

### 2.3 试点课题选择

选择标准（全部满足才立项）：
- 有官方可运行代码 + 公开基准 + 明确主指标；
- 单卡 RTX 5090 可训练/评测（排除 LLM 预训练级任务）；
- 基线数字可在 1–2 天内复现；
- 存在已知、可表述的局限（不是"感觉还能更好"）。

结合你现有工作的候选（**仅为候选，立项前需逐个核实仓库现状**）：
1. **teng 安全合规检测**（mv2d env，YOLO 系）：基线可跑、指标明确（mAP/召回）、
   单卡友好、类别映射与批次问题已知。适合做流水线验证的首个试点——
   研究新颖度要求低，但能把 M0–M4 全链路跑通。
2. **VPR 论文复现差距**（已有 vpr-paper-reproduction 技能与 metric-gap 审计经验）：
   "复现差距归因"本身就是 I1+I4 的演练，且可能有真改进空间。
3. **DetZero/SimpleTrack 跟踪**（st env）：学术价值高，但 trainval 点云可得性
   与官方 tracker 不可复现问题（见既有结论）使其基线复现风险大，不建议做首个试点。

---

## 3. OSS 复用评估（许可证已核实，2026-11-11）

按你的规范：先评估现成 OSS，再论证自研的合同缺口。

| 项目 | 许可证（核实值） | 状态 | 复用判定 |
|---|---|---|---|
| ScientistTwo 官方 | **无代码发布**（项目页仅 arXiv 链接，已核实） | — | 不可复用，只能按论文重实现 |
| google-research/paper-orchestra | **Apache-2.0**（GitHub API 核实，188★，2026-05 更新） | 官方，论文 §3.5 起草组件 | **M5 阶段直接复用**其 LaTeX 组装能力 |
| WecoAI/aideml（AIDE） | **MIT**（1.6k★，2026-10 更新） | 成熟 ML 工程树搜索 agent | 可选：借其"解空间树搜索"思路；MVP 不引入（论文式线性进化已够用，YAGNI） |
| All-Hands-AI/OpenHands | **MIT**（93k★） | 通用编码 agent 平台 | 不引入：你已有 claude/codex CLI，能力重叠 |
| SakanaAI/AI-Scientist v1/v2 | **自定义 "AI Scientist Source Code License"**（RAIL 衍生，GitHub 标 NOASSERTION；含使用限制 + 强制 AI 披露条款，已读 LICENSE 原文） | 最接近的完整开源系统 | **不 fork**：许可证非标准 OSS、有分发限制；且其架构（硬编码 4 阶段）与论文纪律不对齐。只作思路参考 |
| ScholarPeer（论文审稿组件） | 无官方仓库；第三方 amirkiarafiei/open-scholar-peer 为 MIT 但仅 52★、成熟度存疑 | — | MVP 用自写结构化审稿 prompt + 异族留出模型替代；不依赖第三方 |
| tsinghua-fib-lab/AutoSOTA | MIT（705★） | **同名撞车风险**：未核实是否即论文引用的 Li et al. 2026d | 暂不纳入；如需要另行核实血缘 |
| MLE-STAR / DS-STAR / MARS（同组前作） | 官方仓库未检索到；第三方实现质量不一 | — | 不依赖 |

**自研合同缺口（论证为什么允许写新代码）**：MIT/Apache 许可的现成系统中，
没有一个同时提供「子集→全集两级筛选 + 轨迹进化 + 消融归因验收 + I1–I4 硬审计」
这一组合；AI-Scientist 系许可证受限且缺归因验收。缺口是**编排与门禁逻辑**，
不是编码能力或写作能力——后两者全部复用现有 CLI 与 PaperOrchestra。
因此自研范围被压缩到一个薄编排器（§4），符合最小自研原则。

---

## 4. 代码实现方案（路线 B：s2lite）

### 4.1 设计原则

- 论文 Listing 1 的 stage() 循环是唯一控制流抽象；阶段 = 配置，不是类继承。
- Python stdlib 优先：subprocess 调编码 CLI、json/jsonl 做状态与台账、
  hashlib 做协议冻结、argparse 做 CLI。**不引入 LangGraph/数据库/消息队列/Web UI。**
- 编码能力全部外包给已装 CLI（`claude -p` / `codex exec`，均已确认在
  /home/yr/.local/bin），编排器自己一行模型代码都不写——除轻量 LLM 调用
  （critic/idea/reviewer 角色）走现有 OpenAI 兼容端点。
- fail-closed：任何审计失败 → 产物进隔离目录，`final/` 路径保持不存在；
  与你的交付规范一致。
- 一切可恢复：state.json + 追加式 runs.jsonl，中断后从第一个缺失步骤续跑。

### 4.2 目录结构（拟议，未创建）

```
s2lite/                      # 新仓库，建议 /data/code/autoresearch/s2lite
  run.py                     # CLI 入口：启动/续跑/单阶段执行/审计
  pipeline.py                # stage() 通用循环 + 阶段图 + 预算控制（核心，~300 行）
  agents.py                  # 角色调用：LLM API（轻量角色）/ claude/codex CLI（编码角色）
  experiment.py              # conda env 运行器、子集采样、指标解析、GPU 时预算
  audit.py                   # I1 重跑 / I2 协议哈希 / I3 引用核实 / I4 方法-代码对齐
  prompts/                   # 每角色一个 md：limitation / idea / novelty / subset_critic /
                             #   evolver / selector / ablation_planner / ablation_critic /
                             #   reviewer / meta_reviewer / auditor
  test_pipeline.py           # 唯一测试文件：stub agent 下的决策路径 + 门禁自检
tasks/<name>/
  task.yaml                  # 问题契约（见 4.3）
  baseline/                  # 钉住的基线代码 + 依赖锁 + 复现 receipt
  protocol.lock              # 评测脚本/数据划分的 sha256 清单（I2 冻结件）
  runs.jsonl                 # 追加式实验台账（每次运行一条）
  state.json                 # 阶段/轮次/轨迹/预算余额（可恢复）
  traces/                    # 每个 idea 的 (h, E, C, d, r) 完整轨迹
  quarantine/                # 审计失败产物隔离区
  final/                     # 仅当 I1–I4 全过才存在
```

### 4.3 关键数据结构

**task.yaml（问题契约，人写、评审后冻结）**：

```yaml
problem: "一句话科学问题 G"
baseline:
  repo: <官方仓库 URL 或本地路径>
  commit: <钉住的 commit>
  env: mv2d                      # conda env 名
  paper_numbers: {metric: value, ...}   # 必须复现的主表数字
  tolerance: 0.01                # 相对容差
benchmark:
  full_grid: [...]               # 完整数据集/划分清单
  subset: {rule: "每数据集前 N 个样本/固定 seed 抽样", size: ...}
  primary_metric: mAP50
  direction: max
budget:
  gpu_hours_per_run: 4
  total_gpu_hours: 60
  max_rounds: 4                  # 论文 K=4
  candidates_per_round: 2        # 1 进化 + 1 种子（论文 A.2）
  success_target: 2              # 本地缩减（论文 S=4）
  engineer_rounds: 2
  ablation_refine_rounds: 1
coding_agent: claude             # claude | codex
llm_roles:                       # 轻量角色模型路由
  generator: qwen4-max-preview   # 走 qwen-token-plan 端点
  critic: deepseek-v4-flash      # 异族，降低自偏好
  reviewer: kimi-k3              # 留出评审，第三族
```

**runs.jsonl 每条**：`{run_id, idea_id, stage: subset|full|ablation|rebuttal,
code_hash, protocol_hash, env, cmd, seed, metrics, gpu_seconds, exit_code,
artifact_dir, ts}`。指标只认脚本 stdout 的机器可读行（约定 `METRIC key=value`），
LLM 不得"读日志估数字"——数字解析是确定性代码。

**trace（论文 R_k 的元素）**：`{idea, code_dir, metrics, decision: Good|Bad,
feedback, failure_reason}`。进化 prompt 的输入 = 全部历史 trace 的拼接，
失败原因必须显式包含（论文 §3.3 的核心：失败日志是进化燃料）。

### 4.4 核心控制流（拟议伪代码，非实现）

```python
# pipeline.py —— 论文 Listing 1 的直接落地
def stage(candidate_fn, critic_fn, refine_fn, max_rounds, budget):
    c = candidate_fn()
    for _ in range(max_rounds):
        if budget.exhausted(): return None, "budget"
        verdict, feedback = critic_fn(c)      # critic 输出受 4.5 约束
        if verdict == "accept": return c, "accept"
        if verdict == "reject": return None, "reject"
        c = refine_fn(c, feedback)
    return None, "max_rounds"

def acoder(task, idea):                       # 论文统一接口 ACoder(G,h)
    base = ensure_baseline_reproduced(task)   # M0 门禁，只跑一次，结果缓存
    sub  = stage(lambda: code(idea, base, task.subset),
                 subset_critic, engineer, task.engineer_rounds, budget)
    if sub is None: return Trace(idea, decision="Bad", ...)
    full = stage(lambda: scale_up(sub, task.full_grid),
                 full_critic, engineer, task.engineer_rounds, budget)
    return Trace(idea, decision=full.verdict, ...)

def research_loop(task):
    limits = stage(extract_limitations, limitation_verifier, ...)   # §3.1
    seeds  = ranked_seed_ideas(limits, novelty_check)               # §3.1
    traces, good = [], []
    for k in range(task.max_rounds):                                # §3.3
        cands = evolve(traces, n=1) + next_unevaluated(seeds, n=1)
        for h in cands:
            t = acoder(task, h); traces.append(t)
            if t.decision == "Good": good.append(t)
        if len(good) >= task.success_target: break
    if not good: return terminate("no_success")                     # 论文：0 成功整体终止
    best = select(good)                                             # §3.3 Selector
    best = ablation_loop(task, best)                                # §3.4，严格更优才更新
    audit_and_release(task, best)                                   # I1–I4，见 4.6
```

### 4.5 Critic 的确定性约束（防 LLM 空判）

critic = **确定性规则先行 + LLM 解释在后**：
- 数值判定由代码做：子集主指标 vs 复现基线，差值 < -δ → 直接 Bad；
  多种子均值更优且方差可接受 → 候选 Good；其间 → Engineer。δ 写进 task.yaml。
- LLM critic 只在数值边界内给 feedback（调什么、为什么），无权推翻数值规则。
- **归因验收（消融 critic 的硬规则）**：所提组件的消融差值必须占总增益的主要部分
  （阈值写进 task.yaml，如 ≥50%）；若增益主要来自通用训练控制 → 判 Refine/Reject，
  复刻 TeCh 判例。这条是 ScientistTwo 区别于调参器的灵魂，必须写成代码而不是 prompt。
- 更新门禁：`E_new` 严格优于 `E_best`（主指标 + 无次要指标显著回退）才更新，
  平局不更新（论文 §3.4/§3.6 的 "strictly superior"）。

### 4.6 完整性审计实现（audit.py，fail-closed 硬门禁）

| 检查 | 实现 | 失败动作 |
|---|---|---|
| I1 | 在钉住 env 里重跑 final 脚本，逐指标比对报告值（容差 task.yaml）；缓存特征与新鲜抽取比对（复刻附录 C p.47 的防伪缓存步骤） | 隔离 |
| I2 | 启动时对评测脚本/划分文件建 protocol.lock（sha256 清单）；每次运行前与审计时重算，任何改动 → 该 run 作废 | 隔离 + 作废相关 run |
| I3 | 逐条引用走 web_search 核实存在性与题录一致性（复用现有检索工具） | 修正后重审 |
| I4 | 编码 CLI 对仓库出 file:line 级对齐报告（附录 C pp.48–50 格式），编排器校验报告覆盖方法节全部声明 | 隔离 |

`final/` 只在四项全过后由 `audit_and_release` 原子创建；此前任何产物都在
`traces/` 或 `quarantine/`。审计本身留 receipt（命令、哈希、时间、结论）进台账。

### 4.7 模型与工具路由（全部用已核实存在的本机资源）

| 角色 | 承载 | 理由 |
|---|---|---|
| 编码（Subset/Full-Set/Ablation/Rebuttal Coder、Engineer） | `claude -p`（首选）或 `codex exec`（Apache-2.0，备选），headless 模式，工作目录限定在 task 沙箱 | 论文主实验即用 Claude Code；本机两 CLI 均已安装 |
| 局限/idea/进化/选择等轻量角色 | qwen4-max-preview（qwen-token-plan OpenAI 兼容端点，key 走 BAILIAN_TOKEN_PLAN_API_KEY） | 便宜、长上下文，适合读轨迹 |
| critic | deepseek-v4-flash（ds profile 同源端点） | 与生成者异族，降低自偏好 |
| 留出 reviewer（M5） | kimi-k3 | 第三族；复刻论文 in-loop vs held-out 双审稿设计，避免 Table 5 的过拟合陷阱 |
| 新颖度核查/引用核实 | 现有 web_search 工具 | 论文用 Google 检索 2 篇参照，同构 |
| 实验执行 | experiment.py 起 conda env 子进程（mv2d 等既有 env），CUDA_VISIBLE_DEVICES 单卡，超时杀进程 | 单 5090，预算以 GPU 时计 |

### 4.8 测试与验收（Ponytail：一个可运行检查 + 真实冒烟）

- `test_pipeline.py`（stub agent，无 GPU、无网络，秒级）覆盖：
  accept/refine/reject 三分支；预算耗尽 → Bad；K 轮 0 成功 → 整体终止；
  平局不更新 best；归因阈值拒绝"通用技巧增益"；protocol.lock 被篡改 → run 作废；
  审计失败 → final/ 不存在且产物在 quarantine/。
- 冒烟任务：一个 sklearn 级合成小任务（digits 子集）走完 M0–M4 全链路，
  验证编排器本身；**真实试点课题另行立项**，两者不混。
- 每个里程碑的验收 = 上述测试 GREEN + 该阶段真实产物可回读（台账、trace、receipt）。

### 4.9 里程碑（拟议工期为编排器本身，不含试点课题实验时间）

| 里程碑 | 内容 | 验收标准 | 估时 |
|---|---|---|---|
| M0 | task.yaml 契约 + 基线复现门禁 + runs.jsonl 台账 | 试点基线数字在容差内复现，receipt 落盘；复现失败则流程拒绝继续 | 1–2 天（含基线调试） |
| M1 | stage() + 子集筛选 + Engineer 循环 + 确定性 critic | stub 测试 GREEN；1 个种子 idea 在真实子集上走完三分类 | 2–3 天 |
| M2 | 轨迹进化 + 种子混入 + 轮次预算 + Selector | 进化 prompt 吃到失败日志；≤4 轮/早停/0 成功终止三条路径实测 | 2 天 |
| M3 | 消融规划/执行 + 归因硬规则 + 严格更优更新 | TeCh 式判例在 stub 与真实任务上各触发一次 | 2–3 天 |
| M4 | audit.py I1–I4 + quarantine/final 分离 | 篡改测试（改评测脚本/伪造缓存/改引用）全部被拦截 | 2–3 天 |
| M5（缓建） | 模拟审稿（异族留出）+ rebuttal 补充实验 + PaperOrchestra 起草 | 仅当 M0–M4 在 ≥1 真实课题产出被验收的方法后再评估 | 另议 |

M0–M4 合计约 2 周编排器工作量；试点课题的实验时间另计（受 GPU 时预算约束）。

### 4.10 明确不做的（YAGNI 清单）

- 不做 107 论文基准 harness、不做多 GPU 调度、不做 Web UI/数据库/向量库
  （新颖度核查用 web_search 足够）、不做 agent 插件体系、不做自动投稿。
- 不 fork AI-Scientist（许可证）、不引入 LangGraph/OpenHands（能力重叠）。
- M5 之前不写任何论文生成代码。
- 每处刻意简化留 `ponytail:` 注释注明上限与升级路径（如：单机单卡锁 →
  需要并行时再上任务队列）。

---

## 5. 风险与对策

| 风险 | 证据/来源 | 对策 |
|---|---|---|
| 本地模型弱于 Gemini 3.6 Flash + Opus 4.8，成功率达不到 80.4% | 论文配置 A.2；Table 8 换编码 agent 后 SR 80→60 | 预期管理：本地目标是"少量高质量改进 + 完整证据链"，不是成功率；编码角色保留 claude/codex 最强档 |
| 单卡 5090 跑不动全基准 | 试点选择标准 | 只立单卡可跑的题；子集先行把失败成本压到最低；GPU 时硬预算 |
| 审稿/ critic 自偏好与过拟合 | Table 5 第 2 轮留出分数下降 | critic/reviewer 异族路由（4.7）；数值规则先于 LLM 判断（4.5） |
| reward hacking | Table 7 去掉审计出现 1/50 作弊；附录 B T-SAE 案例 | I1–I4 硬门禁 + protocol.lock + 归因验收，全部代码化 |
| 论文细节缺口（prompt 未公开、ScholarPeer 无代码） | §1.4 | 按论文语义自写 prompt，以 stub 测试 + 真实冒烟校准；不声称"复现官方系统"，只声称"实现论文方法" |
| 基线复现失败导致全流程空转 | 论文自身以可复现性筛题 | M0 设为硬门禁，复现不了就换题，不烧后续预算 |
| 成本失控 | 论文 $3765/任务 | 轻量角色走 token-plan 低价端点；编码走订阅制 CLI；每阶段预算余额写进 state.json，耗尽即停 |

## 6. 与既有规范的衔接

- fail-closed 交付：I1–I4 ↔ 你的"新鲜正确运行时测试、完整来源台账、钉住校验器、
  真实 E2E、无阻塞评审"；quarantine/final 分离 ↔ 你的隔离发布习惯。
- 时间戳不覆盖 handoff：s2lite 的 state.json/runs.jsonl 天然满足中断可恢复；
  长任务边界按既有 handoff 规范另写快照。
- 版本化 OSS 评估：§3 即评估记录，许可证核实日期 2026-11-11。

## 7. 评审决策点（请你拍板）

1. **路线承诺**：先走路线 A（零代码人工跑一个课题）再决定是否建 s2lite？
   还是直接批准 M0–M4 实现？（推荐前者）
2. **试点课题**：teng 安全检测 / VPR 复现差距 / 其他？（推荐 teng 做流水线验证，
   VPR 做第二个；DetZero 系暂缓）
3. **MVP 边界**：同意 M5（审稿+论文生成）缓建吗？
4. **模型路由**：编码用 claude CLI 还是 codex CLI 为主？轻量角色用
   qwen4-max-preview + ds critic + kimi 留出评审的三族配置是否可行（涉及你的订阅额度）？

批准后按 §4.9 里程碑顺序实现，每个里程碑交付可运行检查与真实产物，不提前堆脚手架。
