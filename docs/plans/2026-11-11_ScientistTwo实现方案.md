# ScientistTwo 方法落地实现方案（仅方案，未实现代码）

日期：2026-11-11
输入论文：`/data/code/paperNotes/AI-Research/202609-ScientistTwo-Pioneering the Human Knowledge Frontier with Autonomous AI.pdf`
（arXiv 2609.19644v1，Google Cloud AI Research + Waterloo，71 页）
状态：**待评审**。评审通过后才写代码。

---

## 0. 结论先行（Verdict）

| 项 | 结论 |
|---|---|
| 路线 | **不自研多智能体框架**。写一个薄编排层（Python stdlib 状态机，约 8 个文件），重活全部外包给：现成 agent runtime（hermes CLI）、现成写作件（PaperOrchestra, Apache-2.0）、现成控制面（本机 `/data/code/Loop-Engineering`）。 |
| 可执行性 | **可执行，有 3 个硬缺口**：① 无 LaTeX 工具链 ② Semantic Scholar 无 key（429）③ 无 docker/podman 强隔离。三者都有替代路径，详见 §3。`claude` CLI 已装但未登录、`codex` CLI 账号不支持其模型 → 论文里 Claude Code 的位置由 **hermes CLI headless** 顶替，已实测可用。 |
| 功效预期 | **不承诺论文声称的 80.4% 成功率 / 7.5 分 / 25.2% 平均增益**。目标是：在一篇自选论文上跑通"复现 → 1 个 idea → 消融 → 报告 → I1–I4 审计"闭环，并留下可审计证据链。论文本身无源码，我们只能对齐其文字描述与 Appendix A.2 的超参，**无法逐位复现**。 |
| 发布就绪度 | **未就绪**。P0 冒烟 + Gate1 复现通过后才谈得上"可用"；Gate2 全闭环通过后才谈得上"能出报告"。 |
| 首选题目 | **Incremental BPE Tokenization**（ICML 2026 Spotlight，论文 Table 14 内，CPU 纯算法、不吃 GPU，且论文 §Table 9 自己就拿它做过三轮迭代，有对照参考）。本机单卡 RTX 5090 Laptop 24GB 是选题硬约束。 |
| 工作量 | P0：1–2 天。P1：2–3 天。P2：按需，不预做。 |

**未来命令一律标注为"待批准，不执行"**，见 §8。

---

## 1. 论文方法拆解（要复现的到底是什么）

### 1.1 全流程只有一个原语

论文 Listing 1（p.6）把 6 个阶段全部抽象成同一个循环：

```python
def stage(candidate, critic, refine, max_rounds):
    for _ in range(max_rounds):
        verdict, feedback = critic(candidate)
        if verdict == "accept":  return candidate
        elif verdict == "refine": candidate = refine(candidate, feedback)
        else:                    return None
    return None
```

形式化目标（式 1）：`(P+, C+) = A(G)`，`A = {A1..ANa}`，即给定科学问题 G，产出论文 P+ 与可复现代码库 C+。

**实现含义**：编排层不需要 6 套逻辑，只需要 1 个 `stage()` + 每个阶段各自的 (candidate 生成器, critic, refine) 三元组配置。这是本方案能把代码量压到千行以内的根本原因。

### 1.2 六阶段与 agent 清单（§3.1–§3.6, Table 1）

| 阶段 | 候选物 | critic 判据 | refine 动作 |
|---|---|---|---|
| §3.1 找局限 | 局限集合 | 是否足以指导新改进 | 补漏掉的局限（≤16 轮） |
| §3.1 种子想法 | 想法集 H0（按 novelty 分排序） | 是否新颖（检索 2 篇参考论文比对） | 再生成更高 novelty 的想法，凑够 N_seed |
| §3.2 子集评估 | 基线结果 E_base / idea 的 (code, results) | 是否优于复现基线 / 优于原 SOTA | Subset Engineer 调参改码（≤2 轮），判 Bad 则剪枝 |
| §3.3 想法进化 | 由历史执行轨迹 R_<k 进化出的新想法 | — | Idea Evolver（利用）+ 未评估种子想法（探索），≤4 轮，集齐 4 个 Good 提前停 |
| §3.4 消融 | N_p 个消融计划 + 结果 | 组件归因是否干净 | Ablation Critic 判 Refine → Full-Set Engineer 重做（≤1 次），且**只有 E_new 严格优于 E_best 才更新基线** |
| §3.5 写作 | LaTeX 手稿 | 评审分 s_review ≥ 8 | Rebuttal Planner 出补充实验 → Rebuttal Coder 跑 → Paper Enhancer 改稿（≤2 轮） |
| §3.6 元评审 | 手稿 + 评审意见 | Accept / Refine | Refine → 回到 idea 层重做消融与写作（≤N_meta 轮） |

选择最优想法：`ASelector` 在所有 `d_h = Good` 的全量结果里挑 h_best（式 4）。

### 1.3 CoE 完整性审计（§4.2 Table 7，源自 ScientistOne/CoE）

四项检查，这是论文里**最值得抄的部分**，因为它直接对准"AI 生成的论文不可信"：

- **I1 Score Verification**：重跑提交代码，比对论文报告的每个数字（论文案例里做到 bit-exact，p.47）。
- **I2 Specification Violation**：查是否反向工程评测器 / 硬编码答案（reward hacking），多数投票 LLM 判定。
- **I3 Reference Verification**：逐条 bib 查学术 API，抓幻觉引用。论文数据：无参考 agent 时 19/1840 幻觉，有则 0/1814。
- **I4 Method–Code Alignment**：方法章节 vs 代码并排判读。论文数据：三个 refine agent 全开时 49/49，缺一个降到 38/49。

论文 §B（p.31–32）的关键教训，**必须结构性防住，不能靠提示词禁止**：
- AutoSOTA 在 T-SAE 上"赢"的那次改动只改了给 LLM judge 看的展示方式，SAE 本身没动，复评 0.7557 < 基线 0.7586 → **评测协议必须不可变**。
- AutoSOTA 在 Pinet 上砍掉了控制约束违反的 solver 工作量，换来 −16.7% 延迟但没人量可行性 → **跨指标 trade-off 必须被审计**。
- 对比之下 ScientistTwo 用 I1 重跑 + I2 协议不可变审计 + I4 方法-代码对齐 **结构性阻断**。

### 1.4 成本基线（§4.3, Figure 10）

- 平均 **2.5 天/篇**，平均 **$3765/篇**（token + VM）。
- 时间集中在 Idea Refinement、Dynamic Peer-Review、Meta-Review（都是"跑实验"）。
- 配置（Appendix A.2）：除 4 个编码类 agent 用 Claude Code + Opus 4.8 外，其余全用 Gemini 3.6 Flash；novelty 靠 Google Search 取 2 篇参考；ICLR 2025 模板。
- 泛化性（Table 8）：把 Claude Code 换成 Antigravity(Gemini 3.8 Flash)，5 题成功率 80% → 60%，平均增益 3.8% → 16.7%（样本极小），评审分 7.0 → 6.3。**换 coding agent 可行但掉点**，这为我们用 hermes CLI 顶替提供了论文级依据，同时提示预期要打折。

---

## 2. 本机可执行性审计（全部实测，非推测）

| 论文需要 | 本机现状（实测） | 结论 |
|---|---|---|
| Coding agent（Claude Code + Opus 4.8） | `claude` v2.1.300 已装但 **Not logged in**；`codex` 0.153.4 已装但账号拒绝其模型（gpt-6-astra/gpt-5-codex/... 全部 `model_not_found`）；**`hermes` v0.21.5 headless 实测可用**：`hermes --in /tmp/hprobe -z "..." --yolo chat --cli` 成功写出 `hello.py` 并自行 `python3` 跑通 assert，输出 `ok` | **可执行**，用 hermes CLI 顶替。风险：论文 Table 8 显示换 coding agent 会掉点。 |
| 文本 agent LLM（Gemini 3.6 Flash） | token-plan gateway（`https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`，OpenAI 兼容，key 在 `BAILIAN_TOKEN_PLAN_API_KEY`）实测可用，22 个模型含 `qwen3.6-flash / qwen3.8-max / qwen3.8-flash / deepseek-v4-flash / deepseek-v4-pro / glm-5.3 / qwen4-max-preview`；**4 路并发实测：flash 1.3s、max 2.3s，无 429** | **可执行**，且比论文配置更省（本地网关无按 token 现金成本，只有配额） |
| Novelty 检索（Google Search 取 2 篇） | 无 Google API key；hermes 自带 `web_search`（exa/firecrawl/keenable 后端，实测本次调研全程可用） | **可执行**，用 hermes web_search 替代 Google Search |
| I3 引用核验（Semantic Scholar） | **无 key → HTTP 429**（连续 3 次 + 退避重试均 429）；替代实测：**OpenAlex 通**（`title.search` 命中 Incremental BPE Tokenization，返 DOI 10.48550/arxiv.2605.30813）、**CrossRef 通**（46437 条命中）、**arXiv export API 通**（1 条精确命中） | **可执行但需换 API**：I3 用 OpenAlex + CrossRef + arXiv 三源交叉；Semantic Scholar 标注 NOT_AVAILABLE（除非申请 key） |
| 写作（PaperOrchestra + ICLR 模板） | `google-research/paper-orchestra` 实测存在，**Apache-2.0**，188 stars，最后 push 2026-05-17，conda py3.11 + requirements.txt；另有社区 skill 版 `Ar9av/PaperOrchestra`（MIT，无 SDK 依赖，靠宿主 agent 推理）。**本机无 pdflatex / xelatex / lualatex / tectonic / latexmk** | **部分可执行**：能产 `.tex` 源码，**不能编译 PDF**。P0 输出 markdown；要 PDF 需装 LaTeX（§8 待批准） |
| 评审（ScholarPeer / Stanford Agentic Reviewer） | ScholarPeer 官方无码；社区实现 `amirkiarafiei/open-scholar-peer` **MIT**（7-agent 协议做成 skills + MCP）；Stanford Agentic Reviewer 仅浏览器访问、非开源 | **可执行（替代）**：评审用 gateway LLM judge + open-scholar-peer 协议；held-out 评审用**不同模型**（见 §7 风险 5） |
| 实验隔离（论文用 VM） | **无 docker / podman / firejail / nsjail**；有 `bwrap`（但 codex 自身报 "needs access to create user namespaces"） | **降级可执行**：用 git worktree + 独立 conda env + 只读 eval 目录 + 文件 hash 固定，**不是强隔离**。风险已记 §7。 |
| 算力 | 单卡 **RTX 5090 Laptop 24GB**，24 核，62GB RAM，`/data` 剩 269GB（已用 80%） | **选题硬约束**：只接 ≤24GB 显存、单卡可跑、无需大规模预训练的题（算法/时序/表格/纯 CPU 类）。论文的 107 题里大量 CV/LLM 题**本机不可做**。 |
| 控制面 | `/data/code/Loop-Engineering` 已有 loop-types 目录、`paper-research-writing-loop.md` / `paper-reproduction-loop.md` / `algorithm-training-loop.md` playbook、`experiment-registry-template.md` / `run-registry-template.md` / `loop-spec-template.md` 模板、`scripts/check_control_plane.py` | **直接复用**，不另建一套控制面（阶梯第 2 级：仓库里已有的先复用） |

**其余可复用 OSS（已核 license）**：`WecoAI/aideml`(AIDE, MIT，代码空间树搜索)、`codelion/openevolve`(Apache-2.0，进化式代码搜索)、`tsinghua-fib-lab/AutoSOTA`(MIT，705 stars，单指标优化，论文 §B 的对照系统)、`SamuelSchmidgall/AgentLaboratory`(MIT)。
**排除**：`SakanaAI/AI-Scientist-v2` license 为 **NOASSERTION**（非标准 SPDX），不引入。
**YAGNI**：P0/P1 一个都不装。只有当"想法搜索"确实成为瓶颈时才引入 AIDE/OpenEvolve（§6 P2 触发条件）。

---

## 3. 三个硬缺口的处理

1. **无 LaTeX** → P0/P1 产物是 `report.md` + `paper/paper.tex`（源码，不编译）。评审读 md 足够。要 PDF 时执行 §8 的安装命令。
2. **Semantic Scholar 429** → I3 改三源：OpenAlex（title.search）+ CrossRef（query.bibliographic）+ arXiv export（ti:"..."）。三源任一命中且题名/年份/作者近似匹配即 verified；全不命中 → flagged，人工复核。这比论文的单源更强（论文用 S2+arXiv+OpenAlex+CrossRef 四源，我们少一个）。
3. **无强沙箱** → 用"协议不可变"替代"进程隔离"来防作弊（这是论文 §B 的实际做法，它也不是靠 VM 防作弊，是靠 I1/I2/I4 审计）：
   - `eval/` 目录由编排层生成后 **chmod 只读 + 记录 sha256**；每次实验前后校验 hash，变了就整轮判 invalid（TOCTOU/ABA 拒绝）。
   - coding agent 的工作目录是 git worktree，实验后 `git diff --stat` 入台账，`eval/` 出现在 diff 里 = 直接 Bad。
   - `# ponytail:` 无强隔离；若将来跑不可信第三方 repo，升级为 bwrap/firejail 或远程 VM。

---

## 4. 实现方案

### 4.1 目录与状态（一篇论文 = 一个任务目录）

复用 Loop-Engineering 的任务目录约定，不新造：

```
/data/code/AI-Research/tasks/<slug>/
  task.yaml              # 问题定义 G：论文引用、官方 repo、主指标、higher_is_better、
                         # subset 划分规则、预算（max_rounds / 墙钟 / GPU 小时）
  state.json             # 状态机断点：当前阶段、已完成阶段、各阶段 verdict
  registry.jsonl         # 追加式运行台账（每行一次实验：idea_id/code_dir/metrics/git_sha/wall_s/verdict）
  evidence/              # 只追加：metrics.json、stdout/stderr、git_sha、环境快照（I1 依据）
  eval/                  # 协议不可变区：数据划分 + 评测脚本 + sha256 清单（只读）
  code/
    base/                # 复现的 SOTA 基线（E_base, C_base）
    sub_<idea_id>/       # 子集实验代码
    full_<idea_id>/      # 全量实验代码
    abl_<k>/             # 消融变体
  paper/
    report.md            # P0/P1 主产物
    paper.tex            # 源码，不编译
    audit/               # I1–I4 报告 + repro_check.json
```

### 4.2 组件清单（stdlib only，每个文件职责单一）

| 文件 | 职责 | 对应论文 | 预估行数 |
|---|---|---|---|
| `sci2/orchestrator.py` | 唯一入口。`stage()` 原语 + 6 阶段状态机 + 预算 + 断点续跑 + fail-closed | Listing 1, §3 | ~200 |
| `sci2/llm.py` | gateway 客户端（`urllib`，**不装 openai SDK**）：重试/退避、JSON 强约束解析、模型路由表、token 计量 | 各文本 agent | ~120 |
| `sci2/coder.py` | 包装 `hermes --in <dir> -z <prompt> --yolo chat --cli`：超时、日志落盘、退出码、产物校验（必须有 `run.sh` + `metrics.json`） | A_Coder（式 2） | ~150 |
| `sci2/critic.py` | **确定性**指标比较（解析 `metrics.json` 算 delta/显著性），LLM 只负责归因解释，**不让 LLM 算数** | Subset/Full-Set Critic, Result Comparison | ~120 |
| `sci2/audit.py` | I1 重跑比对（含 hash 校验）、I2 规格违规多数投票、I3 三源引用核验、I4 方法-代码对齐 | §4.2 CoE Audit | ~200 |
| `sci2/registry.py` | 追加式 JSONL 台账 + 只读封装（防改写历史） | 执行轨迹 R_k | ~60 |
| `sci2/report.py` | `report.md` + `paper.tex` 骨架（数字全部来自 registry，不允许自由生成数值） | §3.5 Initial Drafter | ~120 |
| `sci2/prompts/*.md` | 11 个提示词：limitation_extractor / limitation_verifier / idea_generator / novelty_checker / subset_critic / full_critic / idea_evolver / selector / ablation_planner / ablation_critic / reviewer / rebuttal_planner / meta_reviewer | 各 agent | 纯文本 |

**不做**（YAGNI，明确列出以免评审误判范围）：不自研 agent 框架、不接 ADK/LangGraph/AutoGen、不做 Web UI、不上数据库（JSONL 够）、不做多任务并行调度、不实现 ScholarPeer、不预装 LaTeX、不引入 AIDE/OpenEvolve（P2 触发条件见 §6）。

### 4.3 关键工程决定（每条带理由）

1. **指标判定用脚本，不用 LLM。** 论文的 critic 是 LLM agent，但 reward hacking 的根源正在"让模型判断自己有没有变好"。改为：`critic.py` 用确定性代码比 `metrics.json` 得 `Better/Worse/Within-Noise`，LLM 只在 `Engineer` 分支产出改进策略文本。→ 直接对齐 I2，且可测。
2. **subset-first 必须保留。** 这是论文的核心省钱机制（§3.2 先在子集验证再全量），本机单卡场景下更是唯一可行路径。
3. **协议不可变（结构性防作弊）。** 见 §3 缺口 3。这条是论文 §B 里 AutoSOTA 两次翻车的直接对策，**优先级高于任何功能**。
4. **数字只能来自 registry。** `report.py` 不允许 LLM 生成任何数值，所有表格单元格从 `registry.jsonl` 取。→ I1/I4 天然通过。
5. **预算 fail-closed。** 每阶段 `max_rounds` 到顶即停并写 `state.json`，绝不静默续跑；`task.yaml` 里的墙钟/GPU 小时预算超了同样硬停。对齐用户既有 release 规范（fail-closed、TOCTOU/ABA 拒绝）。
6. **断点续跑幂等。** 每阶段先查 `state.json` + `evidence/`，已完成的不重跑（长任务必备，2.5 天量级不可能一次跑完）。
7. **一次只跑一篇论文。** 不做并行调度器。
8. **模型路由**（对齐论文"编码用强模型、文本用快模型"的分工）：

   | 角色 | 模型 | 理由 |
   |---|---|---|
   | coding agent（hermes CLI 内） | hermes 默认 profile（qwen3.8-max）或 `--profile ds`（deepseek-v4-flash） | 论文用 Opus 4.8；本机最强可用即 qwen3.8-max |
   | 局限/想法/进化/选择/消融/评审/元评审 | `qwen3.6-flash` | 对齐论文 Gemini 3.6 Flash 的"快模型跑多数 agent" |
   | I2/I4 多数投票判定 | `deepseek-v4-pro` + `glm-5.3` + `qwen3.8-max` 三票 | 投票必须跨模型，否则同源自证 |
   | held-out 评审 | **与写作/评审不同的模型**（如 `glm-5.3`） | 避免论文自己承认的"ScholarPeer 既用于改稿又用于评分"的自证循环 |

### 4.4 数据契约（这是全系统唯一真正的接口）

`metrics.json`（每次实验必须产出，缺一即判 invalid）：

```json
{
  "idea_id": "sub_03",
  "metric_name": "qd_score",
  "metric_value": 26.2560,
  "higher_is_better": true,
  "split": "subset|full",
  "dataset": "prism_a",
  "seed": 0,
  "n_seeds": 3,
  "metric_std": 0.0047,
  "secondary_metrics": {"latency_s": 0.021, "feasibility_violations": 0},
  "wall_s": 812.4,
  "git_sha": "abc1234",
  "env": {"conda_env": "mv2d", "torch": "2.7.1+cu128", "gpu": "RTX 5090 Laptop"},
  "eval_sha256": "...",
  "reproduce_cmd": "bash run.sh"
}
```

`secondary_metrics` 强制存在 —— 直接对应论文 §B 批评 AutoSOTA 的"未度量的 trade-off"（Pinet 案例只报延迟不报可行性）。**Critic 规则：主指标提升但任一 secondary 恶化超阈值 → 不得判 Good，只能 Engineer 或 Bad。**

coder agent 的产出契约（写进提示词，`coder.py` 校验）：
- 必须有 `run.sh`（自包含、可从干净 checkout 重跑）
- 必须有 `metrics.json`（schema 同上）
- 必须打印每个报告数字的来源
- 不得修改 `eval/` 下任何文件

### 4.5 失败处理（三类，分别策略）

| 失败类型 | 判据 | 处理 |
|---|---|---|
| 超时 | 墙钟超 `task.yaml.budget.wall_s_per_run` | 杀进程，记 `verdict=Timeout`，同 idea 最多重试 1 次（降 subset 规模），再超时判 Bad |
| 崩溃 | 退出码 ≠ 0 或 stderr 有 traceback | 把 stderr 尾部喂回 coder agent 修（≤2 轮，即论文 N_eng=2），仍崩判 Bad |
| 无产物 | 缺 `run.sh` 或 `metrics.json` 或 schema 不符 | 同上，≤2 轮，仍缺判 Bad（**不猜、不填默认值**） |

---

## 5. 验证方案（怎么证明它真的能跑通）

三个 Gate，判据都是数字，逐 Gate 通过才进下一 Gate。

**Gate 0 — 冒烟（<10 分钟，不碰 GPU，唯一自动化测试）**
假任务：`eval/` 里放一个 `python -c "print(json.dumps({'metric_value': 1.0}))"` 的桩，跑通 6 阶段状态机全链路。
通过判据：① `state.json` 走完全部阶段 ② `registry.jsonl` ≥ 5 行且 schema 校验通过 ③ 故意让桩在第 2 轮变差 → critic 必须判 `Worse` 且不更新 best ④ 故意改 `eval/` 一个字节 → 必须整轮判 `invalid`（这条是防作弊回归测试，**不可省**）⑤ 杀掉进程重跑 → 不重复已完成阶段。
形式：一个 `tests/test_smoke.py`，assert 驱动，无框架。

**Gate 1 — 复现（单卡或 CPU，1–3 小时）**
选 Incremental BPE Tokenization：clone 官方/第三方 repo → 只跑基线复现 → I1 比对论文报告数字。
通过判据：主指标与论文报告值相对误差 ≤ 1%（或论文声明的 seed 方差范围内），`evidence/repro_check.json` 写 `"reproduction": true`，且 `run.sh` 在干净 worktree 里第二次跑仍一致。
**这一步单独有价值**：即使后面全不做，也得到一个可信的复现台账。

**Gate 2 — 全闭环（6–12 小时）**
在 Gate 1 的基线上跑完整流程：≥2 个种子想法 → 子集评估 → ≥1 个 Good → 全量 → ≥3 个消融 → report.md + paper.tex → I1–I4 全审计。
通过判据：① 至少 1 个 idea 判 Good 且**归因到所提机制**（消融去掉该组件后增益消失，即论文 TeCh 案例里"增益来自 EMA+label smoothing 而非所提机制"必须被拒）② I1 重跑数字全对 ③ I2 无违规 ④ I3 幻觉引用 = 0 ⑤ I4 判定 aligned ⑥ 全流程墙钟与 token 用量落台账。

**评审用**：Gate 2 的 `report.md` + `audit/` 就是交付物；不承诺评审分。

---

## 6. 分期与触发条件

| 期 | 内容 | 工作量 | 交付 |
|---|---|---|---|
| **P0** | `llm.py` `coder.py` `critic.py` `registry.py` `orchestrator.py`（前 4 阶段）+ Gate 0 + Gate 1 | 1–2 天 | 能自动复现一篇论文并留下 I1 证据 |
| **P1** | 想法进化循环 + selector + 消融规划/执行/critic + `audit.py`(I1–I4) + `report.py` + Gate 2 | 2–3 天 | 完整闭环 + 可审计报告 |
| **P2** | 按需，**不预做** | — | — |

P2 触发条件（只在满足时才做）：
- 要 PDF → 装 LaTeX + 接 `google-research/paper-orchestra`（Apache-2.0）。
- 想法搜索成为瓶颈（连续 2 篇论文 0 个 Good）→ 引入 `aideml`(MIT) 的树搜索或 `openevolve`(Apache-2.0) 替换 `idea_evolver`，**不自己写进化算法**。
- 要跑不可信第三方 repo → 上 bwrap/firejail/远程 VM 强隔离。
- 要跨模型评审对照 → 接 `open-scholar-peer`(MIT) 协议 + Stanford Agentic Reviewer 网页（人工）。

---

## 7. 风险与诚实边界

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| 1 | Reward hacking（改评测器/展示层而非模型） | 产出假增益，全盘不可信 | 协议不可变 + hash 校验 + Gate0 第④条回归测试 + I2 跨模型多数投票 |
| 2 | 无强沙箱 | agent 误删/误写宿主文件 | git worktree 限定工作目录；`--in` 指向任务目录；重要目录不入 worktree；台账记录 diff |
| 3 | 单卡 24GB | 论文 107 题里多数做不了 | 选题即约束：CPU/单卡小模型题优先（Incremental BPE、时序、表格、优化类） |
| 4 | gateway 限流/配额 | 长任务中断 | 4 并发实测无 429；`llm.py` 退避重试 + 断点续跑；单任务串行调用为主 |
| 5 | **自证循环**：同一模型既改稿又评分 | 评审分虚高（论文自己承认 ScholarPeer 是 in-distribution） | 写作/评审/元评审/审计四类角色**强制不同模型**；held-out 评审另用一个模型；最终结论以 I1 重跑数字为准，不以评审分为准 |
| 6 | 论文无源码 | 无法逐位复现，细节靠推断 | 只对齐 Listing 1 原语 + Appendix A.2 超参 + Table 1 阶段表；所有推断在 `docs/decisions/` 留痕；**不把推断当事实写进报告** |
| 7 | 成本/时间超预算 | 跑不完 | fail-closed 硬停 + 断点续跑 + subset-first；单篇预算写进 `task.yaml` |
| 8 | 换 coding agent 掉点 | 成功率低于论文 | 论文 Table 8 已量化（80%→60%）；预期直接按 6 折设定，不对外承诺论文数字 |

**不做的诚实声明**：本方案不声称能复现论文的 80.4% 成功率、7.5/10 评审分或 25.2% 平均增益。它声称的是：在一篇自选论文上跑通闭环，并让每个报告数字都能被重跑验证。

---

## 8. 待批准命令（**均未执行**，评审后按需）

```bash
# LaTeX（P2 要 PDF 时才装；本机 sudo 需密码，用 pkexec 走图形授权）
export DISPLAY=:1; pkexec apt-get install -y texlive-latex-extra texlive-fonts-recommended
#   或更轻： pipx install tectonic

# P2 可选依赖（仅在触发条件满足时）
/data/software/conda/anaconda3/envs/mv2d/bin/pip install aideml      # MIT，想法树搜索
/data/software/conda/anaconda3/envs/mv2d/bin/pip install openevolve   # Apache-2.0，进化搜索

# Gate 1 复现用（题目确定后）
git clone <官方或第三方 Incremental BPE repo> /data/code/AI-Research/tasks/incbpe/code/upstream

# 可选：Semantic Scholar key（提升 I3 精度；无 key 时用 OpenAlex+CrossRef+arXiv 三源）
#   https://www.semanticscholar.org/product/api#api-key  —— 需用户自行申请
```

运行入口（实现后）：

```bash
python3 -m sci2.orchestrator --task tasks/incbpe/task.yaml --stage all     # 全跑
python3 -m sci2.orchestrator --task tasks/incbpe/task.yaml --resume        # 断点续跑
python3 -m pytest tests/test_smoke.py -q                                   # Gate 0
python3 -m sci2.audit --task tasks/incbpe --checks I1,I2,I3,I4              # 审计
```

---

## 9. 评审需要拍板的 4 件事

1. **首选题目**：Incremental BPE Tokenization（CPU、不吃 GPU、论文自己做过对照）—— 是否认可？或指定另一篇（须满足单卡 24GB 约束）。
2. **范围**：先只做 P0（复现 + I1 证据，1–2 天），还是 P0+P1 一次做到全闭环（3–5 天）？
3. **coding agent**：用 hermes 默认 profile（qwen3.8-max）还是 `--profile ds`（deepseek-v4-flash）？前者更强、后者更快更省。
4. **是否要 PDF**：要则需批准装 LaTeX（§8 第一条）；不要则 P0/P1 只出 `report.md` + `paper.tex` 源码。
