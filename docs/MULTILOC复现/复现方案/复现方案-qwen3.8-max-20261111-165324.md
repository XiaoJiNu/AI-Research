# MULTILOC 论文复现方案（仅方案，不实现代码）

- 模型：qwen3.8-max（Hermes default profile）；撰写时间 2026-11-11 16:53 CST（本机 `date` 实测）
- 对象论文：`docs/论文/202610_MULTILOC.pdf`（arXiv:2603.27170v2，正文+附录 A/B/C 文本层全读）
- 方法纪律：套用 `docs/plans/2026-11-11_ScientistTwo实现方案.md`（基线复现硬门禁 → subset-first → 消融归因验收 → I1–I4 完整性审计 → 评测协议不可变）
- 本文档全部命令为**拟议、未执行**（见 §8）；标注为"未来命令"的一律待批准
- 同目录另有 qwen3.8-flash 版方案（14:29），本文独立核实全部环境事实后重写，结论一致处以本文为准

---

## 0. 结论先行（Verdict）

| 项 | 结论 |
|---|---|
| 能否原样复现 MultiLoc 主结果（0.29 m / 0.35°） | **不能**。论文承诺 "Code will be released"，2026-11-11 GitHub 检索无任何公开仓库/checkpoint。没有官方权重，论文自报数字一行都抄不来 |
| 推荐路线 | **三段式**：Stage A 复现评测管线 + ReLoc3r 基线（论文 Table 2 对照行，本机代码权重齐全）；Stage B 复现 Table 4 "Frozen VGGT" 行（纯推理，论文中唯一不需 MultiLoc 权重即可精确对照的自研数字：41.0 cm / 0.374°）；Stage C 按论文 §3.1–3.3 规格自研 MultiLoc-lite（冻结 VGGT-1B 主干，只训相机 token 注入 + camera head），MegaDepth 子集训练、Cambridge 零样本评测，复现**方向性结论**（位姿条件化多视角 > 无位姿冻结基线）而非绝对数值 |
| 数据集 | **主评测：Cambridge Landmarks**（通用重定位基准，ReLoc3r/ACE/GLACE/E5+1 全用它对比，reloc3r 仓库原生支持；**本机没有，需下载 ~10GB**）。**副评测：MegaDepth-1500**（相对位姿，本机已有 3.1GB，零下载）。**训练：MegaDepth 子集**（论文 8 个训练集之一，reloc3r 自带预处理脚本，按需下载）。放弃：Indoor6、WaySpots、ACID、ScanNet-1500（体积/许可/管线成本，本期不做） |
| 可执行性 | Stage A/B 可执行（依赖已实测齐全，见 §2）；Stage C 可行但 24GB 显存有 OOM 风险（11 帧×518px VGGT 前向），降级路径：448/392px 并在报告标注偏差 |
| 功效预期 | **不承诺**论文 0.29 m/0.35°、ACID AUC@5 76.56 等数字。承诺的是：① 评测协议本机跑通且可重跑验证；② ReLoc3r 行与 Frozen-VGGT 行可与论文数字直接对照（容差内）；③ MultiLoc-lite 若训出与论文 Table 4 同向的增益（frozen→trained 中位平移误差显著下降），即复现其核心论断 |
| 发布就绪度 | 未就绪。M-A 过 = 基线可信；M-B 过 = 论文评测协议已还原；M-C 是研究性延伸，训不出增益也是有价值的负结果（按 I1/I2 纪律如实报告） |
| 工作量 | Stage A ≈1 天；Stage B ≈1–2 天；Stage C ≈3–5 天（含冒烟与消融） |

---

## 1. 论文要复现什么（方法拆解，全部出自论文文本层）

**方法一句话**：k+1 张图（k=10 个带位姿参考 + 1 查询）一起进 VGGT 式交替注意力 transformer（L=24、1.1B、DINOv2 patch-14 + VGGT 初始化）。参考位姿 SE(3) 展平为 R¹²，经小 MLP 编码为相机 token y，与可学习 token l 相加得 g=l+y（查询图 yq=0）；**每层 frame attention 后重注入** gn=go+y；4 层自注意力 camera head 输出 [q, c, fx, fy]。位姿按参考相机中心均距 s=mean‖ci‖ 归一化，推理时乘回 s、复合第一个参考位姿得米制绝对位姿（§3.1–3.2）。检索用 NetVLAD top-10（§3.2）。训练：4×H200、1M iter、AdamW 1e-5→1e-8、最长边 518px、宽高比 0.33–1 随机、8 数据集（Co3D/DL3DV/VKITTI2/BlendedMVS/Re10K/MegaDepth/ARKitScenes/ScanNet++）、平移/旋转/焦距三项的 homoscedastic 不确定性加权损失（§3.3 式 1–2）。

**复现目标数字清单**（按可对照性排序）：

| 论文位置 | 内容 | 本机可行性 |
|---|---|---|
| Table 2 / Table 7 | Cambridge ReLoc3r 行（avg 0.55 m / 0.56°） | **可**——代码权重管线全在本机，只缺数据（Stage A） |
| Table 1 | MegaDepth-1500 ReLoc3r Pair 行（AUC@5/10/20 = 49.59/67.84/81.22） | **可**——数据已在本机，且本机曾复现成功（Stage A 第二锚点） |
| Table 4 frozen 行 | Cambridge 41.0 cm / 0.374°（冻结 VGGT + motion averaging，附录 B.1） | **可，纯推理**——这是论文自研方法中唯一不需 MultiLoc 权重就能精确对照的行，评测协议正确性的黄金锚点（Stage B） |
| Table 5 | 尺度恢复三法对照（Umeyama / motion avg / s） | **部分可**——前两法可跑；s 法依赖 MultiLoc 权重 |
| Figure 5 | k∈{1,2,4,10,20} 精度-延迟曲线 | 结构可复现（frozen 变体改 topk 重跑），数值依赖权重 |
| Table 2 MultiLoc 行 | 0.29 m / 0.35° | **否**——无权重；只能 Stage C lite 逼近方向 |
| Table 4 trained 行 | 29.5 cm / 0.35°、ACID AUC 76.56/87.15 | **否**——同上 |
| Table 6 / Table 3 | Indoor6 / WaySpots | 本期不做 |

**诚实折扣**（论文自身披露，§2/§5/附录 B.5）：GeLoc3r 无码；MultiLoc 增益依赖检索质量（非共视检索失败案例见附录 B.5）；4×H200×1M iter 全参微调决定了冻结主干的 lite 版绝对精度必然低于论文。

---

## 2. 本机可执行性审计（2026-11-11 16:53 实测，非推测）

| 依赖 | 实测结果 | 判定 |
|---|---|---|
| GPU | RTX 5090 Laptop，24463 MiB，当前空闲（used 64 MiB） | 可 |
| 磁盘 | /data 剩 268 GB（80% 已用） | 可（Cambridge ~10GB + MegaDepth 子集 ≤100GB + VGGT-1B ~5GB） |
| conda env mv2d | torch 2.7.1+cu128，CUDA 可用，认出 5090 | 可（用户指定环境） |
| ReLoc3r 代码 | `/data/code/reLocation-all/reloc3r`（fork），`scripts/eval_cambridge.sh`、`eval_megadepth1500.sh`、`datasets_preprocess/preprocess_megadepth.py` 均在 | 可 |
| ReLoc3r 权重 | `checkpoints/reloc3r-512.pth`（1.7GB）已在 | 可 |
| NetVLAD 检索权重 | `~/.cache/torch/hub/netvlad/VGG16-NetVLAD-Pitts30K.mat`（554MB）已在 | 可 |
| 检索+评测管线验证 | 本机在 zc_*/7scenes 上有既往成功评测产物，同管线换数据即可 | 可 |
| MegaDepth-1500 | `reloc3r/data/megadepth1500` 3.1GB 已在，且曾复现成功（`docs/复现记录/Reloc3r-MegaDepth1500推理复现_20260923_103737.md`） | 可，零下载 |
| Cambridge 数据 | **不在本机**。reloc3r README 给出官方 Google Drive 直链（file id `1XcJIVRMma4_IClJdRq6rwBKX3ZPet5az`），解压到 `data/cambridge`；gdown 已装 | **待下载**（§8 命令 1） |
| MultiLoc 代码/权重 | 无公开仓库（论文承诺录用后开源+checkpoint） | **NOT_AVAILABLE**——契约缺口成立：核心机制（相机 token 注入 + 每层 y 重注入 + s 尺度恢复）无任何已发布实现，按论文 §3.1+附录 A.1 自研最薄模块是唯一路径 |
| VGGT-1B 权重 | huggingface.co 直连不通；**hf-mirror.com 实测 308→200 可达**，`facebook/VGGT-1B` 存在（CC BY-NC-4.0） | 可（镜像下载，§8 命令 2） |

## 3. OSS 许可审计（先评估现成 OSS，再论证自研缺口）

| 项目 | 许可（实测来源） | 复用判定 |
|---|---|---|
| ReLoc3r | CC BY-NC-SA 4.0（本地 LICENSE） | **复用**——非商业研究用途合规；作对比基线正是论文自己的用法 |
| VGGT-1B 权重 | CC BY-NC-4.0（HF API） | **复用**——冻结主干做推理/训练 |
| vggt 代码仓库 | NOASSERTION | 研究用途按仓库条款使用；Stage C 若受阻再评估替代 |
| NetVLAD (Pitts30K) 权重 | hloc 分发，本机已缓存 | 复用 |
| Cambridge Landmarks | PoseNet 原始数据，学术公开 | 下载复用 |
| MegaDepth | CC BY-NC 4.0 | 子集下载复用 |
| **MultiLoc** | **无码可审** | **自研最薄模块的唯一依据**：论文附录 C 保证每个 baseline 都有公开实现，唯独它自己没有；新增模块规格论文已给全（§3.1、式 1–2、附录 A.1/B.1），不依赖未公开细节 |

---

## 4. 复现方案（三 Stage，套 ScientistTwo 纪律）

### Stage A — 评测管线 + ReLoc3r 基线复现（≈1 天）

对应论文：Table 2 ReLoc3r 行 + Table 1 MegaDepth-1500 ReLoc3r Pair 行。

1. 下载 Cambridge → `reloc3r/data/cambridge`，解压前先 `unzip -l` 看结构；解压后做完整性清单（6 场景目录、query/database 划分、pose 文件计数），**清单+哈希固化进 `eval/manifest.sha256`**（I2 协议不可变的起点）。
2. 跑 `scripts/eval_cambridge.sh`（NetVLAD top-10 → ReLoc3r 相对位姿 → motion averaging 恢复绝对位姿），逐场景输出固定 schema 的 `metrics.json`。
3. 判据：Cambridge 逐场景中位数误差与 Table 2 ReLoc3r 行同数量级、**avg 相对差 ≤20%**（论文未报 seed/检索缓存细节，不承诺 bit-exact）；MegaDepth-1500 AUC@5/10/20 与 Table 1 Pair 行（49.59/67.84/81.22）对照，本机曾复现成功，回归成本低。
4. **I1 纪律**：干净目录重跑一次，数字一致才算过。

产物：`evidence/stageA_reloc3r.json` + 与论文表格逐行并排的对照表（数字全部从台账生成，不由 LLM 转写）。

### Stage B — Table 4 "Frozen VGGT" 行复现（≈1–2 天，纯推理）

对应论文：Table 4 frozen 行（Cambridge 41.0 cm / 0.374°）+ Table 5 前两行（Umeyama / motion avg）+ Figure 5 的 k 曲线结构。

1. hf-mirror 下载 `facebook/VGGT-1B`（§8 命令 2）。
2. 按附录 B.1：VGGT aggregator + 自带 camera head 对 (10+1) 图前向 → 相对位姿 → **motion averaging** 解全局位姿与尺度；另跑 Umeyama 7-DoF 对齐对照。**数据加载、检索缓存、误差函数全部复用 Stage A 的，只换预测器**——保证两个预测器在同一协议下可比。
3. 判据：Cambridge avg ≈ 41.0 cm / 0.374°（±15%）；Umeyama vs motion avg 的次序与 Table 5 一致（motion avg 更接近）。这一行过了，说明"检索→多视角前向→尺度恢复→误差统计"整条论文评测协议已在本机还原，Stage C 的任何增益才可信。
4. 顺带产出 Figure 5 结构：k∈{1,2,4,10,20} 重跑 frozen 变体，画精度-延迟曲线（数值不承诺对上论文，论文该曲线是 trained 模型）。
5. 显存预案：11 帧×518px 单次前向若 OOM → 逐 query 处理 / 降分辨率，风险低于 Stage C。

产物：`evidence/stageB_frozen.json`、`evidence/stageB_scale_recovery.json`（Table 5 前两行）、`evidence/stageB_kcurve.json`。

### Stage C — MultiLoc-lite：核心机制归因验证（≈3–5 天，训练）

对应论文：贡献 1/2/3（位姿条件化多视角回归）+ Table 4 的 frozen→trained 方向性增益。

不自研框架，只写论文相对 VGGT 新增的最小三件（新代码目标 <800 行）：

1. `PoseTokenMLP`：SE(3)→R¹² 展平 → MLP → d 维 y；可学习 l（全图都有，附录 A.1 证明这对训练稳定性必要）；g=l+y，查询 yq=0。
2. 每层注入：frame attention 后 gn=go+y（改 VGGT global-attention 路径的残差入口）。
3. 训练环：式 (1)(2) 不确定性损失（平移/旋转/焦距三项 + 可学习 sc/sq/sf）+ AdamW lr 1e-5 cosine 截短到 1e-6 量级（1M iter 不现实，iter 数按冒烟结果定并写入台账），**只训 MLP、l、camera head**；主干冻结、bf16、gradient checkpointing。是否加最后 4 block LoRA：默认不加，见 §7 决策点 3。

- 训练数据：MegaDepth 子集（先 5 个场景）用 `reloc3r/datasets_preprocess/preprocess_megadepth.py` 现成脚本处理；采样/增强按论文 §3.3 "similar to VGGT"（518px、宽高比 0.33–1）。
- **subset-first**（ScientistTwo §3.2）：先 1 场景 × 2000 iter 冒烟——验证三项 loss 均下降、"注入开/关"前向输出存在差异（证明 y 真的进了计算图），再放大到 5 场景。
- 评测：Cambridge **零样本**（不训 Cambridge，保持论文范式），对照三个锚点：Stage B frozen（应显著优于）、Stage A ReLoc3r（应优于）、论文 MultiLoc 0.29/0.35（**预期达不到**，如实报差距与归因：冻结主干 + 截短训练 vs 全参 1M iter）。
- **消融归因验收**（ScientistTwo §3.4，硬门禁）：去掉 y 注入（置零）后精度必须回落到 frozen 水平——增益归因不到"位姿条件化"这一所提机制，则判 Refuse，不得声称复现成功（TeCh 判例）。
- 显存预案：11 帧 OOM → 448/392px，全部结论标注分辨率偏差。
  `# ponytail: 冻结主干版是 24GB 单卡的上限；升级路径是解冻更多 block / 多卡全参微调，不改结构。`

产物：`evidence/stageC_lite.json`、`evidence/stageC_ablation_no_pose.json`、`code/lite/`（三个模块 + train.py）。

### 全阶段共用硬门（直接沿用 ScientistTwo 方案 §3/§4.4）

- `eval/` 只读 + sha256 清单，每次运行前后校验，被改即整轮 invalid（防 reward hacking 的结构性手段，不靠自觉）；
- 指标由确定性脚本解析固定 schema 的 `metrics.json`，LLM 不碰数字；报告数值只从 `registry.jsonl` 台账取；
- secondary 指标强制：per-query 延迟（对照论文 Lat. 列）、focal 预测误差、显存峰值——主指标升而 secondary 恶化不得判 Good；
- 预算 fail-closed：每 Stage 墙钟/GPU 时上限写 `task.yaml`，超了硬停，断点续跑幂等。

### 任务目录布局（`/data/code/AI-Research/tasks/multiloc/`）

```
task.yaml  state.json  registry.jsonl
data/cambridge -> 软链到 reloc3r/data/cambridge
eval/            # 协议不可变区：检索缓存 + 划分清单 + 误差脚本 + manifest.sha256
evidence/stageA/ stageB/ stageC/
code/lite/       # Stage C 三个新模块 + train.py
report.md        # 数字全部从 registry.jsonl 生成
docs/decisions/  # 论文未明说、靠推断的实现细节逐条留痕（MLP 层数、d 维、注入点等）
```

---

## 5. 里程碑与验收判据（全是数字，可机器判定）

| 里程碑 | 判据 | 单独价值 |
|---|---|---|
| M-A | Cambridge ReLoc3r avg 与 Table 2 行相对差 ≤20%；MegaDepth-1500 AUC 回归通过；干净重跑一致（I1） | 可信评测管线 + 基线台账 |
| M-B | Frozen 行 41.0 cm/0.374° ±15%；Table 5 前两行次序一致；k 曲线结构完整 | **论文评测协议已还原的证明**——本方案最硬的可交付物 |
| M-C | lite 相对 frozen：Cambridge 中位平移误差降低 ≥30%（配对中位数检验），且"去 y 注入"消融回落到 frozen 水平；延迟报告完整 | 核心论断（位姿条件化多视角）方向性复现 |

任一里程碑未达 → 按 ScientistTwo I1/I2 纪律写负结果报告，不修饰数字。

## 6. 风险与边界

| 风险 | 缓解 |
|---|---|
| Cambridge Drive 直链失效/需人工确认页 | 备选：PoseNet 原站镜像、hloc 分发；再失败则请用户人工下载（gdown 对大文件偶发需要 cookie） |
| 11 帧×518px VGGT 前向 OOM（24GB） | Stage B 逐 query 推理；Stage C 降 448/392px + checkpointing，报告标注偏差 |
| lite 训不出增益 | 如实报负结果并归因（冻结主干容量/训练量不足 vs 论文全参 1M iter）——这本身是 Table 4 消融逻辑的延伸结论 |
| 论文无码，细节靠文字推断 | 所有推断（MLP 层数、d、注入实现点、loss 权重初值）写 `docs/decisions/`，**不当事实**；MultiLoc 一旦开源，立即补 Stage D：真权重纯推理复现全部表格行 |
| hf-mirror 镜像不稳定 | VGGT-1B 权重一次下载落盘后本地缓存，不重复依赖网络 |
| 非商业许可（CC BY-NC） | 全程研究用途，产物不分发权重 |

## 7. 请拍板的 4 件事

1. **数据集组合**：接受 "Cambridge（下载 ~10GB，主评测）+ MegaDepth-1500（已有，副评测）+ MegaDepth 子集（训练用，≤100GB）"？还是只要 Cambridge（砍掉 MegaDepth 副锚点，Stage A 验证强度降低）？
2. **范围**：只做到 M-B（≈2–3 天、纯推理、可精确对照论文一行）还是含 M-C 训练（+3–5 天、只承诺方向性结论）？**推荐做到 M-B 后再决定是否进 M-C**——M-B 通过前谈训练没有可信基线。
3. **Stage C 是否允许最后 4 个 block 加 LoRA**：更可能出增益，但离论文"全参微调"更远。推荐默认不加、跑通后作为消融变体加一轮。
4. **Indoor6 / WaySpots / ACID / ScanNet-1500**：本期确认不做？（体积与管线成本都高，且不影响核心结论的复现）

## 8. 拟议命令（**全部未执行**，批准后按序）

```bash
# 未来命令 1：下载 Cambridge（reloc3r README 官方直链）
cd /data/code/reLocation-all/reloc3r
/data/software/conda/anaconda3/envs/mv2d/bin/gdown 1XcJIVRMma4_IClJdRq6rwBKX3ZPet5az -O data/cambridge_download.zip
unzip -l data/cambridge_download.zip        # 先看结构
unzip data/cambridge_download.zip -d data/cambridge

# 未来命令 2：VGGT-1B（HF 直连不通，走 hf-mirror，实测可达）
HF_ENDPOINT=https://hf-mirror.com /data/software/conda/anaconda3/envs/mv2d/bin/python \
  -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/VGGT-1B', local_dir='/data/code/reLocation-all/reloc3r/data/vggt-1b')"

# 未来命令 3：Stage A（reloc3r 现成脚本）
bash scripts/eval_cambridge.sh
bash scripts/eval_megadepth1500.sh

# 未来命令 4：Stage B 显存冒烟（单场景单 query，实现后）
/data/software/conda/anaconda3/envs/mv2d/bin/python tasks/multiloc/code/prof_vggt_fwd.py --scene GreatCourt --k 10

# 未来命令 5：MegaDepth 子集预处理（Stage C，场景清单定了后填充参数）
/data/software/conda/anaconda3/envs/mv2d/bin/python datasets_preprocess/preprocess_megadepth.py --subsample ...

# 运行入口（Stage B/C 实现后）
python eval_cambridge_multiloc.py --predictor reloc3r|vggt_frozen|multiloc_lite --topk 10 --scene all
```
