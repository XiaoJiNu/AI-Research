# MULTILOC 复现方案（仅方案，未实现代码）

- 模型：qwen3.8-flash（Hermes default profile）；撰写时间 2026-11-11 14:29 CST（`date` 实测）
- 对象论文：`docs/论文/202610_MULTILOC.pdf`（arXiv:2603.27170v2，18 页）
- 方法框架：`docs/plans/2026-11-11_ScientistTwo实现方案.md` 的纪律（基线复现 → subset-first → 消融归因 → I1 重跑审计 → 协议不可变），本文档全程套用
- 阅读覆盖：正文 + 附录 A–C 文本层全部可读，无图片独占页缺口
- 本文档所有命令均为**拟议，未执行**（见 §7）

---

## 0. 结论先行（Verdict）

| 项 | 结论 |
|---|---|
| 能否"原样"复现 MultiLoc 主结果 | **不能**。2026-11-11 实测 GitHub API 检索：MultiLoc 无公开代码仓库（论文写 "Code will be released"，尚未发布），无 checkpoint。没有权重就没有 0.29 m/0.35° 这一行可抄 |
| 推荐路线 | **三段式**：Stage A 复现评测管线+ReLoc3r 基线（论文 Table 2 的对照行，本机已有代码权重，最硬的一步）；Stage B 复现论文 Table 4 的 "Frozen VGGT" 行（纯推理，可精确对照 41.0 cm/0.374°）；Stage C 自研 MultiLoc-lite（冻结 VGGT-1B 主干，只训练论文 §3.1 新增的相机 token 注入 + camera head），在 MegaDepth 子集上训练，Cambridge 零样本评测，**复现"方向性结论"而非绝对数值** |
| 数据集选择 | **Cambridge Landmarks**（重定位主战场）：论文/ReLoc3r/ACE/GLACE 通用基准，reloc3r 仓库原生支持（`cambridge_retrieval.py` + `eval_cambridge.sh`），需下载（~10GB 级单 zip）；**MegaDepth-1500**（相对位姿）本机已有 3.1GB，零下载；**辅助训练集 MegaDepth 子集**（论文 8 个训练集之一，reloc3r 仓库自带预处理脚本）。放弃：WaySpots（Waymo 许可+体积）、Indoor6（体积大、无现成管线）、ACID/ScanNet-1500（非重定位主基准） |
| 可执行性 | Stage A/B 可执行（实测依赖全齐）；Stage C 可行但有 24GB 显存风险（11 帧×518px 的 VGGT 前向），降级路径：输入降到 448/392px，报告标注偏差 |
| 功效预期 | **不承诺论文的 0.29 m/0.35°、ACID AUC@5 76.56**。声称的是：评测协议本机跑通、ReLoc3r 行和 Frozen-VGGT 行可数字对照、MultiLoc-lite 若训出"位姿条件化多视角 > 冻结无位姿基线"的同向增益即为核心论断（贡献 2/3）的复现 |
| 发布就绪度 | 未就绪。Stage A 通过才算基线可信；Stage B 通过才算论文评测协议已还原；C 是研究性延伸 |

---

## 1. 论文要复现的到底是什么（带表格出处）

**方法一句话**：k+1 张图（k=10 个带位姿参考 + 1 查询）一起进 VGGT 式交替注意力 transformer（L=24, 1.1B, DINOv2 patch + VGGT 初始化），参考位姿经 MLP(SE(3)→R12→d) 编成相机 token `y`，与可学习 token `l` 相加、**每层注入** `gn = go + y`；camera head（4 层自注意力+线性）输出 [q, c, fx, fy]；平移除以场景尺度 `s = mean‖ci‖` 归一化，推理时乘回 s 再复合第一个参考位姿得米制绝对位姿（§3.1–3.2）。检索用 NetVLAD top-10（§3.2）。训练：VGGT 初始化，4×H200，1M iter，AdamW 1e-5→1e-8，最长边 518px、宽高比 0.33–1 随机，8 个数据集（Co3D/DL3DV/VKITTI2/BlendedMVS/Re10K/MegaDepth/ARKitScenes/ScanNet++），损失为平移/旋转/焦距的 homoscedastic 不确定性加权和（§3.3 式 1–2）。

**可复现的目标数字**（全部来自论文文本层）：

| 论文位置 | 内容 | 本机复现可行性 |
|---|---|---|
| Table 2 | Cambridge 各场景 ReLoc3r 行（0.55 m/0.56° avg）| **可**（权重/代码/管线全在本机，缺数据） |
| Table 2 | Cambridge MultiLoc 行（0.29/0.35）| **否**——无权重；只能 lite 逼近 |
| Table 1 | MegaDepth-1500 ReLoc3r Pair/MV 行（49.59/67.84/81.22 等）| **可**（数据已在本机 `data/megadepth1500`） |
| Table 4 "Frozen (no pose)" 行 | Cambridge 41.0 cm/0.374°（VGGT 冻结 + motion averaging，附录 B.1）| **可，纯推理** |
| Table 5 | 尺度恢复三法对照（Umeyama / motion avg / s）| **部分可**：前两种可，s 法依赖 MultiLoc 权重 |
| Figure 5 | k∈{1,2,4,10,20} 精度-延迟曲线 | 结构可（改 topk 重跑），数值依赖权重 |
| Table 6 / 3 | Indoor6 / WaySpots | 数据集未下载，本期**不做** |

**诚实折扣**（§2 论文自身披露）：GeLoc3r 行无码；MultiLoc 增益依赖检索质量（附录 B.5 失败案例）；训练规模 4×H200×1M iter 决定了 lite 版绝对精度必低于论文。

---

## 2. 本机可执行性审计（2026-11-11 实测）

| 依赖 | 实测结果 | 判定 |
|---|---|---|
| GPU | RTX 5090 Laptop 24463 MiB（空闲 64 MiB used） | 可 |
| 磁盘 | /data 剩 269 GB | 可（Cambridge ~10GB + MegaDepth 子集 ≤100GB + VGGT ~5GB 均容纳） |
| conda env mv2d | torch 2.7.1+cu128 认到 5090；transformers 4.57.1、gdown 6.2、opencv、imagesize 已装 | 可（用户指定环境） |
| ReLoc3r 代码 | `/data/code/reLocation-all/reloc3r`（fork: XiaoJiNu/reloc3r），`scripts/eval_cambridge.sh` 现成 | 可 |
| ReLoc3r-512 权重 | `checkpoints/reloc3r-512.pth` 1.7GB 已在 | 可 |
| NetVLAD 检索权重 | `~/.cache/torch/hub/netvlad/VGG16-NetVLAD-Pitts30K.mat` 已在 | 可 |
| 检索管线本机验证 | `_db-q_pair_info/` 与 `data/zc_*`、`data/7scenes` 均有既往成功产物（2026-09/10 的评测记录在 `docs/zc数据集上测试`） | 可（同管线换数据即可） |
| MegaDepth-1500 | `data/megadepth1500` 3.1GB 已在，且本机已复现过一次（`docs/复现记录/Reloc3r-MegaDepth1500推理复现_20260923_103737.md`） | 可，零下载 |
| Cambridge 数据 | **不在本机**；reloc3r README 给出 Google Drive 直链（file id `1XcJIVRMma4_IClJdRq6rwBKX3ZPet5az`）；gdown 已装。下载本身未执行（plan-only） | 待下载 |
| MultiLoc 代码/权重 | GitHub API 搜索（2026-11-11，`q=MultiLoc+pose+regression` total_count=0，189 个 "multiloc" 结果均为无关同名项目）；论文承诺录用后开源 | **NOT_AVAILABLE**，替代=按 §1 描述自研 lite（论文正文+附录 A.1 已给全部新增模块规格） |
| VGGT-1B | huggingface.co 直连超时；**hf-mirror.com 实测可用**，`facebook/VGGT-1B` 存在（safetensors + model.pt，下载量 57 万） | 可（镜像下载） |
| Semantic Scholar/检索类 API | 本任务不需要 | — |

---

## 3. OSS 许可审计（2026-11-11 实测）

| 项目 | 许可（实测来源） | 官方/第三方 | 复用判定 |
|---|---|---|---|
| ReLoc3r | CC BY-NC-SA 4.0（本地 LICENSE 首行） | 官方（CVPR 2025） | **复用**——非商业研究用途符合条款；作对比基线正是论文自己的用法 |
| VGGT-1B 权重 | CC BY-NC-4.0（HF API `facebook/VGGT-1B`） | 官方 | **复用**（冻结主干做推理/训练，非商业） |
| vggt 代码仓库 | GitHub API `license.spdx_id=NOASSERTION` | 官方 | 代码按仓库条款用于研究；若 Stage C 受阻再评估替代实现 |
| NetVLAD 权重 (Pitts30K) | hloc 分发（cvg-data.ethz.ch），本机已缓存 | 官方系 | 复用 |
| Cambridge Landmarks | PoseNet 原始数据，学术公开分发 | 官方 | 下载复用 |
| MegaDepth | CC BY-NC 4.0（数据集条款） | 官方 | 子集下载复用 |
| MultiLoc | **无码可审** | — | 契约缺口成立：位姿条件化多视角回归的核心（camera token 注入 + 每层 y + s 尺度恢复）不存在任何"已发布可许可"的实现，自研最薄模块是唯一路径（论文附录 C 保证每个 baseline 都有公开实现，唯独它自己的没有） |

## 4. 复现方案（三 Stage，套用 ScientistTwo 纪律）

### Stage A — 评测管线 + 对照基线复现（≈1 天，CPU 检索 + GPU 推理）

1. 下载 Cambridge 到 `reloc3r/data/cambridge`（§7 命令 1），解压后先做完整性清单（场景目录、query/database 划分、pose 文件），**哈希固化**。
2. 跑 `scripts/eval_cambridge.sh`（NetVLAD top-10 → ReLoc3r 相对位姿 → 尺度恢复），逐场景写 `evidence/metrics.json`。
3. 对照论文 Table 2 ReLoc3r 行（期望 avg≈0.55 m/0.56°）与 ReLoc3r 原论文。**判据：逐场景中位数误差同数量级、avg 相对差 ≤20%**（论文未报 seed，不承诺 bit-exact）。同时用本机已有的 MegaDepth-1500 重跑一遍 Table 1 ReLoc3r Pair 行作第二锚点（本机曾复现成功，回归成本低）。
4. I1 纪律：`run.sh` 在干净目录重跑一次，数字一致才算过。

产物：`evidence/stageA_reloc3r.json` + 与论文表格并排对照表。

### Stage B — 论文 Table 4 "Frozen VGGT" 行复现（≈1–2 天，纯推理）

1. 从 hf-mirror 下载 `facebook/VGGT-1B`（§7 命令 2）。
2. 按论文附录 B.1：VGGT aggregator + 自带 camera weights 对 (10+1) 图前向 → 相对位姿 → **motion averaging** 解全局位姿与尺度；另跑 Umeyama 对齐对照（Table 5 前两行）。评测代码**复用 Stage A 的 Cambridge 数据加载与误差函数**，只换预测器。
3. **判据：Cambridge avg ≈ 41.0 cm / 0.374°（Table 4 frozen 行）**。这一行是论文里唯一不需要 MultiLoc 权重就能精确对照的自研方法数字，是评测协议正确性的黄金锚点。
4. 顺带产出 Figure 5 的 k 消融结构（k=1,2,4,10,20 重跑 frozen 变体）。

### Stage C — MultiLoc-lite：核心机制的归因验证（≈3–5 天，训练）

不自研框架，只写论文新增的最小三件（相对 VGGT 冻结主干）：
1. `PoseTokenMLP`：SE(3) 展平 R12 → MLP → d 维 `y`；可学习 `l`，`g = l + y`，查询 `yq=0`。
2. 每层注入 `gn = go + y`（改 global-attention 路径的残差入口）。
3. 训练环：论文式 (1)(2) 不确定性损失 + AdamW（lr 1e-5，cosine 到 1e-8 截短），**只训练 MLP、l、camera head**（可加最后 4 个 block 的 LoRA，默认不加）；主干冻结、bf16、gradient checkpointing。
- 训练数据：MegaDepth 子集（5–10 个场景）用 `reloc3r/datasets_preprocess/preprocess_megadepth.py` 现成脚本处理；采样/增强按论文 §3.3 "similar to VGGT"。subset-first：先在 1 个场景、2000 iter 冒烟，验证 loss 三组件下降且 `y` 注入消融（注入开/关）前向差异存在，再放大。
- 评测：Cambridge 零样本（**不训练 Cambridge**，保持论文范式），对照三个锚点：Stage B frozen（应显著优于）、Table 2 ReLoc3r 行（应优于）、论文 MultiLoc 0.29/0.35（**预期达不到**，只报差距与原因：1M iter 全参微调 vs 冻结主干）。
- 显存风险预案：11 帧×518px OOM → 降 448/392px，全部结论标注分辨率偏差。`# ponytail: 冻结主干版，全参微调等 H200 级算力再说——升级路径是解冻更多 block，不改结构。`

### 全阶段共用的硬门（直接沿用 ScientistTwo 方案 §4）

- `eval/` 目录（数据划分清单 + 误差脚本 + 场景哈希 manifest）生成后只读 + sha256，前后校验，变更即整轮 invalid；
- 指标判定由确定性脚本解析固定 schema 的 `metrics.json`，LLM 不碰数字；报告数值只从 `registry.jsonl` 台账取；
- secondary 指标强制：延迟（对照论文 Lat. 列）、focal 预测误差、k-曲线——主指标提升而 secondary 恶化不得判 Good；
- 预算 fail-closed：每 Stage 墙钟上限写 `task.yaml`，断点续跑幂等。

### 目录布局（`/data/code/AI-Research/tasks/multiloc/`）

```
task.yaml  state.json  registry.jsonl
data/cambridge -> (软链到 reloc3r/data/cambridge)
eval/            # 协议不可变区 + manifest.sha256
evidence/stageA|B|C/...
code/lite/       # Stage C 三个新模块 + train.py（全部新代码 < 800 行目标）
report.md        # 数字全部从 registry 生成
```

## 5. 里程碑与验收（数字判据）

| 里程碑 | 判据 | 单独价值 |
|---|---|---|
| M-A | ReLoc3r×Cambridge 与 Table 2 行 avg 相对差 ≤20%；MegaDepth-1500 回归通过；干净重跑一致（I1） | 可信评测管线 + 基线台账 |
| M-B | Frozen 行 ≈41.0 cm/0.374°（±15%）；Umeyama vs motion avg 次序与 Table 5 一致 | 论文评测协议还原的证明 |
| M-C | lite 在 Cambridge 显著优于 M-B frozen 基线（中位平移误差降低 ≥30%，配对中位数检验）且延迟报告完整；消融"去 y 注入"回到 frozen 水平 | 核心论断（位姿条件化多视角）方向性复现 |

## 6. 风险与边界

| 风险 | 缓解 |
|---|---|
| Cambridge Drive 直链失效/需确认页 | 备选：ReLoc3r 官方仓库同链接、PoseNet 原站镜像；失败则告知用户人工下载 |
| 11 帧 VGGT 前向 OOM | 降分辨率预案（Stage C）；Stage B 纯推理可逐 query 处理，风险低 |
| lite 训不出增益 | 仍是有价值的负结果：按 I1/I2 报告"冻结主干容量不足"，对照论文 Table 4 增益来自全参 1M iter——这本身就是消融结论 |
| 论文无码，细节靠文字 | 所有推断（MLP 层数/d 维/y 注入实现点）写 `docs/decisions/`，不当事实；MultiLoc 一旦开源立即用真权重补 Stage D（纯推理复现全部表格行） |
| 非商业许可 | 全程研究用途；产物不分发权重 |

## 7. 拟议命令（**全部未执行**，批准后按序）

```bash
# 1 下载 Cambridge（reloc3r README 直链；先 -O 到暂存，验证后再解压进 data/cambridge）
cd /data/code/reLocation-all/reloc3r
/data/software/conda/anaconda3/envs/mv2d/bin/gdown 1XcJIVRMma4_IClJdRq6rwBKX3ZPet5az -O data/cambridge_download.bin
unzip -l data/cambridge_download.bin   # 先看结构再解

# 2 VGGT-1B（HF 直连超时，走镜像）
HF_ENDPOINT=https://hf-mirror.com /data/software/conda/anaconda3/envs/mv2d/bin/python \
  -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/VGGT-1B', local_dir='data/vggt-1b')"

# 3 Stage B 冒烟（单场景单 query，验证显存）
/data/software/conda/anaconda3/envs/mv2d/bin/python scripts/prof_vggt_fwd.py --scene GreatCourt --k 10

# 4 MegaDepth 子集下载与预处理（场景清单定了后填充；脚本现成）
/data/software/conda/anaconda3/envs/mv2d/bin/python datasets_preprocess/preprocess_megadepth.py --subsample ...

# 运行入口（实现后）
python eval_cambridge_multiloc.py --predictor reloc3r|vggt_frozen|multiloc_lite --topk 10 --scene all
```

## 8. 请拍板的 4 件事

1. 数据集：接受 "Cambridge(下载~10GB) + MegaDepth-1500(已有) + MegaDepth 子集(训练用, ≤100GB)" 组合？还是只要 Cambridge？
2. 范围：只做到 M-B（两天内、纯推理、可精确对照论文一行），还是含 M-C 训练（3–5 天、只承诺方向性）？
3. Stage C 允许最后 4 个 block 加 LoRA 吗（更可能出增益，偏离论文"全参微调"更远）？
4. Indoor6/WaySpots 留待后续（体积/许可成本高）——确认本期不做？
