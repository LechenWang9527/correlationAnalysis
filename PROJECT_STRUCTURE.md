# 项目目录导航（PROJECT_STRUCTURE）

本文件是工作区的地图。第一次接手本项目时，先读这里，再读 `reports/step3_no_leakage_attribution_report.md`。

项目主题：轮胎制造实际工时偏差（ActualVsIEGap = 实际节拍 - IE 标准节拍，单位：秒）的
无泄露建模、跨月盲测与物理特征归因。全流程以 CSV 快照为边界，开发期（8-10 月）与
盲测期（11 月）严格分离，代理工单为最小分析单元。

维护约定：本文件只描述结构与职责，不复制结论；数据与报表的数字以对应产物文件为准。

---

## 1. 根目录一览

```text
相关性分析/
|-- AGENTS.md                     协作与交付规范
|-- PROJECT_STRUCTURE.md          本导航文件
|-- data/                         原始与清洗后的数据快照
|-- src/                          全部生产脚本（清洗、筛选、建模、归因、渲染）
|-- reports/                      正式报表、指标 CSV 与出版级图件
|-- models/                       第一阶段模型包（大文件）
|-- logs/                         运行日志（含 cli_tests/ 临时命令行日志归档）
|-- docs/                         方案文档与文献综述
|-- scratch/                      调研草稿、审计笔记、冻结源码、本地运行时
|-- archive_exploratory/          早期探索脚本与旧口径报表（只读存档）
`-- catboost_info/                CatBoost 训练过程自动产物（只读存档）
```

根目录不放置临时文件。临时命令行输出统一进入 `logs/cli_tests/`。

---

## 2. 核心最终无泄露产物（优先阅读）

第二阶段（Phase 2，无泄露重构）的正式交付物只有两处，其余文件均为支撑材料或历史版本。

### 2.1 主报告

```text
reports/step3_no_leakage_attribution_report.md
```

内容覆盖：快照身份与血缘裁定、字段白名单与验证协议、开发期嵌套交叉验证、
11 月冻结回放、模型表现、四项预测归因、条件置换与支撑区间、发布边界与阻塞项。

关键口径（详见报告正文）：

- 标签：源表 `ActualVsIEGap`，单位秒；每个可评分代理单等权。
- 规模：19,052 个候选代理组，开发期可评分 15,348 个，11 月可评分 3,567 个。
- 主模型：A_CatBoost（物理字段 + 4 项计划字段）与 B_CatBoost（纯物理字段）。
- 11 月回放：A 组 MAE 比 B 组低 0.265 秒；条件置换覆盖率最高 16.8%，
  六个模块均只满足子群发布条件，不满足全局发布条件。

### 2.2 图件目录

```text
reports/step3_phase2/figures/
|-- model_performance.png                     开发期与 11 月模型指标对比
|-- module_attribution.png                    物理模块归因（TreeSHAP / 簇级）
|-- treeshap_physical.png                     物理特征 TreeSHAP 分布
|-- ale_supported_ranges.png                  有支撑区间的 ALE 响应曲线
`-- step3_no_leakage_synthesis_dashboard.png  2x2 综合看板（由前四张合成）
```

报告正文通过相对路径引用这些图件，移动图件会导致 Markdown 预览断链。

### 2.3 产物校验与冻结凭据

```text
reports/step3_phase2/freeze_manifest.json     冻结清单：源码、数据、模型、阈值、折成员校验值
reports/step3_phase2/provenance_summary.json  行级血缘与异常标记汇总
reports/step3_phase2/replay_seal.json         11 月评分封存记录（UTC 时间戳）
reports/step3_phase2/frozen_models.pkl        Phase 2 冻结模型（约 2 MB）
```

`freeze_manifest.json` 与 `replay_seal.json` 是复现与审计的锚点，任何重跑都应先备份。

---

## 3. 源码职责清单（src/）

所有脚本以项目根目录为工作目录运行，模块之间通过同目录导入协作。

### 3.1 主链路

| 脚本 | 阶段 | 职责 |
| --- | --- | --- |
| `step1_data_cleaner.py` | 第一步 | 物理提纯与无失真清洗：修正负值异常，剔除索引列、常数列、严格派生列与文本双胞胎列，按四批规则解码 24 个名义/代号列，输出 `clean_dataset.csv` 与 `feature_schema.json`。 |
| `step2a_strip_a_screening.py` | 第二步 A | 单特征公平体检。数值轨使用 Pearson、Spearman、15 分箱非线性 R²、单因素 F 检验与抽样距离相关；文本轨使用 Eta²、无偏 Omega²、ANOVA、Kruskal-Wallis 与 0-1 离散距离相关；内置随机噪声哨兵。 |
| `step2b_strip_b_team_and_simpson.py` | 第二步 B | 多特征团战探针。Dummy、Ridge、HistGBDT 三路 5 折对比；高基数文本走折内 TargetEncoder，低中基数走 OrdinalEncoder；19 次 Y-Scrambling 置换检验；影子噪声特征重要性对照；全表辛普森悖论分家排查。 |
| `step3_model_benchmark_and_voting.py` | 第三步 Part 1 | 五基模型（Ridge、ExtraTrees、CatBoost、LightGBM、XGBoost）与三类加权投票（Equal、SLSQP、Caruana）；两套候选池、两套防泄露协议（工单分组 GroupKFold OOF 与 11 月跨月盲测）；产出 `models/step3_models_bundle.pkl`。 |
| `step3_part2_feature_attribution.py` | 第三步 Part 2 | 六阶段归因：TreeSHAP 与物理簇 Group Shapley、跨月置换重要性、Top-10 分组 SAGE、定向 LOCO 重训、Top-6 连续参数 ALE、5 折工单隔离部分线性 DML 与安慰剂反驳。 |

### 3.2 Phase 2 无泄露重构链路

| 脚本 | 职责 |
| --- | --- |
| `step3_no_leakage.py` | 主编排入口。`python src/step3_no_leakage.py develop|score|audit`，依次完成开发冻结、11 月独立评分与审计核对。 |
| `step3_no_leakage_data.py` | 快照血缘、无结果信息的代理身份构造、可评分资格判定、代理族折分配与输入读取。 |
| `step3_no_leakage_models.py` | 嵌套开发验证、折内调参、冻结回放模型与指标计算。 |
| `step3_no_leakage_attribution.py` | 折内约束、条件置换块、ALE、簇均值与配对区间、支撑区间判定。 |
| `render_enhanced_figures.py` | 从封存预测与汇总表渲染四张证据面板（300 dpi）。 |
| `render_step3_phase2.py` | 校验 Phase 2 产物完整性并生成工业级 Markdown 报告。 |
| `plot_no_leakage_dashboard.py` | 将四张面板合成 2x2 综合看板，保持原始长宽比。 |
| `diagnostics_step3_phase2.py` | 仅限开发期的冷启动与含/不含异常样本重拟合诊断，不参与正式评分。 |
| `test_step3_no_leakage.py` | 快照契约单元测试：特征清单数量、禁止字段、折分配、支撑区间等。 |

### 3.3 典型运行顺序

```text
python src/step1_data_cleaner.py
python src/step2a_strip_a_screening.py
python src/step2b_strip_b_team_and_simpson.py
python src/step3_model_benchmark_and_voting.py
python src/step3_part2_feature_attribution.py
python src/step3_no_leakage.py develop
python src/step3_no_leakage.py score
python src/step3_no_leakage.py audit
python src/render_step3_phase2.py
python src/plot_no_leakage_dashboard.py
```

注意：`develop` 会重写冻结清单与冻结模型；若只为出图，直接运行最后两步即可，
不要重跑 `develop` 与 `score`。

---

## 4. 数据资产（data/）

```text
data/raw/output.csv                 原始导出，53,313 行，只读，禁止覆盖
data/processed/clean_dataset.csv    清洗后快照，53,145 行 x 76 列，建模唯一输入
data/processed/feature_schema.json  特征白名单、类型与模块归属
```

原始导出行数与清洗行数的差额来自多源行映射到同一清洗记录，属于血缘差额而非丢数，
详见 `reports/step3_phase2/provenance_summary.json` 与主报告第 2 节。

---

## 5. 报表与历史对比数据位置

### 5.1 当前正式口径（Phase 2）

```text
reports/step3_no_leakage_attribution_report.md          主报告
reports/step3_phase2/figures/                           正式图件
reports/step3_phase2/development_metrics.csv            开发期指标
reports/step3_phase2/november_metrics.csv               11 月指标
reports/step3_phase2/development_oof.csv                开发期 OOF 预测
reports/step3_phase2/november_predictions.csv           11 月封存预测
reports/step3_phase2/treeshap_A_ranking.csv             A 组 TreeSHAP 排名
reports/step3_phase2/treeshap_B_ranking.csv             B 组 TreeSHAP 排名
reports/step3_phase2/treeshap_A_modules.csv             A 组模块汇总
reports/step3_phase2/treeshap_B_modules.csv             B 组模块汇总
reports/step3_phase2/ale_bins.csv                       ALE 分箱明细
reports/step3_phase2/loco_and_paired_comparisons.csv    LOCO 与配对比较
reports/step3_phase2/permutation_support.csv            条件置换支撑判定
reports/step3_phase2/permutation_module_summary.csv     置换模块汇总
reports/step3_phase2/cohort_audit.csv                   队列审计
reports/step3_phase2/*_audit.csv                        身份、标签、历史分组审计明细
```

### 5.2 第一阶段（Part 1）基线与投票

```text
reports/step3_model_benchmark_report.csv     六模型 x 两协议指标矩阵
reports/step3_voting_weights.csv             投票权重（Equal / SLSQP / Caruana）
reports/step3_model_benchmark_summary.png    指标对比图
models/step3_models_bundle.pkl               基座模型包（约 2.4 GB，勿随意复制）
logs/step3_train.log                         训练日志
```

### 5.3 第二阶段归因初版（Part 2）

```text
reports/stage1_shap_executive_brief.md                 TreeSHAP 阶段简报
reports/step3_part2_shap_ranking.csv                   单特征 SHAP 排名
reports/step3_part2_permutation_nov_blind.csv          11 月盲测置换重要性
reports/step3_part2_lightweight_sage_top10.csv         Top-10 分组 SAGE
reports/step3_part2_loco_ablation.csv                  LOCO 消融
reports/step3_part2_ale_summary.csv                    ALE 汇总
reports/step3_part2_dml_causal_effects.csv             DML 因果净效应
reports/step3_part2_shap_summary.png                   归因总览
reports/step3_part2_ale_curves.png                     ALE 曲线
reports/step3_part2_final_synthesis_dashboard.png      归因综合看板
logs/step3_part2.log                                   归因运行日志
```

此组为 Phase 2 之前的归因口径。与主报告冲突时，以 Phase 2 主报告为准；
历史数字保留用于方法对比与演进说明，不要覆盖。

### 5.4 第二步筛选报表

```text
reports/step2a_strip_a_report.csv        单特征体检明细
reports/step2a_strip_a_summary.png       单特征体检图
reports/step2b_strip_b_simpson_report.csv 团战与辛普森排查明细
reports/step2b_strip_b_summary.png        团战与辛普森排查图
```

### 5.5 早期探索存档

```text
archive_exploratory/old_reports/     旧口径健康检查与归因报表
archive_exploratory/scripts/         早期探索脚本（数据体检、派生列排查、基线核对等）
```

该目录只读。新分析不要在此继续叠加文件，避免与正式口径混淆。

---

## 6. 其他目录

### logs/

```text
logs/step3_phase2_develop.log        Phase 2 开发阶段日志
logs/step3_phase2_sensitivity.log    Phase 2 敏感性分析日志
logs/step3_part2.log                 Part 2 归因日志
logs/step3_train.log                 Part 1 训练日志
logs/cli_tests/                      临时命令行测试输出归档（本文件所在整理动作的目标目录）
```

`logs/cli_tests/` 仅存放命令行冒烟测试产生的 stdout、stderr 与 JSON 回执，
不承载正式实验记录；正式实验日志直接放在 `logs/` 一级。

### models/

```text
models/step3_models_bundle.pkl   Part 1 五基模型与投票集成的序列化包
```

体积约 2.4 GB，属于大文件。读取前确认内存与磁盘余量，不要纳入版本控制。

### docs/

```text
docs/step3_part1_plan.md                         Part 1 实施方案
docs/step3_part2_plan.md                         Part 2 实施方案
docs/step3_part2_50_papers_attribution_guide.md  50 篇文献综述与 25 篇精读
```

### scratch/

```text
scratch/codex_no_leakage_plan.md          Phase 2 无泄露方案初稿
scratch/attribution_audit_critique.md     归因方法审计与反驳意见
scratch/literature_review_report.md       文献调研报告
scratch/deepseek_paper_lookup_20.md       补充文献检索
scratch/arxiv_verified.json               arXiv 核验记录
scratch/openalex_verified.json            OpenAlex 核验记录
scratch/discover*.mjs                     文献发现脚本
scratch/fetch_abstracts.mjs               摘要抓取脚本
scratch/verify_*.mjs                      文献元数据核验脚本
scratch/step3_phase2_requirements.txt     Phase 2 依赖清单
scratch/phase2_frozen_source/             评分封存时的冻结源码副本（审计基准）
scratch/.phase2-runtime/                  本地便携 Python 运行时（含 Lib、Scripts）
```

`phase2_frozen_source/step3_no_leakage_data.py` 是封存时点的源码证据，
用于解释冻结后对血缘函数读取类型的修正。任何情况下不得删除或覆盖。

`.phase2-runtime/` 是独立的便携解释器环境，不是缓存；删除会导致重跑环境缺失。

### archive_exploratory/ 与 catboost_info/

两者均为只读存档。`catboost_info/` 由 CatBoost 训练自动生成，
包含 `catboost_training.json`、`learn_error.tsv`、`test_error.tsv` 与 `time_left.tsv`，
可作为训练曲线与早停行为的旁证，不参与正式评分。

---

## 7. 文件安全边界

以下内容不得删除、移动或覆盖：

- `data/raw/` 与 `data/processed/` 下的全部数据文件；
- `src/` 下全部脚本，包括测试与诊断脚本；
- `reports/` 下全部报表、指标 CSV 与图件；
- `docs/` 下全部方案与文献文档；
- `models/` 下模型包；
- `scratch/` 下冻结源码、方案稿与本地运行时。

允许的整理动作仅限于：在 `logs/` 下新增归档子目录、新增导航与说明文档。
需要清理时先确认文件是否被报告、清单或脚本引用，再决定是否归档而非删除。

---

## 8. 快速定位

| 我想知道 | 去哪里 |
| --- | --- |
| 最终结论与发布边界 | `reports/step3_no_leakage_attribution_report.md` |
| 最终图件 | `reports/step3_phase2/figures/` |
| 开发期与 11 月指标 | `reports/step3_phase2/development_metrics.csv`、`november_metrics.csv` |
| 特征归因排名 | `reports/step3_phase2/treeshap_A_ranking.csv`、`treeshap_B_ranking.csv` |
| 复现凭据 | `reports/step3_phase2/freeze_manifest.json`、`replay_seal.json` |
| 历史模型对比 | `reports/step3_model_benchmark_report.csv` |
| 归因方法演进 | `reports/stage1_shap_executive_brief.md` 与 `reports/step3_part2_*` |
| 清洗规则 | `src/step1_data_cleaner.py` 与 `data/processed/feature_schema.json` |
| 方案与文献 | `docs/` 与 `scratch/literature_review_report.md` |
| 运行日志 | `logs/` |
