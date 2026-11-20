# TireCycle-Attribution: 轮胎制造实际工时偏差分析与无泄露因果归因系统

Industrial Tire Manufacturing Cycle-Time Deviation Modeling & Leak-Free Causal Attribution Pipeline

---

## 1. 项目概述与业务背景

在轮胎工业离散制造场景中，成型工序的单胎实际生产节拍与标准工时之间存在复杂的非线性波动。本项目针对某头部制造企业现场 53,145 行原始工单记录，旨在解决以下三大工业工程（IE）痛点：
1. **事前风险预警**：在生产工单释放前，仅凭出厂设计规格（BOM）与排产计划，提前预测该工单是否存在显著工时超差风险；
2. **工艺瓶颈归因**：定量拆解轮胎物理尺寸、胶料规格、骨架结构对单胎用时的边际影响，为工艺改进与模具设计提供客观依据；
3. **消除统计伪相关**：破除传统工业数据挖掘中普遍存在的定义项泄露与操作工多重记录偏差，建立无泄露的因果分析闭环。

---

## 2. 项目全流程架构与研发阶段

本项目严格遵循工业级数据科学工程规范，经历了五个递进的研发阶段：

```text
原始数据 (output.csv: 53,313 行 x 94 列)
    │
    ▼
[Step 1: 数据清洗与标准化 Schema 冻结]
├── 投影 76 维标准化字段 (49 数值, 26 类别, 1 标签)，确立 feature_schema.json
└── 输出 clean_dataset.csv (53,145 行 x 76 列)，记录清洗血缘与去重规则
    │
    ▼
[Step 2: 统计筛选与辛普森悖论排查]
├── 试纸 A: 单变量相关性、相关比 Eta 与互信息初筛 (reports/step2a_strip_a_report.csv)
└── 试纸 B: 按机台/班次分层排查，捕获 44 个存在正负翻转的辛普森悖论特征 (step2b)
    │
    ▼
[Step 3 Part 1: 多模型基座横评与投票集成]
├── 建立时间顺序划分 (8-10 月开发折，11 月盲测)
├── 评估 CatBoost、LightGBM、XGBoost、RandomForest、Ridge、MLP 六大模型
└── 输出 5 折 CV R² = 82.61%，11 月盲测 R² = 78.52%
    │
    ▼
[关键转折: 红队审计与 99.968% 算术恒等式泄露发现]
├── 审计发现: 53,049 行数据中 99.968% 严格符合算术公式:
│   Y = ActualVsIEGap = O / (C x Q) - (IE + TOL) / 100
│   (其中 O=打卡秒数, C=覆盖率, Q=完成量, IE=标准工时, TOL=公差)
└── 结构陷阱: 该 5 项定义项占模型方差解释度的 96.4%，物理参数被严重掩盖；
    且 53,145 行记录仅对应约 19,052 个独立工单，行级推断存在严重伪重复。
    │
    ▼
[Step 3 Part 2 (Phase 2): 无泄露工单级重构与五维因果归因]
├── 样本净化: 将 5.3 万行操作工打卡记录干净聚合为 19,052 个独立代理工单
├── 白名单机制: 彻底剥离 O, C, Q, IE, TOL 等事后项，确立纯物理 B 组 (42 列) 与事前 A 组 (46 列)
├── 工单族隔离验证: 8-10 月 5 折嵌套 CV 与 11 月独立冻结回放
└── 五维归因输出: TreeSHAP 排名、模块条件置换、模块级 LOCO 消融、带支撑区间的 1D-ALE
```

---

## 3. 核心实验结论与物理归因发现

### 3.1 无泄露双基座模型对比

在彻底剔除定义项泄露与事后变量后，模型完全退回至“工单未开工前、仅凭静态图纸与计划批量”的严谨事前设定：

| 模型架构 | 输入特征集 | 8~10 月 5 折 CV R² | 11 月独立回放 R² | 单胎平均误差 MAE | 工业工程意义 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Schedule_CatBoost** | 仅 4 项事前排产特征 | 17.9% | 19.6% | 6.82 秒/胎 | 排产批量与计划时点带来的基准解释力 |
| **B_CatBoost (纯物理)** | 42 项纯轮胎出厂图纸 | **26.4%** | **26.7%** | 6.43 秒/胎 | 纯静态图纸参数独立解释超过 1/4 的制造波动 |
| **A_CatBoost (合体)** | **46 项全量事前特征** | **32.3%** | **30.9%** | **6.12 秒/胎** | **物理图纸为排产模型带来 +14.4% 的显著净增益** |
| **A_Ridge (线性对照)** | 46 项全量事前特征 | 30.8% | 29.1% | 6.22 秒/胎 | 线性正则对比基准，树模型具备非线性交互优势 |

注：在充满工人疲劳度、交接班、温湿度与微停机等随机扰动的离散制造现场，仅凭静态图纸即锁定超过 30% 的方差，是极具工业实践价值的真实有效信号。

### 3.2 Top 5 物理归因发现与车间改善建议

基于工单级 TreeSHAP 归因，对纯物理特征（B 组 42 列）的贡献度进行定量排序：

1. **三角胶宽度 (`1_apex_-_width`)**：平均绝对影响为 **0.636 秒/胎**（排名第一）。三角胶尺寸过宽导致胎圈包布与滚压阻力显著增加，是成型机台的主要瓶颈环节；
2. **胎面-带束层组件重量 (`1_tread-belt_assembly_-_weight`)**：平均绝对影响为 **0.524 秒/胎**（排名第二）。大规格胎面自重大，多轴机械臂抓取定位与复合贴合时的惯性衰减导致节拍延长；
3. **胎面底胶宽度 (`1_tread_base_-_width`)**：平均绝对影响为 **0.446 秒/胎**。底胶尺寸决定贴合压实接触面积，直接关联滚压循环时长；
4. **三角胶胶料配方 (`1_apex_-_compound`)**：平均绝对影响为 **0.401 秒/胎**。高硬度配方在常温成型时流动延展性低，压延贴合耗时较长；
5. **胎冠胶料配方 (`1_tread_cap_-_compound`)**：平均绝对影响为 **0.332 秒/胎**。耐磨与高抓地力胶料配方增加了模具挤压成型周期的波动。

---

## 4. 运行说明与复现指南

为了防止企业核心商业机密泄露，本项目原始制造数据集已从仓库中排除。仓库保留完整的清洗规则、模型架构、测试用例与分析流水线。

### 4.1 环境准备

支持 Python 3.10 ~ 3.13 环境：

```bash
# 克隆仓库
git clone https://github.com/your-username/TireCycle-Attribution.git
cd TireCycle-Attribution

# 安装核心依赖
pip install -r requirements.txt
```

### 4.2 运行自动化单元测试

本测试套件覆盖数据聚合守恒性校验、血缘追溯断言与 ALE 支撑逻辑：

```bash
python -m unittest discover -s src -p test_step3_no_leakage.py
```
预期输出：`Ran 5 tests in ~0.04s ... OK`。

### 4.3 执行无泄露建模与归因流水线

在配置本地数据路径后，可一键执行全流程复算：

```bash
# 1. 运行工单级无泄露建模与 5 折交叉验证（结果输出至 reports/step3_phase2/）
python src/step3_no_leakage.py develop

# 2. 运行 11 月独立盲测集评分封存
python src/step3_no_leakage.py score

# 3. 运行一致性与敏感性审计诊断
python src/step3_no_leakage.py audit
python src/diagnostics_step3_phase2.py

# 4. 重新渲染出版级图表与技术总报告
python src/render_enhanced_figures.py

# 5. 合成 5100x2640 px 综合全景看板
python src/plot_no_leakage_dashboard.py
```

---

## 5. 项目目录结构说明

```text
TireCycle-Attribution/
|-- README.md                                 # 本项目主说明书（全流程与复现指南）
|-- PROJECT_STRUCTURE.md                      # 详尽的 300 行模块导航与血缘索引文档
|-- requirements.txt                          # 生产依赖版本锁定清单
|-- .gitignore                                # Git 忽略规范（严格拦截数据与大模型）
|
|-- src/                                      # 生产源码目录
|   |-- step1_data_cleaner.py                 # Step 1: 原始数据清洗与 76 维 Schema 定义
|   |-- step2a_strip_a_screening.py           # Step 2A: 单变量相关性与互信息初筛
|   |-- step2b_strip_b_team_and_simpson.py    # Step 2B: 班次/机台分层与辛普森悖论排查
|   |-- step3_model_benchmark_and_voting.py   # Step 3 Part 1: 六大基座模型基准评测
|   |-- step3_no_leakage.py                   # Step 3 Part 2: 无泄露工单级建模与评分入口
|   |-- step3_no_leakage_data.py              # 工单聚合、白名单过滤与血缘守恒校验
|   |-- step3_no_leakage_models.py            # CatBoost / Ridge 嵌套交叉验证与超参搜索
|   |-- step3_no_leakage_attribution.py       # TreeSHAP / 模块置换 / 模块LOCO / ALE 引擎
|   |-- render_enhanced_figures.py            # 出版级学术图表渲染引擎
|   |-- plot_no_leakage_dashboard.py          # 2x2 全景大看板拼装合成模块
|   `-- test_step3_no_leakage.py              # 自动化回归测试套件
|
|-- reports/                                  # 交付报告与图表
|   |-- step3_no_leakage_attribution_report.md# Phase 2 工业技术总报告
|   `-- step3_phase2/
|       |-- figures/                          # 300 DPI 出版级图表目录
|       |   |-- step3_no_leakage_synthesis_dashboard.png # 全景综合大看板
|       |   |-- model_performance.png         # 子图 (a) 模型竞技场对比
|       |   |-- treeshap_physical.png         # 子图 (b) 物理重要性 Top 20 榜单
|       |   |-- module_attribution.png        # 子图 (c) 模块级消融与条件置换
|       |   `-- ale_supported_ranges.png      # 子图 (d) 局部有效 ALE 响应曲线
|       `-- *.csv                             # 32 份指标明细与特征重要性报表
|
`-- docs/                                     # 阶段技术方案与 50 篇顶会文献综述
```

---

## 6. 学术引用与合规声明

- 本项目采用的方法学遵循 Kaufman et al. (ACM TKDD 2012) 关于数据挖掘泄露防御的标准、Lundberg et al. (Nature Machine Intelligence 2020) 树模型可解释性理论以及 Apley & Zhu (JRSS-B 2020) 累积局部效应 (ALE) 理论。
- 遵循保密合规要求，本项目代码库中不包含任何受知识产权保护的企业原始工艺数据、配方成分比例或客户专有规格。
