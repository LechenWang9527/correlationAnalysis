# 第三步 Part 1 实施计划：6 大模型 × 双重防泄露验证与加权投票融合

## 1. 目标概述
针对清洗后的洁净数据集 `data/processed/clean_dataset.csv`（53,145 行 × 76 列），针对三大核心数据特性——**19,189 个生产工单多行拆分的泄露风险**、**4 个 >250 类的超高基数无序文本特征**、**44 个辛普森悖论特征**——构建并运行工业级、零泄露的多模型对比与加权投票流水线（`src/step3_model_benchmark_and_voting.py`）。

**核心评估矩阵：共训练并评估 6 个模型（5 大门派单模型 + 1 个多模型加权投票集成），每个模型均严格接受 2 种防泄露验证方式检验。**

---

## 2. 两种防泄露验证方式

1. **验证方式 1：8~10 月开发集内部 5 折工单组隔离交叉验证（`GroupKFold` / `Nested OOF`）**
   - **数据范围**：`ActualStartMonth` 为 `8, 9, 10` 月的 `43,451` 行（占 `81.76%`）。
   - **隔离机制**：根据 `['RecipeID', 'WorkCenterNameCoupled', 'ActualStartMonth', 'ActualStartDayOfWeek', 'ActualStartHour', 'PlannedQuantity', 'QuantityDone', 'ActualVsIEGap']` 生成唯一工单编号 `work_order_id`，同一工单的多位操作工记录强制绑定在同一折，彻底堵死 `88.87%` 的同工单目标泄露。
   - **防权重过拟合**：第 6 个模型（加权投票）采用 **Nested OOF**（内 4 折求解权重、外 1 折无偏评分），杜绝乐观偏倚。

2. **验证方式 2：11 月份独立未来外推盲测**
   - **数据范围**：留出 `ActualStartMonth == 11` 的 `9,694` 行（占 `18.24%`）。
   - **隔离机制**：全程不参与任何训练、早停与权重计算，仅在最后一次性前向预测验收，检验跨月趋势稳定性及 **16.01% 未见新配方冷启动**表现。

---

## 3. 参赛的 6 个模型阵容

| 序号 | 模型名称 | 所属门派 | 高基数类别适配策略 | 核心使命 |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **`Ridge`** | 线性正则门派 | 低基数（<=250）`OneHotEncoder` + 4 个超高基数（>250）折内 `TargetEncoder` + 数值 `StandardScaler` | 纯线性基准对照 |
| **2** | **`ExtraTrees`** | Bagging 极度随机树 | 低基数 `OrdinalEncoder` + 4 个超高基数折内 `TargetEncoder` | Bagging 降方差代表 |
| **3** | **`CatBoost`** | 对称树 + Ordered TS | 26 个无序文本列全量原生 `cat_features` 直入（Ordered Target Statistics） | 专治高基数类别与冷启动的主力王牌 |
| **4** | **`LightGBM`** | Leaf-wise 提升树 | 26 个类别列转 `category` 原生切分（配 `cat_smooth=20`, `min_data_per_group=50`） | 高阶非线性交互专家 |
| **5** | **`XGBoost`** | Depth-wise 提升树 | 26 个类别列转 `category` + `enable_categorical=True` + `tree_method='hist'` | 工业界经典对称结构 GBDT 标杆 |
| **6** | **`Weighted_Voting`** | 多模型加权投票集成 | 在 Dev 集 OOF 上对比 **纯树三巨头（CatBoost+LightGBM+XGBoost）SLSQP / Caruana 贪心选择** 与 **跨门派全模型 SLSQP**（约束 $w_k \ge 0, \sum w_k = 1$） | 检验融合增益并兼顾后续 `TreeSHAP` 线性可加性 |

---

## 4. 工程交付规范与产出清单
- **编码铁律**：严格遵循 `AGENTS.md` 工业规范（PEP 8、标准 `logging`、`[INFO]`/`[PASS]`/`[WARN]` 纯文本标签，严禁任何花哨 Emoji）。
- **交付脚本**：`src/step3_model_benchmark_and_voting.py`
- **输出文件**：
  - `reports/step3_model_benchmark_report.csv`（6 个模型双考场 $R^2$ / RMSE / MAE 汇总表）
  - `reports/step3_voting_weights.csv`（加权投票权重与基模型残差相关性矩阵）
  - `reports/step3_model_benchmark_summary.png`（双拼专业对比图）
  - `models/step3_models_bundle.pkl`（持久化模型与权重包，供 Step 3 Part 2 直接加载做归因）
