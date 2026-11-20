# -*- coding: utf-8 -*-
"""
====================================================================
第一步：物理提纯与无失真数据清洗流水线 (Industrial Data Cleaner)
====================================================================
本脚本严格遵循“物理提纯 + 保留真实名义类别”的原则，坚决杜绝把无序文本/代号
强行编成 1,2,3 或频次占比（避免制造虚假线性关系），具体执行以下 5 大动作：

1. 【物理异常修正】：
   - 将 OperatorOverlapSeconds 和 OperatorCoverageRatio 中小于 0 的异常负数截断为 0。
2. 【剔除垃圾、常数与派生/双胞胎列】：
   - 剔除索引、92%纯空格、绝对常数列（6 列）；
   - 剔除数学严格派生列 & 文本双胞胎列（8 列）；
   - 剔除单值占比 >= 98% 的伪常数死值列（但显式保护并保留 1_belt_ply_assembly_-_1_strip_rubber_-_width）。
3. 【34 个非数值/代号列的四批精准物理提纯】：
   - 先将伪装成数字的代号/类别列 RecipeID、1_belt_ply_assembly_-_local_plant_workcode、OperatorUserID、DestinationInfo 转为干净字符串；
   - 批次 A（二元开关类，2 列）：oeindicator、1_flange_cushion_-_compound -> 映射为 0/1；
   - 批次 B（天然有序/内嵌真数值类，7 列）：
     * 1_sidewall_-_bom_component_setting -> 去掉 'A=' 转为 float；
     * speed_symbol -> 映射为真实速度上限（km/h）；
     * 1_wound_cap_ply_-_cf_construction -> 修复空格脏数据后映射为总纤度（dtex）；
     * bead_-_bead_construction -> 解析 a x b 计算钢丝总根数；
     * ply_lock -> 按层数结构序数编码；
     * article_cluster_link_status -> 按 PLM 生命周期序数编码；
     * tire_size -> 既保留原字符串作为规范规格分组列，又额外拆出 tire_section_width 与 tire_aspect_ratio 两个物理数值列；
   - 批次 C（低基数无序名义类，18 列，含 DestinationInfo）与 批次 D（高基数无序代号类，7 列，含 OperatorUserID）+ tire_size：
     * 严禁乱编数字！仅做首尾空格清理，原封不动保留干净文本字符串（共 26 个类别字符串列）。
4. 【全局去重】：
   - 删除完全一模一样的重复行，并重置行索引。
5. 【输出与严格自检】：
   - 导出洁净数据集：data/processed/clean_dataset.csv
   - 导出特征结构元数据：data/processed/feature_schema.json
   - 清理旧版易引发误会的 text_encoding_dictionary.json
   - 对输出结果执行 0 NaN、0 Inf 断言校验，并打印全列类型统计与关键样例。
"""

import os
import re
import json
import numpy as np
import pandas as pd


class IndustrialDataCleaner:
    def __init__(self, data_path="data/raw/output.csv", target_col="ActualVsIEGap", output_dir="data/processed"):
        """
        初始化清洗器。
        :param data_path: 原始数据 CSV 路径（默认 data/raw/output.csv）
        :param target_col: 目标变量列名（默认 ActualVsIEGap）
        :param output_dir: 输出文件存放目录（默认 data/processed）
        """
        self.data_path = data_path
        self.target_col = target_col
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 显式受保护的列：即使占比接近阈值也绝不误删
        self.protected_cols = {
            self.target_col,
            "1_belt_ply_assembly_-_1_strip_rubber_-_width",
        }

        # 记录四批提纯的完整元数据，最终写入 feature_schema.json
        self.schema = {
            "target_col": self.target_col,
            "dataset_shape": {},
            "dropped_columns": {},
            "numeric_features": [],
            "binary_and_ordinal_mappings": {},
            "derived_physical_features": {},
            "categorical_features": [],
            "categorical_unique_counts": {},
        }

    @staticmethod
    def _parse_bead_total_wires(val: str) -> float:
        """
        将胎圈钢丝结构字符串（如 '6x6'）解析为钢丝总根数（36.0）。
        若遇到未知格式则直接抛出异常，防止静默产生 NaN。
        """
        match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", str(val))
        if not match:
            raise ValueError(f"无法解析 bead_-_bead_construction 取值: {val!r}")
        a, b = int(match.group(1)), int(match.group(2))
        return float(a * b)

    @staticmethod
    def _parse_tire_size(val: str) -> tuple:
        """
        从轮胎规格字符串（如 '235/50 R 19' 或 '285/30 ZR 20'）中
        提取 (胎宽 tire_section_width, 扁平比 tire_aspect_ratio)。
        """
        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*Z?R\s*\d+(?:\.\d+)?\s*",
            str(val),
        )
        if not match:
            raise ValueError(f"无法解析 tire_size 取值: {val!r}")
        return float(match.group(1)), float(match.group(2))

    @staticmethod
    def _apply_strict_map(series: pd.Series, mapping: dict, col_name: str) -> pd.Series:
        """
        严格字典映射辅助函数：先去首尾空格，再校验全列无未知取值，最后映射为 float。
        绝不允许悄悄变成 NaN！
        """
        cleaned = series.astype(str).str.strip()
        unknown_vals = set(cleaned.unique()) - set(mapping.keys())
        if unknown_vals:
            raise ValueError(f"列 [{col_name}] 存在未覆盖的未知取值: {sorted(unknown_vals)}")
        return cleaned.map(mapping).astype(float)

    def run(self) -> pd.DataFrame:
        print("=" * 75)
        print("【第一步：物理提纯与无失真数据清洗流水线 · 启动】")

        # 0. 读取原始数据（low_memory=False 防止混合类型警告）
        df = pd.read_csv(self.data_path, low_memory=False)
        print(f"[0] 原始数据读取完成: {len(df):,} 行 × {df.shape[1]} 列")

        # ------------------------------------------------------------------
        # 1. 物理异常修正：负数工时与覆盖率安全截断为 0
        # ------------------------------------------------------------------
        for col in ["OperatorOverlapSeconds", "OperatorCoverageRatio"]:
            if col in df.columns:
                neg_cnt = int((df[col] < 0).sum())
                df[col] = df[col].clip(lower=0)
                print(f"[1] 物理异常修正 [{col}]: 已将 {neg_cnt} 行异常负数截断为 0")

        # ------------------------------------------------------------------
        # 2. 剔除无用、纯空格、绝对常数、派生与双胞胎列
        # ------------------------------------------------------------------
        drop_basic = [
            "Unnamed: 0",        # 无意义导出行号
            "PhaseInOut",        # 92% 纯空格死列
            "variant",           # 绝对常数列
            "spec_type",         # 绝对常数列
            "base_variant",      # 绝对常数列
            "base_spec_type",    # 绝对常数列
        ]
        drop_numeric_derived = [
            "quantity gap ",              # == QuantityDone - PlannedQuantity
            "bead_-_bead_circumference",  # == pi * bead_inner_diameter
            "1_belt_-_angle",             # == 2_belt_-_angle
            "1_belt_-_width",             # == 2_belt_-_width + 10
        ]
        drop_text_twins = [
            "base_article_development_cluster_name",  # 与 article_development_cluster_name 1:1 镜像
            "base_article_cluster_link_status",       # 与 article_cluster_link_status 1:1 镜像
            "2_belt_-_cf_code",                       # 与 1_belt_-_cf_code 1:1 镜像
            "2_belt_-_gradient",                      # 与 1_belt_-_gradient 1:1 镜像
        ]

        to_drop_basic = [c for c in drop_basic if c in df.columns]
        to_drop_derived = [c for c in drop_numeric_derived if c in df.columns]
        to_drop_twins = [c for c in drop_text_twins if c in df.columns]
        all_explicit_drop = to_drop_basic + to_drop_derived + to_drop_twins

        df.drop(columns=all_explicit_drop, inplace=True)
        print(f"[2.1] 已剔除无用/常数/派生/双胞胎列共 {len(all_explicit_drop)} 列")

        # 剔除伪常数列（单一值占比 >= 98%），显式跳过目标列和受保护特征列
        pseudo_drop = []
        for col in df.columns:
            if col in self.protected_cols:
                continue
            top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top_freq >= 0.98:
                pseudo_drop.append(col)

        if pseudo_drop:
            df.drop(columns=pseudo_drop, inplace=True)
        print(
            f"[2.2] 已剔除伪常数列 (单值占比 >= 98%) 共 {len(pseudo_drop)} 列: {pseudo_drop}"
        )
        assert "1_belt_ply_assembly_-_1_strip_rubber_-_width" in df.columns, (
            "受保护列 1_belt_ply_assembly_-_1_strip_rubber_-_width 不得被剔除！"
        )

        self.schema["dropped_columns"] = {
            "basic_and_constants": to_drop_basic,
            "numeric_derived": to_drop_derived,
            "text_twins": to_drop_twins,
            "pseudo_constants_ge_98pct": pseudo_drop,
        }

        # ------------------------------------------------------------------
        # 3. 34 个非数值/代号列的“四批精准物理提纯”
        # ------------------------------------------------------------------
        # 先把伪装成大整数的代号/类别列强制转为干净字符串 (id_as_cat)
        id_as_cat = [
            "RecipeID",
            "1_belt_ply_assembly_-_local_plant_workcode",
            "OperatorUserID",
            "DestinationInfo",
        ]
        for id_col in id_as_cat:
            if id_col in df.columns:
                df[id_col] = df[id_col].astype(str).str.strip()

        # 确认当前所有非数值列数量应正好为 34 列
        all_non_numeric_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
        print(f"[3.0] 识别到待处理非数值/代号列共 {len(all_non_numeric_cols)} 列")

        # --- 批次 A：二元开关类（2 列） ---
        oe_map = {"N": 0.0, "Y": 1.0}
        df["oeindicator"] = self._apply_strict_map(df["oeindicator"], oe_map, "oeindicator")

        flange_map = {"R00037": 0.0, "R00218": 1.0}
        df["1_flange_cushion_-_compound"] = self._apply_strict_map(
            df["1_flange_cushion_-_compound"], flange_map, "1_flange_cushion_-_compound"
        )

        self.schema["binary_and_ordinal_mappings"]["oeindicator"] = {
            "batch": "A (二元开关)",
            "description": "是否原配胎 (N->0, Y->1)",
            "mapping": oe_map,
        }
        self.schema["binary_and_ordinal_mappings"]["1_flange_cushion_-_compound"] = {
            "batch": "A (二元开关)",
            "description": "轮辋保护胶料配方二元编码",
            "mapping": flange_map,
        }
        print("[3.1] 批次 A（二元开关 2 列）映射完成：oeindicator, 1_flange_cushion_-_compound")

        # --- 批次 B：天然有序 / 内嵌真数值类（7 列） ---
        # B1: 1_sidewall_-_bom_component_setting -> 去掉 'A=' 直接转 float
        sw_cleaned = df["1_sidewall_-_bom_component_setting"].astype(str).str.strip()
        assert sw_cleaned.str.startswith("A=").all(), "1_sidewall_-_bom_component_setting 存在非 'A=' 开头的数据！"
        df["1_sidewall_-_bom_component_setting"] = sw_cleaned.str.removeprefix("A=").astype(float)
        self.schema["binary_and_ordinal_mappings"]["1_sidewall_-_bom_component_setting"] = {
            "batch": "B (内嵌连续真数值)",
            "description": "去掉前缀 'A=' 后直接转为 float 连续数值",
            "min": float(df["1_sidewall_-_bom_component_setting"].min()),
            "max": float(df["1_sidewall_-_bom_component_setting"].max()),
        }

        # B2: speed_symbol -> 真实速度上限 (km/h)
        speed_map = {
            "T": 190.0,
            "H": 210.0,
            "V": 240.0,
            "W": 270.0,
            "Y": 300.0,
            "(Y": 320.0,  # 行业标准 (Y) 表示 >300 km/h 极高速级别，赋序数物理值 320 km/h
        }
        df["speed_symbol"] = self._apply_strict_map(df["speed_symbol"], speed_map, "speed_symbol")
        self.schema["binary_and_ordinal_mappings"]["speed_symbol"] = {
            "batch": "B (内嵌物理速度真值)",
            "description": "轮胎速度级别映射为真实允许最高车速 (km/h)",
            "mapping": speed_map,
        }

        # B3: 1_wound_cap_ply_-_cf_construction -> 先修复空格脏数据，再映射为总纤度 (dtex)
        cap_cleaned = (
            df["1_wound_cap_ply_-_cf_construction"]
            .astype(str)
            .str.strip()
            .replace({"1400 dtex x2": "1400 dtex x 2"})
        )
        cap_dtex_map = {
            "470 dtex x 2": 940.0,            # 470 * 2 = 940 dtex
            "940 dtex x 2": 1880.0,           # 940 * 2 = 1880 dtex
            "1400 dtex x 2": 2800.0,          # 1400 * 2 = 2800 dtex
            "(1670 or 1680) + 1400": 3075.0,  # (1670+1680)/2 + 1400 = 3075 dtex（最高规格复合帘线）
        }
        df["1_wound_cap_ply_-_cf_construction"] = self._apply_strict_map(
            cap_cleaned, cap_dtex_map, "1_wound_cap_ply_-_cf_construction"
        )
        self.schema["binary_and_ordinal_mappings"]["1_wound_cap_ply_-_cf_construction"] = {
            "batch": "B (内嵌物理纤度真值)",
            "description": "先将 '1400 dtex x2' 规范为 '1400 dtex x 2'，再按总旦数(dtex)映射为有序物理数值",
            "dirty_value_fixes": {"1400 dtex x2": "1400 dtex x 2"},
            "mapping": cap_dtex_map,
        }

        # B4: bead_-_bead_construction -> 解析 a x b 计算钢丝总根数
        bead_cleaned = df["bead_-_bead_construction"].astype(str).str.strip()
        bead_map = {v: self._parse_bead_total_wires(v) for v in sorted(bead_cleaned.unique())}
        df["bead_-_bead_construction"] = self._apply_strict_map(
            bead_cleaned, bead_map, "bead_-_bead_construction"
        )
        self.schema["binary_and_ordinal_mappings"]["bead_-_bead_construction"] = {
            "batch": "B (内嵌物理总根数)",
            "description": "按 a x b 解析出胎圈钢丝总根数",
            "mapping": bead_map,
        }

        # B5: ply_lock -> 按层数结构序数编码
        ply_lock_map = {
            "C+0": 0.0,
            "1+0": 1.0,
            "1+3/4": 2.0,
            "2+0": 3.0,
        }
        df["ply_lock"] = self._apply_strict_map(df["ply_lock"], ply_lock_map, "ply_lock")
        self.schema["binary_and_ordinal_mappings"]["ply_lock"] = {
            "batch": "B (天然有序结构编码)",
            "description": "按胎体帘布反包锁边层数结构从简到繁有序编码",
            "mapping": ply_lock_map,
        }

        # B6: article_cluster_link_status -> 按 PLM 生命周期序数编码
        plm_map = {
            "Unknown": 0.0,
            "draft": 1.0,
            "introduced": 2.0,
            "confirmed": 3.0,
            "historicized": 4.0,
        }
        df["article_cluster_link_status"] = self._apply_strict_map(
            df["article_cluster_link_status"], plm_map, "article_cluster_link_status"
        )
        self.schema["binary_and_ordinal_mappings"]["article_cluster_link_status"] = {
            "batch": "B (天然有序 PLM 生命周期)",
            "description": "按产品生命周期阶段从未知/草稿到确认/归档有序编码",
            "mapping": plm_map,
        }

        # B7: tire_size -> 保留原字符串作为规格分组类别列 + 额外拆出两个物理数值列
        df["tire_size"] = df["tire_size"].astype(str).str.strip()
        parsed_sizes = df["tire_size"].map(self._parse_tire_size)
        df["tire_section_width"] = parsed_sizes.map(lambda x: x[0]).astype(float)
        df["tire_aspect_ratio"] = parsed_sizes.map(lambda x: x[1]).astype(float)
        self.schema["derived_physical_features"] = {
            "tire_section_width": "从 tire_size 解析出的轮胎名义断面宽度 (mm)",
            "tire_aspect_ratio": "从 tire_size 解析出的轮胎扁平比 (%)",
        }
        print("[3.2] 批次 B（天然有序/内嵌真数值 7 列）提纯完成，并拆出 tire_section_width 与 tire_aspect_ratio")

        # --- 批次 C（18 列，含 DestinationInfo） & 批次 D（7 列，含 OperatorUserID）+ 保留原字符串的 tire_size（共 26 个类别字符串列） ---
        # 注意：严禁乱编 1,2,3 或频次占比！仅清理首尾空格，原封不动保留干净字符串
        batch_a_cols = ["oeindicator", "1_flange_cushion_-_compound"]
        batch_b_numeric_cols = [
            "1_sidewall_-_bom_component_setting",
            "speed_symbol",
            "1_wound_cap_ply_-_cf_construction",
            "bead_-_bead_construction",
            "ply_lock",
            "article_cluster_link_status",
        ]
        categorical_cols = [
            c for c in all_non_numeric_cols if c not in batch_a_cols + batch_b_numeric_cols
        ]
        for col in categorical_cols:
            df[col] = df[col].astype(str).str.strip()

        # 统计并区分低基数 (Batch C) 与高基数 (Batch D + tire_size)
        cat_unique_counts = {c: int(df[c].nunique()) for c in categorical_cols}
        print(f"[3.3] 批次 C+D 及 tire_size（共 {len(categorical_cols)} 列）已保留干净原始文本字符串，严禁伪数字编码！")

        # ------------------------------------------------------------------
        # 4. 去除完全一模一样的重复行
        # ------------------------------------------------------------------
        before_dup = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed_dup = before_dup - len(df)
        print(f"[4] 已清除完全重复记录: {removed_dup:,} 行（保留 {len(df):,} 行唯一记录）")

        # ------------------------------------------------------------------
        # 5. 整理元数据 schema、导出文件并执行严格断言校验
        # ------------------------------------------------------------------
        numeric_features = [
            c for c in df.columns if c not in categorical_cols and c != self.target_col
        ]
        self.schema["dataset_shape"] = {
            "rows": int(len(df)),
            "total_columns": int(df.shape[1]),
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_cols),
        }
        self.schema["numeric_features"] = numeric_features
        self.schema["categorical_features"] = categorical_cols
        self.schema["categorical_unique_counts"] = cat_unique_counts

        clean_path = os.path.join(self.output_dir, "clean_dataset.csv")
        schema_path = os.path.join(self.output_dir, "feature_schema.json")
        old_dict_path = os.path.join(self.output_dir, "text_encoding_dictionary.json")

        # 写出 clean_dataset.csv 与 feature_schema.json
        df.to_csv(clean_path, index=False, encoding="utf-8-sig")
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(self.schema, f, ensure_ascii=False, indent=2)

        # 删除旧的容易引起误会的 text_encoding_dictionary.json
        if os.path.exists(old_dict_path):
            os.remove(old_dict_path)
            print(f"[5.0] 已移除旧版易产生误导的字典文件: {old_dict_path}")

        # 执行落地后的回读严格自检
        self._verify_outputs(df, clean_path, schema_path, numeric_features, categorical_cols)
        return df

    def _verify_outputs(
        self,
        df_mem: pd.DataFrame,
        clean_path: str,
        schema_path: str,
        numeric_features: list,
        categorical_cols: list,
    ):
        """
        严格断言校验：回读导出的 clean_dataset.csv，确保 0 NaN、0 Inf、类别列数量精确、样例无误。
        """
        print("\n" + "-" * 75)
        print("【严格自检报告 · 逐项断言验证】")

        # 回读时显式将 26 个类别列按 str 读取，防止 RecipeID / OperatorUserID / DestinationInfo 等纯数字代号被误读为整型
        dtype_map = {c: str for c in categorical_cols}
        df_disk = pd.read_csv(clean_path, dtype=dtype_map, low_memory=False)

        # 断言 1：行列数完全一致
        assert df_disk.shape == df_mem.shape, f"磁盘文件尺寸 {df_disk.shape} 与内存 {df_mem.shape} 不一致！"

        # 断言 2：全表 0 NaN
        total_nan = int(df_disk.isna().sum().sum())
        assert total_nan == 0, f"发现 {total_nan} 个 NaN 缺失值！请检查！"

        # 断言 3：所有数值列 + 目标列 0 Inf 且均为数值型
        num_and_target = numeric_features + [self.target_col]
        for col in num_and_target:
            assert pd.api.types.is_numeric_dtype(df_disk[col]), f"数值列 [{col}] 类型异常: {df_disk[col].dtype}"
            inf_cnt = int(np.isinf(df_disk[col]).sum())
            assert inf_cnt == 0, f"数值列 [{col}] 存在 {inf_cnt} 个 Inf！"

        # 断言 4：26 个类别列均为字符串且无空字符串或 'nan'
        assert len(categorical_cols) == 26, f"类别字符串列应为 26 列，当前为 {len(categorical_cols)} 列！"
        assert len(numeric_features) == 49, f"数值特征列应为 49 列，当前为 {len(numeric_features)} 列！"
        for col in categorical_cols:
            s = df_disk[col].astype(str).str.strip()
            assert not s.isin(["", "nan", "None", "NaN"]).any(), f"类别列 [{col}] 存在异常空值！"

        # 断言 5：受保护列与新拆物理列均存在
        for must_col in [
            "1_belt_ply_assembly_-_1_strip_rubber_-_width",
            "tire_size",
            "tire_section_width",
            "tire_aspect_ratio",
            "OperatorUserID",
            "DestinationInfo",
        ]:
            assert must_col in df_disk.columns, f"关键列 [{must_col}] 缺失！"

        print(f"  [PASS] 洁净矩阵尺寸: {df_disk.shape[0]:,} 行 × {df_disk.shape[1]} 列")
        print(f"  [PASS] 目标列 (1 列): {self.target_col} ({df_disk[self.target_col].dtype})")
        print(f"  [PASS] 数值特征列 ({len(numeric_features)} 列): 全部为数值类型，0 个 NaN，0 个 Inf")
        print(f"  [PASS] 无序类别/分组字符串列 ({len(categorical_cols)} 列): 原文保留，0 个空值/NaN")
        print(f"  [PASS] 洁净数据路径: {clean_path}")
        print(f"  [PASS] 特征元数据路径: {schema_path}")

        print("\n【关键物理提纯列 & 类别列样例抽查】")
        sample_cols = [
            "oeindicator",
            "1_flange_cushion_-_compound",
            "1_sidewall_-_bom_component_setting",
            "speed_symbol",
            "1_wound_cap_ply_-_cf_construction",
            "bead_-_bead_construction",
            "ply_lock",
            "article_cluster_link_status",
            "tire_size",
            "tire_section_width",
            "tire_aspect_ratio",
            "RecipeID",
            "1_belt_ply_assembly_-_local_plant_workcode",
            "OperatorUserID",
            "DestinationInfo",
        ]
        for col in sample_cols:
            vals = df_disk[col].unique()[:5].tolist()
            print(f"  • {col:<46} | dtype={str(df_disk[col].dtype):<8} | nunique={df_disk[col].nunique():<4} | sample={vals}")
        print("=" * 75)


if __name__ == "__main__":
    cleaner = IndustrialDataCleaner()
    cleaner.run()
