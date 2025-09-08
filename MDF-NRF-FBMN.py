import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


# ---------------------- 第一步：读取文件并预处理 ----------------------
def read_files():
    """读取理论数据、实验数据和MSP文件，并预处理"""
    try:
        theory_df = pd.read_excel("MDF_Window.xlsx", engine="openpyxl")
        required_columns = {"m/z", "Group", "NRF"}
        if not required_columns.issubset(theory_df.columns):
            missing = required_columns - set(theory_df.columns)
            raise ValueError(f"理论数据缺少必要列: {', '.join(missing)}")
        theory_df["Da"] = theory_df["m/z"].astype(int)
        theory_df["mDa"] = (theory_df["m/z"] - theory_df["Da"]) * 1000

        # 读取MSP文件，提取所有有二级谱的化合物名称
        try:
            with open("ID-FBMN-HQY-month.msp", "r", encoding="utf-8") as f:
                msp_content = [block.strip() for block in f.read().split("\n\n") if block.strip()]
        except FileNotFoundError:
            print("警告: MSP文件未找到，跳过MSP过滤")
            msp_content = []
            valid_compounds = set()
        else:
            # 提取MSP中所有化合物名称
            valid_compounds = set()
            for block in msp_content:
                for line in block.split("\n"):
                    if line.startswith("NAME:") or line.startswith("Comment:"):
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            compound = parts[1].strip()
                            valid_compounds.add(compound)
                        break

        try:
            exp_df = pd.read_csv("ID-FBMN-HQY-month.csv")
        except FileNotFoundError:
            raise FileNotFoundError("实验数据文件未找到")

        if "m/z" not in exp_df.columns or "Compound" not in exp_df.columns:
            raise ValueError("实验数据缺少必要列(m/z, Compound)")

        if valid_compounds:
            exp_df_filtered = exp_df[exp_df["Compound"].str.strip().isin(valid_compounds)].copy()
            original_count = len(exp_df)
            filtered_count = len(exp_df_filtered)
            print(f"实验数据原始总数：{original_count}, MSP过滤保留数：{filtered_count}")
        else:
            exp_df_filtered = exp_df.copy()
            print("警告: 无有效MSP化合物，跳过MSP过滤")

    except Exception as e:
        raise RuntimeError(f"文件读取失败: {str(e)}")

    return theory_df, exp_df_filtered, msp_content


# ---------------------- 第二步：构建MDF窗口并关联NRF规则 ----------------------
def build_mdf_windows(theory_df):
    """构建多分组凸包并关联氮规则过滤要求"""
    windows = {}
    group_nrf_rules = {}  # 存储每个Group的氮规则要求

    for group, group_df in theory_df.groupby("Group"):
        unique_nrf = group_df["NRF"].dropna().unique()

        if len(unique_nrf) == 0:
            print(f"警告: Group '{group}'无有效NRF值，跳过")
            continue
        if len(unique_nrf) > 1:
            print(f"警告: Group '{group}'包含不一致的NRF值: {unique_nrf}，使用第一个值")
        group_nrf_rules[group] = unique_nrf[0]

    # 构建每个Group的凸包窗口
    for group, group_df in theory_df.groupby("Group"):
        if group not in group_nrf_rules:
            continue
        points = group_df[["Da", "mDa"]].values
        if len(points) < 3:
            print(f"警告: Group '{group}'点数不足({len(points)}<3)，无法构建凸包")
            continue

        try:
            hull = ConvexHull(points)
            if len(hull.vertices) == 0:
                print(f"警告: Group '{group}'凸包计算失败，顶点为空")
                continue

            hull_points = group_df.iloc[hull.vertices][["Da", "mDa"]]
            hull_points = pd.concat([hull_points, hull_points.head(1)])
            polygon = Polygon(hull_points.values)
            if polygon.is_empty:
                print(f"警告: Group '{group}'创建的多边形为空")
            else:
                windows[group] = (polygon, group_nrf_rules[group])
        except Exception as e:
            print(f"Group '{group}'凸包计算错误: {str(e)}")

    if not windows:
        raise ValueError("无有效MDF窗口")

    polygons = [poly for poly, _ in windows.values() if not poly.is_empty]
    if not polygons:
        raise ValueError("所有多边形均为空")

    combined_polygon = unary_union(polygons)

    return combined_polygon, windows


# ---------------------- 第三步：复合过滤 ----------------------
def filter_and_save(exp_df, combined_polygon, group_windows, msp_content):
    """MDF过滤 -> 分组氮规则过滤"""
    if exp_df.empty:
        print("警告: 实验数据为空，跳过过滤")
        return
    exp_df = exp_df.copy()
    exp_df["Da"] = exp_df["m/z"].astype(int)
    exp_df["mDa"] = (exp_df["m/z"] - exp_df["Da"]) * 1000

    # 第一阶段：MDF过滤
    points = []
    thresholds = []
    for _, row in exp_df.iterrows():
        mz = row["Da"] + row["mDa"] / 1000
        threshold = mz * 15 / 1e6 * 1000  # 15ppm转mDa
        points.append(Point(row["Da"], row["mDa"]))
        thresholds.append(threshold)

    # 空间过滤：点在多边形内或距离小于阈值
    mask = []
    for i, (p, t) in enumerate(zip(points, thresholds)):
        try:
            if combined_polygon.contains(p) or combined_polygon.distance(p) <= t:
                mask.append(True)
            else:
                mask.append(False)
        except Exception as e:
            print(f"点 {i} 空间关系计算错误: {str(e)}")
            mask.append(False)

    filtered_mdf = exp_df.loc[mask].copy()
    mdf_count = len(filtered_mdf)

    if filtered_mdf.empty:
        print("MDF过滤后无数据，跳过后续处理")
        return

    # 第二阶段：分组氮规则过滤
    # 为每个点确定所属的Group和对应的NRF规则
    filtered_mdf["Group"] = ""
    filtered_mdf["NRF_Rule"] = ""

    for idx, row in filtered_mdf.iterrows():
        point = Point(row["Da"], row["mDa"])
        matched = False

        for group, (polygon, nrf_rule) in group_windows.items():
            try:
                if polygon.contains(point) or polygon.distance(point) <= thresholds[idx]:
                    filtered_mdf.at[idx, "Group"] = group
                    filtered_mdf.at[idx, "NRF_Rule"] = nrf_rule
                    matched = True
                    break
            except Exception as e:
                print(f"点 {idx} 与Group '{group}'空间关系计算错误: {str(e)}")

        if not matched:
            filtered_mdf.at[idx, "Group"] = "Unmatched"
            filtered_mdf.at[idx, "NRF_Rule"] = "Unknown"
    filtered_results = []

    for group, group_df in filtered_mdf.groupby("Group"):
        if group == "Unmatched" or group_df.empty:
            continue

        nrf_rules = group_df["NRF_Rule"].unique()
        if len(nrf_rules) == 0:
            print(f"警告: Group '{group}'无NRF规则，跳过")
            continue

        nrf_rule = nrf_rules[0].lower() if pd.notna(nrf_rules[0]) else "unknown"

        # 根据NRF规则进行过滤
        if nrf_rule == "odd":
            group_filtered = group_df[group_df["Da"] % 2 == 0]
        elif nrf_rule == "even":
            group_filtered = group_df[group_df["Da"] % 2 == 1]
        else:
            print(f"警告: Group '{group}'有未知NRF规则 '{nrf_rule}'，跳过氮规则过滤")
            group_filtered = group_df

        if not group_filtered.empty:
            filtered_results.append(group_filtered)

    # 合并所有分组结果
    if filtered_results:
        filtered_odd = pd.concat(filtered_results)
    else:
        filtered_odd = pd.DataFrame(columns=filtered_mdf.columns)

    odd_count = len(filtered_odd)

    # 统计MSP匹配结果
    compounds_odd = set()
    if not filtered_odd.empty and "Compound" in filtered_odd.columns:
        compounds_odd = set(filtered_odd["Compound"].str.strip().dropna())

    matched_blocks = []
    for block in msp_content:
        for line in block.split("\n"):
            if line.startswith(("NAME:", "Comment:")):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    compound = parts[1].strip()
                    if compound in compounds_odd:
                        matched_blocks.append(block)
                        break

    msp_final_count = len(matched_blocks)

    print(
        f"MDF过滤保留数：{mdf_count}\n"
        f"氮规则过滤保留数：{odd_count}\n"
        f"MSP文件最终保留数：{msp_final_count}"
    )

    if not filtered_odd.empty:
        filtered_odd.to_csv("match_result.csv", index=False)
        print(f"保存匹配结果到 match_result.csv ({len(filtered_odd)}行)")
    else:
        print("警告: 氮规则过滤后无数据可保存")

    if matched_blocks:
        with open("match_result.msp", "w", encoding="utf-8") as f:
            f.write("\n\n".join(matched_blocks))
        print(f"保存匹配谱图到 match_result.msp ({len(matched_blocks)}个谱图)")
    else:
        print("警告: 无匹配的MSP谱图块可保存")


# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    try:
        print("开始处理数据...")
        theory_df, exp_df, msp_content = read_files()
        print("理论数据读取成功，开始构建MDF窗口...")
        combined_polygon, group_windows = build_mdf_windows(theory_df)
        print(f"构建MDF窗口完成，共{len(group_windows)}个分组")
        print("开始复合过滤...")
        filter_and_save(exp_df, combined_polygon, group_windows, msp_content)
        print("处理完成")
    except Exception as e:
        import traceback

        print(f"运行错误: {str(e)}")
        print(traceback.format_exc())