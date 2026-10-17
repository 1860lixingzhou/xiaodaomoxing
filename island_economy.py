# -*- coding: utf-8 -*-
"""
小岛经济投入产出与货币收益模拟器
================================

经济结构: 3 个部门 —— 农业(1)、工业/铁匠(2)、矿业(3)

核心逻辑:
    各部门为生产需要互相消耗实物(中间投入), 同时满足岛外外部需求(出口)。
    先由实物投入产出表反推消耗系数矩阵 C, 再解平衡总产出 x,
    最后叠加价格体系 p, 得到出口总收入与各行业增加值(GDP/利润)。

    数学公式:
        C    = 各行业消耗量 / 各行业基础产出量      (3x3 消耗系数矩阵)
        x    = (E - C)^(-1) d                      (平衡总产出向量)
        Z    = C @ diag(x)                          (实物中间投入矩阵)
        出口总收入      = sum(d * p)
        行业总产出价值  = x * p
        行业中间投入价值 = p @ Z
        行业增加值      = 行业总产出价值 - 行业中间投入价值
        岛内总 GDP      = sum(行业增加值)

运行方式:
    # 方式一(本机已装好依赖): 双击/执行 run_app.ps1
    .\run_app.ps1

    # 方式二(手动):
    $env:PYTHONPATH = "$(Get-Location)\streamlit_env"   # 指向本地依赖目录
    python -m streamlit run island_economy.py
"""

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# 页面基础配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="小岛经济投入产出模拟器",
    page_icon="🏝️",
    layout="wide",
)

# 三个部门的中文名称, 用于表格展示
SECTORS = ["农业", "工业/铁匠", "矿业"]


# ===========================================================================
# 一、核心后台算法(与界面完全解耦, 便于单独测试)
# ===========================================================================

def build_consumption_matrix(ag_self, ag_ind, ag_min,
                             in_ag, in_self, in_min,
                             mi_ag, mi_ind, mi_self,
                             agriculture_output, industrial_output, mining_output):
    """
    第一步: 反推消耗系数矩阵 C (3x3)。

    消耗系数 C[i][j] 表示「部门 j 每生产 1 单位, 需要消耗部门 i 的数量」。
    即:  C[i][j] = 部门 j 对部门 i 的消耗量 / 部门 j 的基础产出量。

    约定:
        - 行 = 被消耗方(投入来源, 顺序: 农业/工业/矿业)
        - 列 = 消耗方(产出部门, 顺序: 农业/工业/矿业)
        - 农业消耗工业量(ag_ind) -> C[工业][农业] = ag_ind / 农业产出
        - 工业消耗矿业量(in_min) -> C[矿业][工业] = in_min / 工业产出

    参数: 9 个实物消耗量(ag_* 农业部门消耗, in_* 工业部门消耗, mi_* 矿业部门消耗,
          后缀 self/ag/ind/min 分别表示消耗自身/农业品/工业品/矿产品),
          以及 3 个部门的基础产出量。

    返回
    ----
    C : np.ndarray, shape (3, 3)
    """
    outputs = np.array([agriculture_output, industrial_output, mining_output],
                       dtype=float)

    if np.any(outputs <= 0):
        raise ValueError("各部门的基础产出量必须为正数, 才能计算消耗系数。")

    # 按「列 = 消耗方」组织: 每列是该部门对各投入来源(行)的消耗量
    # 列 0 = 农业消耗: 自耗 ag_self, 耗工业 ag_ind, 耗矿业 ag_min
    # 列 1 = 工业消耗: 耗农业 in_ag, 自耗 in_self, 耗矿业 in_min
    # 列 2 = 矿业消耗: 耗农业 mi_ag, 耗工业 mi_ind, 自耗 mi_self
    consumption = np.array([
        [ag_self, in_ag,  mi_ag ],   # 农业品被各行业消耗
        [ag_ind,  in_self, mi_ind],  # 工业品被各行业消耗
        [ag_min,  in_min,  mi_self], # 矿产品被各行业消耗
    ], dtype=float)

    # 列归一化: 每列除以该列对应部门的产出量 -> 消耗系数矩阵 C
    C = consumption / outputs          # 广播: (3,3) / (3,) 逐列相除
    return C


def solve_economy(C, d, p):
    """
    第二步~第四步: 完成从实物账到货币账的全部计算。

    参数
    ----
    C : np.ndarray, shape (3, 3)  消耗系数矩阵
    d : array_like, shape (3,)    外部需求向量(出口量)
    p : array_like, shape (3,)    价格向量

    返回
    ----
    dict, 包含以下键:
        C, d, p                输入矩阵/向量
        L                      E - C (列昂惕夫矩阵)
        det, cond              行列式(可逆性)与条件数(数值稳定性)
        x                      平衡总产出向量 (3,)
        Z                      实物中间投入矩阵 (3,3)
        export_revenue         出口总收入(标量)
        output_value           行业总产出价值 (3,)
        intermediate_value     行业中间投入价值 (3,)
        value_added            行业增加值 (3,)
        gdp                    岛内总 GDP(标量)
        physical_residual      实物平衡恒等式残差 x - (d + Z^T·1)
        monetary_residual      货币核算恒等式残差 gdp - Σ(d·p)
    """
    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)

    n = 3
    # 维度校验, 避免广播错误
    if C.shape != (n, n):
        raise ValueError(f"消耗系数矩阵 C 必须为 3x3, 当前形状 {C.shape}")
    if d.shape != (n,):
        raise ValueError(f"外部需求向量 d 必须为长度 3, 当前长度 {d.size}")
    if p.shape != (n,):
        raise ValueError(f"价格向量 p 必须为长度 3, 当前长度 {p.size}")

    E = np.eye(n)                # 3x3 单位矩阵
    L = E - C                    # 列昂惕夫矩阵 (Leontief matrix)

    # ---- 防崩溃校验: 行列式过小 -> 矩阵近似不可逆 ----
    det = np.linalg.det(L)
    if abs(det) < 1e-6:
        raise ValueError("经济系统参数不满足矩阵可逆条件，技术系数相互矛盾，无法计算")

    # ---- 求解平衡总产出 x = (E - C)^(-1) d ----
    # 使用 np.linalg.solve 数值更稳定; 允许 x 出现负数(物理失衡), 不做任何拦截
    x = np.linalg.solve(L, d)

    # ---- 实物中间投入矩阵 Z = C @ diag(x) ----
    # Z[i][j] = 部门 j 生产中实际消耗部门 i 的实物量
    Z = C @ np.diag(x)

    # ---- 货币账本 ----
    export_revenue = float(np.sum(d * p))          # 岛外出口总收入(标量)
    output_value = x * p                          # 行业总产出价值 (3,)
    intermediate_value = p @ Z                    # 行业中间投入价值 (3,) = (1x3)@(3x3)
    value_added = output_value - intermediate_value   # 行业增加值(利润, 允许负)
    gdp = float(np.sum(value_added))              # 岛内总 GDP(标量)

    # ---- 内部核算校验(供界面展示, 也是结果正确性的自检) ----
    # 实物平衡恒等式: x_j = d_j + Σ_k Z[j][k]
    #   (部门 j 的总产出 = 出口 j 的数量 + j 产品被各行业作为中间投入买走的数量,
    #    即 Z 的行和; 由 (E-C)x = d 移项 x = d + Cx = d + Z·1 保证)
    physical_residual = x - (d + Z.sum(axis=1))
    # 货币核算恒等式: Σ行业增加值 = Σ(d * p) (中间投入在岛内互相抵消)
    monetary_residual = gdp - export_revenue
    # 条件数: 衡量解的数值稳定性(越大越不稳定)
    cond = float(np.linalg.cond(L))

    return {
        "C": C, "d": d, "p": p,
        "L": L, "det": det, "cond": cond,
        "x": x, "Z": Z,
        "export_revenue": export_revenue,
        "output_value": output_value,
        "intermediate_value": intermediate_value,
        "value_added": value_added,
        "gdp": gdp,
        "physical_residual": physical_residual,
        "monetary_residual": monetary_residual,
    }


def classify_status(x, value_added, eps=1e-9):
    """
    根据平衡总产出 x 与增加值, 判定每个部门的系统状态。

    优先级(物理失衡 > 亏损 > 健康):
        x < -eps                     -> 🔴 物理失衡 (倒贴生产)
        增加值 < -eps 且 x >= -eps    -> 🟡 亏损运营
        增加值 >= -eps               -> 🟢 健康盈利

    eps 为浮点容差: 数值噪声级的小负数(如 -1e-15)不误报失衡/亏损。
    """
    status = []
    for xi, va in zip(x, value_added):
        if xi < -eps:
            status.append("🔴 物理失衡 (倒贴生产)")
        elif va < -eps:
            status.append("🟡 亏损运营")
        else:
            status.append("🟢 健康盈利")
    return status


# ===========================================================================
# 二、界面输入区
# ===========================================================================

st.title("🏝️ 小岛经济投入产出与货币收益模拟器")
st.caption(
    "农业 · 工业/铁匠 · 矿业 三部门模型 | 先算实物平衡产出, 再算货币收益(GDP)。"
    " 允许出现负产出(倒贴生产)与负增加值(亏损), 用于观察物理失衡。"
)

with st.expander("📋 模型说明与公式", expanded=False):
    st.markdown(
        """
        **消耗系数矩阵** `C[i][j]` = 部门 j 每生产 1 单位所消耗部门 i 的数量：
        `C = 消耗量 / 基础产出量`（按列归一化）。

        **平衡总产出**：`x = (E - C)⁻¹ d`，其中 `E` 为单位矩阵，`d` 为外部需求（出口）。

        **实物中间投入**：`Z = C · diag(x)`，表示各部门之间的实际实物流转量。

        **货币账本**：
        - 出口总收入 = `Σ(d × p)`
        - 行业总产出价值 = `x × p`
        - 行业中间投入价值 = `p · Z`
        - 行业增加值 = 总产出价值 − 中间投入价值
        - 岛内 GDP = `Σ` 行业增加值
        """
    )

col_left, col_right = st.columns([1, 1])

# ---- 左侧: 基础产出与消耗 ----
with col_left:
    st.subheader("① 基础产出与实物消耗")

    st.markdown("**农业（产出：粮食）**")
    a1, a2, a3, a4 = st.columns(4)
    agriculture_output = a1.number_input("农业基础产出量", 0.0, 1e9, 100.0, step=1.0, key="ag_out")
    ag_self = a2.number_input("农业消耗农业（种子/口粮）", 0.0, 1e9, 20.0, step=1.0, key="ag_self")
    ag_ind = a3.number_input("农业消耗工业（铁锹）", 0.0, 1e9, 1.0, step=1.0, key="ag_ind")
    ag_min = a4.number_input("农业消耗矿业", 0.0, 1e9, 0.0, step=1.0, key="ag_min")

    st.markdown("**工业/铁匠（产出：铁锹）**")
    i1, i2, i3, i4 = st.columns(4)
    industrial_output = i1.number_input("工业基础产出量", 0.0, 1e9, 10.0, step=1.0, key="in_out")
    in_ag = i2.number_input("工业消耗农业（工人吃粮）", 0.0, 1e9, 10.0, step=1.0, key="in_ag")
    in_self = i3.number_input("工业消耗工业", 0.0, 1e9, 0.0, step=1.0, key="in_self")
    in_min = i4.number_input("工业消耗矿业（煤炭）", 0.0, 1e9, 5.0, step=1.0, key="in_min")

    st.markdown("**矿业（产出：煤炭）**")
    m1, m2, m3, m4 = st.columns(4)
    mining_output = m1.number_input("矿业基础产出量", 0.0, 1e9, 50.0, step=1.0, key="mi_out")
    mi_ag = m2.number_input("矿业消耗农业（矿工吃粮）", 0.0, 1e9, 15.0, step=1.0, key="mi_ag")
    mi_ind = m3.number_input("矿业消耗工业（工具损耗）", 0.0, 1e9, 1.0, step=1.0, key="mi_ind")
    mi_self = m4.number_input("矿业消耗矿业", 0.0, 1e9, 0.0, step=1.0, key="mi_self")

# ---- 右侧: 外部需求 + 价格体系 ----
with col_right:
    st.subheader("② 外部需求（岛外出口量）")
    d1, d2, d3 = st.columns(3)
    d_ag = d1.number_input("出口粮食", 0.0, 1e9, 30.0, step=1.0, key="d_ag")
    d_in = d2.number_input("出口铁锹", 0.0, 1e9, 2.0, step=1.0, key="d_in")
    d_mi = d3.number_input("出口煤炭", 0.0, 1e9, 10.0, step=1.0, key="d_mi")

    st.subheader("③ 价格体系")
    p1, p2, p3 = st.columns(3)
    p_ag = p1.number_input("每斤粮食价格", 0.0, 1e9, 2.0, step=1.0, key="p_ag")
    p_in = p2.number_input("每把铁锹价格", 0.0, 1e9, 50.0, step=1.0, key="p_in")
    p_mi = p3.number_input("每吨煤炭价格", 0.0, 1e9, 100.0, step=1.0, key="p_mi")


# ===========================================================================
# 三、组装输入数据并执行计算
# ===========================================================================

# 9 个实物消耗量直接来自界面输入; 组装顺序按「列=消耗部门」在
# build_consumption_matrix 内部显式排布(见该函数 docstring)。
d = np.array([d_ag, d_in, d_mi], dtype=float)   # 外部需求向量
p = np.array([p_ag, p_in, p_mi], dtype=float)   # 价格向量

st.divider()
st.subheader("④ 计算结果")

try:
    # ---- 第一步: 反推消耗系数矩阵 C ----
    C = build_consumption_matrix(
        ag_self, ag_ind, ag_min,
        in_ag, in_self, in_min,
        mi_ag, mi_ind, mi_self,
        agriculture_output, industrial_output, mining_output,
    )

    # ---- 第二~四步: 完整求解(可逆性校验、平衡产出、货币账本) ----
    result = solve_economy(C, d, p)

except (ValueError, np.linalg.LinAlgError) as e:
    # 矩阵不可逆 / 参数非法 / 数值求解失败: 显式报错, 程序不崩溃
    st.error(f"⚠️ {e}")
    st.stop()

# ===========================================================================
# 四、结果展示与预警
# ===========================================================================

x = result["x"]
value_added = result["value_added"]
status = classify_status(x, value_added)

# ---- 总体预警: 只要存在负数(失衡或亏损), 就在最上方给出警告 ----
# 同样使用容差, 避免浮点噪声误触发
EPS = 1e-9
has_problem = bool(np.any(x < -EPS) or np.any(value_added < -EPS))
if has_problem:
    st.warning("⚠️ 当前经济系统存在亏损或物理失衡状态，请查看下方报表。")

# ---- 核心指标大字展示 ----
m1, m2, m3 = st.columns(3)
m1.metric("💵 岛外出口总收入", f"{result['export_revenue']:,.2f}")
m2.metric("📈 岛内增加值 GDP", f"{result['gdp']:,.2f}")
m3.metric("🧮 det(E−C)", f"{result['det']:.6f}")

# ---- 完整账本表(含系统状态预警列) ----
ledger = pd.DataFrame({
    "行业名称": SECTORS,
    "平衡总产出": np.round(x, 4),
    "总产出价值": np.round(result["output_value"], 2),
    "中间投入成本": np.round(result["intermediate_value"], 2),
    "增加值(利润)": np.round(value_added, 2),
    "系统状态": status,
})
st.dataframe(ledger, width="stretch", hide_index=True)

# ---- 增加值对比柱状图(支持负数柱子) ----
chart_df = pd.DataFrame({"行业": SECTORS, "增加值": value_added})
chart_df = chart_df.set_index("行业")
st.subheader("各行业增加值对比")
st.bar_chart(chart_df, color="#1f77b4")

# ===========================================================================
# 五、明细附录(消耗系数矩阵、中间投入矩阵、核算恒等式校验)
# ===========================================================================

with st.expander("🔍 查看计算明细与核算校验", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**消耗系数矩阵 C**（行=投入来源, 列=产出部门）")
        st.dataframe(pd.DataFrame(C, index=SECTORS, columns=SECTORS),
                     width="stretch")
    with c2:
        st.markdown("**列昂惕夫矩阵 E − C**")
        st.dataframe(pd.DataFrame(result["L"], index=SECTORS, columns=SECTORS),
                     width="stretch")

    st.markdown("**平衡总产出向量 x**")
    st.dataframe(pd.DataFrame({"行业": SECTORS, "x": np.round(x, 4)}),
                 width="stretch", hide_index=True)

    st.markdown("**实物中间投入矩阵 Z = C · diag(x)**（行=投入来源, 列=产出部门）")
    st.dataframe(pd.DataFrame(np.round(result["Z"], 4),
                              index=SECTORS, columns=SECTORS),
                 width="stretch")

    # 核算恒等式校验(自检结果正确性):
    #   ① 实物平衡: x_j = d_j + Σ_i Z[i][j], 即 总产出 = 最终需求 + 中间需求
    #   ② 货币核算: Σ行业增加值 = Σ(d·p), 因中间投入在岛内互相抵消
    #   ③ 条件数: 衡量解 x 的数值稳定性(远大于 1e10 时结果需谨慎)
    phys = result["physical_residual"]
    st.markdown(
        f"**① 实物平衡校验**：x − (d + 中间需求) = "
        f"`{phys[0]:+.2e}, {phys[1]:+.2e}, {phys[2]:+.2e}` "
        f"（最大残差 `{np.max(np.abs(phys)):.2e}`，应接近 0）。"
    )
    st.markdown(
        f"**② 货币核算校验**：Σ行业增加值 − 出口总收入 = "
        f"`{result['monetary_residual']:.2e}`（应接近 0，因各部门中间投入在岛内互相抵消）。"
    )
    st.markdown(f"**③ 矩阵条件数**：cond(E−C) = `{result['cond']:.3e}`")
