# 🏝️ 小岛经济投入产出与货币收益模拟器

一个基于 **投入产出模型（Input-Output / Leontief Model）** 的 3 部门经济模拟器：
农业、工业/铁匠、矿业。调整实物消耗、出口需求与价格，实时查看平衡总产出、
出口总收入、各行业增加值（GDP/利润），并自动诊断经济系统状态。

## 功能

- **实物账**：由 9 个部门间消耗量与 3 个基础产出反推消耗系数矩阵 `C`，
  解线性方程组 `(E − C)·x = d` 得到平衡总产出 `x`
- **货币账**：叠加价格向量 `p`，计算出口总收入、行业总产出价值、中间投入成本、增加值与总 GDP
- **状态预警**：
  - 🔴 物理失衡（倒推出现负产出，倒贴生产）
  - 🟡 亏损运营（有产出但增加值为负）
  - 🟢 健康盈利
- **自检**：实物平衡恒等式 `xⱼ = dⱼ + Σ Z[j][k]`、货币核算恒等式 `Σ增加值 = Σ(d·p)`、条件数

## 数学框架

```
C[i][j] = 部门 j 对部门 i 的消耗量 / 部门 j 的基础产出量
x       = (E − C)⁻¹ d          # 平衡总产出
Z       = C · diag(x)          # 实物中间投入矩阵
出口总收入      = Σ(d × p)
行业总产出价值  = x × p
行业中间投入价值 = p · Z
行业增加值      = 总产出价值 − 中间投入成本
岛内总 GDP      = Σ 行业增加值
```

## 本地运行

```bash
pip install -r requirements.txt
streamlit run island_economy.py
```

浏览器打开 http://localhost:8501 即可。

## 部署到 Streamlit Community Cloud

1. 在 GitHub 新建一个仓库（如 `island-economy-simulator`，Public 或 Private 均可）
2. 把本目录的 4 个文件上传到仓库根目录：
   `island_economy.py`、`requirements.txt`、`README.md`、`.gitignore`
3. 打开 <https://share.streamlit.io> → **Create app** → 选择你的仓库、分支 `main`
4. **Main file path** 填：`island_economy.py`
5. 点 **Deploy**，等待构建完成即可得到公开访问链接

> 注意：不要上传本地的 `streamlit_env/` 依赖目录（体积极大且已被 `.gitignore` 排除）。
> Streamlit Cloud 会按 `requirements.txt` 自动安装依赖。

## 文件说明

| 文件 | 作用 |
|---|---|
| `island_economy.py` | 主程序（算法函数 + Streamlit 界面） |
| `requirements.txt` | 云端依赖清单 |
| `README.md` | 本说明 |
| `.gitignore` | 排除本地依赖与缓存 |
