# -*- coding: utf-8 -*-
"""文章C1 图 2：ROC 曲线（只作图，不做新分析）
=========================================================================
- 参照与纳入规则直接复用 c04_evaluate（c4.D、c4.read、c4.TARGETS["IRF"]）：gradable == "yes"；C1–C3 = 1，C0 = 0，其余排除。
- 面板 a：OCTDL 测试组；面板 b：外部 DME 合并（TowardPi + OLIVES）。ROC 用 prob_IRF 对 C≥1。
- 分类模型工作点：主阈值（OCTDL 设计组 Youden，按 c04 同一函数 c4.youden 复算精确值，并与 xlsx「阈值」表 0.7021 核对）实心；
  固定 0.5 空心。
- 7 个大模型工作点：取 c04 输出 xlsx「与大模型比较」表中「主分析：IRF→C≥1」对应评估集的「大模型判出」「大模型特异」换算。
- 自查：n、阳性/阴性数、AUC（两位小数）、主阈值点的判出/特异须与 c04 一致，否则报错退出、不写任何文件。
- 输出：结果\\figures\\ 下 Fig2_ROC.svg/.pdf/.tiff(600 dpi)/_preview.png(200 dpi) 与 fig2_roc_points.csv（只含汇总曲线点）。
  任一同名文件已存在即停止，不覆盖。

用法：python c05_fig2_roc.py
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # 不在工作目录生成 __pycache__ 文件

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c04_evaluate as c4  # noqa: E402

PREDS = HERE / "runs" / "final_cnxt_lr1e-4_bce_e2" / "preds_external.csv"
XLSX = HERE / "结果" / "分类模型_外部测试_20261112.xlsx"
OUTDIR = HERE / "结果" / "figures"
VER = "_v3"  # v3：在 v2 基础上只改图中文字（千问标注、面板 b 标题、标题第二行），数据与画法不变
OUTS = {k: OUTDIR / v for k, v in dict(svg=f"Fig2_ROC{VER}.svg", pdf=f"Fig2_ROC{VER}.pdf", tiff=f"Fig2_ROC{VER}.tiff",
                                         png=f"Fig2_ROC{VER}_preview.png", csv=f"fig2_roc_points{VER}.csv").items()}
LABEL = "主分析：IRF→C≥1"
PANELS = [  # (面板, c04 评估集名, 图中标题, 期望 n/阳/阴, 期望 AUC, 期望主阈值判出/特异)
    ("a", "OCTDL 测试组（主要）", "OCTDL test set", (74, 60, 14), 0.83, ("56/60", "8/14")),
    ("b", "外部 DME 合并", "Combined external set (TowardPi + OLIVES TREX-DME)", (143, 114, 29), 0.98, ("104/114", "26/29")),
]
QWEN = "千问"
C_QWEN, C_GREY, C_MODEL = "#1f5fa8", "#6b6b6b", "black"


def frac(s):
    a, b = s.split("/")
    return int(a), int(b)


def roc_points(y, p):
    """按概率从高到低逐个阈值计算 (fpr, tpr, threshold)；首点 (0,0) 阈值为 inf。"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    P, N = (y == 1).sum(), (y == 0).sum()
    thr = np.unique(p)[::-1]
    tpr = [0.0] + [((p >= t) & (y == 1)).sum() / P for t in thr]
    fpr = [0.0] + [((p >= t) & (y == 0)).sum() / N for t in thr]
    return np.array(fpr), np.array(tpr), np.concatenate([[np.inf], thr])


def main():
    exist = [str(p) for p in OUTS.values() if p.exists()]
    if exist:
        sys.exit("!! 以下输出文件已存在，停止，不覆盖：\n" + "\n".join(exist))

    # ---------- 预测（与 c04 相同的读法） ----------
    pr = pd.read_csv(PREDS, encoding="utf-8-sig", dtype={"image_id": str})
    prob = {s: g.set_index("image_id") for s, g in pr.groupby("set")}
    all_prob = pd.concat([g.drop(columns="set") for s, g in prob.items() if s != "dev_octdl"])
    all_prob = all_prob[~all_prob.index.duplicated()]

    # ---------- 主阈值：与 c04 第 151–154 行完全相同的计算 ----------
    field, fn = c4.TARGETS["IRF"]
    dg = c4.read(c4.D["dev_gold"]); dg = dg[dg.gradable == "yes"].set_index("image_id")
    t_main, _ = c4.youden(dg.C.map(fn).values, prob["dev_octdl"].loc[dg.index].prob_IRF.values)
    xl_thr = pd.read_excel(XLSX, sheet_name="阈值")
    xl_t = float(xl_thr.loc[xl_thr["阈值来源"] == "OCTDL 设计组 Youden（主分析）", "阈值"].iloc[0])
    if round(t_main, 4) != round(xl_t, 4):
        sys.exit(f"!! 复算主阈值 {t_main:.6f} 与 xlsx 阈值 {xl_t} 不一致，停止。")

    # ---------- 参照（与 c04 第 175–179 行相同） ----------
    tr = c4.read(c4.D["test_ref"]); tr = tr[tr.gradable == "yes"].set_index("image_id")
    er = c4.read(c4.D["ext_ref"]); er = er[er.gradable == "yes"].set_index("image_id")
    refs = {"OCTDL 测试组（主要）": tr, "外部 DME 合并": er[er.dataset.isin(["TowardPi", "OLIVES"])]}

    # ---------- 大模型工作点（只读 c04 输出） ----------
    cmp_ = pd.read_excel(XLSX, sheet_name="与大模型比较", dtype=str)
    cmp_ = cmp_[cmp_["分析"] == LABEL]

    data, rows, report = [], [], []
    for pan, sname, title, (en, ep, eneg), eauc, (ehit, espec) in PANELS:
        ref = refs[sname]
        ids = [i for i in ref.index if i in all_prob.index and fn(ref.loc[i, field]) == fn(ref.loc[i, field])]
        y = ref.loc[ids, field].map(fn).values.astype(float)
        p = all_prob.loc[ids, "prob_IRF"].values.astype(float)
        n, P, N = len(y), int((y == 1).sum()), int((y == 0).sum())
        a = c4.auc(y, p)
        pts = {}
        for tn, t in (("main", t_main), ("0.5", 0.5)):
            pred = p >= t
            hit, sp = int((pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
            pts[tn] = dict(t=t, hit=hit, sp=sp, sens=hit / P, spec=sp / N)
        errs = []
        if (n, P, N) != (en, ep, eneg):
            errs.append(f"n/阳/阴 = {n}/{P}/{N}，期望 {en}/{ep}/{eneg}")
        if f"{a:.2f}" != f"{eauc:.2f}":
            errs.append(f"AUC = {a:.4f}，期望 {eauc:.2f}")
        if (f"{pts['main']['hit']}/{P}", f"{pts['main']['sp']}/{N}") != (ehit, espec):
            errs.append(f"主阈值点 {pts['main']['hit']}/{P}、{pts['main']['sp']}/{N}，期望 {ehit}、{espec}")
        sub = cmp_[cmp_["评估集"] == sname]
        if len(sub) != 7:
            errs.append(f"xlsx 中该评估集的大模型行数 = {len(sub)}，期望 7")
        if errs:
            sys.exit(f"!! 面板 {pan}（{sname}）自查不通过，停止，未写任何文件：\n  " + "\n  ".join(errs))
        llm = []
        for _, r in sub.iterrows():
            h, hp = frac(r["大模型判出"]); s, sn = frac(r["大模型特异"])
            llm.append((r["大模型"], h / hp, 1 - s / sn))
        fpr, tpr, thr = roc_points(y, p)
        for f_, t_, th in zip(fpr, tpr, thr):
            rows.append(dict(panel=pan, fpr=round(f_, 6), tpr=round(t_, 6), threshold=th))
        data.append(dict(pan=pan, title=title, n=n, P=P, N=N, auc=a, fpr=fpr, tpr=tpr, pts=pts, llm=llm))
        report.append(f"面板 {pan}（{sname}）：n={n}，阳性 {P}/阴性 {N}，AUC={a:.4f}（{a:.2f}）；"
                      f"阈值 {t_main:.4f}：敏感度 {pts['main']['hit']}/{P}={pts['main']['sens']:.3f}，"
                      f"特异度 {pts['main']['sp']}/{N}={pts['main']['spec']:.3f}；"
                      f"阈值 0.5：敏感度 {pts['0.5']['hit']}/{P}={pts['0.5']['sens']:.3f}，"
                      f"特异度 {pts['0.5']['sp']}/{N}={pts['0.5']['spec']:.3f}")

    # ---------- 作图 ----------
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    mm = 1 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(174 * mm, 80 * mm))
    for ax, d in zip(axes, data):
        ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color="#b0b0b0", zorder=1)
        ax.plot(d["fpr"], d["tpr"], color=C_MODEL, lw=1.0, zorder=2, label="Classifier ROC", clip_on=False)
        for name, sens, fpr1 in d["llm"]:
            if name != QWEN:
                ax.scatter(fpr1, sens, s=14, marker="^", color=C_GREY, edgecolor="none", zorder=3, clip_on=False)
        for name, sens, fpr1 in d["llm"]:
            if name == QWEN:
                ax.scatter(fpr1, sens, s=18, marker="^", color=C_QWEN, edgecolor="none", zorder=4, clip_on=False)
                ax.annotate("qwen3.8-max", (fpr1, sens), xytext=(8, -12), textcoords="offset points", color=C_QWEN,
                            fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.5, color=C_QWEN))
        m, h = d["pts"]["main"], d["pts"]["0.5"]
        ax.scatter(1 - m["spec"], m["sens"], s=22, marker="o", color=C_MODEL, zorder=5, clip_on=False)
        ax.scatter(1 - h["spec"], h["sens"], s=22, marker="o", facecolor="white", edgecolor=C_MODEL, lw=0.8,
                   zorder=5, clip_on=False)
        ax.text(0.97, 0.05, f"AUC = {d['auc']:.2f}", ha="right", va="bottom", transform=ax.transAxes, fontsize=7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.0); ax.set_aspect("equal")
        ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
        ax.set_title(f"{d['title']}\nn = {d['n']} ({d['P']} C ≥1, {d['N']} C0)", fontsize=7, pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.22, 1.10, d["pan"], transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C_MODEL, lw=1.0, label="Classifier ROC"),
               Line2D([], [], ls="", marker="o", color=C_MODEL, ms=4, label=f"Classifier, threshold {t_main:.3f}"),
               Line2D([], [], ls="", marker="o", mfc="white", mec=C_MODEL, ms=4, label="Classifier, threshold 0.5"),
               Line2D([], [], ls="", marker="^", color=C_QWEN, ms=4.5, label="qwen3.8-max"),
               Line2D([], [], ls="", marker="^", color=C_GREY, ms=4, label="Other 6 MLLMs")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.25, wspace=0.35)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTS["svg"]); fig.savefig(OUTS["pdf"])
    fig.savefig(OUTS["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUTS["png"], dpi=200)
    pd.DataFrame(rows).to_csv(OUTS["csv"], index=False, encoding="utf-8-sig")
    print(f"主阈值（c4.youden 复算）= {t_main:.6f}；xlsx 阈值 = {xl_t}")
    print("\n".join(report))
    for name, d in (("a", data[0]), ("b", data[1])):
        print(f"面板 {name} 大模型工作点（敏感度, 1−特异度）：" +
              "；".join(f"{nm} ({s:.3f}, {f:.3f})" for nm, s, f in d["llm"]))
    print("自查全部通过。已保存：\n" + "\n".join(str(p) for p in OUTS.values()))


if __name__ == "__main__":
    main()
