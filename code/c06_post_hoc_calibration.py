# -*- coding: utf-8 -*-
"""c06（本人电脑上运行，只读）：登记外的事后描述分析，2026-11-12 杨医生同意
=========================================================================
不重跑 c03/c04，不改任何已有文件，只读：
  1. 从 c04 结果表（分类模型_外部测试_20261112.xlsx）取出主阈值下的敏感度、特异度及其 95% CI
     （c04 已用聚类 bootstrap 2000 次、seed 20261023 算好，这里只是摘出来）。
  2. 参照标准中 ESASO C 分级（C0/C1/C2/C3）在各评估集的分布（只计数）。
  3. IRF 概率对 C≥1 的校准（calibration）：平均预测概率与实际患病率、校准截距（斜率固定为 1）、
     校准斜率、Brier 分数；95% CI 用与 c04 相同的聚类 bootstrap（2000 次，seed 20261023）；
     另画可靠性图（Supplementary Fig. S2）。
第 2、3 项不在 OSF 登记中，稿件中须标明为事后（post hoc）描述分析，不涉及任何假设检验。

用法（Miniforge Prompt，conda activate dme）：
  cd /d "%USERPROFILE%\\Desktop\\文献 DME AI\\08_文章C分类模型"
  python c06_post_hoc_calibration.py --preds runs\\final\\preds_external.csv
输出：结果\\分类模型_事后校准_20261112.xlsx 和 结果\\FigS2_calibration.{png,pdf,svg}
屏幕上只打印汇总数字（没有逐图参照），请把屏幕输出整段复制给 Claude。
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c04_evaluate import D, TARGETS, read, boot_idx, ci, SEED  # noqa: E402

FN = TARGETS["IRF"][1]
EPS = 1e-6


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_logistic(x, y, offset=None, slope=True, iters=50):
    """y ~ a + b*x（slope=True）或 y ~ a + offset（slope=False）；牛顿法，返回 (a, b)。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    off = np.zeros_like(x) if offset is None else np.asarray(offset, float)
    X = np.column_stack([np.ones_like(x), x]) if slope else np.ones((len(x), 1))
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta + off
        mu = 1 / (1 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return (beta[0], beta[1]) if slope else (beta[0], 1.0)


def calib(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    lp = logit(p)
    out = dict(n=len(y), 患病率=y.mean(), 平均预测概率=p.mean(), Brier=np.mean((p - y) ** 2))
    if 0 < y.sum() < len(y):
        out["校准截距"] = fit_logistic(lp, y, offset=lp, slope=False)[0]
        out["校准斜率"] = fit_logistic(lp, y)[1]
    else:
        out["校准截距"] = out["校准斜率"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--c04-xlsx", default=str(HERE / "结果" / "分类模型_外部测试_20261112.xlsx"))
    ap.add_argument("--out", default=str(HERE / "结果" / "分类模型_事后校准_20261112.xlsx"))
    for k in ("dev_gold", "test_ref", "ext_ref"):
        ap.add_argument("--" + k.replace("_", "-"), default=str(D[k]))
    a = ap.parse_args()
    inputs = [a.preds, a.c04_xlsx, a.dev_gold, a.test_ref, a.ext_ref]
    before = {p: sha(p) for p in inputs}
    print("== 输入文件 SHA-256（运行前）=="); [print(v[:16], p) for p, v in before.items()]

    # ---- 1. c04 已有的敏感度/特异度 CI ----
    res = pd.read_excel(a.c04_xlsx, sheet_name="结果")
    main = res[(res["分析"].str.contains("IRF")) & (res["阈值"].astype(str).str.startswith("设计组Youden"))]
    sens = main[["评估集", "阈值", "n", "敏感度", "特异度", "判出", "特异"]]
    print("\n== 1. c04 结果表中主阈值的敏感度、特异度（95% CI，聚类 bootstrap）==")
    print(sens.to_string(index=False))

    # ---- 参照与评估集（与 c04 相同的取法）----
    pr = pd.read_csv(a.preds, encoding="utf-8-sig", dtype={"image_id": str})
    prob = {s: g.set_index("image_id") for s, g in pr.groupby("set")}
    dg_all = read(a.dev_gold)
    tr_all = read(a.test_ref); er_all = read(a.ext_ref)
    tr = tr_all[tr_all.gradable == "yes"].set_index("image_id"); tr["cluster"] = "octdl_" + tr.patient_id
    er = er_all[er_all.gradable == "yes"].set_index("image_id")
    er["cluster"] = er.dataset + "_" + er.patient + "_" + er.eye.where(er.dataset != "OCTDL_NO", "")
    sets = {"OCTDL 测试组（主要）": tr, "OCTDL 正常眼": er[er.dataset == "OCTDL_NO"], "TowardPi": er[er.dataset == "TowardPi"],
            "OLIVES TREX": er[er.dataset == "OLIVES"], "外部 DME 合并": er[er.dataset.isin(["TowardPi", "OLIVES"])]}

    # ---- 2. C 分级分布 ----
    dist = []
    raw = {"OCTDL 设计组": dg_all, "OCTDL 测试组（主要）": tr_all, "OCTDL 正常眼": er_all[er_all.dataset == "OCTDL_NO"],
           "TowardPi": er_all[er_all.dataset == "TowardPi"], "OLIVES TREX": er_all[er_all.dataset == "OLIVES"],
           "外部 DME 合并": er_all[er_all.dataset.isin(["TowardPi", "OLIVES"])]}
    for name, d in raw.items():
        g = d[d.gradable == "yes"]
        row = dict(评估集=name, 图像=len(d), 不可判读=int((d.gradable != "yes").sum()))
        for c in ("C0", "C1", "C2", "C3"):
            row[c] = int((g.C == c).sum())
        row["C未定"] = int((~g.C.isin(["C0", "C1", "C2", "C3"])).sum())
        dist.append(row)
    dist = pd.DataFrame(dist)
    print("\n== 2. 参照标准 ESASO C 分级分布 =="); print(dist.to_string(index=False))

    # ---- 3. 校准 ----
    all_prob = pd.concat([g.drop(columns="set") for s, g in prob.items() if s != "dev_octdl"])
    all_prob = all_prob[~all_prob.index.duplicated()]
    rng = np.random.default_rng(SEED)
    cal_rows, rel_rows, curves = [], [], {}
    for sname, ref in sets.items():
        ids = [i for i in ref.index if i in all_prob.index and FN(ref.loc[i, "C"]) == FN(ref.loc[i, "C"])]
        r = ref.loc[ids]; y = r.C.map(FN).values.astype(float); p = all_prob.loc[ids, "prob_IRF"].values.astype(float)
        s = calib(y, p)
        bs = [calib(y[b], p[b]) for b in boot_idx(r.cluster.values, rng)]
        row = dict(评估集=sname, n=s["n"], 患病率=round(s["患病率"], 3), 平均预测概率=round(s["平均预测概率"], 3))
        for k in ("校准截距", "校准斜率", "Brier"):
            lo, hi = ci([x[k] for x in bs])
            row[k] = "" if s[k] != s[k] else f"{s[k]:.2f} ({lo:.2f} 至 {hi:.2f})"
        cal_rows.append(row)
        if sname in ("OCTDL 测试组（主要）", "外部 DME 合并"):
            q = pd.qcut(p, 5, labels=False, duplicates="drop")
            for k in sorted(set(q)):
                m = q == k
                rel_rows.append(dict(评估集=sname, 组=int(k) + 1, n=int(m.sum()), 平均预测概率=round(p[m].mean(), 3),
                                     实际C1以上比例=round(y[m].mean(), 3)))
            curves[sname] = (p, y, q)
    cal = pd.DataFrame(cal_rows); rel = pd.DataFrame(rel_rows)
    print("\n== 3. IRF 概率对 C≥1 的校准（事后，95% CI 为聚类 bootstrap）=="); print(cal.to_string(index=False))
    print("\n可靠性分组（按预测概率五分位）："); print(rel.to_string(index=False))

    # ---- 图 S2 ----
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
                         "axes.spines.right": False, "axes.spines.top": False})
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 80 / 25.4))
    for ax, (lab, sname, title) in zip(axes, [("a", "OCTDL 测试组（主要）", "OCTDL test set"),
                                               ("b", "外部 DME 合并", "Combined external set")]):
        p, y, q = curves[sname]
        ax.plot([0, 1], [0, 1], ls="--", lw=0.7, color="#999999")
        sub = rel[rel["评估集"] == sname]
        ax.plot(sub["平均预测概率"], sub["实际C1以上比例"], "o-", color="#2b6cb0", ms=3.5, lw=1)
        for _, rr in sub.iterrows():
            ax.annotate(f"n={rr.n}", (rr["平均预测概率"], rr["实际C1以上比例"]), textcoords="offset points",
                        xytext=(3, -8), fontsize=5.5, color="#444444")
        ax.axvline(0.702, color="black", lw=0.6, ls=":")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Mean predicted IRF probability"); ax.set_ylabel("Observed proportion C ≥1")
        ax.set_title(f"{title} (n = {len(y)})", fontsize=7)
        ax.text(-0.18, 1.06, lab, transform=ax.transAxes, fontsize=8, fontweight="bold")
    fig.tight_layout()
    outdir = Path(a.out).parent; outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"FigS2_calibration.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")

    notes = pd.DataFrame({"说明": [__doc__, f"预测文件：{a.preds}", f"c04 结果表：{a.c04_xlsx}",
                                   "校准截距：斜率固定为 1 时的 logistic 截距（0 表示整体不偏）；校准斜率：logit(p) 的系数（1 为理想）。",
                                   "参照 C≥1 = C1/C2/C3；评估集取法与 c04 完全相同。"]})
    with pd.ExcelWriter(a.out) as w:
        sens.to_excel(w, sheet_name="敏感度特异度_取自c04", index=False)
        dist.to_excel(w, sheet_name="C分级分布", index=False)
        cal.to_excel(w, sheet_name="校准_事后", index=False)
        rel.to_excel(w, sheet_name="可靠性分组", index=False)
        notes.to_excel(w, sheet_name="说明", index=False)

    after = {p: sha(p) for p in inputs}
    print("\n== 输入文件哈希运行后", "全部不变" if after == before else "!! 有变化，请停下告诉 Claude", "==")
    print("已保存", a.out, "和", outdir / "FigS2_calibration.png")
    print("输出 xlsx SHA-256", sha(a.out))


if __name__ == "__main__":
    main()
