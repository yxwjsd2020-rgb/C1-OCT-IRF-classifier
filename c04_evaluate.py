# -*- coding: utf-8 -*-
"""第 4 步（本人电脑上运行）：定阈值 → 外部测试 → 与文章A 的 7 个大模型和人类读者比较
=========================================================================
主分析（预先规定）：IRF 概率 → C≥1（有视网膜内囊腔/积液）。
- 阈值：OCTDL 设计组 40 幅上取 Youden 指数最大的阈值（并列时取最接近 0.5 的）；只用设计组，不看任何测试图。
- 参照：直接用文章A 已冻结的参照——测试组三人多数票；外部集 R2/R3 协商结果。参照 C1–C3 即 C≥1。
- 指标：Gwet AC1（主）、敏感度、特异度；分类模型另报 AUC（大模型没有概率，不报）。
- 95% CI：聚类 bootstrap 2000 次（OCTDL 按患者；TowardPi、OLIVES 按眼），seed 20261023，与文章A 相同；
  分类模型 − 大模型的配对差值用同一批重抽样。
- 人类基准：测试组 R1 vs R2、外部集 R2 vs R3（独立判读，两人都可判读的图），口径与文章A 相同。
次要分析（预先规定）：SRF 概率 → F1（视网膜下液，文章A 中大模型漏判最多的参数）。
- 阈值取训练集（OLIVES + RETOUCH）折外预测的 Youden 阈值（需 --oof）。不用设计组：设计组的 F 是在补充规则 4
  （区分视网膜下液与大囊腔）加入之前判读的，口径与测试组、外部集不同。
- 阳性数少（测试组 7、外部 DME 26），以判出数和敏感度/特异度为主，AC1 作参考。
敏感性分析：阈值固定 0.5；C≥1 另用折外 Youden 阈值。
探索性：HRF、EZ 两个头对 H1、E≥1 只报 AUC（定义与 ESASO 不完全对应）；OLIVES 官方 IRF 标签作参照（需 --olives-index）。

用法：
  python c04_evaluate.py --preds runs\\final\\preds_external.csv --oof runs\\cv_xxx\\oof_pred.csv \
      [--olives-index D:\\公共数据集\\OLIVES_TREX\\olives_trex_index.csv] --out 结果\\分类模型_外部测试.xlsx
  不加 --no-llm 时会读取文章A 的大模型结果（只读）。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import gwet_ac1_binary, auc  # noqa: E402

DME_AI = HERE.parent
MLLM = DME_AI / "04_提示词与MLLM" / "MLLM调用"
A07 = DME_AI / "07_文章A外部验证"
D = dict(
    dev_gold=MLLM / "inputs" / "gold_final_dev.csv",
    test_ref=DME_AI / "OCTDL测试组判读" / "OCTDL测试组_最终参照标准_三人多数票.csv",
    test_r=DME_AI / "OCTDL测试组判读" / "OCTDL测试组_人工判读_{}.csv",
    ext_ref=DME_AI / "外部集判读" / "外部集_最终参照标准_R2R3协商.csv",
    ext_r=DME_AI / "外部集判读" / "外部集_人工判读_{}.csv",
    test_llm=A07 / "测试组_七模型_P1v10_三人参照_排除缺失.xlsx",
    ext_raw=A07 / "results_ext",
)
LLMS = [("千问", "qwen"), ("GLM", "glm"), ("Kimi", "kimi"), ("Gemma", "gemma"), ("Gemini", "gemini"),
        ("GPT", "openai"), ("DeepSeek", "deepseek")]
# 模型头 → (参照字段, 参照值 → 1/0/NaN)
TARGETS = {
    "IRF": ("C", lambda v: 1.0 if v in ("C1", "C2", "C3") else 0.0 if v == "C0" else np.nan),
    "SRF": ("F", lambda v: 1.0 if v == "F1" else 0.0 if v == "F0" else np.nan),
    "HRF": ("H", lambda v: 1.0 if v == "H1" else 0.0 if v == "H0" else np.nan),
    "EZ": ("E", lambda v: 1.0 if v in ("E1", "E2") else 0.0 if v == "E0" else np.nan),
}
B, SEED = 2000, 20261023


def read(p):
    return pd.read_csv(p, encoding="utf-8-sig", dtype=str).fillna("")


def youden(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~np.isnan(y); y, p = y[ok], p[ok]
    best, thr = -1, 0.5
    for t in np.unique(np.concatenate([p, [0.5]])):
        pred = p >= t
        j = pred[y == 1].mean() + (~pred[y == 0]).mean() - 1
        if j > best + 1e-12 or (abs(j - best) <= 1e-12 and abs(t - 0.5) < abs(thr - 0.5)):
            best, thr = j, float(t)
    return thr, best


def summ(pred, ref):
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    ok = ~np.isnan(ref)
    pred, ref = pred[ok], ref[ok]
    pred = np.where(np.isnan(pred), 1 - ref, pred)          # 无效输出按判错计（文章A 规则）
    P, N = (ref == 1).sum(), (ref == 0).sum()
    return dict(n=int(ok.sum()), AC1=gwet_ac1_binary(pred, ref),
                敏感度=(pred[ref == 1] == 1).mean() if P else np.nan, 特异度=(pred[ref == 0] == 0).mean() if N else np.nan,
                判出=f"{int((pred[ref == 1] == 1).sum())}/{int(P)}", 特异=f"{int((pred[ref == 0] == 0).sum())}/{int(N)}")


def boot_idx(cluster, rng):
    u = np.unique(cluster); idx = {c: np.where(cluster == c)[0] for c in u}
    return [np.concatenate([idx[c] for c in rng.choice(u, len(u), replace=True)]) for _ in range(B)]


def ci(v):
    v = np.asarray(v, float)
    return (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) if np.isfinite(v).any() else (np.nan, np.nan)


def fmt(x, c):
    return "" if x != x else f"{x:.2f} ({c[0]:.2f} 至 {c[1]:.2f})"


def evaluate(head, thrs, sets, all_prob, llm, readers, label):
    """head：模型头；thrs：{名称: 阈值}，第一项为主；返回 (结果行, 与大模型比较行, 人类基准行)"""
    field, fn = TARGETS[head]
    rng = np.random.default_rng(SEED)
    rows, diffs, hums = [], [], []
    main_t = next(iter(thrs))
    for sname, ref in sets.items():
        ids = [i for i in ref.index if i in all_prob.index and fn(ref.loc[i, field]) == fn(ref.loc[i, field])]
        if not ids:
            continue
        r = ref.loc[ids]; y = r[field].map(fn).values; p = all_prob.loc[ids, f"prob_{head}"].values
        boots = boot_idx(r.cluster.values, rng)
        for tname, t in thrs.items():
            pred = (p >= t).astype(float)
            s = summ(pred, y); bs = [summ(pred[b], y[b]) for b in boots]
            row = dict(分析=label, 评估集=sname, 阈值=f"{tname}（{t:.3f}）", n=s["n"],
                       AC1=fmt(s["AC1"], ci([x["AC1"] for x in bs])), 敏感度=fmt(s["敏感度"], ci([x["敏感度"] for x in bs])),
                       特异度=fmt(s["特异度"], ci([x["特异度"] for x in bs])), 判出=s["判出"], 特异=s["特异"])
            if tname == main_t:
                row["AUC"] = fmt(auc(y, p), ci([auc(y[b], p[b]) for b in boots]))
                for name, ser in llm.get(field, {}).items():
                    q = ser.reindex(ids).map(fn).values
                    sl = summ(q, y)
                    d = [summ(pred[b], y[b])["AC1"] - summ(q[b], y[b])["AC1"] for b in boots]
                    diffs.append(dict(分析=label, 评估集=sname, 大模型=name, 大模型AC1=round(sl["AC1"], 3),
                                      大模型判出=sl["判出"], 大模型特异=sl["特异"], 分类模型AC1=round(s["AC1"], 3),
                                      分类模型判出=s["判出"], 分类模型特异=s["特异"],
                                      差值_分类减大模型=fmt(s["AC1"] - sl["AC1"], ci(d))))
                for hname, (ha, hb) in readers.items():
                    if (hname == "R1 vs R2") != sname.startswith("OCTDL 测试组"):
                        continue
                    hid = [i for i in ids if i in ha.index and i in hb.index and ha.loc[i, field] and hb.loc[i, field]]
                    if hid:
                        sa, sb = ha.loc[hid, field].map(fn).values, hb.loc[hid, field].map(fn).values
                        sh = summ(sa, sb)
                        hums.append(dict(分析=label, 评估集=sname, 比较=hname, n=len(hid), AC1=round(sh["AC1"], 3),
                                         说明="R1 vs R2 以 R1 为参照；R2 vs R3 以 R2 为参照（同文章A）", 判出=sh["判出"], 特异=sh["特异"]))
            rows.append(row)
    return rows, diffs, hums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--oof"); ap.add_argument("--olives-index"); ap.add_argument("--no-llm", action="store_true")
    for k, v in D.items():
        ap.add_argument("--" + k.replace("_", "-"), default=str(v))
    a = ap.parse_args()
    pr = pd.read_csv(a.preds, encoding="utf-8-sig", dtype={"image_id": str})
    prob = {s: g.set_index("image_id") for s, g in pr.groupby("set")}

    # ---------- 阈值 ----------
    dg = read(a.dev_gold); dg = dg[dg.gradable == "yes"].set_index("image_id")
    dev = prob["dev_octdl"].loc[dg.index]
    y_dev = dg.C.map(TARGETS["IRF"][1]).values
    t_c, j_c = youden(y_dev, dev.prob_IRF.values)
    thr_rows = [dict(标志物="IRF→C≥1", 阈值来源="OCTDL 设计组 Youden（主分析）", 阈值=round(t_c, 4), Youden=round(j_c, 3),
                     n=len(dg), 阳性=int(np.nansum(y_dev)), AUC=round(auc(y_dev, dev.prob_IRF.values), 3))]
    thr_c = {"设计组Youden": t_c, "固定0.5": 0.5}
    thr_f = {"固定0.5": 0.5}
    if a.oof:
        oof = pd.read_csv(a.oof, encoding="utf-8-sig")
        for head, tdict in (("IRF", thr_c), ("SRF", thr_f)):
            o = oof.dropna(subset=[head])
            t, j = youden(o[head].values, o[f"prob_{head}"].values)
            thr_rows.append(dict(标志物=head, 阈值来源="训练集折外预测 Youden" + ("（主分析）" if head == "SRF" else "（敏感性分析）"),
                                 阈值=round(t, 4), Youden=round(j, 3), n=len(o), 阳性=int(o[head].sum()),
                                 AUC=round(auc(o[head].values, o[f"prob_{head}"].values), 3)))
            if head == "SRF":
                thr_f = {"折外Youden": t, "固定0.5": 0.5}
            else:
                thr_c["折外Youden"] = t
    else:
        print("!! 未给 --oof：SRF 只能用固定 0.5 阈值，C≥1 少一项敏感性分析。")

    # ---------- 评估集与参照 ----------
    tr = read(a.test_ref); tr = tr[tr.gradable == "yes"].set_index("image_id"); tr["cluster"] = "octdl_" + tr.patient_id
    er = read(a.ext_ref); er = er[er.gradable == "yes"].set_index("image_id")
    er["cluster"] = er.dataset + "_" + er.patient + "_" + er.eye.where(er.dataset != "OCTDL_NO", "")
    sets = {"OCTDL 测试组（主要）": tr, "OCTDL 正常眼": er[er.dataset == "OCTDL_NO"], "TowardPi": er[er.dataset == "TowardPi"],
            "OLIVES TREX": er[er.dataset == "OLIVES"], "外部 DME 合并": er[er.dataset.isin(["TowardPi", "OLIVES"])]}
    all_prob = pd.concat([g.drop(columns="set") for s, g in prob.items() if s != "dev_octdl"])
    all_prob = all_prob[~all_prob.index.duplicated()]

    # ---------- 大模型（只读文章A 结果） ----------
    llm = {}
    if not a.no_llm:
        wb = pd.read_excel(a.test_llm, sheet_name="逐图", dtype=str).set_index("image_id")
        sys.path.insert(0, str(MLLM))
        from analyze import load_model, majority
        for name, prov in LLMS:
            m = majority(load_model(str(Path(a.ext_raw) / prov / "raw_responses.jsonl"), "P1_v10", prov)).set_index("image_id")
            for field in ("C", "F"):
                col = f"{field}|{name}"
                s = pd.concat([wb[col] if col in wb.columns else pd.Series(dtype=str), m[field]])
                llm.setdefault(field, {})[name] = s[~s.index.duplicated()]

    # ---------- 人类读者 ----------
    r1, r2 = (read(a.test_r.format(k)).set_index("image_id") for k in ("R1", "R2"))
    e2, e3 = (read(a.ext_r.format(k)).set_index("image_id") for k in ("R2", "R3"))
    for rr in (r1, r2, e2, e3):
        for f in ("C", "F", "H", "E"):
            rr.loc[rr.gradable != "yes", f] = ""
    readers = {"R1 vs R2": (r2, r1), "R2 vs R3": (e3, e2)}

    rows, diffs, hums = evaluate("IRF", thr_c, sets, all_prob, llm, readers, "主分析：IRF→C≥1")
    r_, d_, h_ = evaluate("SRF", thr_f, sets, all_prob, llm, readers, "次要分析：SRF→F1")
    rows += r_; diffs += d_; hums += h_

    explo = []
    for head in ("HRF", "EZ"):
        field, fn = TARGETS[head]
        for sname, ref in sets.items():
            ids = [i for i in ref.index if i in all_prob.index]
            yy = ref.loc[ids, field].map(fn).values
            explo.append(dict(评估集=sname, 模型头=head, 对应参数=field, n=int((~np.isnan(yy)).sum()),
                              参照阳性=int(np.nansum(yy)), AUC=round(auc(yy, all_prob.loc[ids, f"prob_{head}"].values), 3)))
    if a.olives_index:
        oi = pd.read_csv(a.olives_index, encoding="utf-8-sig")
        oi["image_id"] = oi["file"].str.replace("images/", "", regex=False).str.replace(".png", "", regex=False)
        for head, pat in (("IRF", "irf"), ("SRF", "srf")):
            col = next(c for c in oi.columns if pat in c.lower())
            o = oi.set_index("image_id").reindex(prob["olives_trex"].index)
            yy = pd.to_numeric(o[col], errors="coerce").values; pp = prob["olives_trex"][f"prob_{head}"].values
            explo.append(dict(评估集="OLIVES TREX（官方标签）", 模型头=head, 对应参数=f"官方 {col}",
                              n=int((~np.isnan(yy)).sum()), 参照阳性=int(np.nansum(yy)), AUC=round(auc(yy, pp), 3)))

    notes = pd.DataFrame({"说明": [__doc__, f"预测文件：{a.preds}", f"折外预测：{a.oof}", f"设计组参照：{a.dev_gold}",
        f"测试组参照：{a.test_ref}", f"外部集参照：{a.ext_ref}",
        "大模型取 P1_v10 三次多数票；C≥1 = C1/C2/C3。分类模型只输出有/无，与文章A 的 C 四级加权 κ 不可直接比较。"]})
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as w:
        pd.DataFrame(thr_rows).to_excel(w, sheet_name="阈值", index=False)
        pd.DataFrame(rows).to_excel(w, sheet_name="结果", index=False)
        pd.DataFrame(diffs).to_excel(w, sheet_name="与大模型比较", index=False)
        pd.DataFrame(hums).to_excel(w, sheet_name="人类基准", index=False)
        pd.DataFrame(explo).to_excel(w, sheet_name="探索性", index=False)
        notes.to_excel(w, sheet_name="说明", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(thr_rows).to_string()); print(pd.DataFrame(rows).to_string())
    if diffs:
        print(pd.DataFrame(diffs).to_string())
    print(pd.DataFrame(hums).to_string())
    print("已保存", out)


if __name__ == "__main__":
    main()
