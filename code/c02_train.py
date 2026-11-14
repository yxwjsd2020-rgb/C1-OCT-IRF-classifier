# -*- coding: utf-8 -*-
"""第 2 步（阿里云 DSW，GPU）：按患者 5 折交叉验证选参数，再用全部训练数据训练最终模型
==========================================================================
训练数据 = c01 的 OLIVES 非 TREX 部分 + c01b 的 RETOUCH 训练集；文章A 的任何测试图都不在这里出现。
--index 与 --root 可各给多个，按顺序一一对应；某来源没有的标志物记为缺失，不计损失。

交叉验证（每组参数一次）：
  python c02_train.py cv --index /mnt/data/olives_prime/train_index.csv /mnt/data/retouch_bscan/train_index.csv \
      --root /mnt/data/olives_prime /mnt/data/retouch_bscan \
      --out runs/cv_cnxt_lr1e-4_bce --arch convnext_tiny --lr 1e-4 --loss bce --epochs 20
  结果：<out>/cv_summary.json（每轮各折平均 AUC；最佳轮次）和 oof_pred.csv（最佳轮次的折外预测）

最终模型（参数由 cv 结果选定）：
  python c02_train.py final --index ... --root ... --out runs/final --arch convnext_tiny --lr 1e-4 --loss bce --epochs 12
  结果：<out>/model.pt、model_sha256.txt、train_config.json

先跑一次冒烟测试（只用 64 幅、2 轮，几分钟内结束，检查代码和显存）：
  python c02_train.py cv ... --smoke
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABEL_PATTERNS, MAIN, ImageTable, auc, build_model, dump_json, masked_loss, seed_all, sha256  # noqa: E402

LABELS = list(LABEL_PATTERNS)


def loaders(df_tr, df_va, a):
    import torch
    from torch.utils.data import DataLoader
    tr = DataLoader(ImageTable(df_tr, a.root, LABELS, a.size, True), batch_size=a.batch, shuffle=True,
                    num_workers=a.workers, pin_memory=True, drop_last=len(df_tr) > a.batch)
    va = None
    if df_va is not None:
        va = DataLoader(ImageTable(df_va, a.root, LABELS, a.size, False), batch_size=a.batch * 2,
                        shuffle=False, num_workers=a.workers, pin_memory=True)
    return tr, va


def predict(model, loader, dev, tta=True):
    import torch
    model.eval(); out = []
    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        for x, _ in loader:
            x = x.to(dev, non_blocking=True)
            p = torch.sigmoid(model(x).float())
            if tta:
                p = (p + torch.sigmoid(model(torch.flip(x, dims=[3])).float())) / 2
            out.append(p.cpu().numpy())
    return np.concatenate(out)


def train_one(df_tr, df_va, a, dev, log_prefix=""):
    """训练一次；返回 (model, 每轮验证集预测列表)。"""
    import torch
    seed_all(a.seed)
    model = build_model(a.arch, len(LABELS), a.weights).to(dev)
    tr, va = loaders(df_tr, df_va, a)
    pos = df_tr[LABELS].mean().clip(0.01, 0.99).values
    pw = torch.tensor((1 - pos) / pos, dtype=torch.float32, device=dev).clamp(max=10) if a.loss == "bce" and a.pos_weight else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * len(tr); warm = len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    preds = []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                logits = model(x)
            loss = masked_loss(logits.float(), y, a.loss, pw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(x)
        msg = f"{log_prefix}第 {ep}/{a.epochs} 轮 loss {tot / len(df_tr):.4f}  {time.time() - t0:.0f}s"
        if va is not None:
            p = predict(model, va, dev)
            preds.append(p)
            msg += "  验证 AUC " + " ".join(f"{n}={auc(df_va[n], p[:, j]):.3f}" for j, n in enumerate(LABELS))
        print(msg, flush=True)
    return model, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cv", "final"])
    ap.add_argument("--index", required=True, nargs="+"); ap.add_argument("--root", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="convnext_tiny", choices=["convnext_tiny", "resnet50", "resnet18"])
    ap.add_argument("--weights", default="imagenet", help="imagenet | none | 本地 .pth")
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--loss", default="bce", choices=["bce", "asl"]); ap.add_argument("--pos-weight", action="store_true")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=384); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--smoke", action="store_true", help="只用 64 幅、2 轮，检查代码能否跑通")
    a = ap.parse_args()

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    assert len(a.index) == len(a.root), "--index 与 --root 个数须相同"
    parts = []
    for ix, rt in zip(a.index, a.root):
        d = pd.read_csv(ix, encoding="utf-8-sig", dtype={"patient": str})
        d["path"] = [str(Path(rt).resolve() / p) for p in d["path"]]
        if "source" not in d:
            d["source"] = "OLIVES"
        for n in LABELS:
            if n not in d:
                d[n] = np.nan
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    a.root = "/"                                            # 路径已是绝对路径
    assert not df.get("trial", pd.Series(dtype=str)).astype(str).str.contains("TREX").any(), "训练表里出现 TREX，停止"
    assert not df.duplicated("path").any(), "训练表里有重复图像"
    print("训练数据：", df.groupby("source").agg(幅=("path", "size"), 人=("patient", "nunique")).to_dict(),
          "阳性率：", df.groupby("source")[LABELS].mean().round(3).to_dict())
    if a.smoke:
        df = df.groupby("patient").head(4).groupby("source").head(32).reset_index(drop=True)
        a.epochs = 2; a.folds = min(a.folds, 2)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in vars(a).items()}
    cfg.update(n_images=len(df), n_patients=int(df.patient.nunique()), labels=LABELS,
               index_sha256={ix: sha256(ix) for ix in a.index})
    dump_json(cfg, out / "train_config.json")

    if a.mode == "cv":
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=a.folds)
        per_epoch = np.full((a.folds, a.epochs, len(LABELS)), np.nan)
        oof = {}
        for k, (i_tr, i_va) in enumerate(gkf.split(df, groups=df.patient)):
            d_tr, d_va = df.iloc[i_tr], df.iloc[i_va]
            print(f"== 第 {k + 1}/{a.folds} 折：训练 {len(d_tr)} 幅 / {d_tr.patient.nunique()} 人，验证 {len(d_va)} 幅 / {d_va.patient.nunique()} 人")
            # 逐折缓存：Colab 运行时被回收后重跑可从中断的折继续。每折开始都用同一种子（train_one 内 seed_all），结果与不中断一致。
            cache = out / f"fold{k + 1}_preds.npz"
            if cache.exists() and np.array_equal(np.load(cache)["va_index"], d_va.index.values):
                preds = list(np.load(cache)["preds"]); print(f"  第 {k + 1} 折已有缓存，跳过训练")
            else:
                _, preds = train_one(d_tr, d_va, a, dev, f"[折{k + 1}] ")
                np.savez(cache, preds=np.stack(preds), va_index=d_va.index.values)
            for e, p in enumerate(preds):
                per_epoch[k, e] = [auc(d_va[n], p[:, j]) for j, n in enumerate(LABELS)]
                oof.setdefault(e, []).append(pd.DataFrame(p, columns=[f"prob_{n}" for n in LABELS], index=d_va.index))
        mean = np.nanmean(per_epoch, axis=0)                 # 轮次 × 标志物
        best = int(np.nanargmax(mean[:, LABELS.index(MAIN)]))
        summary = {"main": MAIN, "best_epoch": best + 1,
                   "best_mean_auc": {n: round(float(mean[best, j]), 4) for j, n in enumerate(LABELS)},
                   "fold_auc_at_best": {n: [round(float(x), 4) for x in per_epoch[:, best, j]] for j, n in enumerate(LABELS)},
                   "mean_auc_by_epoch": {n: [round(float(x), 4) for x in mean[:, j]] for j, n in enumerate(LABELS)}}
        o = pd.concat(oof[best]).sort_index()
        oo = pd.concat([df[["path", "patient", "source", *LABELS]], o], axis=1)
        summary["oof_auc_by_source"] = {s: {n: round(auc(g[n], g[f"prob_{n}"]), 4) for n in LABELS}
                                        for s, g in oo.groupby("source")}
        dump_json(summary, out / "cv_summary.json")
        oo.to_csv(out / "oof_pred.csv", index=False, encoding="utf-8-sig")
        print(f"最佳轮次 {best + 1}，各折平均 AUC：{summary['best_mean_auc']}")
    else:
        model, _ = train_one(df, None, a, dev)
        torch.save({"state_dict": model.state_dict(), "arch": a.arch, "labels": LABELS, "size": a.size}, out / "model.pt")
        (out / "model_sha256.txt").write_text(sha256(out / "model.pt") + "  model.pt\n", encoding="utf-8")
        print("已保存", out / "model.pt")


if __name__ == "__main__":
    main()
