# -*- coding: utf-8 -*-
"""第 3 步（阿里云 DSW）：用冻结的最终模型预测外部图像，并封存
=========================================================
输入是电脑上 c00_prep_external.py 打的包（只有图像和清单，不含任何参照标准）。
输出 preds.csv（每幅图各标志物的概率）和封存记录（模型、清单、预测文件的 SHA-256）。
概率→阳性/阴性的阈值不在这里定，回到电脑上由 c04 用设计组确定，避免在云上接触参照。

  python c03_predict.py --model runs/final/model.pt --manifest /mnt/data/articleC_ext/manifest.csv \
      --root /mnt/data/articleC_ext --out runs/final/preds_external.csv
"""
import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ImageTable, build_model, sha256  # noqa: E402
from c02_train import predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.model, map_location="cpu")
    model = build_model(ck["arch"], len(ck["labels"]), "none")
    model.load_state_dict(ck["state_dict"]); model.to(dev)
    man = pd.read_csv(a.manifest, encoding="utf-8-sig")
    dl = DataLoader(ImageTable(man, a.root, [], ck["size"], False), batch_size=a.batch, num_workers=a.workers)
    p = predict(model, dl, dev)
    out = man[["image_id", "set"]].copy()
    for j, n in enumerate(ck["labels"]):
        out[f"prob_{n}"] = p[:, j].round(6)
    out.to_csv(a.out, index=False, encoding="utf-8-sig")
    seal = Path(a.out).with_suffix(".seal.txt")
    seal.write_text("\n".join([
        f"时间（UTC）：{datetime.datetime.utcnow().isoformat(timespec='seconds')}",
        f"模型 {a.model}  SHA-256 {sha256(a.model)}",
        f"清单 {a.manifest}  SHA-256 {sha256(a.manifest)}",
        f"预测 {a.out}  SHA-256 {sha256(a.out)}",
        f"图像数 {len(out)}；各集合：{out.set.value_counts().to_dict()}"]) + "\n", encoding="utf-8")
    print(seal.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
