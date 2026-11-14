# -*- coding: utf-8 -*-
"""第 1 步（Google Colab 上运行）：不下载整个 33.9 GB 的 OLIVES.zip，用 HTTP 分段读取只抽训练用的 B-scan
================================================================================
- 训练图 = OLIVES 中有逐幅标志物标签、且不属于 TREX DME 的 B-scan（默认每个容积第 13–37 幅）。
- 同时出现在 TREX 的患者整个剔除（文章A 的 OLIVES 外部测试图来自 TREX）。
- OLIVES.zip 里可能嵌套 zip：内层 zip 若为“存储”方式（不压缩），直接分段读取；
  若为压缩方式，则把该内层 zip 整个下载到 /content 后再读取（会先打印大小）。
- 输出与 c01_olives_prime.py 相同：<out>/images/*.png + <out>/train_index.csv

用法（Colab 单元格）：
  !python c01_colab_olives.py probe                       # 只列出 zip 结构和标签概况，几分钟
  !python c01_colab_olives.py extract --out /content/olives_prime
"""
import argparse
import io
import re
import struct
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABEL_PATTERNS  # noqa: E402

import os
IMG_URL = os.environ.get("OLIVES_IMG_URL", "https://zenodo.org/records/7105232/files/OLIVES.zip?download=1")
LAB_URL = os.environ.get("OLIVES_LAB_URL", "https://zenodo.org/records/7105232/files/OLIVES_Dataset_Labels.zip?download=1")
INNER_DIR = os.environ.get("OLIVES_INNER_DIR", "/content")
TREX = "TREX DME"
SES = requests.Session()


class HTTPRange(io.RawIOBase):
    """把远程文件（或其中一段）当作可随机访问的本地文件。"""

    def __init__(self, url, start=0, length=None):
        if length is None:
            r = SES.head(url, allow_redirects=True, timeout=60)
            url = r.url
            length = int(r.headers["Content-Length"])
        self.url, self.start, self.len, self.pos = url, start, length, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.len + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.len:
            return 0
        n = min(len(b), self.len - self.pos)
        a = self.start + self.pos
        for k in range(8):
            try:
                r = SES.get(self.url, headers={"Range": f"bytes={a}-{a + n - 1}"}, timeout=120)
                if r.status_code == 206:
                    d = r.content
                    b[:len(d)] = d
                    self.pos += len(d)
                    return len(d)
                raise IOError(f"HTTP {r.status_code}")
            except Exception as e:  # 网络波动：等待后重试
                print(f"  分段读取失败（{e}），{5 * (k + 1)} 秒后重试", flush=True)
                time.sleep(5 * (k + 1))
        raise IOError("分段读取连续失败")


def remote(url, start=0, length=None, buf=1 << 20):
    return io.BufferedReader(HTTPRange(url, start, length), buffer_size=buf)


def member_data_offset(f, info):
    f.seek(info.header_offset)
    h = f.read(30)
    n, m = struct.unpack("<HH", h[26:30])
    return info.header_offset + 30 + n + m


def norm(p):
    parts = [x for x in str(p).replace("\\", "/").split("/") if x]
    return "/".join(parts[-4:]).lower()


def open_zips(download=True):
    """返回 [(zipfile 对象, 说明)]：外层 zip，以及其中非 TREX 的内层 zip。
    TREX 的内层 zip 一律跳过（不下载、不读取）。download=False 时只列出内层 zip 信息。"""
    raw = HTTPRange(IMG_URL)
    outer = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))
    out = [(outer, "外层")]
    for info in outer.infolist():
        if not info.filename.lower().endswith(".zip"):
            continue
        is_trex = "trex" in info.filename.lower()
        kind = "存储" if info.compress_type == zipfile.ZIP_STORED else f"压缩(type {info.compress_type})"
        print(f"内层 zip：{info.filename}  {kind}  压缩后 {info.compress_size / 1e9:.2f} GB"
              + ("  → TREX，跳过" if is_trex else ""))
        if is_trex or not download:
            continue
        if info.compress_type == zipfile.ZIP_STORED:
            f = io.BufferedReader(HTTPRange(raw.url, 0, raw.len), buffer_size=1 << 16)
            off = member_data_offset(f, info)
            out.append((zipfile.ZipFile(remote(raw.url, off, info.file_size)), f"内层(存储) {info.filename}"))
        else:
            local = Path(INNER_DIR) / Path(info.filename).name
            if not (local.exists() and local.stat().st_size == info.file_size):
                print(f"  整个下载到 {local} ……", flush=True)
                t0 = time.time()
                with outer.open(info) as src, open(local, "wb") as dst:
                    while True:
                        b = src.read(1 << 24)
                        if not b:
                            break
                        dst.write(b)
                        print(f"    {dst.tell() / 1e9:.2f} / {info.file_size / 1e9:.2f} GB，{time.time() - t0:.0f} 秒", flush=True)
            out.append((zipfile.ZipFile(local), f"内层(已下载) {info.filename}"))
    return raw, out


def load_labels():
    r = SES.get(LAB_URL, timeout=600)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    cand = [n for n in z.namelist() if n.lower().endswith("biomarker_clinical_data_images.csv")]
    name = next((n for n in cand if "full_labels" in n.lower()), cand[0])
    print("标签文件：", name)
    return pd.read_csv(io.BytesIO(z.read(name)))


def select(t, scans):
    path_col, scan_col = t.columns[0], t.columns[1]
    cols = {}
    for name, pat in LABEL_PATTERNS.items():
        hit = [c for c in t.columns if re.search(pat, c.lower())]
        if len(hit) != 1:
            sys.exit(f"!! 标志物 {name} 用 /{pat}/ 匹配到 {hit}，应恰好 1 列。")
        cols[name] = hit[0]
    pid = next(c for c in t.columns if c.lower().replace("_", "") in ("patientid", "patient"))
    eid = next((c for c in t.columns if c.lower().replace("_", "") in ("eyeid", "eye")), None)
    t["trial"] = t[path_col].astype(str).str.strip("/").str.split("/").str[0]
    print("各部分行数：", t.trial.value_counts().to_dict())
    is_trex = t.trial.str.contains(TREX, regex=False)
    tr = t[~is_trex].copy()
    overlap = set(tr[pid].dropna()) & set(t.loc[is_trex, pid].dropna())
    if overlap:
        print(f"!! {len(overlap)} 位患者同时出现在 TREX 与训练部分，整个剔除")
        tr = tr[~tr[pid].isin(overlap)]
    if scans != "all":
        lo, hi = map(int, scans.split("-"))
        tr = tr[tr[scan_col].between(lo, hi)]
    tr = tr.dropna(subset=[cols["IRF"]])
    parts = tr[path_col].astype(str).str.strip("/").str.split("/")
    tr["visit"], tr["eye"] = parts.str[-3], parts.str[-2]
    tr["patient"] = tr[pid].astype(str)
    tr["eye_id"] = tr[eid].astype(str) if eid else tr["patient"] + "_" + tr["eye"]
    for name, c in cols.items():
        tr[name] = pd.to_numeric(tr[c], errors="coerce")
    print(f"训练用 B-scan {len(tr)} 幅；患者 {tr.patient.nunique()} 人；眼 {tr.eye_id.nunique()} 只")
    print("阳性率：", {n: round(float(tr[n].mean()), 3) for n in cols})
    return tr, path_col, scan_col, list(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "extract"])
    ap.add_argument("--out", default="/content/olives_prime")
    ap.add_argument("--scans", default="13-37")
    a = ap.parse_args()

    t = load_labels()
    print("标签列：", list(t.columns))
    tr, path_col, scan_col, labels = select(t, a.scans)
    raw, zips = open_zips(download=(a.mode == "extract"))
    print(f"OLIVES.zip 大小 {raw.len / 1e9:.2f} GB")
    for z, tag in zips:
        names = [n for n in z.namelist() if not n.endswith("/")]
        print(f"[{tag}] 成员 {len(names)} 个；示例：{names[:3]}")
    if a.mode == "probe":
        print("训练表路径示例：", tr[path_col].head(3).tolist())
        return

    idx = {}
    for z, tag in zips:
        for n in z.namelist():
            if not n.endswith("/") and not n.lower().endswith(".zip"):
                idx.setdefault(norm(n), (z, n))
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    from PIL import Image
    miss, t0 = 0, time.time()
    for k, (i, r) in enumerate(tr.iterrows(), 1):
        hit = idx.get(norm(r[path_col]))
        if hit is None:
            miss += 1; tr.loc[i, "path"] = None; continue
        z, n = hit
        name = f"P{r.patient}_{r.eye}_{r.visit}_s{int(r[scan_col]):02d}.png".replace(" ", "")
        if not (out / "images" / name).exists():
            Image.open(io.BytesIO(z.read(n))).convert("L").save(out / "images" / name)
        tr.loc[i, "path"] = "images/" + name
        if k % 100 == 0:
            print(f"  {k}/{len(tr)}，用时 {time.time() - t0:.0f} 秒", flush=True)
    if miss:
        print(f"!! {miss} 幅在 zip 里没找到；训练表路径示例 {tr[path_col].head(2).tolist()}")
    tr = tr.dropna(subset=["path"])
    keep = ["path", "patient", "eye_id", "visit", scan_col, "trial", *labels]
    tr[keep].rename(columns={scan_col: "scan"}).to_csv(out / "train_index.csv", index=False, encoding="utf-8-sig")
    print(f"完成：{len(tr)} 幅 → {out / 'train_index.csv'}")


if __name__ == "__main__":
    main()
