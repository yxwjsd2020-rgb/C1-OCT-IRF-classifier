# -*- coding: utf-8 -*-
"""第 1a 步（Colab）：多线程分段下载 OLIVES.zip 里的 Prime_FULL.zip，再在本地解压成完整的内层 zip
=========================================================================
Zenodo 单连接只有约 3–4 MB/s，14.4 GB 要一个多小时；这里用多个连接并行下载同一段字节，
下载完再用 zlib 解压，并核对大小和 CRC32。完成后 c01_colab_olives.py extract 会直接使用本地文件。
TREX_DME.zip 不下载。

  !python c01a_fetch_inner.py --member Prime_FULL.zip --threads 16
"""
import argparse
import os
import sys
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c01_colab_olives import IMG_URL, INNER_DIR, HTTPRange, member_data_offset  # noqa: E402
import io  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="Prime_FULL.zip")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=64)
    a = ap.parse_args()

    raw = HTTPRange(IMG_URL)
    outer = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))
    info = next(i for i in outer.infolist() if i.filename.endswith(a.member))
    assert "trex" not in info.filename.lower(), "不下载 TREX"
    f = io.BufferedReader(HTTPRange(raw.url, 0, raw.len), buffer_size=1 << 16)
    off = member_data_offset(f, info)
    csize, usize = info.compress_size, info.file_size
    print(f"{info.filename}：压缩 {csize / 1e9:.2f} GB，解压后 {usize / 1e9:.2f} GB，偏移 {off}，压缩方式 {info.compress_type}")

    out = Path(INNER_DIR) / Path(info.filename).name
    if out.exists() and out.stat().st_size == usize:
        print("已存在完整的本地文件，跳过：", out); return
    comp = Path(INNER_DIR) / (Path(info.filename).name + ".deflate")
    step = a.chunk_mb << 20
    segs = [(off + s, off + min(s + step, csize) - 1) for s in range(0, csize, step)]
    done_flag = Path(str(comp) + ".done")
    if not done_flag.exists():
        fd = os.open(comp, os.O_RDWR | os.O_CREAT)
        os.ftruncate(fd, csize)
        t0, got = time.time(), 0
        with ThreadPoolExecutor(a.threads) as ex:
            futs = [ex.submit(fetch_rel, raw.url, s, e, fd, off) for s, e in segs]
            for k, fu in enumerate(as_completed(futs), 1):
                got += fu.result()
                if k % 10 == 0 or k == len(futs):
                    el = time.time() - t0
                    print(f"  {got / 1e9:.2f} / {csize / 1e9:.2f} GB，{got / 1e6 / el:.0f} MB/s，{el:.0f} 秒", flush=True)
        os.close(fd)
        done_flag.write_text("ok")
    print("解压并核对 CRC32 ……", flush=True)
    if info.compress_type == zipfile.ZIP_STORED:
        os.replace(comp, out)
    else:
        d = zlib.decompressobj(-15)
        crc, n = 0, 0
        with open(comp, "rb") as src, open(out, "wb") as dst:
            while True:
                b = src.read(1 << 24)
                if not b:
                    break
                u = d.decompress(b)
                crc = zlib.crc32(u, crc); n += len(u); dst.write(u)
            u = d.flush(); crc = zlib.crc32(u, crc); n += len(u); dst.write(u)
        ok = (n == usize) and ((crc & 0xFFFFFFFF) == info.CRC)
        print(f"解压 {n} 字节（应为 {usize}），CRC32 {'一致' if ok else '不一致'}")
        if not ok:
            sys.exit("!! 校验失败，请把以上输出发给 Claude")
        comp.unlink(); done_flag.unlink()
    print("完成：", out)


def fetch_rel(url, a, b, fd, base):
    """下载远程 [a, b]，写到本地文件的 (a - base) 位置。"""
    for k in range(10):
        try:
            r = requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=300)
            if r.status_code != 206 or len(r.content) != b - a + 1:
                raise IOError(f"HTTP {r.status_code}, {len(r.content)} 字节")
            os.pwrite(fd, r.content, a - base)
            return b - a + 1
        except Exception as e:
            last = e
            time.sleep(3 * (k + 1))
    raise IOError(f"段 {a}-{b} 连续失败：{last}")


if __name__ == "__main__":
    main()
