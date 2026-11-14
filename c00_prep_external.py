# -*- coding: utf-8 -*-
"""第 0 步（本人电脑上运行）：把外部测试图像打包，准备上传到阿里云
============================================================
只打包图像和清单（image_id、所属集合、相对路径），不带任何参照标准、读者判读或大模型结果。
参照标准一直留在电脑上，最后由 c04 在电脑上计算——相当于云上的预测是“盲”的。

图像全部是文章A 用过的公开数据（与送大模型的完全相同的文件）：
  dev_octdl     OCTDL 设计组 40 幅（30 DME + 10 正常）——只用来定阈值
  test_octdl    OCTDL 测试组 77 幅 DME
  normal_octdl  OCTDL 正常眼 40 幅
  towardpi      TowardPi 36 幅
  olives_trex   OLIVES TREX-DME 112 幅
Duke 不打包（与文章A 一致，不送任何模型）。

用法（Miniforge Prompt，conda activate dme）：
  cd /d "%USERPROFILE%\\Desktop\\文献 DME AI\\08_文章C分类模型"
  python c00_prep_external.py
  → 生成 articleC_ext 文件夹（images 子文件夹 + manifest.csv）和 articleC_ext.zip，把 zip 上传到 DSW 的 /mnt/data/ 下解压。
"""
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DME_AI = HERE.parent
MLLM = DME_AI / "04_提示词与MLLM" / "MLLM调用"
A07 = DME_AI / "07_文章A外部验证"
OCTDL_ROOT = DME_AI / "05_OCTDL数据集" / "OCTDL"
EXT_ROOT = A07 / "work" / "ext"

SOURCES = [  # 集合名, 清单, 图像根目录
    ("dev_octdl", MLLM / "images_octdl_dev.csv", OCTDL_ROOT),
    ("test_octdl", MLLM / "images_octdl_test.csv", OCTDL_ROOT),
    ("normal_octdl", A07 / "work" / "推理清单" / "images_octdl_normal40.csv", OCTDL_ROOT),
    ("towardpi", A07 / "work" / "推理清单" / "images_towardpi.csv", EXT_ROOT),
    ("olives_trex", A07 / "work" / "推理清单" / "images_olives.csv", EXT_ROOT),
]
EXPECT = {"dev_octdl": 40, "test_octdl": 77, "normal_octdl": 40, "towardpi": 36, "olives_trex": 112}


def main():
    out = HERE / "articleC_ext"
    if out.exists():
        sys.exit(f"{out} 已存在；如需重做请先手动删除。")
    (out / "images").mkdir(parents=True)
    rows = []
    for name, lst, root in SOURCES:
        d = pd.read_csv(lst, encoding="utf-8-sig")
        if len(d) != EXPECT[name]:
            sys.exit(f"!! {name} 清单有 {len(d)} 行，应为 {EXPECT[name]}：{lst}")
        for _, r in d.iterrows():
            src = root / r["path"]
            if not src.exists():
                sys.exit(f"!! 找不到图像：{src}")
            dst = f"images/{name}__{r['image_id']}{src.suffix.lower()}"
            shutil.copy2(src, out / dst)                     # 复制，不改动原图
            rows.append({"image_id": r["image_id"], "set": name, "path": dst})
    man = pd.DataFrame(rows)
    assert not man.duplicated(["set", "image_id"]).any()
    man.to_csv(out / "manifest.csv", index=False, encoding="utf-8-sig")
    with zipfile.ZipFile(HERE / "articleC_ext.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(HERE))
    print(man.set.value_counts().to_string())
    print(f"完成：{len(man)} 幅 → {HERE / 'articleC_ext.zip'}（不含任何参照标准）")


if __name__ == "__main__":
    main()
