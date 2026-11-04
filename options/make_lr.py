"""高解像度(GT)画像から、HATのファインチューニング用に低解像度(LR)画像を作るスクリプト。

出力:
  <out>/train/GT, <out>/train/LR   学習用
  <out>/val/GT,   <out>/val/LR     検証用(--val-num 枚を無作為に取り分け)

- LRはbicubicで 1/scale に縮小します(HAT/BasicSRの標準的な作り方)。
- GTは縦横が scale の倍数になるよう端を切り落とします(GTmod2 と同じ考え方)。
- GTとLRは同じファイル名(拡張子も同じ .png)で保存します。
  BasicSRのPairedImageDatasetは、同じファイル名でペアを対応づけるためです。

使い方の例:
  python make_lr.py --src /path/to/hr_images --out datasets --scale 2 --val-num 100
"""
import argparse
import random
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def process(args):
    src, gt_dir, lr_dir, scale, min_size = args
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            w2, h2 = w - w % scale, h - h % scale
            if min(w2, h2) < min_size:
                return f"skip(小さすぎ {w}x{h}): {src.name}"
            gt = im.crop((0, 0, w2, h2))
            lr = gt.resize((w2 // scale, h2 // scale), Image.BICUBIC)
            name = f"{src.stem}.png"
            gt.save(gt_dir / name)
            lr.save(lr_dir / name)
    except Exception as e:  # 壊れた画像などは飛ばして続行
        return f"error({e}): {src.name}"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="元の高解像度画像のフォルダ")
    p.add_argument("--out", default="datasets", help="出力先フォルダ")
    p.add_argument("--scale", type=int, default=2, help="縮小倍率(2, 3, 4 など)")
    p.add_argument("--val-num", type=int, default=100, help="検証用に取り分ける枚数(0なら分けない)")
    p.add_argument("--min-size", type=int, default=128,
                   help="GTの短辺がこれ未満の画像は除外(gt_size以上にする)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    src_dir, out = Path(a.src), Path(a.out)
    files = sorted(f for f in src_dir.rglob("*") if f.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"画像が見つかりません: {src_dir}")

    # 別フォルダに同名ファイルがあると上書きされるので検出する
    stems = [f.stem for f in files]
    if len(stems) != len(set(stems)):
        raise SystemExit("ファイル名(拡張子を除く)が重複しています。リネームしてから再実行してください。")

    random.Random(a.seed).shuffle(files)
    val_files = files[: a.val_num]
    train_files = files[a.val_num:]

    jobs = []
    for split, group in (("train", train_files), ("val", val_files)):
        gt_dir, lr_dir = out / split / "GT", out / split / "LR"
        gt_dir.mkdir(parents=True, exist_ok=True)
        lr_dir.mkdir(parents=True, exist_ok=True)
        jobs += [(f, gt_dir, lr_dir, a.scale, a.min_size) for f in group]

    print(f"学習用 {len(train_files)} 枚 / 検証用 {len(val_files)} 枚 / scale=x{a.scale}")
    msgs = []
    with Pool(a.workers) as pool:
        for i, m in enumerate(pool.imap_unordered(process, jobs, chunksize=16), 1):
            if m:
                msgs.append(m)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}")
    for m in msgs:
        print(m)
    print(f"完了(スキップ/エラー {len(msgs)} 件)。出力先: {out.resolve()}")


if __name__ == "__main__":
    main()
