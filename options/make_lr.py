"""datasets/GT の画像を縮小して datasets/LR に出力するスクリプト(HATのファインチューニング用)。

- LR は bicubic で 1/scale に縮小します。
- ファイル名は GT と同じ(拡張子も同じ)にします。BasicSR は同じ名前でペアを対応づけるためです。
- GT の縦横が scale の倍数でない画像は、LR との大きさが合わず学習時にエラーになります。
  既定ではそのような画像を一覧表示して作成を飛ばします。
  --crop-gt を付けると、GT の端を切り落として倍数にそろえます(元のGTファイルを上書きします)。

使い方:
  python make_lr.py                                   # datasets/GT -> datasets/LR, x2
  python make_lr.py --gt datasets/GT --lr datasets/LR --scale 2
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def process(job):
    src, lr_dir, scale, crop_gt = job
    try:
        with Image.open(src) as im:
            fmt_ext = src.suffix.lower()
            im = im.convert("RGB")
            w, h = im.size
            w2, h2 = w - w % scale, h - h % scale
            if (w2, h2) != (w, h):
                if not crop_gt:
                    return f"skip(縦横が{scale}の倍数でない {w}x{h}): {src.name}"
                im = im.crop((0, 0, w2, h2))
                if fmt_ext in (".jpg", ".jpeg"):
                    im.save(src, quality=100, subsampling=0)
                else:
                    im.save(src)
            lr = im.resize((w2 // scale, h2 // scale), Image.BICUBIC)
            out = lr_dir / src.name
            if fmt_ext in (".jpg", ".jpeg"):
                lr.save(out, quality=100, subsampling=0)
            else:
                lr.save(out)
    except Exception as e:  # 壊れた画像などは飛ばして続行
        return f"error({e}): {src.name}"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", default="datasets/GT", help="高解像度画像のフォルダ")
    p.add_argument("--lr", default="datasets/LR", help="低解像度画像の出力先フォルダ")
    p.add_argument("--scale", type=int, default=2, help="縮小倍率(YAMLのscaleと同じにする)")
    p.add_argument("--crop-gt", action="store_true", help="GTを倍数に切り落とす(GTを上書き)")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    gt_dir, lr_dir = Path(a.gt), Path(a.lr)
    files = sorted(f for f in gt_dir.iterdir() if f.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"画像が見つかりません: {gt_dir}")
    lr_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(files)} 枚を x{a.scale} に縮小します: {gt_dir} -> {lr_dir}")
    jobs = [(f, lr_dir, a.scale, a.crop_gt) for f in files]
    msgs = []
    with Pool(a.workers) as pool:
        for i, m in enumerate(pool.imap_unordered(process, jobs, chunksize=16), 1):
            if m:
                msgs.append(m)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}")
    for m in msgs:
        print(m)
    print(f"完了(スキップ/エラー {len(msgs)} 件)")
    if msgs:
        print("※ スキップした画像はLRがないため、このままだとGTとLRの枚数が合わず学習でエラーになります。"
              "--crop-gt を付けて再実行するか、該当GTを削除してください。")


if __name__ == "__main__":
    main()
