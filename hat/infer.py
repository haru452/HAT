# flake8: noqa
"""重みと入力フォルダを引数で指定して、HAT で超解像する (正解画像は不要)。

例 (HAT フォルダで実行):
  python hat/infer.py --weight experiments/train_HAT-L_SRx2_finetune/models/net_g_100000.pth
  python hat/infer.py --weight experiments/train_HAT-L_SRx2_finetune_realesrgan/models/net_g_100000.pth \
      --input datasets/infer_input --tile_size 128

ほかの設定 (モデルの形・タイル処理など) は options/test/HAT-L_SRx2_finetune_infer.yml から読み、
引数で指定したものだけを上書きして、basicsr の test と同じ処理を行う。
出力: results/<name>/visualization/<入力フォルダ名>/<元の名前>_<name>.png
"""
import argparse
import os
import os.path as osp
import sys
import tempfile

import yaml

import hat.archs
import hat.data
import hat.models
from basicsr.test import test_pipeline

DEFAULT_OPT = 'options/test/HAT-L_SRx2_finetune_infer.yml'


def default_name(weight):
    """experiments/<実験名>/models/net_g_100000.pth -> <実験名>_net_g_100000 (結果の置き場がぶつからないように)"""
    stem = osp.splitext(osp.basename(weight))[0]
    exp = osp.basename(osp.dirname(osp.dirname(osp.abspath(weight))))
    return f'{exp}_{stem}' if exp else stem


def main():
    p = argparse.ArgumentParser(description='HAT で超解像する')
    p.add_argument('--weight', required=True, help='使う重み (.pth)')
    p.add_argument('--input', default=None, help='超解像したい画像のフォルダ (省略時は yml の dataroot_lq)')
    p.add_argument('--name', default=None, help='結果フォルダの名前 (省略時は重みのパスから自動で付ける)')
    p.add_argument('--tile_size', type=int, default=None, help='タイルの大きさ (16 の倍数。メモリ不足なら下げる)')
    p.add_argument('--param_key', default=None, help='重みの中のキー (省略時は yml の値 = params_ema)')
    p.add_argument('-opt', default=DEFAULT_OPT, help='元にする yml')
    args = p.parse_args()

    if not osp.isfile(args.weight):
        raise SystemExit(f'重みが見つかりません: {args.weight}')
    with open(args.opt, encoding='utf-8') as f:
        opt = yaml.safe_load(f)

    opt['path']['pretrain_network_g'] = args.weight
    if args.param_key:
        opt['path']['param_key_g'] = args.param_key
    opt['name'] = args.name or default_name(args.weight)
    ds = opt['datasets']['test_1']
    if args.input:
        ds['dataroot_lq'] = args.input
        ds['name'] = osp.basename(osp.normpath(args.input))
    if not osp.isdir(ds['dataroot_lq']):
        raise SystemExit(f'入力フォルダが見つかりません: {ds["dataroot_lq"]}')
    if args.tile_size:
        if args.tile_size % 16:
            raise SystemExit('--tile_size は 16 の倍数にしてください')
        opt.setdefault('tile', {'tile_pad': 32})['tile_size'] = args.tile_size

    print(f'重み: {args.weight}\n入力: {ds["dataroot_lq"]}\n結果: results/{opt["name"]}/visualization/{ds["name"]}/')

    # 上書きした設定を一時ファイルに書き、basicsr の test にそのまま渡す
    fd, tmp = tempfile.mkstemp(suffix='.yml')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        yaml.safe_dump(opt, f, allow_unicode=True, sort_keys=False)
    try:
        sys.argv = [sys.argv[0], '-opt', tmp]
        root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
        test_pipeline(root_path)
    finally:
        os.remove(tmp)


if __name__ == '__main__':
    main()
