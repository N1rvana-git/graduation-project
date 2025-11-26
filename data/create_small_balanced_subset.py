"""从已生成的 YOLO 数据集中抽取小规模、近平衡的子集用于快速训练/调试。

示例用法：
python data/create_small_balanced_subset.py --src data/yolo_dataset --dst data/yolo_dataset_small \
    --train_size 2000 --val_size 500 --test_size 500 --mask_ratio 0.3
"""
from pathlib import Path
import argparse
import random
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_label_class(label_path: Path):
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if not lines:
            return None
        # 取第一个目标的 class id
        parts = lines[0].split()
        return int(parts[0])
    except Exception:
        return None


def collect_samples(src: Path, split: str):
    img_dir = src / 'images' / split
    label_dir = src / 'labels' / split
    imgs = list(img_dir.glob('*.jpg'))
    samples = []
    for img in imgs:
        label = label_dir / f"{img.stem}.txt"
        cls = read_label_class(label) if label.exists() else None
        samples.append({'img': img, 'label': label, 'class': cls})
    return samples


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_samples(samples, dst_src_dir: Path, split: str):
    out_img_dir = dst_src_dir / 'images' / split
    out_label_dir = dst_src_dir / 'labels' / split
    ensure_dir(out_img_dir)
    ensure_dir(out_label_dir)
    for s in samples:
        dst_img = out_img_dir / f"{s['img'].stem}.jpg"
        dst_lab = out_label_dir / f"{s['img'].stem}.txt"
        shutil.copy2(s['img'], dst_img)
        if s['label'].exists():
            shutil.copy2(s['label'], dst_lab)


def create_dataset_yaml(dst: Path, nc=2, names=('no_mask', 'mask')):
    cfg = dst / 'dataset.yaml'
    with open(cfg, 'w', encoding='utf-8') as f:
        f.write(f"path: {dst.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write(f"nc: {nc}\n")
        f.write("names: ['no_mask', 'mask']\n")
    logger.info("创建 dataset.yaml: %s", cfg)


def build_small_set(src: Path, dst: Path, train_size: int, val_size: int, test_size: int, mask_ratio: float):
    dst_src_dir = dst
    # collect from each split proportionally
    splits = ['train', 'val', 'test']
    demands = {'train': train_size, 'val': val_size, 'test': test_size}

    for split in splits:
        samples = collect_samples(src, split)
        if not samples:
            logger.warning('split %s has no samples, skipping', split)
            continue

        mask_samples = [s for s in samples if s['class'] == 1]
        nomask_samples = [s for s in samples if s['class'] == 0]

        n_total = demands[split]
        n_mask = int(n_total * mask_ratio)
        n_nomask = n_total - n_mask

        chosen = []
        if mask_samples:
            chosen_mask = random.choices(mask_samples, k=n_mask) if len(mask_samples) < n_mask else random.sample(mask_samples, n_mask)
        else:
            chosen_mask = []

        if nomask_samples:
            chosen_nomask = random.choices(nomask_samples, k=n_nomask) if len(nomask_samples) < n_nomask else random.sample(nomask_samples, n_nomask)
        else:
            chosen_nomask = []

        chosen = chosen_mask + chosen_nomask
        random.shuffle(chosen)

        copy_samples(chosen, dst_src_dir, split)
        logger.info('为 split=%s 创建样本: %d (mask=%d, no_mask=%d)', split, len(chosen), len(chosen_mask), len(chosen_nomask))

    create_dataset_yaml(dst_src_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='源 YOLO 数据集 目录（data/yolo_dataset）')
    parser.add_argument('--dst', required=True, help='目标小数据集 目录')
    parser.add_argument('--train_size', type=int, default=2000)
    parser.add_argument('--val_size', type=int, default=500)
    parser.add_argument('--test_size', type=int, default=500)
    parser.add_argument('--mask_ratio', type=float, default=0.3, help='目标 mask 比例（0-1）')

    args = parser.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    random.seed(42)

    if not src.exists():
        raise FileNotFoundError(f"源目录不存在: {src}")

    if dst.exists():
        logger.warning('目标目录已存在，可能会覆盖: %s', dst)

    build_small_set(src, dst, args.train_size, args.val_size, args.test_size, args.mask_ratio)
    logger.info('小数据集创建完成: %s', dst)


if __name__ == '__main__':
    main()
