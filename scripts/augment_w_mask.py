import os
import glob
import cv2
import albumentations as A
from tqdm import tqdm

# === 配置 ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "combined_dataset")  # 训练数据集根目录
TARGET_CLASS = 1  # W_mask 类别索引
AUG_CYCLES = 3    # 每张图片生成的增强数量

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.ImageCompression(quality_range=(60, 90), p=0.3),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)


def augment_minority_class():
    img_dir = os.path.join(DATA_ROOT, "images", "train")
    lbl_dir = os.path.join(DATA_ROOT, "labels", "train")

    target_files = []
    print("🔍 正在扫描 W_mask 样本...")

    for lbl_file in glob.glob(os.path.join(lbl_dir, "*.txt")):
        with open(lbl_file, "r", encoding="utf-8") as f:
            if any(int(line.split()[0]) == TARGET_CLASS for line in f if line.strip()):
                target_files.append(lbl_file)

    print(f"✅ 发现 {len(target_files)} 张包含 W_mask 的图片，准备生成 {len(target_files) * AUG_CYCLES} 张增强样本。")

    for lbl_path in tqdm(target_files):
        basename = os.path.basename(lbl_path).rsplit(".", 1)[0]
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = os.path.join(img_dir, basename + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if not img_path:
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        bboxes = []
        class_labels = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    bboxes.append(coords)
                    class_labels.append(cls)

        if not bboxes:
            continue

        for i in range(AUG_CYCLES):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_labels = augmented["class_labels"]

                if not aug_bboxes:
                    continue

                new_basename = f"{basename}_aug_{i}"
                cv2.imwrite(os.path.join(img_dir, new_basename + ".jpg"), aug_img)

                with open(os.path.join(lbl_dir, new_basename + ".txt"), "w", encoding="utf-8") as f_out:
                    for cls, box in zip(aug_labels, aug_bboxes):
                        box = [min(max(x, 0.0), 1.0) for x in box]
                        f_out.write(f"{cls} {' '.join(map(str, box))}\n")
            except Exception as e:
                print(f"⚠️ 增强失败: {basename} -> {e}")


if __name__ == "__main__":
    augment_minority_class()
