import pandas as pd
import matplotlib.pyplot as plt

# 读取训练结果
csv_path = r"e:/graduation_project/runs/yolov11_mask_custom_test/custom/results.csv"
df = pd.read_csv(csv_path)

# 损失曲线
plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['train/box_loss'], label='train/box_loss')
plt.plot(df['epoch'], df['val/box_loss'], label='val/box_loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='train/cls_loss')
plt.plot(df['epoch'], df['val/cls_loss'], label='val/cls_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Box/Cls Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('e:/graduation_project/runs/yolov11_mask_custom_test/custom/loss_curve.png')
plt.close()

# 精度曲线
plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
plt.plot(df['epoch'], df['metrics/precision(B)'], label='precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Precision/Recall/mAP Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('e:/graduation_project/runs/yolov11_mask_custom_test/custom/metric_curve.png')
plt.close()

print('曲线已保存至runs/yolov11_mask_custom_test/custom/')
