import pandas as pd

# 读取训练结果CSV
csv_path = r"e:/graduation_project/runs/yolov11_mask_custom_test/custom/results.csv"
df = pd.read_csv(csv_path)

# 输出每轮主要指标
print("每轮主要指标：")
print(df[['epoch', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss']])

# 自动输出最佳mAP50轮次
best_row = df.loc[df['metrics/mAP50(B)'].idxmax()]
print('\n最佳mAP50出现在第{}轮，值为{:.3f}'.format(int(best_row['epoch']), best_row['metrics/mAP50(B)']))

# 自动输出最佳mAP50-95轮次
best_row_95 = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
print('最佳mAP50-95出现在第{}轮，值为{:.3f}'.format(int(best_row_95['epoch']), best_row_95['metrics/mAP50-95(B)']))
