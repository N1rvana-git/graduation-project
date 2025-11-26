#!/usr/bin/env python3
"""
检查模型精度和训练结果
"""

import torch
import os
from pathlib import Path
import pandas as pd

def check_model_metrics():
    """检查模型精度指标"""
    
    # 检查可用的模型文件
    model_paths = [
        'models/quick_runs/quick_mask_detection/weights/best.pt',
        'models/runs/mask_detection/weights/best.pt',
        'models/weights/best.pt'
    ]

    print('=== 模型文件检查 ===')
    current_model = None
    
    for path in model_paths:
        if os.path.exists(path):
            print(f'✓ 找到模型: {path}')
            current_model = path
            try:
                # 尝试加载模型并查看信息
                checkpoint = torch.load(path, map_location='cpu')
                if 'epoch' in checkpoint:
                    print(f'  - 训练轮数: {checkpoint["epoch"]}')
                if 'best_fitness' in checkpoint:
                    print(f'  - 最佳适应度: {checkpoint["best_fitness"]:.4f}')
                if 'model' in checkpoint:
                    print(f'  - 模型类型: {type(checkpoint["model"])}')
                print()
                break  # 使用第一个找到的模型
            except Exception as e:
                print(f'  - 加载失败: {e}')
        else:
            print(f'✗ 未找到: {path}')

    # 检查CSV结果文件
    csv_paths = [
        'models/quick_runs/quick_mask_detection/results.csv',
        'models/runs/mask_detection/results.csv'
    ]
    
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            print(f'=== 训练结果 ({csv_path}) ===')
            try:
                # 使用pandas读取CSV文件
                df = pd.read_csv(csv_path)
                
                if len(df) > 0:
                    # 获取最后一行的结果
                    last_row = df.iloc[-1]
                    
                    print('最终训练指标:')
                    
                    # 检查各种可能的列名
                    metric_mappings = {
                        'mAP@0.5': ['metrics/mAP50(B)', 'mAP50', 'mAP_0.5'],
                        'mAP@0.5:0.95': ['metrics/mAP50-95(B)', 'mAP50-95', 'mAP_0.5:0.95'],
                        'Precision': ['metrics/precision(B)', 'precision', 'P'],
                        'Recall': ['metrics/recall(B)', 'recall', 'R'],
                        'F1-Score': ['F1', 'f1']
                    }
                    
                    for metric_name, possible_cols in metric_mappings.items():
                        for col in possible_cols:
                            if col in df.columns:
                                value = last_row[col]
                                print(f'  {metric_name}: {value:.4f} ({value*100:.2f}%)')
                                break
                    
                    print(f'\n训练总轮数: {len(df)}')
                    print(f'当前使用模型: {current_model}')
                    
            except Exception as e:
                print(f'读取CSV文件失败: {e}')
            
            break  # 只处理第一个找到的结果文件
    
    # 如果没有找到结果文件，尝试手动解析
    if not any(os.path.exists(path) for path in csv_paths):
        print('=== 未找到训练结果文件 ===')
        print('建议重新训练模型以获取详细的精度指标')

if __name__ == '__main__':
    check_model_metrics()