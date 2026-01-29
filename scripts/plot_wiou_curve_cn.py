import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def generate_wiou_curve_cn():
    # 虽然是在 Windows，但为了保险起见，尝试配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False 

    # 1. 设置参数
    alpha = 1.9
    delta = 3.0
    
    # 2. 生成数据点 (beta 从 0 到 10)
    beta = np.linspace(0, 8, 500)
    
    # 3. 计算 r
    r = beta / (delta * np.power(alpha, beta - delta))
    
    # 4. 绘图设置
    plt.figure(figsize=(9, 6), dpi=300) #稍微大一点
    
    # 画主曲线
    plt.plot(beta, r, color='#FF5722', linewidth=3, label='WIoU v3 聚焦曲线')
    
    # 5. 调整坐标轴范围，确保峰值可见
    # 计算真实峰值位置 beta = 1/ln(alpha)
    peak_beta_real = 1 / np.log(alpha)
    peak_r_real = peak_beta_real / (delta * np.power(alpha, peak_beta_real - delta))
    
    # 设置 Y 轴上限为峰值得 1.1 倍左右
    plt.ylim(0, peak_r_real * 1.15)
    plt.xlim(0, 8)

    # 6. 标注关键区域和点
    
    # 标注 delta 点 (论文中的核心参设点)
    r_delta = 3.0 / (3.0 * np.power(alpha, 0)) # r=1.0
    plt.scatter([delta], [r_delta], color='red', s=60, zorder=5)
    plt.text(delta + 0.2, r_delta + 0.05, f'聚焦锚点 ($\\beta={delta}, r=1.0$)\n普通困难样本', 
             ha='left', va='bottom', fontsize=10, fontweight='bold')

    # 标注真实峰值点 (数学上的最高点)
    plt.scatter([peak_beta_real], [peak_r_real], color='orange', s=40, zorder=5, marker='x')
    plt.vlines(peak_beta_real, 0, peak_r_real, linestyles='--', colors='orange', alpha=0.5)
    plt.text(peak_beta_real, peak_r_real + 0.05, f'最大增益峰值\n($\\approx {peak_r_real:.2f}$)', 
             ha='center', fontsize=9, color='#E65100')

    # 简单样本区域 (左侧)
    plt.axvspan(0, 1.2, color='#4CAF50', alpha=0.1)
    plt.text(0.6, 0.2, '简单样本\n(降低权重)', ha='center', color='#2E7D32', fontsize=11, fontweight='bold')
    
    # 极端样本区域 (右侧)
    plt.axvspan(5, 8, color='#9E9E9E', alpha=0.1)
    plt.text(6.5, 0.2, '离群/脏数据\n(梯度抑制)', ha='center', color='#616161', fontsize=11, fontweight='bold')
    
    # 中间区域文字
    plt.text(3.5, 0.6, '优质梯度贡献区', ha='center', color='#D84315', fontsize=12, fontweight='bold', alpha=0.3, rotation=-30)

    # 7. 坐标轴修饰
    plt.title(r'WIoU v3 动态非单调聚焦机制图解 ($\alpha=1.9, \delta=3.0$)', fontsize=14, pad=20)
    plt.xlabel(r'离群度 Outlierness ($\beta$)', fontsize=12)
    plt.ylabel(r'梯度增益 Gradient Gain ($r$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 保存
    save_path = 'models/wiou_curve_cn.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Chart saved to {save_path}")

if __name__ == "__main__":
    generate_wiou_curve_cn()
