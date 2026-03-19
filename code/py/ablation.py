import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

# --- 1. 样式配置 ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri'] + plt.rcParams['font.sans-serif']

hzysfontsize = 21
plt.rcParams['font.size'] = hzysfontsize

# --- 2. 数据准备 ---
labels = [f'#{i}' for i in range(1, 11)]
#                    1   2   3      4  5   6  7   8  9   10
# data = {
#     'MoT':       [60, 1, 2206, 19, 2, 14, 6, 97, 4, 103],
#     'MoT_dl':    [60, 1,    0, 19, 2, 14, 6, 97, 4, 103],
#     'MoT_svm':   [53, 0, 2206, 13, 1,  5, 4, 37, 4,  84]
# }


data = {
    'MoT':       [1,  19, 2, 14, 6, 97, 4, 103, 60, 2206],
    'MoT_dl':    [1, 19, 2, 14, 6, 97, 4, 103, 0, 0],
    'MoT_svm':   [1,  13, 1,  5, 4, 37, 4,  84, 60, 2206]
}

colors = {'MoT': '#FA7F6F', 'MoT_dl': '#82B0D2', 'MoT_svm': '#FFBE7A'}
hatches = {'MoT': '///', 'MoT_dl': '\\\\\\', 'MoT_svm': '...'}

# 柱状图位置计算
x = np.arange(len(labels))
width = 0.25 
bar_keys = list(data.keys()) # ['MoT', 'MoT_dl', 'MoT_svm']

# --- 3. 绘图 ---
fig, ax = plt.subplots(figsize=(10, 3.3))

# 3.1 绘制柱子 (不在此处标注数值)
for i, name in enumerate(bar_keys):
    offset = (i - 1) * width
    
    if '_' in name:
        base, suffix = name.split('_')
        label_display = f'{base}$^{{{suffix}}}$'
    else:
        label_display = name

    ax.bar(x + offset, data[name], width, 
           label=label_display, 
           color=colors[name], 
           hatch=hatches[name], 
           edgecolor='black', 
           linewidth=1.0, 
           zorder=3)

# 3.2 [核心修改] 智能标注数值
# 定义一个辅助函数来画字
def add_text(pos_x, value):
    # 处理 0 值在 symlog 下的显示位置
    pos_y = value if value > 0 else 0.1
    
    # [修改点]：如果数值是 2206，则显示为 '2.2k'，否则显示整数
    if int(value) == 2206:
        display_text = '2.2k'
    else:
        display_text = f'{int(value)}'
        
    ax.text(pos_x, pos_y, display_text, 
            ha='center', va='bottom', fontsize=12, color='black')

# 遍历每一组数据 (x=0 到 9)
for i in range(len(labels)):
    # 获取当前组的三个值
    v1 = data['MoT'][i]
    v2 = data['MoT_dl'][i]
    v3 = data['MoT_svm'][i]
    
    # 对应的 x 坐标中心点
    x1 = x[i] - width  # 左边柱子中心
    x2 = x[i]          # 中间柱子中心
    x3 = x[i] + width  # 右边柱子中心
    
    # --- 判断逻辑 ---
    
    # 情况 1: 三个值都相等 -> 在中间标注一次
    if v1 == v2 == v3:
        add_text(x2, v2)
        
    # 情况 2: 前两个相等 (且与第三个不等) -> 在 1和2 之间标注，3 单独标注
    elif v1 == v2:
        add_text((x1 + x2) / 2, v1) # 1和2的中点
        add_text(x3, v3)
        
    # 情况 3: 后两个相等 (且与第一个不等) -> 1 单独标注，在 2和3 之间标注
    elif v2 == v3:
        add_text(x1, v1)
        add_text((x2 + x3) / 2, v2) # 2和3的中点
        
    # 情况 4: 都不相等 (或者 1==3 但 !=2，这种不相邻的情况也分开标)
    else:
        add_text(x1, v1)
        add_text(x2, v2)
        add_text(x3, v3)

# --- 4. 坐标轴设置 (Symlog) ---
ax.set_yscale('symlog', linthresh=1)
ax.set_yticks([0, 1, 10, 100, 2500])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# --- 5. 细节调整 ---
ax.set_xticks(x)
ax.set_xticklabels(labels)

# [保持您的修改] 图例位置
ax.legend(fontsize=18, 
          frameon=False, 
          loc='upper center', 
          bbox_to_anchor=(0.5, 1.07), 
          ncol=3, 
          columnspacing=1.5)

padding_value = 10 
ax.tick_params(axis='x', which='major', pad=padding_value)

ax.set_xlabel('Exploitation types', fontsize=hzysfontsize, labelpad=10)
ax.set_ylabel('# of detections', fontsize=hzysfontsize)

ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.6, zorder=0)

ax.set_ylim(0, 20000)
ax.set_xlim(-0.6, len(labels) - 0.4)

# --- 6. 保存 ---
plt.tight_layout()
output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, 'exp_ablation.pdf')
plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')

print(f"Chart saved to {save_path}")