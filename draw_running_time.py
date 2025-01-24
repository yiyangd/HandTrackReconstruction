import matplotlib.pyplot as plt

# 数据
methods = ['Powell', 'bfgs', 'nelder-mead', 'trust-constr']
angles = [6.01476812, 3.77822521, 2.18451971, 11.6789441]
errors = [0.108923296, 0.0733223448, 0.565140053, 0.177203538]

# 画图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

# 角度的柱状图
ax1.bar(methods, angles, color='skyblue')
ax1.set_ylabel('Time (s)', fontsize=16)
ax1.set_xlabel('Method', fontsize=16)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)

# 误差的柱状图
ax2.bar(methods, errors, color='lightcoral')
ax2.set_ylabel('Error', fontsize=14)
ax2.set_xlabel('Method', fontsize=14)

# 显示图像
plt.show()