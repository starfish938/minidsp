import matplotlib.pyplot as plt
import numpy as np

# 读取数据
# 用绝对路径
data = np.loadtxt('C:/Users/Magewell/Desktop/EE/miniDSP_library/build/signal_data.txt', skiprows=1)  # skiprows=1 跳过表头
time = data[:, 0]      # 第一列：时间
original = data[:, 1]  # 第二列：原始信号
filtered = data[:, 2]  # 第三列：滤波后信号

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制原始信号P
plt.plot(time, original, label='origin signal', alpha=0.7)

# 绘制滤波后信号
plt.plot(time, filtered, label='filtered signal', alpha=0.7)

# 添加标签和图例
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.title('Comparation')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()
data = np.loadtxt(fname=r"./data.csv", dtype=float, skiprows=1, delimiter=",")



# 可选：保存图片
plt.savefig('signal_comparison.png')