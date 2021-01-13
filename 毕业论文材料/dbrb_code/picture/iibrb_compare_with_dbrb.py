import numpy as np
import matplotlib.pyplot as plt

# 均值；KNN；EM；MI；this
name = ['均值方法', 'KNN方法', 'EM方法', 'MI方法', '本文方法']
single_attr_acc_avg_2CV = [78.43333335, 75.9666665, 80.03333325, 94.29999975, 92.7]
single_attr_acc_avg_5CV = [76.8333335, 77.81666675, 81.65, 92, 93.3833335]
single_attr_acc_avg_10CV = [76.46666575, 80.85, 82.60000025, 94.1999999175, 94.833]

# multi_attr_acc_avg_2CV =[73.378, 69.556, 76.044, 92.667, ]
# multi_attr_acc_avg_5CV =[67.222, 72.067, 76.978, 93.933, ]
# multi_attr_acc_avg_10CV =[,,,, 94.667]
# 开启一个窗口
fig = plt.figure()
# 使用add_subplot在窗口加子图，其本质就是添加坐标系
# 三个参数分别为：行数，列数，本子图是所有子图中的第几个，最后一个参数设置错了子图可能发生重叠
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
# 绘制曲线
ax1.bar(range(len(name)), single_attr_acc_avg_2CV, tick_label=name, label=' 2CV')
ax1.set_xlabel("(a)")
ax1.set_ylabel("percentage of classes correctly classified(%)")
ax1.legend()

ax2.bar(range(len(name)), single_attr_acc_avg_5CV, tick_label=name, label=' 5CV')
ax2.set_xlabel("(b)")
ax2.set_ylabel("percentage of classes correctly classified(%)")
ax2.legend()

ax3.bar(range(len(name)), single_attr_acc_avg_10CV, tick_label=name, label='10CV')
ax2.set_xlabel("(c)")
ax3.set_ylabel("percentage of classes correctly classified(%)")
ax3.legend()

plt.tight_layout()
plt.show()
