import numpy as np
import matplotlib.pyplot as plt         # 导入pyplot，文档里面的例子通常将其别名为plt

# plt.figure()    # 创建一幅图

plt.plot([2, 4, 3, 5], linewidth=3.0)                  # matplotlib 默认是y轴的数值序列
line, = plt.plot([1, 3, 2, 4], linewidth=3.0)          # matplotlib 默认是y轴的数值序列
line.set_antialiased(False)     # 关闭抗锯齿像素

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro', linewidth=6.0)

plt.axis([0, 6, 0, 20])         # 函数传入形式为[xmin,xmax,ymin,ymax]的列表，指定了坐标轴的范围

plt.ylabel('some numbers')      # 为y轴加注释

plt.show()      # 显示


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

# np.arange(start, stop, step):从start开始输出，每隔step输出一个数，直到大于stop截止
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)                               # 创建一幅图

plt.subplot(211)                            # 指定一个坐标系，2行1列，指定第一个坐标系
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)                            # 指定一个坐标系，2行1列，指定第二个坐标系
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)                              # 显示网格
plt.show()



