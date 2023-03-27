import matplotlib.pyplot as plt

# 创建图形和子图
fig, ax = plt.subplots()

# 添加虚线
y_value = 0.5
ax.axhline(y=y_value, linestyle='dashed')

# 在坐标轴上添加文本标签
label_text = '{:.1f}'.format(y_value)
ax.text(-0.05, y_value, label_text, ha='center', va='center')

# 显示图形
plt.show()
