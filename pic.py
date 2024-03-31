import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
line1 = []
line2 = []
line3 = []
line4 = []
line5 = []

line11 = []
line22 = []
line33 = []
line44 = []


# 轮数
epochs = []
plt.rcParams.update({'font.size': 12, 'text.color': 'black'})
for i in range(1, 201):
    epochs.append(i)
with open('./result/mthfl_local_non_iid.txt', 'r') as file:
    # 逐行读取文件内容
    lines = file.readlines()
    for line in lines:
        match = re.search(r'cbs (\d+)模型测试： acc: (\d+\.\d+)', line)
        if match:
            # 提取浮点数部分并转换为浮点型
            id = int(match.group(1))
            accuracy = float(match.group(2))
            if id == 0:
                line1.append(accuracy)
            elif id == 1:
                line2.append(accuracy)
            elif id == 2:
                line3.append(accuracy)
            elif id == 3:
                line4.append(accuracy)


# 打开文件
with open('./result/rohfl-non_iid.txt', 'r') as file:
    # 逐行读取文件内容
    lines = file.readlines()
    for line in lines:
        match = re.search(r'全局测试： acc: (\d+\.\d+)', line)
        if match:
            # 提取浮点数部分并转换为浮点型
            accuracy = float(match.group(1))
            line5.append(accuracy)
        match2 = re.search(r'cbs (\d+)模型测试： acc: (\d+\.\d+)', line)
        if match2:
            # 提取浮点数部分并转换为浮点型
            id = int(match2.group(1))
            accuracy = float(match2.group(2))
            if id == 0:
                line11.append(accuracy)
            elif id == 1:
                line22.append(accuracy)
            elif id == 2:
                line33.append(accuracy)
            elif id == 3:
                line44.append(accuracy)
# 画折线图
plt.plot(epochs, line1, 'r', label='mthfl-cbs0')
plt.plot(epochs, line2, 'b', label='mthfl-cbs1')
plt.plot(epochs, line3, 'g', label='mthfl-cbs2')
plt.plot(epochs, line4, 'c', label='mthfl-cbs3')
plt.plot(epochs, line5, 'k', label='rohfl')
#plt.plot(epochs, line11, 'r--', label='rohfl-cbs0')
#plt.plot(epochs, line22, 'b--', label='rohfl-cbs1')
#plt.plot(epochs, line33, 'g--', label='rohfl-cbs2')
#plt.plot(epochs, line44, 'c--', label='rohfl-cbs3')

# 添加标题和标签
plt.title('准确率随轮数变化')
plt.xlabel('轮数')
plt.ylabel('准确率')
plt.legend(fontsize=12)
# 添加图例
plt.legend()
plt.savefig('output_figure.png', dpi=600)
# 显示图形
plt.show()