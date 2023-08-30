import matplotlib.pyplot as plt

filepath  = "traindata.txt"

with open(filepath, 'r', encoding='utf-8') as infile:
    # 每行分开读取
    data1_1 = []
    data2_1 = []
    data3_1 = []
    data4_1 = []

    data1_2 = []
    data2_2 = []
    data3_2 = []
    data4_2 = []

    data1_3 = []
    data2_3 = []
    data3_3 = []
    data4_3 = []
    for line in infile:
        data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
        if int(data_line[4]) == 1:
            data1_1.append(float(data_line[0]))
            data2_1.append(float(data_line[1]))
            data3_1.append(float(data_line[2]))
            data4_1.append(float(data_line[3]))
        elif int(data_line[4]) == 2:
            data1_2.append(float(data_line[0]))
            data2_2.append(float(data_line[1]))
            data3_2.append(float(data_line[2]))
            data4_2.append(float(data_line[3]))
        elif int(data_line[4]) == 3:
            data1_3.append(float(data_line[0]))
            data2_3.append(float(data_line[1]))
            data3_3.append(float(data_line[2]))
            data4_3.append(float(data_line[3]))
labels = ["1","2","3"]


plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(221)
plt.grid(True)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.boxplot([data1_1,data1_2,data1_3],labels=labels)
plt.title("特征一的区间分布",fontsize=8)

plt.subplot(222)
plt.grid(True)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.boxplot([data2_1,data2_2,data2_3],labels=labels)
plt.title("特征二的区间分布",fontsize=8)

plt.subplot(223)
plt.grid(True)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.boxplot([data3_1,data3_2,data3_3],labels=labels)
plt.title("特征三的区间分布",fontsize=8)

plt.subplot(224)
plt.grid(True)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.boxplot([data4_1,data4_2,data4_3],labels=labels)
plt.title("特征四的区间分布",fontsize=8)

plt.suptitle("训练集的数据分析")
plt.show()

