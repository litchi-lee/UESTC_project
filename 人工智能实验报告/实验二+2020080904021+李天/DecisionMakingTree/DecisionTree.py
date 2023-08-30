import time

import numpy as np


class DecisionTree():
    def __init__(self, train_filepath, test_filepath):
        self.range_fea = [[5.5, 6, 7, 10], [2.5, 3, 3.5, 10], [3, 5, 10], [1, 1.5, 10]]
        self.fea_class_num = []
        for fea_range in self.range_fea:
            self.fea_class_num.append(len(fea_range))
        self.train_data, self.train_label = self.loadData(train_filepath)
        self.test_data, self.test_label = self.loadData(test_filepath)

    def loadData(self, fileName):
        '''
        加载文件
        :param fileName:要加载的文件路径
        :return: 数据集和标签集
        '''
        # 存放数据及标记
        dataArr = []
        labelArr = []
        with open(fileName, 'r', encoding='utf-8') as infile:
            for line in infile:
                data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
                feature = [float(data) for data in data_line[0:4]]
                for i in range(len(feature)):
                    fea = feature[i]
                    range_each = self.range_fea[i]
                    for j in range(len(range_each)):
                        if fea <= float(range_each[j]):
                            feature[i] = j
                            break
                dataArr.append(feature)
                labelArr.append(int(data_line[4]))
        # 返回数据集和标记
        return dataArr, labelArr

    def majorClass(self, labelArr):
        '''
        找到当前标签集中占数目最大的标签
        :param labelArr: 标签集
        :return: 最大的标签
        '''
        # 建立字典，用于不同类别的标签技术
        classDict = {}
        # 遍历所有标签
        for i in range(len(labelArr)):
            if labelArr[i] in classDict.keys():
                classDict[labelArr[i]] += 1
            else:
                classDict[labelArr[i]] = 1
        # 对字典依据值进行降序排序
        classSort = sorted(classDict.items(), key=lambda x: x[1], reverse=True)
        # 返回最大一项的标签，即占数目最多的标签
        return classSort[0][0]

    def calc_H_D(self, trainLabelArr):
        '''
        计算数据集D的经验熵
        :param trainLabelArr:当前数据集的标签集
        :return: 经验熵
        '''
        H_D = 0
        trainLabelSet = set([label for label in trainLabelArr])
        for i in trainLabelSet:
            # 计算|Ck|/|D|
            p = trainLabelArr[trainLabelArr == i].size / trainLabelArr.size
            # 对经验熵的每一项累加求和
            H_D += -1 * p * np.log2(p)
        return H_D

    def calcH_D_A(self, trainDataArr_DevFeature, trainLabelArr):
        '''
        计算经验条件熵
        :param trainDataArr_DevFeature:切割后只有feature那列数据的数组
        :param trainLabelArr: 标签集数组
        :return: 经验条件熵
        '''
        H_D_A = 0
        trainDataSet = set([label for label in trainDataArr_DevFeature])

        # 对于每一个特征取值遍历计算条件经验熵的每一项
        for i in trainDataSet:
            # 计算H(D|A)
            # trainDataArr_DevFeature[trainDataArr_DevFeature == i].size / trainDataArr_DevFeature.size:|Di| / |D|
            # calc_H_D(trainLabelArr[trainDataArr_DevFeature == i]):H(Di)
            H_D_A += trainDataArr_DevFeature[trainDataArr_DevFeature == i].size / trainDataArr_DevFeature.size \
                     * self.calc_H_D(trainLabelArr[trainDataArr_DevFeature == i])
        # 返回得出的条件经验熵
        return H_D_A

    def calcBestFeature(self, trainDataList, trainLabelList):
        '''
        计算信息增益最大的特征
        :param trainDataList: 当前数据集
        :param trainLabelList: 当前标签集
        :return: 信息增益最大的特征及最大信息增益值
        '''
        trainDataArr = np.array(trainDataList)
        trainLabelArr = np.array(trainLabelList)
        featureNum = trainDataArr.shape[1]

        maxG_D_A = -1
        maxFeature = -1

        # 1.计算数据集D的经验熵H(D)
        H_D = self.calc_H_D(trainLabelArr)
        for feature in range(featureNum):
            trainDataArr_DevideByFeature = np.array(trainDataArr[:, feature].flat)
            # 2.计算信息增益G(D|A)    G(D|A) = H(D) - H(D | A)
            G_D_A = H_D - self.calcH_D_A(trainDataArr_DevideByFeature, trainLabelArr)
            if G_D_A > maxG_D_A:
                maxG_D_A = G_D_A
                maxFeature = feature
        return maxFeature, maxG_D_A

    def getSubDataArr(self, trainDataArr, trainLabelArr, A, a):
        '''
        更新数据集和标签集
        :param trainDataArr:要更新的数据集
        :param trainLabelArr: 要更新的标签集
        :param A: 要去除的特征索引
        :param a: 当data[A]== a时，说明该行样本时要保留的
        :return: 新的数据集和标签集
        '''
        retDataArr = []
        retLabelArr = []
        # 对当前数据的每一个样本进行遍历
        for i in range(len(trainDataArr)):
            # 如果当前样本的特征为指定特征值a
            if trainDataArr[i][A] == a:
                # 那么将该样本的第A个特征切割掉，放入返回的数据集中
                retDataArr.append(trainDataArr[i][0:A] + trainDataArr[i][A + 1:])
                retLabelArr.append(trainLabelArr[i])
        return retDataArr, retLabelArr

    def createTree(self, *dataSet):
        '''
        递归创建决策树
        :param dataSet:(trainDataList， trainLabelList) <<-- 元祖形式
        :return:新的子节点或该叶子节点的值
        '''
        Epsilon = 0.1
        trainDataList = dataSet[0][0]
        trainLabelList = dataSet[0][1]
        # 打印信息：开始一个子节点创建，打印当前特征向量数目及当前剩余样本数目
        print('创建一个子节点，此时数据集中特征维度为{}，剩余样本数量为{}。'.format(len(trainDataList[0]), len(trainLabelList)))

        classDict = set(trainLabelList)
        # 如果D中所有实例属于同一类Ck，则置T为单节点数，并将Ck作为该节点的类，返回T
        if len(classDict) == 1:
            return trainLabelList[0]

        # 如果A为空集，则置T为单节点数，并将D中实例数最大的类Ck作为该节点的类，返回T
        if len(trainDataList[0]) == 0:
            # 返回当前标签集中占数目最大的标签
            return self.majorClass(trainLabelList)

        # 否则，计算A中特征值的信息增益，选择信息增益最大的特征Ag
        Ag, EpsilonGet = self.calcBestFeature(trainDataList, trainLabelList)

        # 如果Ag的信息增益比小于阈值Epsilon，则置T为单节点树，并将D中实例数最大的类Ck作为该节点的类，返回T
        if EpsilonGet < Epsilon:
            return self.majorClass(trainLabelList)

        # 否则，对Ag的每一可能值ai，依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的类作为标记，构建子节点，由节点及其子节点构成树T，返回T
        treeDict = {Ag: {}}
        class_num = self.fea_class_num[Ag]
        class_set = set([row[Ag] for row in trainDataList])
        for num in range(class_num):
            if num in class_set:
                treeDict[Ag][num] = self.createTree(self.getSubDataArr(trainDataList, trainLabelList, Ag, num))

        return treeDict

    def predict(self, testDataList, tree):
        '''
        预测标签
        :param testDataList:样本
        :param tree: 决策树
        :return: 预测结果
        '''
        # 死循环，直到找到一个有效地分类
        while True:
            (key, value), = tree.items()
            # 如果当前的value是字典，说明还需要遍历下去
            if type(tree[key]).__name__ == 'dict':
                # 获取目前所在节点的feature值，需要在样本中删除该feature
                dataVal = testDataList[key]
                del testDataList[key]
                # 将tree更新为其子节点的字典
                tree = value[dataVal]
                # 如果当前节点的子节点的值是int，就直接返回该int值
                if type(tree).__name__ == 'int':
                    return tree
            else:
                return value

    def model_test(self, testDataList, testLabelList, tree):
        '''
        测试准确率
        :param testDataList:待测试数据集
        :param testLabelList: 待测试标签集
        :param tree: 训练集生成的树
        :return: 准确率
        '''
        # 错误次数计数
        errorCnt = 0
        for i in range(len(testDataList)):
            # 判断预测与标签中结果是否一致
            pre_label = self.predict(testDataList[i], tree)
            print("测试集样本{}，标签为{}，预测结果为{}。".format(i, testLabelList[i], pre_label))
            if testLabelList[i] != pre_label:
                errorCnt += 1
        return 1 - errorCnt / len(testDataList)


if __name__ == '__main__':
    # 开始时间
    start = time.time()

    # 创建决策树
    print('开始创建决策树...')
    model = DecisionTree("traindata.txt", "testdata.txt")
    tree = model.createTree((model.train_data, model.train_label))
    print('决策树结构为：', tree)
    print()

    # 测试准确率
    print('调用测试集测试中...')
    accur = model.model_test(model.test_data, model.test_label, tree)
    print('最终测试准确率为：', accur)
    print()

    # 结束时间
    end = time.time()
    print('总用时: {}s'.format(end - start))
