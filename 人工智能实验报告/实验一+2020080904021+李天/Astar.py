from operator import attrgetter

import numpy as np

# 定义open表与close表
open_list = []
close_list = []
start_state = np.zeros((3, 3), dtype=int)  # 开始状态
target_state = np.zeros((3, 3), dtype=int)  # 目标状态


# 定义节点类
class Node:
    G = 0  # g函数，已经消耗的步数
    H = 0  # h函数，预计到达目标还需要的步数
    F = 0  # f = g + h，总步数
    state = np.zeros((3, 3), dtype=int)
    parent = []  # 到达该状态的前一个状态

    # state是当前状态，prt是该状态的前一个状态
    def __init__(self, state, prt=[]):
        self.state = state
        if prt:
            self.parent = prt
            self.G = prt.G + 1
        for i in range(len(state)):
            for j in range(len(state[i])):
                x, y = self.find_pos(target_state, state[i][j])
                # 计算曼哈顿距离
                self.H = self.H + abs(x - i) + abs(y - j)
        self.F = self.G * 1 + self.H * 10

    # 找num在state状态中的位置x,y
    def find_pos(self, state, num):
        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] == num:
                    return i, j

    # 移动该状态
    def moveto(self, x, y):
        x0, y0 = self.find_pos(self.state, 0)
        newstate = (self.state).copy()
        newstate[x0][y0] = newstate[x][y]
        newstate[x][y] = 0
        return newstate


# 得到逆序数，用于判断解的存在性
def get_reverse_num(state):
    ans = 0
    s = ""
    for i in range(len(state)):
        for j in range(len(state[i])):
            # 0即空格，不在考虑范围内
            if not state[i][j] == 0:
                s += str(state[i][j])

    for i in range(len(s)):
        for j in range(i):
            if s[j] > s[i]:
                ans += 1
    return ans


# 输出状态及深度
def display(cur_node):
    alist = []
    tmp = cur_node
    while tmp:
        alist.append(tmp)
        tmp = tmp.parent
    alist.reverse()
    for node in alist:
        step = node.G
        if step == 0:
            print("原图：")
        elif step == 1:
            print("移动过程：")
            print()
            print("Step %d：" % node.G)
        else:
            print("Step %d：" % node.G)
        mat = node.state
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if mat[i][j] == 0:
                    print(" ", end=" ")
                else:
                    print(mat[i][j], end=" ")
            print()
        print()
    print("移动结束！")


# 检查state状态是否在list中（可能是open或close表）
def is_in_list(alist, state):
    for stat in alist:
        if (stat.state == state).all():
            return stat
    return -1


if __name__ == "__main__":
    # 初始状态和目标状态
    start_state = np.array([[2, 8, 3],
                            [1, 6, 4],
                            [7, 0, 5]])
    target_state = np.array([[1, 2, 3],
                             [8, 0, 4],
                             [7, 6, 5]])

    # 可行解判断
    if get_reverse_num(target_state) % 2 != get_reverse_num(start_state) % 2:
        print("找不到可行解！")
        exit(-1)

    # 可行解存在时，开始启发搜索
    open_list.append(Node(start_state))
    while open_list:
        current_node = open_list.pop(0)
        close_list.append(current_node)

        # 当open表中取出的恰好为目标状态时
        if (current_node.state == target_state).all():
            print("可行解已找到！")
            display(current_node)
            exit(0)

        # 否则对当前节点进行拓展
        x, y = current_node.find_pos(current_node.state, 0)
        for [x_, y_] in [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]:
            if 0 <= x_ < len(start_state) and 0 <= y_ < len(start_state):
                new_state = current_node.moveto(x_, y_)
                # 判断新状态是否在close表
                if is_in_list(close_list, new_state) == -1:
                    # 如果不在close表
                    if is_in_list(open_list, new_state) == -1:
                        # 如果也不在open表
                        open_list.append(Node(new_state, current_node))
                    else:
                        # 如果open表中已存在这种状态,则选取G值较小的
                        sta = is_in_list(open_list, new_state)
                        if current_node.G + 1 < sta.G:
                            # 如果新路线更好，则放弃旧路线而选择新路线
                            open_list.remove(sta)
                            open_list.append(Node(new_state, current_node))

        # 对open表按F值从小到大进行排序
        open_list.sort(key=attrgetter("F"))