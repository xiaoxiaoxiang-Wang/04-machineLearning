import sys

import numpy as np


class Node():
    def __init__(self, left, right, f_idx, x_val, y_val, is_leaf):
        self.left = left
        self.right = right
        self.f_idx = f_idx
        self.x_val = x_val
        self.y_val = y_val
        self.is_leaf = is_leaf


class CartRegression():
    def __init__(self):
        self.max_depth = 5

    def fit(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("input must be ndarray")
        self.feature_idxs = [i for i in range(x.shape[1])]
        data_nums = [i for i in range(x.shape[0])]
        self.tree = self.build_tree(x, y, data_nums, 1)

    def build_tree(self, x, y, data_nums, depth):
        if depth > self.max_depth or len(data_nums) == 0:
            return None
        if len(data_nums) == 1:
            return Node(None, None, 0, x[data_nums[0], 0], y[data_nums[0]], True)
        min_mse = sys.maxsize
        best_f_idx = 0
        best_n_idx = 0
        y_val = 0
        for f_idx in self.feature_idxs:
            for n_idx in data_nums:
                mse, avr = self.get_mse(x[:, f_idx], y, x[n_idx, f_idx], data_nums)
                if mse < min_mse:
                    min_mse = mse
                    best_f_idx = f_idx
                    best_n_idx = n_idx
                    y_val = avr
        left_num_idxs = []
        right_num_idxs = []
        for n_idx in data_nums:
            if x[n_idx, best_f_idx] <= x[best_n_idx, best_f_idx]:
                left_num_idxs.append(n_idx)
            else:
                right_num_idxs.append(n_idx)

        left_node = self.build_tree(x, y, left_num_idxs, depth + 1)
        right_node = self.build_tree(x, y, right_num_idxs, depth + 1)
        if left_node is not None and left_node.is_leaf:
            print("left_node=", left_node.x_val, left_node.y_val, depth + 1)
        if right_node is not None and right_node.is_leaf:
            print("right_node=", right_node.x_val, right_node.y_val, depth + 1)
        return Node(left_node, right_node, best_f_idx, x[best_n_idx, best_f_idx], y_val,
                    left_node is None and right_node is None)

    def get_mse(self, x_idx, y, val, data_nums):
        sum1 = 0
        num1 = 0
        sum2 = 0
        num2 = 0
        for i in data_nums:
            if x_idx[i] <= val:
                sum1 += y[i]
                num1 += 1
            else:
                sum2 += y[i]
                num2 += 1
        avr1 = sum1 / num1 if num1 != 0 else 0
        avr2 = sum2 / num2 if num2 != 0 else 0
        mse = 0
        for i in data_nums:
            if x_idx[i] <= val:
                mse += (y[i] - avr1) ** 2
            else:
                mse += (y[i] - avr2) ** 2
        return (mse / (len(data_nums) - 1)) ** 0.5, (sum1 + sum2) / (num1 + num2)

    def predict(self, x):
        y = []
        for n_idx in range(x.shape[0]):
            node = self.tree
            while not node.is_leaf:
                if x[n_idx, node.f_idx] <= node.x_val:
                    node = node.left
                else:
                    node = node.right

            y.append(node.y_val)
        return y


if __name__ == '__main__':
    cr = CartRegression()
    x = np.array([[1.5], [2.5], [3.5], [4.5], [5.5], [6.5], [7.5], [8.5], [9.5]])
    y = np.array([15.72, 12.07, 8.36, 5.78, 3.91, 1.93, 8.01, 11.73, 15.74])
    cr.fit(x, y)
    x_ = np.array([[1.6], [2.6], [3.6], [4.6], [5.6], [6.6], [7.6], [8.6], [9.6]])
    print(cr.predict(x))
