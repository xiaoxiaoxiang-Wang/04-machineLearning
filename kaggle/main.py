import os

import numpy as np
import pandas as pd
import xgboost as xgb


# 统计销售总量
def generate_sales_sum():
    items_path = os.path.join(os.path.dirname("__file__"),
                              "data//competitive-data-science-predict-future-sales//items.csv")
    item_categories_path = os.path.join(os.path.dirname("__file__"),
                                        "data//competitive-data-science-predict-future-sales//item_categories.csv")
    sales_train_path = os.path.join(os.path.dirname("__file__"),
                                    "data//competitive-data-science-predict-future-sales//sales_train.csv")
    shops_path = os.path.join(os.path.dirname("__file__"),
                              "data//competitive-data-science-predict-future-sales//shops.csv")
    test_path = os.path.join(os.path.dirname("__file__"),
                             "data//competitive-data-science-predict-future-sales//test.csv")
    my_submission_path = os.path.join(os.path.dirname("__file__"),
                                      "data//competitive-data-science-predict-future-sales//my_submission.csv")
    sample_submission_path = os.path.join(os.path.dirname("__file__"),
                                          "data//competitive-data-science-predict-future-sales//sample_submission.csv")

    sales_train = pd.read_csv(sales_train_path)
    sales_train = sales_train.drop(
        sales_train[sales_train.item_price < 0].index | sales_train[sales_train.item_price >= 100000].index)
    sales_train = sales_train.drop(
        sales_train[sales_train.item_cnt_day < 0].index | sales_train[sales_train.item_cnt_day >= 1000].index)
    sales_train_sort = sales_train.sort_values(["shop_id", "item_id", "date_block_num"], inplace=False)
    sales_train_sort.to_csv(os.path.join(os.path.dirname("__file__"),
                                         "data//competitive-data-science-predict-future-sales//123.csv"), index=False)

    test = pd.read_csv(test_path)
    test_sort = test.sort_values(["shop_id", "item_id"], inplace=False)
    test_sort.to_csv(os.path.join(os.path.dirname("__file__"),
                                  "data//competitive-data-science-predict-future-sales//1234.csv"), index=False)
    test_idx = 0
    my_submission = pd.read_csv(sample_submission_path)

    # item cnt
    item_cnt = 0
    date_block = -1
    sum_price = 0
    X, y = init_x_y()
    for i in range(0, len(sales_train_sort)):
        if less(sales_train_sort.iloc[i, 2], test_sort.iloc[test_idx, 1], sales_train_sort.iloc[i, 3],
                test_sort.iloc[test_idx, 2]):
            continue
        # 同一个店铺的所有数据
        if same(sales_train_sort.iloc[i, 2], test_sort.iloc[test_idx, 1], sales_train_sort.iloc[i, 3],
                test_sort.iloc[test_idx, 2]):
            if date_block == -1:
                date_block = sales_train_sort.iloc[i, 1]
            # 同一个月
            if sales_train_sort.iloc[i, 1] == date_block:
                item_cnt += sales_train_sort.iloc[i, 5]
                sum_price += sales_train_sort.iloc[i, 4] * item_cnt
            else:
                X[date_block][2] = sum_price / item_cnt
                y[date_block] = item_cnt
                item_cnt = sales_train_sort.iloc[i, 5]
                sum_price = sales_train_sort.iloc[i, 4] * item_cnt
                date_block = sales_train_sort.iloc[i, 1]
        else:
            if item_cnt != 0:
                y[date_block] = item_cnt
                X[date_block][2] = sum_price / item_cnt
                model = xgb.XGBRegressor(max_depth=4, learning_rate=0.1, n_estimators=32, objective="reg:squarederror")
                model.fit(X=X, y=y, eval_metric='rmse')
                pre = round(model.predict([[10, 2, X[date_block][2]]])[0], 2)
                print(" y=", y, " index=", i, " shop_id=", test_sort.iloc[test_idx, 1], " item_id=",
                      test_sort.iloc[test_idx, 2], "ID=", test_sort.iloc[test_idx, 0], "price=", X[date_block][2],
                      " pre=", pre)
                if pre > 20:
                    pre = 20
                elif pre < 0:
                    pre = 0
                my_submission.iloc[test_sort.iloc[test_idx, 0], 1] = pre
                item_cnt = 0
                sum_price = 0
                date_block = -1
                X, y = init_x_y()
            test_idx += 1
            if test_idx == len(test):
                break
    my_submission.to_csv(my_submission_path, index=False)


def predictFutureSales():
    sales_train_path = os.path.join(os.path.dirname("__file__"),
                                    "data//competitive-data-science-predict-future-sales//sales_train.csv")
    items_path = os.path.join(os.path.dirname("__file__"),
                              "data//competitive-data-science-predict-future-sales//items.csv")
    test_path = os.path.join(os.path.dirname("__file__"),
                             "data//competitive-data-science-predict-future-sales//test.csv")

    sales_train = pd.read_csv(sales_train_path)
    sales_train = sales_train.drop(
        sales_train[sales_train.item_price < 0].index | sales_train[sales_train.item_price >= 100000].index)
    sales_train = sales_train.drop(
        sales_train[sales_train.item_cnt_day < 0].index | sales_train[sales_train.item_cnt_day >= 1000].index)

    sales_train = pd.pivot_table(sales_train, index=['shop_id', 'item_id', 'date_block_num'],
                                 aggfunc={'item_price': np.mean, 'item_cnt_day': np.sum}, fill_value=0).reset_index()
    sales_train.insert(3, 'month', sales_train['date_block_num'] % 12)
    sales_train.insert(3, 'year', sales_train['date_block_num'] // 12)

    item = pd.read_csv(items_path)
    test = pd.read_csv(test_path)
    sales_train = pd.merge(sales_train, item.iloc[:, [1, 2]], on=['item_id'], how='left')
    sales_train = pd.merge(sales_train, test, on=['shop_id', 'item_id'], how='left')

    cols = ['shop_id', 'item_id', 'date_block_num', 'year', 'month', 'item_cnt_day', 'item_price', 'item_category_id',
            'ID']
    groups = sales_train.groupby(['shop_id', 'item_id'])
    patchs = []
    for name, group in groups:
        patch = []
        for i in range(35):
            patch.append([name[0], name[1], i, i // 12, i % 12, 0, group.loc[group.index[-1]]['item_price'],
                          group.loc[group.index[-1]]['item_category_id'],
                          -1 if pd.isnull(group.loc[group.index[-1]]['ID']) else group.loc[group.index[-1]]['ID']])
        start = 0
        for idx, row in group.iterrows():
            for i in range(start, int(row['date_block_num'])):
                patch[i][6] = row['item_price']
            patch[int(row['date_block_num'])][5] = row['item_cnt_day']
            start = int(row['date_block_num'] + 1)
        patchs.append(patch)

    sales_train_final = pd.DataFrame(np.vstack(patchs), columns=cols)

    print(sales_train_final)

    sales_train_final['shop_id'] = sales_train_final['shop_id'].astype(np.int)
    sales_train_final['item_id'] = sales_train_final['item_id'].astype(np.int)
    sales_train_final['date_block_num'] = sales_train_final['date_block_num'].astype(np.int)
    sales_train_final['year'] = sales_train_final['year'].astype(np.int)
    sales_train_final['month'] = sales_train_final['month'].astype(np.int)
    sales_train_final['item_category_id'] = sales_train_final['item_category_id'].astype(np.int)
    sales_train_final['ID'] = sales_train_final['ID'].astype(np.int)
    sales_train_final_path = os.path.join(os.path.dirname("__file__"),
                                          "data//competitive-data-science-predict-future-sales//sales_train_final.csv")

    sales_train_final.to_csv(sales_train_final_path, index=False)

    # model = xgb.XGBRegressor(max_depth=4, learning_rate=0.1, n_estimators=32, objective="reg:squarederror")
    # model.fit(X=sales_train, y=y, eval_metric='rmse')


def regression():
    sales_train_final_path = os.path.join(os.path.dirname("__file__"),
                                          "data//competitive-data-science-predict-future-sales//sales_train_final.csv")
    sales_train_final = pd.read_csv(sales_train_final_path)

    test_path = os.path.join(os.path.dirname("__file__"),
                             "data//competitive-data-science-predict-future-sales//test.csv")
    test = pd.read_csv(test_path)

    sample_submission_path = os.path.join(os.path.dirname("__file__"),
                                          "data//competitive-data-science-predict-future-sales//sample_submission.csv")
    my_submission_path = os.path.join(os.path.dirname("__file__"),
                                      "data//competitive-data-science-predict-future-sales//my_submission.csv")
    my_submission = pd.read_csv(sample_submission_path)

    sales_train_final['item_cnt_day'] = sales_train_final['item_cnt_day'].fillna(0).clip(0, 20)
    v = sales_train_final['item_cnt_day'].values
    v = np.insert(v, 0, values=1, axis=0)
    v = np.delete(v, [-1], axis=0)
    # 这一列等于下一列
    v[::34] = v[1::34]

    sales_train_final.insert(8, 'item_cnt_month_pre', v)

    print(sales_train_final.iloc[:, [5, 8]])
    X = sales_train_final[(sales_train_final.date_block_num != 34) & (sales_train_final.date_block_num != 34)].iloc[:,
        [0, 1, 2, 3, 4, 6, 7, 8]].values
    y = sales_train_final[(sales_train_final.date_block_num != 34) & (sales_train_final.date_block_num != 34)].iloc[:,
        [5]].values
    x_val = sales_train_final[sales_train_final.date_block_num == 22].iloc[:, [0, 1, 2, 3, 4, 6, 7, 8]].values
    y_val = sales_train_final[sales_train_final.date_block_num == 22].iloc[:, [5]].values
    X_test_sort = sales_train_final[
        (sales_train_final.date_block_num == 34) & (sales_train_final.ID != -1)].sort_values(['ID'], inplace=False)
    X_test = X_test_sort.iloc[:, [0, 1, 2, 3, 4, 6, 7, 8]].values
    X_test_id = X_test_sort.iloc[:, [9]]

    # my_submission.iloc[int(X_test_sort.iloc[:, [8]].values),[1]] = X_test_sort.iloc[:, [7]]
    #
    #
    # print(my_submission)
    model = xgb.XGBRegressor(max_depth=4, colsample_btree=0.1, learning_rate=0.1, n_estimators=32, min_child_weight=2,
                             objective="reg:squarederror")
    model.fit(X=X, y=y)
    y_test = model.predict(X_test)
    print(y_test)
    X_test_id.insert(1, 'item_cnt_month_pre', y_test)

    my_submission = pd.merge(my_submission, X_test_id, on=['ID'], how='left')
    my_submission['item_cnt_month'] = np.where(np.isnan(my_submission.item_cnt_month_pre), my_submission.item_cnt_month,
                                               my_submission.item_cnt_month_pre)
    my_submission['item_cnt_month'] = my_submission['item_cnt_month'].fillna(0).clip(0, 20)
    my_submission = my_submission.drop(['item_cnt_month_pre'], axis=1)
    my_submission.to_csv(my_submission_path, index=False)

    print(my_submission)


def zero():
    my_submission_path = os.path.join(os.path.dirname("__file__"),
                                      "data//competitive-data-science-predict-future-sales//my_submission.csv")
    my_submission = pd.read_csv(my_submission_path)
    for i in range(0, len(my_submission)):
        if (my_submission.iloc[i, 1] == 0.5):
            my_submission.iloc[i, 1] = 0
    my_submission.to_csv(my_submission_path, index=False)


def getSum():
    my_submission_path = os.path.join(os.path.dirname("__file__"),
                                      "data//competitive-data-science-predict-future-sales//my_submission.csv")
    my_submission = pd.read_csv(my_submission_path)
    sum = 0
    for i in range(0, len(my_submission)):
        if my_submission.iloc[i, 1] < 0:
            print(my_submission.iloc[i, 1])
    print(sum)


def init_x_y():
    # month,year,ave_price
    X = []
    # cnt
    y = []
    for i in range(34):
        X.append([i % 12, i // 12, 0])
        y.append(0)
    return X, y


def less(a1, b1, a2, b2):
    if a1 < b1:
        return True
    if a1 == b1 and a2 < b2:
        return True
    return False


def same(a1, b1, a2, b2):
    return a1 == b1 and a2 == b2


# getSum()
# predictFutureSales()
regression()
# generate_sales_sum()
# zero()
