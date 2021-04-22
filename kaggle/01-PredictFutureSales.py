import os

import matplotlib.pyplot as plt
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

    items = pd.read_csv(items_path)
    item_categories = pd.read_csv(item_categories_path)
    sales_train = pd.read_csv(sales_train_path)
    shops = pd.read_csv(shops_path)
    item_id_sort = sales_train.sort_values(["shop_id", "item_id"], inplace=False)
    sales_train_sum = pd.DataFrame(columns=['shop_id', 'item_id', 'item_cnt_sum', 'sale_price', 'sale_date'])
    print(len(item_id_sort))
    item_cnt_sum = 0
    sale_price = []
    sale_date = []
    for i in range(0, len(item_id_sort) - 1):
        print(i, "/", len(item_id_sort))
        item_cnt_sum += item_id_sort.iloc[i, 5]
        sale_date.append(item_id_sort.iloc[i, 0])
        sale_price.append(item_id_sort.iloc[i, 4])
        if item_id_sort.iloc[i, 3] != item_id_sort.iloc[i + 1, 3] or item_id_sort.iloc[i, 2] != item_id_sort.iloc[
            i + 1, 2]:
            series = pd.Series({
                'shop_id': item_id_sort.iloc[i, 2],
                'item_id': item_id_sort.iloc[i, 3],
                'item_cnt_sum': item_cnt_sum,
                'sale_price': sale_price,
                'sale_date': sale_date})
            sales_train_sum = sales_train_sum.append(series, ignore_index=True)
            item_cnt_sum = 0
            sale_date = []
            sale_price = []
    print(sales_train_sum.head())
    print(sales_train_sum.tail())
    sales_train_sum_path = os.path.join(os.path.dirname("__file__"),
                                        "data//competitive-data-science-predict-future-sales//sales_train_sum.csv")
    sales_train_sum.to_csv(sales_train_sum_path, index=False)


# 设置商品对应id
def set_sum_id():
    sales_train_sum_path = os.path.join(os.path.dirname("__file__"),
                                        "data//competitive-data-science-predict-future-sales//sales_train_sum.csv")
    test_path = os.path.join(os.path.dirname("__file__"),
                             "data//competitive-data-science-predict-future-sales//test.csv")
    sales_train_sum = pd.read_csv(sales_train_sum_path)
    sales_train_sum.loc[:, 'ID'] = 0
    test = pd.read_csv(test_path)
    test_dirt = {}
    for index, row in test.iterrows():
        test_dirt[str(row['shop_id']) + '-' + str(row['item_id'])] = row['ID']
    print(test_dirt)
    for index, row in sales_train_sum.iterrows():
        if str(row['shop_id']) + '-' + str(row['item_id']) == "2-5037":
            print(str(row['shop_id']) + '-' + str(row['item_id']),
                  test_dirt[str(row['shop_id']) + '-' + str(row['item_id'])])
        if str(row['shop_id']) + '-' + str(row['item_id']) in test_dirt:
            sales_train_sum.at[index, 'ID'] = test_dirt[str(row['shop_id']) + '-' + str(row['item_id'])]
    print(sales_train_sum.head())
    print(sales_train_sum.tail())
    sales_train_sum_id_path = os.path.join(os.path.dirname("__file__"),
                                           "data//competitive-data-science-predict-future-sales//sales_train_sum_id.csv")
    sales_train_sum.to_csv(sales_train_sum_id_path, index=False)


# 店铺和销量的关系
def shop_sales():
    sales_train_path = os.path.join(os.path.dirname("__file__"),
                                    "data//competitive-data-science-predict-future-sales//sales_train.csv")
    sales_train = pd.read_csv(sales_train_path)
    print(len(sales_train))
    sales_train = sales_train.append(pd.Series({
        'date': '02.01.2013',
        'date_block_num': 0,
        'shop_id': 60,
        'item_id': 22154,
        'item_price': 999,
        'item_cnt_day': 0}), ignore_index=True)
    print(len(sales_train))
    shop_cnt = pd.DataFrame(columns=['shop_id', 'sale_cnt'])
    shop_id_sort = sales_train.sort_values(["shop_id"], inplace=False)
    cnt = 0
    for i in range(0, len(shop_id_sort) - 1):
        cnt += shop_id_sort.iloc[i, 5]
        if shop_id_sort.iloc[i, 2] != shop_id_sort.iloc[i + 1, 2]:
            series = pd.Series({
                'shop_id': shop_id_sort.iloc[i, 2],
                'sale_cnt': cnt})
            shop_cnt = shop_cnt.append(series, ignore_index=True)
            cnt = 0
    shop_cnt_path = os.path.join(os.path.dirname("__file__"),
                                 "data//competitive-data-science-predict-future-sales//shop_cnt.csv")
    shop_cnt.to_csv(shop_cnt_path, index=False)


def date_block_sales():
    # sales_train_path = os.path.join(os.path.dirname("__file__"),
    #                                 "data//competitive-data-science-predict-future-sales//sales_train.csv")
    # sales_train = pd.read_csv(sales_train_path)
    # sales_train = sales_train.append(pd.Series({
    #     'date': '02.01.2013',
    #     'date_block_num': 34,
    #     'shop_id': 60,
    #     'item_id': 22154,
    #     'item_price': 999,
    #     'item_cnt_day': 0}), ignore_index=True)
    # date_block_sort = sales_train.sort_values(["date_block_num"], inplace=False)
    # date_block_cnt = pd.DataFrame(columns=['date_block_num', 'sale_cnt'])
    # cnt = 0
    # for i in range(0, len(date_block_sort) - 1):
    #     cnt += date_block_sort.iloc[i, 5]
    #     if date_block_sort.iloc[i, 1] != date_block_sort.iloc[i + 1, 1]:
    #         series = pd.Series({
    #             'date_block_num': date_block_sort.iloc[i, 1],
    #             'sale_cnt': cnt})
    #         date_block_cnt = date_block_cnt.append(series, ignore_index=True)
    #         cnt = 0
    date_block_path = os.path.join(os.path.dirname("__file__"),
                                   "data//competitive-data-science-predict-future-sales//date_block_cnt.csv")
    date_block_cnt = pd.read_csv(date_block_path)
    date_block_cnt.to_csv(date_block_path, index=False)
    print(date_block_cnt.iloc[0:12, 1].values)
    plt.bar(x=[i for i in range(12)], height=date_block_cnt.iloc[0:12, 1].values, width=0.3, color='red', label="第一年")
    plt.bar(x=[i + 0.3 for i in range(12)], height=date_block_cnt.iloc[12:24, 1].values, width=0.3, color='blue',
            label="第二年")
    plt.bar(x=[i + 0.6 for i in range(11)], height=date_block_cnt.iloc[24:35, 1].values, width=0.3, color='green',
            label="第三年")
    # plt.bar(x=[i for i in range(34)], height=date_block_cnt.iloc[0:34, 1].values, color='red', label="历史")
    plt.show()

def price():
    sales_train_path = os.path.join(os.path.dirname("__file__"),
                                    "data//competitive-data-science-predict-future-sales//sales_train.csv")
    sales_train = pd.read_csv(sales_train_path)
    shop_id_sort = sales_train.sort_values(["item"], inplace=False)



def generate_submission():
    sales_train_sum_id_path = os.path.join(os.path.dirname("__file__"),
                                           "data//competitive-data-science-predict-future-sales//sales_train_sum_id.csv")
    sample_submission_path = os.path.join(os.path.dirname("__file__"),
                                          "data//competitive-data-science-predict-future-sales//sample_submission.csv")
    test_path = os.path.join(os.path.dirname("__file__"),
                             "data//competitive-data-science-predict-future-sales//test.csv")

    sales_train_sum_id = pd.read_csv(sales_train_sum_id_path)
    sample_submission = pd.read_csv(sample_submission_path)
    sample_submission['item_cnt_month'] = 0
    test = pd.read_csv(test_path)

    sales_train_sum_id['item_cnt_sum'] = round(sales_train_sum_id['item_cnt_sum'] * 30 / 1034)
    test_sort = test.sort_values("item_id", inplace=False)
    print(test_sort.head())

    for index, row in sales_train_sum_id.iterrows():
        # print(row['ID'],row['item_cnt_sum'])
        sample_submission.iloc[row['ID'], 1] = row['item_cnt_sum']
        # print(sample_submission.at[row['ID'],'item_cnt_month'])
        # sample_submission.iloc[row['item_id'],1] = row[1]

    # sample_submission.set_index('ID').join(sales_train_sum.set_index('item_id'))
    print(sample_submission.head())
    print(sample_submission.tail())

    my_submission_path = os.path.join(os.path.dirname("__file__"),
                                      "data//competitive-data-science-predict-future-sales//my_submission.csv")
    sample_submission.to_csv(my_submission_path, index=False)


# generate_sales_sum()
# set_sum_id()
# generate_submission()
# shop_sales()
date_block_sales()
# item_category
# item
# date_block_num 每个月的数量
# price
# item cnt day
# item cnt month

xgb.XGBRegressor(max_depth=3,learning_rate=0.1)