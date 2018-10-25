# -*- coding:utf-8 -*-

import json
import math
import os
import collections
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import collections
import numpy as np
from tqdm import tqdm
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from PIL import Image
from pylab import array
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest ,chi2
from mpl_toolkits.mplot3d import Axes3D
import  operator

# FILE_PATH = '/Users/weijie/level/leveldata/valid/loactioninfo_'
# test是小米全部 test1是mate全部 test2是小米中空 test3是mate中空
# FILE_PATH = 'E://learn//Senior//Huawei//test//loactioninfo_
FILE_PATH = 'E://learn//Senior//Huawei//test1//loactioninfo_'
# FILE_PATH = 'E://learn//Senior//Huawei//test2//loactioninfo_'
# FILE_PATH = 'E://learn//Senior//Huawei//test3//loactioninfo_'
# FILE_PATH = 'E://learn//Senior//Huawei//H2test-h//loactioninfo_'
# FILE_PATH_84 = '/Users/weijie/level/leveldata/valid/loactioninfo_'
FILE_PATH_84 = 'E://learn//Senior//Huawei//test//loactioninfo_84_'
LGB_MODEL_PATH = '/Users/weijie/level/lgb_model'
# XGB_MODEL_PATH = '/Users/weijie715/level/xgb_model'
XGB_MODEL_PATH='E://learn//Senior//Huawei//xgboost_floorIdentification//level//xgb_model//model.model'
MAC_LIST_PATH = 'E://learn//Senior//Huawei//xgboost_floorIdentification//level//level//mac_list.txt'
OMH_MODEL_PATH = '/Users/weijie/level/omh_xgb_model'
OMH_MAX_LIST_PATH = '/Users/weijie/level/omh_mac_list.txt'
OMH_ICT_MODEL_PATH = '/Users/weijie/level/omh_ict_xgb_model'
OMH_ICT_MAC_LIST_PATH = '/Users/weijie/level/omh_ict_mac_list.txt'
SZ_MODEL_PATH = '/Users/weijie/level/sz_xgb_model'
SZ_MAC_LIST_PATH = '/Users/weijie/level/sz_mac_list.txt'
NEW_OMH_ICT_MODEL_PATH = '/Users/weijie/level/new_omh_ict_xgb_model'
NEW_OMH_ICT_MAC_LIST_PATH = '/Users/weijie/level/new_omh_ict_mac_list.txt'


# 深圳测试集Wi-Fi训练数据解析
# def get_total_test_data():
#     df_total = pd.DataFrame()
#     # for file_path in (FILE_PATH, FILE_PATH_84):
#     file_path = FILE_PATH
#     for level in (1, 2, 3,-1):
#         path = file_path + str(level) + 'F.txt'
#         with open(path, 'r') as f:
#             data = f.readlines()
#             id_list = [x.split('\t')[0] for x in data]
#             finger_list = [x.split('\t')[1] for x in data]
#         df = pd.DataFrame()
#         selected_id_list = []
#         rssi_list = []
#         for idx, finger_info in enumerate(finger_list):
#             fingers_info = json.loads(finger_info)
#             # 有些指纹没有wifi指纹
#             if 'wifilist' in fingers_info['fingerinfo'].keys():
#                 wifi_list_info = fingers_info['fingerinfo']['wifilist']
#                 rssi_dict = {}
#                 for info in wifi_list_info:
#                     rssi_dict[info['mac']] = info['rssi']
#             rssi_str = json.dumps(rssi_dict)
#             rssi_list.append(rssi_str)
#             selected_id_list.append(id_list[idx])
#         df['id'] = selected_id_list
#         df['rssi'] = rssi_list
#         label = 0 if level == -1 else level
#         df['label'] = label
#         df_total = pd.concat([df_total, df], axis=0)
#     df_total.reset_index(drop=True, inplace=True)
#     return df_total

# 测试报文解析
# def get_total_test_data(path):
#     with open(path, 'r') as f:
#         data = f.readlines()
#         id_list = [x.split('\t')[0] for x in data]
#         finger_list = [x.split('\t')[1] for x in data]
#     df = pd.DataFrame()
#     df_total = pd.DataFrame()
#     selected_id_list = []
#     rssi_list = []
#     for idx, finger_info in enumerate(finger_list):
#         fingers_info = json.loads(finger_info)
#         # 有些指纹没有wifi指纹
#         if 'wifilist' in fingers_info['fingerinfo'].keys():
#             wifi_list_info = fingers_info['fingerinfo']['wifilist']
#             rssi_dict = {}
#             for info in wifi_list_info:
#                 rssi_dict[info['mac']] = info['rssi']
#             rssi_str = json.dumps(rssi_dict)
#             rssi_list.append(rssi_str)
#             selected_id_list.append(id_list[idx])
#     df['id'] = selected_id_list
#     df['rssi'] = rssi_list
#     df['label'] = 0
#     df_total = pd.concat([df_total, df], axis=0)
#     df_total.reset_index(drop=True, inplace=True)
#     return df_total

# 测试报文解析2 wifilist格式
# def get_total_test_data(path):
#     df = pd.DataFrame()
#     df_total = pd.DataFrame()
#     rssi_list = []
#     with open(path, 'r') as f:
#         data = f.readlines()
#         wifilist = json.loads(data)
#         if 'wifilist' in wifilist.keys():
#             wifi_list_info = wifilist['wifilist']
#             rssi_dict = {}
#             for info in wifi_list_info:
#                 rssi_dict[info['mac']] = info['rssi']
#         rssi_str = json.dumps(rssi_dict)
#         rssi_list.append(rssi_str)
#     df['rssi'] = rssi_list
#     df['label'] = 0
#     df_total = pd.concat([df_total, df], axis=0)
#     df_total.reset_index(drop=True, inplace=True)
#     return df_total

# 获取mac_id list
def get_total_mac_list(df):
    def write_mac_list_to_file(mac_list):
        with open(MAC_LIST_PATH, 'w') as f:
            for mac_id in mac_list:
                f.write(mac_id)
                f.write('\n')
                f.flush()
        f.close()
    rssi_str_list = df['rssi_list'].tolist()
    mac_list = []
    for rssi_info in rssi_str_list:
        rssi_dict = json.loads(rssi_info)
        mac_list += list(set(rssi_dict.keys()))
    mac_list = list(set(mac_list))
    mac_list = sorted(mac_list)
    write_mac_list_to_file(mac_list)
    return mac_list


# 检查样本的分布
def check_label_cnt(df):
    label_cnt = df.groupby(['label'], as_index=False).size()
    return label_cnt


# 对mac rssi进行OneHot编码
def get_one_hot_transformation(df, mac_list):
    rssi_str_list = df['rssi_list'].tolist()
    template_dict = collections.OrderedDict()
    for mac_id in mac_list:
        template_dict[mac_id] = 0.0
    total_wifi_list = []
    for rssi_info in tqdm(rssi_str_list):
        rssi_dict = json.loads(rssi_info)
        cur_dict = template_dict.copy()
        for mac_id, rssi in rssi_dict.items():
            if mac_id in cur_dict.keys():
                cur_dict[mac_id] = math.pow(1.035, int(rssi))
        rssi_values = cur_dict.values()
        total_wifi_list.append(rssi_values)

    df_train = pd.DataFrame(total_wifi_list, columns=template_dict.keys())
    df_train['label'] = df['label'].tolist()
    return df_train


# 基于分层采样的模型验证
def lgb_evaluation(df):
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for label in (0, 1, 2, 3):
        df_temp = df.loc[df.label == label]
        part_size = len(df_temp)
        train_part = df_temp.iloc[: int(part_size * 0.8), :]
        test_part = df_temp.iloc[int(part_size * 0.8):, :]
        df_train = pd.concat([df_train, train_part], axis=0)
        df_test = pd.concat([df_test, test_part], axis=0)
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    X_train = df_train.drop(['label'], axis=1)
    y_train = df_train['label']
    X_test = df_test.drop(['label'], axis=1)
    y_test = df_test['label']

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test)

    params = {
        'boosting_type': 'gbdt',
        'application': 'multiclass',
        'metric': 'multi_error',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'verbose': 1,
        'num_classes': 4
    }

    gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_eval)
    pre_label_list = [np.argmax(gbm.predict(X_test.loc[i: i, :])) for i in range(len(X_test))]
    print(accuracy_score(pre_label_list, y_test))

    # save model to file
    # gbm.save_model('model.txt')


def xgb_evaluation(df):
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for label in (0, 1, 2, 3):
        df_temp = df.loc[df.label == label]
        part_size = len(df_temp)
        train_part = df_temp.iloc[: int(part_size * 0.8), :]
        test_part = df_temp.iloc[int(part_size * 0.8):, :]
        df_train = pd.concat([df_train, train_part], axis=0)
        df_test = pd.concat([df_test, test_part], axis=0)
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    train_X = df_train.drop(['label'], axis=1)
    train_Y = df_train['label']
    test_X = df_test.drop(['label'], axis=1)
    test_Y = df_test['label']

    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    param = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'slient': 1,
        'num_class': 10,
        'max_depth': 8,
        'nthread': 4,
        'eval_metric': 'merror',
        'seed': 777,
    }
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 39
    bst = xgb.train(param, xg_train, num_round, watchlist)
    pre = bst.predict(xg_test)
    acc = accuracy_score(test_Y, pre)
    print(acc)
    bst.save_model('xgb_newest_model.model')
    con_m = confusion_matrix(test_Y, pre, labels=[0, 1, 2, 3])
    print('the confusion matrix:')
    print(con_m)


# 深圳分楼层准确率xgb模型验证
def xgb_advanced_evaluation():
    # TRAIN_PATH = '/Users/weijie/level/leveldata/train.csv'
    TRAIN_PATH = 'E://learn//Senior//Huawei//trainmodel//train.csv'
    # TEST_PATH = '/Users/weijie/level/leveldata/test.csv'
    TEST_PATH = 'E://learn//Senior//Huawei//trainmodel//test.csv'
    if os.path.exists(TRAIN_PATH):
        df_train = pd.read_csv(TRAIN_PATH)
        # df_valid = pd.read_csv(TEST_PATH)
    else:
        df_train = get_total_train_data()
        df_valid = get_total_test_data()
        df_valid.drop(['id'], axis=1, inplace=True)
        df_valid.columns = ['rssi_list', 'label']
        df_valid.drop_duplicates(['rssi_list'], inplace=True)
        train_mac_list = get_total_mac_list(df_train)
        mac_list = list(set(train_mac_list))
        df_train = get_one_hot_transformation(df_train, mac_list)
        df_valid = get_one_hot_transformation(df_valid, mac_list)
        df_train.to_csv(TRAIN_PATH, index=False)
        df_valid.to_csv(TEST_PATH, index=False)
    train_X = df_train.drop(['label'], axis=1)
    train_Y = df_train['label']
    test_X = df_valid.drop(['label'], axis=1)
    test_Y = df_valid['label']
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    param = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'slient': 1,
        'num_class': 4,
        'max_depth': 8,
        'nthread': 4,
        'eval_metric': 'merror',
        'seed': 777,
    }
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 39
    bst = xgb.train(param, xg_train, num_round, watchlist)
    pre = bst.predict(xg_test)
    acc = accuracy_score(test_Y, pre)
    print(acc)
    bst.save_model(XGB_MODEL_PATH)
    con_m = confusion_matrix(test_Y, pre, labels=[0, 1, 2, 3])
    print('the confusion matrix:')
    print(con_m)
    floor_map_dict = {}
    floor_map_dict[0] = '-1 floor'
    floor_map_dict[1] = '1 floor'
    floor_map_dict[2] = '2 floor'
    floor_map_dict[3] = '3 floor'
    write_floor_map_file(floor_map_dict, 'shenzhen.txt')


def map_level(s):
    if s == 'f1':
        return 1
    elif s == 'f1t1':
        return 1
    elif s == 'f1t2':
        return 1
    elif s == 'f2':
        return 2
    elif s == 'f3':
        return 3
    elif s == 'f31':
        return 3


# 新的深圳数据解析
def get_new_shenzhen_data():
    f1_path = '/Users/weijie/level/new_shenzhen_data/P9_f1_wifiinfo.txt'
    f1t1_path = '/Users/weijie/level/new_shenzhen_data/P9_f1t1_wifiinfo.txt'
    f1t2_path = '/Users/weijie/level/new_shenzhen_data/P9_f1t2_wifiinfo.txt'
    f2_path = '/Users/weijie/level/new_shenzhen_data/P9_f2_wifiinfo.txt'
    f3_path = '/Users/weijie/level/new_shenzhen_data/P9_f3_wifiinfo.txt'
    f31_path = '/Users/weijie/level/new_shenzhen_data/P9_f31_wifiinfo.txt'
    path_list = [f1_path, f1t1_path, f1t2_path, f2_path, f3_path, f31_path]
    df_total = pd.DataFrame()
    for file_path in path_list:
        dataset = file_path.split('/')[-1].split('_')[1]
        with open(file_path, 'r') as f:
            finger_list = f.readlines()
            df = pd.DataFrame()
            rssi_list = []
            for idx, finger_info in enumerate(finger_list):
                fingers_info = json.loads(finger_info)
                # 有些指纹没有wifi指纹
                if 'wifilist' in fingers_info['fingerinfo'].keys():
                    wifi_list_info = fingers_info['fingerinfo']['wifilist']
                    rssi_dict = {}
                    for info in wifi_list_info:
                        rssi_dict[info['mac']] = int(info['rssi'])
                    rssi_str = json.dumps(rssi_dict)
                    rssi_list.append(rssi_str)
            df['rssi'] = rssi_list
            df['dataset'] = dataset
            df_total = pd.concat([df_total, df], axis=0)
    df_total.reset_index(drop=True, inplace=True)
    df_total['label'] = df_total['dataset'].apply(map_level)
    test_path = '/Users/weijie/level/new_shenzhen_data/test.csv'
    df_total.to_csv(test_path, index=False)
    return df_total


# 新的深圳时间序列数据解析
def get_new_seq_shenzhen_data():
    f1_path = '/Users/weijie/level/new_shenzhen_data/P9_f1_wifiinfo.txt'
    f1t1_path = '/Users/weijie/level/new_shenzhen_data/P9_f1t1_wifiinfo.txt'
    f1t2_path = '/Users/weijie/level/new_shenzhen_data/P9_f1t2_wifiinfo.txt'
    f2_path = '/Users/weijie/level/new_shenzhen_data/P9_f2_wifiinfo.txt'
    f3_path = '/Users/weijie/level/new_shenzhen_data/P9_f3_wifiinfo.txt'
    f31_path = '/Users/weijie/level/new_shenzhen_data/P9_f31_wifiinfo.txt'
    path_list = [f1_path, f1t1_path, f1t2_path, f2_path, f3_path, f31_path]
    df_total = pd.DataFrame()
    for file_path in path_list:
        dataset = file_path.split('/')[-1].split('_')[1]
        with open(file_path, 'r') as f:
            finger_list = f.readlines()
            df = pd.DataFrame()
            rssi_list = []
            time_list = []
            for idx, finger_info in enumerate(finger_list):
                fingers_info = json.loads(finger_info)
                # 有些指纹没有wifi指纹
                if 'wifilist' in fingers_info['fingerinfo'].keys():
                    wifi_list_info = fingers_info['fingerinfo']['wifilist']
                    rssi_dict = {}
                    for info in wifi_list_info:
                        rssi_dict[info['mac']] = int(info['rssi'])
                    rssi_str = json.dumps(rssi_dict)
                    time = fingers_info['timestamp']
                    rssi_list.append(rssi_str)
                    time_list.append(time)
            df['rssi'] = rssi_list
            df['dataset'] = dataset
            df['time'] = time_list
            df_total = pd.concat([df_total, df], axis=0)
    df_total.reset_index(drop=True, inplace=True)
    df_total['label'] = df_total['dataset'].apply(map_level)

    df_total = df_total.sort_values(['dataset', 'time'], ascending=[True, True])
    test_path = '/Users/weijie/level/new_shenzhen_data/sq_test.csv'
    df_total.to_csv(test_path, index=False)
    return df_total


def sliding_window_from_test_data(window=2):
    def get_generated_data(df, level, dataset, window_size=2):
        rssi_list = df['rssi'].tolist()
        rssi_list = [json.loads(x) for x in rssi_list]
        new_rssi_list = []
        for idx in range(len(rssi_list) - window_size):
            common_ap_list = []
            # 计算若干个元素的交集
            ap_cnt = []
            for j in range(idx, idx + window_size):
                ap_cnt += rssi_list[idx].keys()
            cnt = collections.Counter(ap_cnt)
            for k, v in cnt.items():
                if cnt[k] == window_size:
                    common_ap_list.append(k)
            ap_dict = {}
            for j in range(idx, idx + window_size):
                for k, v in rssi_list[idx].items():
                    if k in common_ap_list and k not in ap_dict.keys():
                        ap_dict[k] = []
                    if k in common_ap_list and k in ap_dict.keys():
                        ap_dict[k].append(v)
            new_ap_dict = {}
            for k, v in ap_dict.items():
                new_ap_dict[k] = np.mean(v)
            new_rssi_list.append(json.dumps(new_ap_dict))
        new_df = pd.DataFrame()
        new_df['rssi'] = new_rssi_list
        new_df['dataset'] = dataset
        new_df['label'] = level
        return new_df

    test_path = '/Users/weijie/level/new_shenzhen_data/sq_test.csv'
    df_test = pd.read_csv(test_path)

    data_name = ['f1', 'f1t1', 'f1t2', 'f2', 'f3', 'f31']

    name = 'f1'
    df_cur = df_test.loc[df_test.dataset == name]
    time_list = df_cur['time'].tolist()
    for i in range(len(time_list) - 1):
        gap = time_list[i + 1] - time_list[i]
        if gap > 100000:
            print(i, gap)
    df_f1 = df_cur
    df_f1_new = get_generated_data(df_f1, 1, name, window_size=window)

    name = 'f1t1'
    df_cur = df_test.loc[df_test.dataset == name]
    time_list = df_cur['time'].tolist()
    for i in range(len(time_list) - 1):
        gap = time_list[i + 1] - time_list[i]
        if gap > 100000:
            print(i, gap)
    df_f1t1_part1 = df_cur.iloc[:1112]
    df_f1t1_part2 = df_cur.iloc[1122:1430]
    df_f1t1_part3 = df_cur.iloc[1440:]
    df_f1t1_part1_new = get_generated_data(df_f1t1_part1, 1, name, window_size=window)
    df_f1t1_part2_new = get_generated_data(df_f1t1_part2, 1, name, window_size=window)
    df_f1t1_part3_new = get_generated_data(df_f1t1_part3, 1, name, window_size=window)

    name = 'f1t2'
    df_cur = df_test.loc[df_test.dataset == name]
    time_list = df_cur['time'].tolist()
    for i in range(len(time_list) - 1):
        gap = time_list[i + 1] - time_list[i]
        if gap > 100000:
            print(i, gap)
    df_f1t2_part1 = df_cur.iloc[:185]
    df_f1t2_part2 = df_cur.iloc[198:]
    df_f1t2_part1_new = get_generated_data(df_f1t2_part1, 1, name, window_size=window)
    df_f1t2_part2_new = get_generated_data(df_f1t2_part2, 1, name, window_size=window)


    name = 'f2'
    df_cur = df_test.loc[df_test.dataset == name]
    time_list = df_cur['time'].tolist()
    for i in range(len(time_list) - 1):
        gap = time_list[i + 1] - time_list[i]
        if gap > 100000:
            print(i, gap)
    df_f2 = df_cur
    df_f2_new = get_generated_data(df_f2, 2, name, window_size=window)

    name = 'f3'
    df_cur = df_test.loc[df_test.dataset == name]
    time_list = df_cur['time'].tolist()
    for i in range(len(time_list) - 1):
        gap = time_list[i + 1] - time_list[i]
        if gap > 100000:
            print(i, gap)
    df_f3 = df_cur
    df_f3_new = get_generated_data(df_f3, 3, name, window_size=window)

    name = 'f31'
    df_cur = df_test.loc[df_test.dataset == name]
    time_list = df_cur['time'].tolist()
    for i in range(len(time_list) - 1):
        gap = time_list[i + 1] - time_list[i]
        if gap > 100000:
            print(i, gap)
    df_f31 = df_cur
    df_f31_new = get_generated_data(df_f31, 3, name, window_size=window)

    df_ret = pd.concat([df_f1_new, df_f1t1_part1_new], axis=0)
    df_ret = pd.concat([df_ret, df_f1t1_part2_new], axis=0)
    df_ret = pd.concat([df_ret, df_f1t1_part3_new], axis=0)
    df_ret = pd.concat([df_ret, df_f1t2_part1_new], axis=0)
    df_ret = pd.concat([df_ret, df_f1t2_part2_new], axis=0)
    df_ret = pd.concat([df_ret, df_f2_new], axis=0)
    df_ret = pd.concat([df_ret, df_f3_new], axis=0)
    df_ret = pd.concat([df_ret, df_f31_new], axis=0)

    return df_ret


# 新的_深圳分楼层准确率xgb模型验证
def new_xgb_advanced_evaluation(size=2):
    # TRAIN_PATH = '/Users/weijie/level/new_shenzhen_data/train.csv'
    TRAIN_PATH = 'E://learn//Senior//Huawei//trainmodel//train.csv'
    df_train = pd.read_csv(TRAIN_PATH)
    # mac_old_list = df_train.columns
    #
    # removed_prefix_list = [
    #     '9c50ee74870',
    #     '9c50ee73914',
    #     '9c50ee7394c',
    #     '9c50ee73928',
    #     '9c50ee7388a',
    #     '9c50ee745dc'
    #  ]
    #
    # removed_list = []
    # for ap in mac_old_list:
    #     if ap[:-1] in removed_prefix_list:
    #         removed_list.append(ap)
    #         print 'removed ap %s' % ap

    df_valid = sliding_window_from_test_data(window=size)
    df_valid.columns = ['rssi_list', 'dataset', 'label']
    df_valid.drop_duplicates(['rssi_list'], inplace=True)

    mac_list = df_train.columns[:-1]
    # mac_list = [x for x in mac_list if x not in removed_list]
    # df_train = df_train.drop(removed_list, axis=1)

    dataset_list = df_valid['dataset'].tolist()
    df_valid = get_one_hot_transformation(df_valid, mac_list)

    train_X = df_train.drop(['label'], axis=1)
    train_Y = df_train['label']
    test_X = df_valid.drop(['label'], axis=1)
    test_Y = df_valid['label']

    feat_model = SelectKBest(chi2, k=3200).fit(train_X, train_Y)
    train_feat_X = feat_model.transform(train_X)
    test_feat_X = feat_model.transform(test_X)

    xg_train = xgb.DMatrix(train_feat_X, label=train_Y)
    xg_test = xgb.DMatrix(test_feat_X, label=test_Y)
    param = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'slient': 1,
        'num_class': 4,
        'max_depth': 10,
        'nthread': 4,
        'eval_metric': 'merror',
        'seed': 777
    }
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 85
    bst = xgb.train(param, xg_train, num_round, watchlist)
    pre = bst.predict(xg_test)
    acc = accuracy_score(test_Y, pre)
    print(acc)
    con_m = confusion_matrix(test_Y, pre, labels=[0, 1, 2, 3])
    print('the confusion matrix:')
    print(con_m)
    floor_map_dict = {}
    floor_map_dict[0] = '-1 floor'
    floor_map_dict[1] = '1 floor'
    floor_map_dict[2] = '2 floor'
    floor_map_dict[3] = '3 floor'
    # write_floor_map_file(floor_map_dict, 'new_shenzhen.txt')



    df_eval = pd.DataFrame()
    df_eval['dataset'] = dataset_list
    df_eval['test_Y'] = test_Y
    df_eval['pre'] = pre

    dataset_set = list(set(dataset_list))
    for data in dataset_set:
        df_cur = df_eval.loc[df_eval.dataset == data]
        print(data, accuracy_score(df_cur['test_Y'], df_cur['pre']))

    # c = xgb.to_graphviz(bst)


# 深圳分楼层准确率lgb模型验证
def lgb_advanced_evaluation():
    TRAIN_PATH = '/Users/weijie/level/leveldata/train.csv'
    TEST_PATH = '/Users/weijie/level/leveldata/test.csv'
    if os.path.exists(TRAIN_PATH):
        df_train = pd.read_csv(TRAIN_PATH)
        df_valid = pd.read_csv(TEST_PATH)
    else:
        df_train = get_total_train_data()
        df_valid = get_total_test_data()
        df_valid.drop(['id'], axis=1, inplace=True)
        df_valid.columns = ['rssi_list', 'label']
        df_valid.drop_duplicates(['rssi_list'], inplace=True)
        train_mac_list = get_total_mac_list(df_train)
        mac_list = list(set(train_mac_list))
        df_train = get_one_hot_transformation(df_train, mac_list)
        df_valid = get_one_hot_transformation(df_valid, mac_list)
        df_train.to_csv(TRAIN_PATH, index=False)
        df_valid.to_csv(TEST_PATH, index=False)
    X_train = df_train.drop(['label'], axis=1)
    y_train = df_train['label']
    X_test = df_valid.drop(['label'], axis=1)
    y_test = df_valid['label']
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test)
    params = {
        'boosting_type': 'gbdt',
        'application': 'multiclass',
        'metric': 'multi_error',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'verbose': -1,
        'num_classes': 4
    }
    gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_eval)
    pre_label_list = [np.argmax(gbm.predict(X_test.loc[i: i, :])) for i in range(len(X_test))]
    print('the accuracy:')
    print(accuracy_score(y_test, pre_label_list))
    con_m = confusion_matrix(y_test, pre_label_list, labels=[0, 1, 2, 3])
    print('the confusion matrix:')
    gbm.save_model(LGB_MODEL_PATH)
    print(con_m)
    # for (true, pre) in zip(y_test, pre_label_list):
    #     print true, pre


# knn模型验证
def knn_evaluation(df):
    # 分层采样
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for label in (0, 1, 2, 3):
        df_temp = df.loc[df.label == label]
        part_size = len(df_temp)
        train_part = df_temp.iloc[: int(part_size * 0.8), :]
        test_part = df_temp.iloc[int(part_size * 0.8):, :]
        df_train = pd.concat([df_train, train_part], axis=0)
        df_test = pd.concat([df_test, test_part], axis=0)
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    X_train = df_train.drop(['label'], axis=1)
    y_train = df_train['label']
    X_test = df_test.drop(['label'], axis=1)
    y_test = df_test['label']

    for k in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
        clf = neighbors.KNeighborsClassifier(k)
        clf.fit(X_train, y_train)
        pre_label_list = [clf.predict(X_test.loc[i: i, :])[0] for i in range(len(X_test))]
        print(accuracy_score(pre_label_list, y_test))





# 检查分布
def check_distribution():
    TRAIN_PATH = '/Users/weijie/level/leveldata/train.csv'
    TEST_PATH = '/Users/weijie/level/leveldata/test.csv'
    if os.path.exists(TRAIN_PATH):
        df_train = pd.read_csv(TRAIN_PATH)
        df_valid = pd.read_csv(TEST_PATH)
    else:
        df_train = get_total_train_data()
        df_valid = get_total_test_data()
        df_valid.drop(['id'], axis=1, inplace=True)
        df_valid.columns = ['rssi_list', 'label']
        df_valid.drop_duplicates(['rssi_list'], inplace=True)
        train_mac_list = get_total_mac_list(df_train)
        mac_list = list(set(train_mac_list))
        df_train = get_one_hot_transformation(df_train, mac_list)
        df_valid = get_one_hot_transformation(df_valid, mac_list)
        df_train.to_csv(TRAIN_PATH, index=False)
        df_valid.to_csv(TEST_PATH, index=False)
    train_cnt = check_label_cnt(df_train)
    test_cnt = check_label_cnt(df_valid)
    return train_cnt, test_cnt


# 过采样尝试
def check_oversampling():
    TRAIN_PATH = '/Users/weijie/level/leveldata/train.csv'
    TEST_PATH = '/Users/weijie/level/leveldata/test.csv'
    if os.path.exists(TRAIN_PATH):
        df_train = pd.read_csv(TRAIN_PATH)
        df_valid = pd.read_csv(TEST_PATH)
    else:
        df_train = get_total_train_data()
        df_valid = get_total_test_data()
        df_valid.drop(['id'], axis=1, inplace=True)
        df_valid.columns = ['rssi_list', 'label']
        df_valid.drop_duplicates(['rssi_list'], inplace=True)
        train_mac_list = get_total_mac_list(df_train)
        mac_list = list(set(train_mac_list))
        df_train = get_one_hot_transformation(df_train, mac_list)
        df_valid = get_one_hot_transformation(df_valid, mac_list)
        df_train.to_csv(TRAIN_PATH, index=False)
        df_valid.to_csv(TEST_PATH, index=False)

    X_train = df_train.drop(['label'], axis=1)
    y_train = df_train['label']
    X_test = df_valid.drop(['label'], axis=1)
    y_test = df_valid['label']
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test)
    params = {
        'boosting_type': 'gbdt',
        'application': 'multiclass',
        'metric': 'multi_error',
        'num_leaves': 16,
        'learning_rate': 0.01,
        'verbose': 1,
        'num_classes': 4
    }
    gbm = lgb.train(params, lgb_train, num_boost_round=1500, valid_sets=lgb_eval)
    pre_label_list = [np.argmax(gbm.predict(X_test.loc[i: i, :])) for i in range(len(X_test))]
    print('the accuracy:')
    print(accuracy_score(y_test, pre_label_list))
    # con_m = confusion_matrix(y_test, pre_label_list, labels=[0, 1, 2, 3])
    # print 'the confusion matrix:'
    '''
    [[29  6  0  0]
     [ 0 26  1  0]
     [ 1 11 17  1]
     [ 0 10  3 32]]
    '''
    # print con_m
    #
    # for (true, pre) in zip(y_test, pre_label_list):
    #     print true, pre


# 输出测试数据
def output_test_data():
    TEST_PATH = '/Users/weijie/level/leveldata/test.csv'
    TEST_X_PATH = '/Users/weijie/level/test_X.csv'
    TEST_Y_PATH = '/Users/weijie/level/test_Y.csv'
    if os.path.exists(TEST_PATH):
        df_test = pd.read_csv(TEST_PATH)
        test_X = df_test.drop(['label'], axis=1)
        test_Y = df_test['label']
        test_X.to_csv(TEST_X_PATH, index=False)
        test_Y.to_csv(TEST_Y_PATH, index=False)


# 欧美汇Wi-Fi数据解析
def read_omh_data():
    COMMON_OMH_PATH = '/Users/weijie/level/omh_data/'
    FLOOR = ['-1/', '1/', '2/']
    df_total_data = pd.DataFrame()
    for floor_str in tqdm(FLOOR):
        df = pd.DataFrame()
        floor = floor_str[:-1]
        floor = int(floor) if int(floor) > 0 else 0
        total_rssi_list = []
        for file in os.listdir(COMMON_OMH_PATH + floor_str):
            with open(COMMON_OMH_PATH + floor_str + file) as f:
                data = f.readline()
                data_dict = json.loads(data)
                for atom_data in data_dict['datas']:
                    if len(atom_data['wifi']) != 0:
                        rssi_dict = {}
                        for tmp_dict in atom_data['wifi']:
                            rssi_dict[tmp_dict['mac']] = int(tmp_dict['rssi'])
                        total_rssi_list.append(json.dumps(rssi_dict))
        total_rssi_list = list(set(total_rssi_list))
        df['rssi_list'] = total_rssi_list
        df['label'] = floor
        df_total_data = pd.concat([df_total_data, df], axis=0)
    df_total_data.drop_duplicates(['rssi_list'], inplace=True)
    return df_total_data


# 新的欧美汇Wi-Fi数据解析
def read_new_omh_data():
    COMMON_NEW_OMH_PATH = '/Users/weijie/level/new_omh_ict_data/'
    FLOOR = ['-1/', '1/', '2/']
    df_total_data = pd.DataFrame()
    for floor_str in tqdm(FLOOR):
        df = pd.DataFrame()
        floor = floor_str[:-1]
        floor = int(floor) if int(floor) > 0 else 0
        total_rssi_list = []
        for file in os.listdir(COMMON_NEW_OMH_PATH + floor_str):
            with open(COMMON_NEW_OMH_PATH + floor_str + file) as f:
                data = f.readline()
                data_dict = json.loads(data)
                for atom_data in data_dict['datas']:
                    if len(atom_data['wifi']) != 0:
                        rssi_dict = {}
                        for tmp_dict in atom_data['wifi']:
                            rssi_dict[tmp_dict['mac']] = int(tmp_dict['rssi'])
                        total_rssi_list.append(json.dumps(rssi_dict))
        total_rssi_list = list(set(total_rssi_list))
        df['rssi_list'] = total_rssi_list
        df['label'] = floor
        df_total_data = pd.concat([df_total_data, df], axis=0)
    df_total_data.drop_duplicates(['rssi_list'], inplace=True)
    return df_total_data


# 计算所Wi-Fi数据解析
def read_ict_data():
    COMMON_ICT_PATH = '/Users/weijie/level/ict_data/'
    FLOOR = ['6/', '7/', '8/']
    df_total_data = pd.DataFrame()
    for floor_str in tqdm(FLOOR):
        df = pd.DataFrame()
        floor = floor_str[:-1]
        floor = int(floor) if int(floor) > 0 else 0
        total_rssi_list = []
        for file in os.listdir(COMMON_ICT_PATH + floor_str):
            with open(COMMON_ICT_PATH + floor_str + file) as f:
                data = f.readline()
                data_dict = json.loads(data)
                for atom_data in data_dict['datas']:
                    if len(atom_data['wifi']) != 0:
                        rssi_dict = {}
                        for tmp_dict in atom_data['wifi']:
                            rssi_dict[tmp_dict['mac']] = int(tmp_dict['rssi'])
                        total_rssi_list.append(json.dumps(rssi_dict))
        total_rssi_list = list(set(total_rssi_list))
        df['rssi_list'] = total_rssi_list
        df['label'] = floor
        df_total_data = pd.concat([df_total_data, df], axis=0)
    df_total_data.drop_duplicates(['rssi_list'], inplace=True)
    return df_total_data


# 苏州研究院Wi-Fi数据解析
def read_sz_data():
    COMMON_SZ_PATH = '/Users/weijie/level/sz_data/'
    FLOOR = ['1/', '2/', '3/']
    df_total_data = pd.DataFrame()
    for floor_str in tqdm(FLOOR):
        df = pd.DataFrame()
        floor = floor_str[:-1]
        floor = int(floor)-1 if int(floor) > 0 else 0
        total_rssi_list = []
        for file in os.listdir(COMMON_SZ_PATH + floor_str):
            with open(COMMON_SZ_PATH + floor_str + file) as f:
                data = f.readline()
                data_dict = json.loads(data)
                for atom_data in data_dict['datas']:
                    if len(atom_data['wifi']) != 0:
                        rssi_dict = {}
                        for tmp_dict in atom_data['wifi']:
                            rssi_dict[tmp_dict['mac']] = int(tmp_dict['rssi'])
                        total_rssi_list.append(json.dumps(rssi_dict))
        total_rssi_list = list(set(total_rssi_list))
        df['rssi_list'] = total_rssi_list
        df['label'] = floor
        df_total_data = pd.concat([df_total_data, df], axis=0)
    df_total_data.drop_duplicates(['rssi_list'], inplace=True)
    return df_total_data


# 欧美汇分楼层准确率测试
def omh_test():
    df_omh = read_omh_data()
    omh_mac_list = get_total_mac_list(df_omh)
    df_omh = get_one_hot_transformation(df_omh, omh_mac_list)

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for label in (0, 1, 2):
        df_temp = df_omh.loc[df_omh.label == label]
        part_size = len(df_temp)
        train_part = df_temp.iloc[: int(part_size * 0.8), :]
        test_part = df_temp.iloc[int(part_size * 0.8):, :]
        df_train = pd.concat([df_train, train_part], axis=0)
        df_test = pd.concat([df_test, test_part], axis=0)
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    train_X = df_train.drop(['label'], axis=1)
    train_Y = df_train['label']
    test_X = df_test.drop(['label'], axis=1)
    test_Y = df_test['label']
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    param = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'slient': 1,
        'num_class': 3,
        'max_depth': 8,
        'nthread': 4,
        'eval_metric': 'merror',
        'seed': 777,
    }
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 39
    bst = xgb.train(param, xg_train, num_round, watchlist)
    pre = bst.predict(xg_test)
    acc = accuracy_score(test_Y, pre)
    print(acc)

    con_m = confusion_matrix(test_Y, pre, labels=[0, 1, 2])
    print('the confusion matrix:')
    print(con_m)

    bst.save_model(OMH_MODEL_PATH)

    def write_omh_mac_list_to_file(mac_list):
        with open(OMH_MAX_LIST_PATH, 'w') as f:
            for mac_id in mac_list:
                f.write(mac_id + '\n')

    write_omh_mac_list_to_file(omh_mac_list)

    floor_map_dict = {}
    floor_map_dict[0] = 'omh -1 floor'
    floor_map_dict[1] = 'omh 1 floor'
    floor_map_dict[2] = 'omh 2 floor'
    write_floor_map_file(floor_map_dict, 'oumeihui.txt')


# 计算所+欧美汇分楼层准确率测试
def omh_ict_test():
    def map_ict_floor(label):
        if label == 6:
            return 3
        elif label == 7:
            return 4
        elif label == 8:
            return 5
        else:
            return label
    df_omh = read_omh_data()
    df_ict = read_ict_data()
    omh_mac_list = get_total_mac_list(df_omh)
    ict_mac_list = get_total_mac_list(df_ict)
    omh_ict_mac_list = omh_mac_list + ict_mac_list

    df_ict_omh = pd.concat([df_omh, df_ict], axis=0)
    df_ict_omh = get_one_hot_transformation(df_ict_omh, omh_ict_mac_list)
    df_ict_omh['label'] = df_ict_omh['label'].apply(map_ict_floor)

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for label in (0, 1, 2, 3, 4, 5):
        df_temp = df_ict_omh.loc[df_ict_omh.label == label]
        part_size = len(df_temp)
        train_part = df_temp.iloc[: int(part_size * 0.8), :]
        test_part = df_temp.iloc[int(part_size * 0.8):, :]
        df_train = pd.concat([df_train, train_part], axis=0)
        df_test = pd.concat([df_test, test_part], axis=0)
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    train_X = df_train.drop(['label'], axis=1)
    train_Y = df_train['label']
    test_X = df_test.drop(['label'], axis=1)
    test_Y = df_test['label']
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    param = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'slient': 1,
        'num_class': 6,
        'max_depth': 8,
        'nthread': 4,
        'eval_metric': 'merror',
        'seed': 777,
    }
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 39
    bst = xgb.train(param, xg_train, num_round, watchlist)
    pre = bst.predict(xg_test)
    acc = accuracy_score(test_Y, pre)
    print(acc)

    con_m = confusion_matrix(test_Y, pre, labels=[0, 1, 2, 3, 4, 5])
    print('the confusion matrix:')
    print(con_m)

    bst.save_model(OMH_ICT_MODEL_PATH)

    def write_omh_ict_mac_list_to_file(mac_list):
        with open(OMH_ICT_MAC_LIST_PATH, 'w') as f:
            for mac_id in mac_list:
                f.write(mac_id + '\n')

    write_omh_ict_mac_list_to_file(omh_ict_mac_list)

    floor_map_dict = {}
    floor_map_dict[0] = 'omh -1 floor'
    floor_map_dict[1] = 'omh 1 floor'
    floor_map_dict[2] = 'omh 2 floor'
    floor_map_dict[3] = 'ict 6 floor'
    floor_map_dict[4] = 'ict 7 floor'
    floor_map_dict[5] = 'ict 8 floor'
    write_floor_map_file(floor_map_dict, 'oumeihui+ict.txt')


# 计算所+欧美汇分楼层准确率测试
def new_omh_ict_test():
    def map_ict_floor(label):
        if label == 6:
            return 3
        elif label == 7:
            return 4
        elif label == 8:
            return 5
        else:
            return label
    df_omh = read_new_omh_data()
    df_ict = read_ict_data()
    omh_mac_list = get_total_mac_list(df_omh)
    ict_mac_list = get_total_mac_list(df_ict)
    omh_ict_mac_list = omh_mac_list + ict_mac_list

    df_ict_omh = pd.concat([df_omh, df_ict], axis=0)
    df_ict_omh = get_one_hot_transformation(df_ict_omh, omh_ict_mac_list)
    df_ict_omh['label'] = df_ict_omh['label'].apply(map_ict_floor)

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for label in (0, 1, 2, 3, 4, 5):
        df_temp = df_ict_omh.loc[df_ict_omh.label == label]
        part_size = len(df_temp)
        train_part = df_temp.iloc[: int(part_size * 0.8), :]
        test_part = df_temp.iloc[int(part_size * 0.8):, :]
        df_train = pd.concat([df_train, train_part], axis=0)
        df_test = pd.concat([df_test, test_part], axis=0)
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    train_X = df_train.drop(['label'], axis=1)
    train_Y = df_train['label']
    test_X = df_test.drop(['label'], axis=1)
    test_Y = df_test['label']
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    param = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'slient': 1,
        'num_class': 6,
        'max_depth': 8,
        'nthread': 4,
        'eval_metric': 'merror',
        'seed': 777,
    }
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 39
    bst = xgb.train(param, xg_train, num_round, watchlist)
    pre = bst.predict(xg_test)
    acc = accuracy_score(test_Y, pre)
    print(acc)

    con_m = confusion_matrix(test_Y, pre, labels=[0, 1, 2, 3, 4, 5])
    print('the confusion matrix:')
    print(con_m)

    bst.save_model(NEW_OMH_ICT_MODEL_PATH)

    def write_omh_ict_mac_list_to_file(mac_list):
        with open(NEW_OMH_ICT_MAC_LIST_PATH, 'w') as f:
            for mac_id in mac_list:
                f.write(mac_id + '\n')

    write_omh_ict_mac_list_to_file(omh_ict_mac_list)

    floor_map_dict = {}
    floor_map_dict[0] = 'omh -1 floor'
    floor_map_dict[1] = 'omh 1 floor'
    floor_map_dict[2] = 'omh 2 floor'
    floor_map_dict[3] = 'ict 6 floor'
    floor_map_dict[4] = 'ict 7 floor'
    floor_map_dict[5] = 'ict 8 floor'
    write_floor_map_file(floor_map_dict, 'new_oumeihui+ict.txt')


# 苏州研究院分楼层准确率测试
def sz_test():
    df_sz = read_sz_data()
    sz_mac_list = get_total_mac_list(df_sz)
    df_sz = get_one_hot_transformation(df_sz, sz_mac_list)

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for label in (0, 1, 2):
        df_temp = df_sz.loc[df_sz.label == label]
        part_size = len(df_temp)
        train_part = df_temp.iloc[: int(part_size * 0.8), :]
        test_part = df_temp.iloc[int(part_size * 0.8):, :]
        df_train = pd.concat([df_train, train_part], axis=0)
        df_test = pd.concat([df_test, test_part], axis=0)
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    train_X = df_train.drop(['label'], axis=1)
    train_Y = df_train['label']
    test_X = df_test.drop(['label'], axis=1)
    test_Y = df_test['label']
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    param = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'slient': 1,
        'num_class': 3,
        'max_depth': 8,
        'nthread': 4,
        'eval_metric': 'merror',
        'seed': 777,
    }
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 39
    bst = xgb.train(param, xg_train, num_round, watchlist)
    pre = bst.predict(xg_test)
    acc = accuracy_score(test_Y, pre)
    print(acc)

    con_m = confusion_matrix(test_Y, pre, labels=[0, 1, 2])
    print('the confusion matrix:')
    print(con_m)

    bst.save_model(SZ_MODEL_PATH)

    def write_sz_mac_list_to_file(mac_list):
        with open(SZ_MAC_LIST_PATH, 'w') as f:
            for mac_id in mac_list:
                f.write(mac_id + '\n')

    write_sz_mac_list_to_file(sz_mac_list)

    floor_map_dict = {}
    floor_map_dict[0] = 'suzhou 1 floor'
    floor_map_dict[1] = 'suzhou 2 floor'
    floor_map_dict[2] = 'suzhou 3 floor'
    write_floor_map_file(floor_map_dict, 'suzhou.txt')


def write_floor_map_file(floor_map_dict, filename):
    FLOOR_MAP_PATH = filename
    with open(FLOOR_MAP_PATH, 'w') as f:
        f.write('prediction_label' + ',' + 'true_floor' + '\n')
        for pred, floor in floor_map_dict.items():
            f.write(str(pred) + ',' + floor + '\n')
            print(pred, floor)
    return floor_map_dict


# 深圳训练集Wi-Fi训练数据解析
def get_total_train_data(COMMON_TRAIN_PATH):
    # FLOOR = ['J1B1//', 'J1F1//', 'J1F2//', 'J1F3//']
    FLOOR = ['J1B1//', 'J1F1//', 'J1F2//', 'J1F3//', 'H2F4//', 'H2F5//', 'H2F6//', 'H2F7//']
    # FLOOR = ['H2B1//', 'H2F1//', 'H2F2//', 'H2F3//']
    df_total_data = pd.DataFrame()
    for floor_str in tqdm(FLOOR):
        df = pd.DataFrame()
        floor = floor_str[-4:]
        floor = int(floor[-3]) if floor[-4] == 'F' else 0
        total_rssi_list = []
        for file in os.listdir(COMMON_TRAIN_PATH +  floor_str):
            with open(COMMON_TRAIN_PATH + floor_str + file,'r') as f:
                data = f.readline()
                data_dict = json.loads(data)
                for atom_data in data_dict['datas']:
                    if 'wifi' in atom_data.keys() and len(atom_data['wifi']) != 0:
                        rssi_dict = {}
                        for tmp_dict in atom_data['wifi']:
                            rssi_dict[tmp_dict['mac']] = int(tmp_dict['rssi'])
                        total_rssi_list.append(json.dumps(rssi_dict))
        # total_rssi_list = list(set(total_rssi_list))
        df['rssi_list'] = total_rssi_list
        df['label'] = floor
        df_total_data = pd.concat([df_total_data, df], axis=0)
    # df_total_data.drop_duplicates(['rssi_list'], inplace=True)
    return df_total_data

#深圳训练集当测试集用获取测试集
def get_total_test_data(COMMON_TRAIN_PATH):
    # FLOOR = ['J1B1//', 'J1F1//', 'J1F2//', 'J1F3//']
    # FLOOR = ['J1B1//', 'J1F1//','J1F2//']
    FLOOR = ['J1B1//', 'J1F1//', 'J1F2//', 'J1F3//','J1F4//','J1F5//','J1F6//','J1F7//']
    df_total_data = pd.DataFrame()
    for floor_str in tqdm(FLOOR):
        df = pd.DataFrame()
        floor = floor_str[-4:]
        floor = int(floor[-3]) if floor[-4] == 'F' else 0
        total_rssi_list = []

        for file in os.listdir(COMMON_TRAIN_PATH +  floor_str):
            with open(COMMON_TRAIN_PATH + floor_str + file,'r') as f:
                data = f.readline()
                data_dict = json.loads(data)
                for atom_data in data_dict['datas']:
                    if 'wifi' in atom_data.keys() and len(atom_data['wifi']) != 0:
                        rssi_dict = {}
                        for tmp_dict in atom_data['wifi']:
                            rssi_dict[tmp_dict['mac']] = int(tmp_dict['rssi'])
                        total_rssi_list.append(json.dumps(rssi_dict))
        # total_rssi_list = list(set(total_rssi_list))
        df['rssi_list'] = total_rssi_list
        df['label'] = floor
        df_total_data = pd.concat([df_total_data, df], axis=0)
    # df_total_data.drop_duplicates(['rssi_list'], inplace=True)
    return df_total_data

def get_mac_from_file():
    # df = pd.read_csv("E://learn//Senior//Huawei//feature.csv")
    maclist = []
    with open('E://learn//Senior//Huawei//mac_list.txt','r') as f:
        data = f.readlines()
    for i in range(0,len(data)):
        maclist.append(data[i].strip('\n'))
    return maclist

def train_model():
    TRAIN_PATH = 'E://learn//Senior//Huawei//trainmodel//train.csv'
    TEST_PATH = 'E://learn//Senior//Huawei//trainmodel//test.csv'
    # newtotal：0704&0628&0712 totaltraindata:0712&0628
    # df_train = get_total_train_data('E://learn//Senior//Huawei//0725total//')
    df_train = get_total_train_data('E://learn//Senior//Huawei//0904total//')
    # df_train = get_total_train_data('E://learn//Senior//Huawei//0712//')
    # df_train = get_total_train_data('E://learn//Senior//Huawei//0628traindata//')
    # df_train = get_total_train_data('E://learn//Senior//Huawei//totaltraindata//')
    # df_train = get_total_train_data('E://learn//Senior//Huawei//newtotal//')
    # df_train = get_total_train_data('E://learn//Senior//Huawei//H2total//')
    # df_valid = get_total_test_data('E://learn//Senior//Huawei//0716-2//')
    # print("total df_valid")
    # print(len(df_valid))
    # df_valid = get_total_test_data()
    df_valid = get_total_test_data('E://learn//Senior//Huawei//0904total//')
    # df_valid.drop(['id'], axis=1, inplace=True)
    df_valid.columns = ['rssi_list', 'label']
    # df_valid.drop_duplicates(['rssi_list'], inplace=True)
    # train_mac_list = get_mac_from_file().values.tolist()
    # train_mac_list = get_mac_from_file()
    train_mac_list = get_total_mac_list(df_train)
    df_train = get_one_hot_transformation(df_train, train_mac_list)
    df_valid = get_one_hot_transformation(df_valid, train_mac_list)

    df_train.to_csv(TRAIN_PATH, index=False)
    df_valid.to_csv(TEST_PATH, index=False)
    train_X = df_train.drop(['label'], axis=1)
    train_Y = df_train['label']
    test_X = df_valid.drop(['label'], axis=1)
    test_Y = df_valid['label']
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    param = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'slient': 1,
        'num_class':8,
        'max_depth': 8,
        'nthread': 4,
        'eval_metric': 'merror',
        'seed': 777,
    }

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 39

    bst = xgb.train(param, xg_train, num_round,watchlist)
    bst.save_model('xgb_model.model')
    # importances = bst.get_score()
    # res = sorted(importances.items(), key=lambda importances:importances[1],reverse=True)
    # importantMac = []
    # for i in res:
    #     importantMac.append(i[0])
    # print(len(importantMac))
    # df_train = get_total_train_data('E://learn//Senior//Huawei//0716-1//')
    # df_train = get_one_hot_transformation(df_train, importantMac)
    # train_X = df_train.drop(['label'], axis=1)
    # train_Y = df_train['label']
    # xg_train = xgb.DMatrix(train_X, label=train_Y)
    # param = {
    #     'objective': 'multi:softmax',
    #     'eta': 0.1,
    #     'slient': 1,
    #     'num_class': 4,
    #     'max_depth': 8,
    #     'nthread': 4,
    #     'eval_metric': 'merror',
    #     'seed': 777,
    # }
    # watchlist = [(xg_train, 'train')]
    # num_round = 39
    # bst = xgb.train(param, xg_train, num_round, watchlist)

    # 测试
    # df_valid = get_total_test_data('E://learn//Senior//Huawei//0716-2//')
    # df_valid.columns = ['rssi_list', 'label']
    # df_valid = get_one_hot_transformation(df_valid, importantMac)
    # test_X = df_valid.drop(['label'], axis=1)
    # test_Y = df_valid['label']
    #
    # xg_test = xgb.DMatrix(test_X, label=test_Y)
    pre = bst.predict(xg_test)
    pd.DataFrame(pre).to_csv("result.csv")
    acc = accuracy_score(test_Y, pre)
    print(acc)
    con_m = confusion_matrix(test_Y, pre, labels=[0, 1, 2, 3,4,5,6,7])
    print('the confusion matrix:')
    print(con_m)
    # floor_map_dict = {}
    # floor_map_dict[0] = 'J1b1 floor'
    # floor_map_dict[1] = 'J1F1 floor'
    # floor_map_dict[2] = 'J1F2 floor'
    # floor_map_dict[3] = 'J1F3 floor'
    # floor_map_dict[4] = 'OMHB1 floor'
    # floor_map_dict[5] = 'OMHF1 floor'
    # floor_map_dict[6] = 'OMHF2 floor'
    # floor_map_dict[7] = 'ICTF6 floor'
    # floor_map_dict[8] = 'ICTF7 floor'
    # floor_map_dict[9] = 'ICTF8 floor'
    # write_floor_map_file(floor_map_dict, 'E://learn//Senior//Huawei//xgboost_floorIdentification//level//shenzhen.txt')

def train_model_prob():
    df_train = get_total_train_data('E://learn//Senior//Huawei//J0928//')
    # df_valid = get_total_test_data('E://learn//Senior//Huawei//0716-1//')
    # df_valid.columns = ['rssi_list', 'label']
    train_mac_list = get_total_mac_list(df_train)
    df_train = get_one_hot_transformation(df_train, train_mac_list)
    # df_valid = get_one_hot_transformation(df_valid, train_mac_list)

    train_X = df_train.drop(['label'], axis=1)
    train_Y = df_train['label']
    # test_X = df_valid.drop(['label'], axis=1)
    # test_Y = df_valid['label']
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    # xg_test = xgb.DMatrix(test_X, label=test_Y)
    param = {
        'objective': 'multi:softprob',
        'eta': 0.1,
        'slient': 1,
        'num_class':8,
        'max_depth': 8,
        'nthread': 4,
        'eval_metric': 'merror',
        'seed': 777,
    }
    watchlist = [(xg_train, 'train')]
    num_round = 39

    bst = xgb.train(param, xg_train, num_round,watchlist)
    bst.save_model('xgb_model_prob.model')

    # yprob = bst.predict(xg_test).reshape(test_Y.shape[0],4)
    # print(yprob)
    # ylabel = np.argmax(yprob, axis=1)# return the index of the biggest pro
    # print(ylabel)

if __name__ == '__main__':
    # train_cnt, test_cnt = check_distribution()
    # lgb_advanced_evaluation()
    # xgb_advanced_evaluation()
    # omh_ict_test()
    # sz_test()
    # omh_test()
    # new_omh_ict_test()
    # new_xgb_advanced_evaluation(size=8)
    # new_xgb_advanced_evaluation(size=8)
    # df = get_total_train_data('E://learn//Senior//Huawei//0712//')
    # train_mac_list = get_total_mac_list(df)
    # mac_list = list(set(train_mac_list))
    # df = get_one_hot_transformation(df, mac_list)
    # xgb_evaluation(df)
    # train_model()
    train_model_prob()
    # get_total_test_data()
    print("finished...")
