import pandas as pd
import os
import urllib
import copy

import tensorflow as tf
import numpy as np

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# 読み込みファイル指定
# column名はデータ読み込み後に付与
TPRO_TRAIN = "tpro_train_noImg.csv"
TPRO_TEST = "tpro_test_noImg.csv"
MODEL_DIR = "./tpro_model_NoImage5"

df_all = pd.read_csv(TPRO_TRAIN)
df_test_all = pd.read_csv(TPRO_TEST)
## 画像特徴量のrename


df_all.columns = ['classVal', 'holiday_flag', 'time_zone', 'receipt_id', 'weather', 'item_id', 'num', 'price'] + \
                 ['v'+str(i+1) for i in range(len(df_all.iloc[:,8:].columns))]

print("------------columns------------")
print(df_all.columns)
print("------------vector num------------")
print(len(df_all.iloc[:,8:].columns))

df_test_all.columns = ['classVal', 'holiday_flag', 'time_zone', 'receipt_id', 'weather', 'item_id', 'num', 'price'] + \
                 ['v'+str(i+1) for i in range(len(df_all.iloc[:,8:].columns))]


# ターゲットの設定
training_y = df_all['item_id']
test_y = df_test_all['item_id']

train_y_label = pd.DataFrame(training_y, columns = ["item_id"])
label_list = train_y_label.groupby("item_id").count().index
CLASS_NUM  = len(label_list)
print('CLASS_NUM: {}'.format(CLASS_NUM))

##【重要】教師ラベルの値を[0, CLASS_NUM]に補正（ラベルの値がCLASS_NUMを超えるとエラーになる）
label_dict = {}
for i, label in enumerate(label_list): label_dict[label] = i
relocate_train_y = copy.deepcopy(training_y)
relocate_test_y = copy.deepcopy(test_y)

for i in range(len(relocate_train_y)):
    relocate_train_y[i] = label_dict[training_y[i]]

for i in range(len(relocate_test_y)):
    relocate_test_y[i] = label_dict[test_y[i]]


# 使用する特徴量の設定
# feature_x：item_id以外の全column
feature_x = {}
feature_test_x = {}
col_list = df_all.columns.drop('item_id')

f_tra_tmp = df_all.drop('item_id', axis=1)
f_tes_tmp = df_test_all.drop('item_id', axis=1)

for i in col_list:
    feature_x[i] = f_tra_tmp[i]
    feature_test_x[i] = f_tes_tmp[i]

classVal = tf.feature_column.categorical_column_with_vocabulary_list(
  'classVal', [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26])
      
holiday_flag = tf.feature_column.categorical_column_with_vocabulary_list(
  'holiday_flag', [0, 1])

time_zone = tf.feature_column.categorical_column_with_vocabulary_list(
  'time_zone', [
      1, 2, 3, 4, 5, 6, 7, 8])

receipt_id = tf.feature_column.numeric_column("receipt_id", shape=[1])

weather = tf.feature_column.categorical_column_with_vocabulary_list(
  'weather', [
      1, 2, 3, 4, 10, 12, 15])  

num = tf.feature_column.numeric_column("num", shape=[1])

price = tf.feature_column.numeric_column("price", shape=[1])

v = []
# for i in range(0, 3):
for i in range(0, len(df_all.iloc[:,8:].columns)):
    v.append(tf.feature_column.numeric_column("v"+str(i+1), shape=[1]))

w_col1 = [classVal, holiday_flag, time_zone, receipt_id,
                weather, num, price]

w_col2 = []
d_col2 = []
for i in range(0, len(df_all.iloc[:,8:].columns)):
    w_col2.append(v[i])
    d_col2.append(v[i])

d_col1 =  [tf.feature_column.indicator_column(classVal), 
                    tf.feature_column.indicator_column(holiday_flag), 
                    tf.feature_column.indicator_column(time_zone), 
                    receipt_id,
                    tf.feature_column.indicator_column(weather), 
                    num, 
                    price]

wide_columns = w_col1
# wide_columns = base_columns + crossed_columns
deep_columns = d_col1 + d_col2
# deep_columns = d_col1

print("--------------wide_columns--------------")
# print(wide_columns)
print("--------------deep_columns--------------")
# print(deep_columns)
                    
# モデルの設定
classifier = tf.estimator.DNNLinearCombinedClassifier(
    n_classes = CLASS_NUM,
    model_dir = MODEL_DIR,

    linear_feature_columns = wide_columns,
    dnn_feature_columns = deep_columns,
    dnn_hidden_units = [1000, 200, 50],
    # config = run_config
    )


# 学習用データ(辞書)を返す関数の作成
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = feature_x,
    y = np.array(relocate_train_y),
    num_epochs = 1,
    shuffle = True)

# 学習の実行。stepsを分割して実行しても同様の結果を得られる
classifier.train(input_fn=train_input_fn, steps=100)

# 評価用データ(辞書)を返す関数の作成
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = feature_test_x,
    y = np.array(relocate_test_y),
    num_epochs = 1,
    shuffle = False)

# 評価
# accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
accuracy_score = classifier.evaluate(input_fn=test_input_fn)


print("Test Accuracy: {0:f}\n".format(accuracy_score["accuracy"]))
print("Test loss: {0:f}\n".format(accuracy_score["loss"]))
print("Test average_loss: {0:f}\n".format(accuracy_score["average_loss"]))