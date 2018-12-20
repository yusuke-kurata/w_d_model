import pandas as pd
import os
import urllib
import copy

import tensorflow as tf
import numpy as np

TPRO_TRAIN = "tpro_train2.csv"
TPRO_TEST = "tpro_test2.csv"

#データ読み込み用のメソッド作成
def read_file(file_name):
    classVal = pd.read_csv(file_name, usecols = ['classVal'])
    holiday_flag = pd.read_csv(file_name, usecols = ['holiday_flag'])
    time_zone = pd.read_csv(file_name, usecols = ['time_zone'])
    receipt_id = pd.read_csv(file_name, usecols = ['receipt_id'])
    weather = pd.read_csv(file_name, usecols = ['weather'])
    num = pd.read_csv(file_name, usecols = ['num'])
    price = pd.read_csv(file_name, usecols = ['price'])
    v1 = pd.read_csv(file_name, usecols = ['v1'])
    v2 = pd.read_csv(file_name, usecols = ['v2'])
    v3 = pd.read_csv(file_name, usecols = ['v3'])
    
    label = pd.read_csv(file_name, usecols = ['item_id'])
    return classVal.values, holiday_flag.values, time_zone.values, receipt_id.values, weather.values, num.values, price.values, v1.values, v2.values, v3.values, label.values

#データ読み込み
classVal_tra, holiday_flag_tra, time_zone_tra, receipt_id_tra, weather_tra, num_tra, price_tra, v1_tra, v2_tra, v3_tra, training_y= read_file(TPRO_TRAIN)
classVal_tes, holiday_flag_tes, time_zone_tes, receipt_id_tes, weather_tes, num_tes, price_tes, v1_tes, v2_tes, v3_tes, test_y = read_file(TPRO_TEST)

train_y_label = pd.DataFrame(training_y, columns = ["item_id"])
label_list = train_y_label.groupby("item_id").count().index
CLASS_NUM  = len(label_list)
print('CLASS_NUM: {}'.format(CLASS_NUM))

##【重要】教師ラベルの値を[0, CLASS_NUM]に補正（ラベルの値がCLASS_NUMを超えるとエラーになる）
label_dict = {}
for i, label in enumerate(label_list): label_dict[label] = i
relocate_train_y = copy.deepcopy(training_y)
relocate_test_y = copy.deepcopy(test_y)

# applyはこの段階ではもう使えない
# relocate_train_y.apply(lambda x: label_dict[x].values)
for i in range(len(relocate_train_y)):
    relocate_train_y[i] = label_dict[training_y[i][0]]

for i in range(len(relocate_test_y)):
    relocate_test_y[i] = label_dict[test_y[i][0]]


# featureのデータ型の指定

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

v1 = tf.feature_column.numeric_column("v1", shape=[1])

v2 = tf.feature_column.numeric_column("v2", shape=[1])

v3 = tf.feature_column.numeric_column("v3", shape=[1])
base_columns = [
  classVal, holiday_flag, time_zone, receipt_id,
  weather, num, price, v1, v2, v3
]

wide_columns = base_columns
# wide_columns = base_columns + crossed_columns

deep_columns =  [tf.feature_column.indicator_column(classVal), 
                    tf.feature_column.indicator_column(holiday_flag), 
                    tf.feature_column.indicator_column(time_zone), 
                    receipt_id,
                    tf.feature_column.indicator_column(weather), 
                    num, 
                    price, 
                    v1, 
                    v2, 
                    v3]

# モデルの設定
# classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
#                                         hidden_units=[10, 20, 10],
#                                         n_classes=CLASS_NUM,
#                                         model_dir="./tpro_model")

classifier = tf.estimator.DNNLinearCombinedClassifier(
    n_classes=CLASS_NUM,
    model_dir="./tpro_model",

    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[10, 20, 10],
    # config=run_config
    )


# 学習用データ(辞書)を返す関数の作成
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"classVal": np.array(classVal_tra), "holiday_flag": np.array(holiday_flag_tra),  "time_zone": np.array(time_zone_tra),  
       "receipt_id": np.array(receipt_id_tra),  "weather": np.array(weather_tra),
       "num": np.array(num_tra),  "price": np.array(price_tra),  "v1": np.array(v1_tra),  "v2": np.array(v2_tra),  "v3": np.array(v3_tra)},
    y=np.array(relocate_train_y),
    num_epochs=None,
    shuffle=True)

# 学習の実行。stepsを分割して実行しても同様の結果を得られる
classifier.train(input_fn=train_input_fn, steps=200000)

# 評価用データ(辞書)を返す関数の作成
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"classVal": np.array(classVal_tes), "holiday_flag": np.array(holiday_flag_tes),  "time_zone": np.array(time_zone_tes),  
       "receipt_id": np.array(receipt_id_tes), "weather": np.array(weather_tes), 
       "num": np.array(num_tes),  "price": np.array(price_tes),  "v1": np.array(v1_tes),  "v2": np.array(v2_tes),  "v3": np.array(v3_tes)},
    y=np.array(relocate_test_y),
    num_epochs=1,
    shuffle=False)

# 評価
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))