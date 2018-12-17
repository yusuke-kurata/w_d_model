# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Download and clean the Census Income Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# pylint: disable=wrong-import-order
from absl import app as absl_app
from absl import flags
from six.moves import urllib
import tensorflow as tf
from flask import session
# pylint: enable=wrong-import-order

from official.utils.flags import core as flags_core

DATA_URL = '/t_proj/models/official/wide_deep/krt-data'
TRAINING_FILE = 'test.csv'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'test_eval.csv'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)


# vXは画像ベクトル(256次元)
# csvのカラム的にはserialは存在するが、学習のための特徴量としては使用しない
#
# item_idがターゲットの多クラス分類
#
#
_CSV_COLUMNS = [
    'classVal', 'holiday_flag', 'time_zone', 'receipt_id',
    'weather', 'item_id', 'num', 'price',"v1","v2","v3"
]

# 設定しなきゃうまく動かないのか？
_CSV_COLUMN_DEFAULTS = [[''], [''], [''], [0], 
                        [''], [0], [0], [0], [0.0], [0.0], [0.0]]

_HASH_BUCKET_SIZE = 300

_NUM_EXAMPLES = {
    'train': 3060,
    'validation': 759,
}


def _download_and_clean_file(filename, url):
  """Downloads data from url, and makes changes to match the CSV format."""
  temp_file, _ = urllib.request.urlretrieve(url)
  with tf.gfile.Open(temp_file, 'r') as temp_eval_file:
    with tf.gfile.Open(filename, 'w') as eval_file:
      for line in temp_eval_file:
        line = line.strip()
        line = line.replace(', ', ',')
        if not line or ',' not in line:
          continue
        if line[-1] == '.':
          line = line[:-1]
        line += '\n'
        eval_file.write(line)
  tf.gfile.Remove(temp_file)


def download(data_dir):
  """Download census data if it is not already present."""
  tf.gfile.MakeDirs(data_dir)

  training_file_path = os.path.join(data_dir, TRAINING_FILE)
  if not tf.gfile.Exists(training_file_path):
    _download_and_clean_file(training_file_path, TRAINING_URL)

  eval_file_path = os.path.join(data_dir, EVAL_FILE)
  if not tf.gfile.Exists(eval_file_path):
    _download_and_clean_file(eval_file_path, EVAL_URL)


# wide and deep用にdatasetを加工する
def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous variable columns
  # age = tf.feature_column.numeric_column('age')

  classVal = tf.feature_column.categorical_column_with_vocabulary_list(
      'classVal', [
          '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
          '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
          '21', '22', '23', '24', '25', '26'])

  # holiday_flag = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'holiday_flag', ['0', '1'])

  time_zone = tf.feature_column.categorical_column_with_vocabulary_list(
      'time_zone', [
          '1', '2', '3', '4', '5', '6', '7', '8'])

  receipt_id = tf.feature_column.numeric_column('receipt_id')

  weather = tf.feature_column.categorical_column_with_vocabulary_list(
      'weather', [
          '1', '2', '3', '4',
          '10', '12', '15'])  

  item_id = tf.feature_column.numeric_column('item_id')     

  num = tf.feature_column.numeric_column('num')

  price = tf.feature_column.numeric_column('price')
  
  v1 = tf.feature_column.numeric_column('v1')

  v2 = tf.feature_column.numeric_column('v2')

  v3 = tf.feature_column.numeric_column('v3')
  
  # wideモデルの特徴量の設定
  # そのまま使う場合　収入に関する以外の情報入っている？
  base_columns = [
      classVal, time_zone, receipt_id,
      weather, item_id, num, price, v1, v2, v3
  ]
  # 交差積を用いて特徴量を設定する場合
  # 下の交差積は上の交差積を内包してるように感じるけど、上のは意味あるのか？
  # crossed_columns = [
  #     tf.feature_column.crossed_column(
  #         ['education', 'occupation'], hash_bucket_size=_HASH_BUCKET_SIZE),
  #     tf.feature_column.crossed_column(
  #         [age_buckets, 'education', 'occupation'],
  #         hash_bucket_size=_HASH_BUCKET_SIZE),
  # ]

  # wide_columns = base_columns + crossed_columns
  wide_columns = base_columns
  
  #deepモデルの特徴量の設定　
  deep_columns = [
      tf.feature_column.indicator_column(classVal),
      # tf.feature_column.indicator_column(holiday_flag),
      tf.feature_column.indicator_column(time_zone),
      receipt_id,
      tf.feature_column.indicator_column(weather),
      item_id,
      num,
      price,
      v1,
      v2,
      v3

      # To show an example of embedding
      # 画像データを入れる時もembeddingする必要があり、そのためのfunctionは要調査
      # tf.feature_column.embedding_column(occupation, dimension=8),
  ]

  return wide_columns, deep_columns


# 学習、評価の際の入力データ作成
# 今回の学習でのbatch sizeは40で、全データのうち40columnのデータが1回の学習に使用される
# 全データ / batch size = iteration となる
# 全iteration終了で、1回のepocが終了となる
def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have run census_dataset.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  #予測するターゲットの設定
  def parse_csv(value):
    tf.logging.info('Parsing {}'.format(data_file))
    # csvのレコードをTensorに変換する
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)

    # key:_CSV_COLUMNS, value:Tensor型のcolumns の辞書　を作成
    features = dict(zip(_CSV_COLUMNS, columns))

    from tensorflow.python.framework import ops
    from tensorflow.python.framework import dtypes
    # labels = features.pop('num')
    labels = features.pop('holiday_flag')
    # labels = tf.cast(labels, tf.int32)
    names_str = '1'
    names_tf = ops.convert_to_tensor(names_str, dtype=dtypes.string)
    # constNum = tf.constant(1)    
    print("--------------------------")
    print(labels)
    print("--------------------------")
    
    classes = tf.equal(labels, names_tf)  # binary classification

    return features, classes

  # Extract lines from input files using the Dataset API.
  #データセットを1行ずつ読み込んでいる？
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])
    
  # datasetを関数適用するための変換？あんまり意味はよくわからない
  # 参照：https://deepage.net/tensorflow/2017/07/18/tensorflow-dataset-api.html
  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset


def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", default="/tmp/census_data/",
      help=flags_core.help_wrap(
          "Directory to download and extract data."))


def main(_):
  download(flags.FLAGS.data_dir)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_data_download_flags()
  absl_app.run(main)
