# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Train DNN on census income dataset."""
import tensorflow as tf

# 商品数
item_num = 180
# 各商品(item_id)のカテゴリ
label_vocab = ['1', '21', '22', '23', '24', '25', '32', '33', '37', '38',
               '44', '50', '56', '67', '68', '79', '80', '83', '85', '87',
               '89', '90', '98', '100', '103', '125', '152', '167', '177', '193',
               '203', '205', '207', '212', '219', '237', '238', '239', '259', '260',
               '261', '264', '265', '266', '272', '280', '282', '284', '285', '286',
               '291', '295', '299', '312', '321', '336', '343', '344', '347', '353',
               '365', '381', '387', '392', '405', '417', '418', '422', '425', '428',
               '430', '431', '432', '434', '443', '445', '453', '465', '476', '487',
               '492', '493', '494', '501', '502', '505', '509', '510', '511', '512',
               '513', '514', '515', '526', '531', '544', '546', '552', '554', '555',
               '562', '572', '581', '585', '589', '591', '595', '596', '600', '604',
               '608', '610', '611', '612', '613', '615', '616', '623', '624', '627',
               '628', '629', '630', '632', '639', '640', '641', '642', '643', '644',
               '645', '651', '656', '658', '660', '664', '667', '670', '671', '672',
               '673', '675', '688', '692', '694', '699', '700', '701', '702', '703',
               '705', '707', '708', '709', '713', '714', '717', '720', '724', '725',
               '729', '732', '733', '737', '738', '739', '744', '746', '752', '763',
               '764', '768', '773', '776', '777', '780', '798', '807', '808', '809'
               ]

import sys
import os

# PYTHONPATHの設定（必要であれば）
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# import pprint
# pprint.pprint(sys.path)
from absl import app as absl_app
from absl import flags

from official.utils.flags import core as flags_core
from official.utils.logs import logger
import census_dataset_krt as census_dataset
from official.wide_deep import wide_deep_run_loop

def define_census_flags():
  wide_deep_run_loop.define_wide_deep_flags()
  flags.adopt_module_key_flags(wide_deep_run_loop) 
  flags_core.set_defaults(data_dir='/home/ubuntu/t_proj/models/official/wide_deep/krt-data',
                          model_dir='/home/ubuntu/t_proj/models/official/wide_deep/krt-model',
                          train_epochs=4,  #エポック数等の設定
                          epochs_between_evals=2,
                          inter_op_parallelism_threads=0,
                          intra_op_parallelism_threads=0,
                          batch_size=40)
# L112から。引数なしでなぜ動くのか？ model_column_fnは呼び出し元L111の引数を使っているのか？
def build_estimator(model_dir, model_type, model_column_fn, inter_op, intra_op):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = model_column_fn()
  hidden_units = [100, 75, 50, 25] #隠れ層の設定

  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0},
                                    inter_op_parallelism_threads=inter_op,
                                    intra_op_parallelism_threads=intra_op))
                                    
#model_typeを指定することでwideだけ、deepだけの手法にすることが可能
  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    
    return tf.estimator.DNNLinearCombinedClassifier(
        #Args:参照 https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier

        #デフォルトでは2になっていて、これを設定することで多クラス分類にすることが可能
        n_classes= item_num, 
        # n_classesの値とセットで、文字列のリストをセットすれば良い？
        label_vocabulary=label_vocab, 
        
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def run_census(flags_obj):
  # データをDLする
  # 一旦コメントアウト
  # if flags_obj.download_if_missing:
    # census_dataset.download(flags_obj.data_dir)
  train_file = os.path.join(flags_obj.data_dir, census_dataset.TRAINING_FILE)
  test_file = os.path.join(flags_obj.data_dir, census_dataset.EVAL_FILE)

  # 学習データを作成する
  def train_input_fn():
    return census_dataset.input_fn(train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size)
  # 評価データを作成する
  def eval_input_fn():
    return census_dataset.input_fn(test_file, 1, False, flags_obj.batch_size)

  tensors_to_log = {
      'average_loss': '{loss_prefix}head/truediv',
      'loss': '{loss_prefix}head/weighted_loss/Sum'
  }

  wide_deep_run_loop.run_loop(
      name="T-project", train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      model_column_fn=census_dataset.build_model_columns,
      build_estimator_fn=build_estimator,
      flags_obj=flags_obj,
      tensors_to_log=tensors_to_log,
      early_stop=True)

def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_census(flags.FLAGS)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_census_flags()
  absl_app.run(main)