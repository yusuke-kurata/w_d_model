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

item_num = 10

import sys
import os

# PYTHONPATHの設定（必要であれば）
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# import pprint
# pprint.pprint(sys.path)
from absl import app as absl_app
from absl import flags
import tensorflow as tf
from official.utils.flags import core as flags_core
from official.utils.logs import logger
import census_dataset_krt as census_dataset
from official.wide_deep import wide_deep_run_loop

#使用データやモデルの吐き出す場所

def define_census_flags():
  wide_deep_run_loop.define_wide_deep_flags()
  flags.adopt_module_key_flags(wide_deep_run_loop) #おまじない
  #data,model Pathの変更
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

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
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
        #linear_optimizer=tf.train.FtrlOptimizer(...),  #wideの手法の選択
        #dnn_optimizer=tf.train.ProximalAdagradOptimizer(...),  #deepの手法の選択
        
        n_classes= item_num, #デフォルトでは2になっていて、これを設定することで多クラス分類にすることが可能
        # label_vocabulary=None, n_classesの値とセットで、文字列のリストをセットすれば良い？
        
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def run_census(flags_obj):
  """Construct all necessary functions and call run_loop.

  Args:
    flags_obj: Object containing user specified flags.
  """
  # データをDLする
  # 一旦コメントアウト
  # if flags_obj.download_if_missing:
    # census_dataset.download(flags_obj.data_dir)
  train_file = os.path.join(flags_obj.data_dir, census_dataset.TRAINING_FILE)
  print('----------------------------------------')
  print(train_file)
  print('----------------------------------------')
  test_file = os.path.join(flags_obj.data_dir, census_dataset.EVAL_FILE)

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
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