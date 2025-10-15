# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import tempfile
import unittest

from google.protobuf import text_format

from tzrec.protos import pipeline_pb2 as tzrec_pipeline_pb2
from tzrec.tools.convert_easyrec_config_to_tzrec_config import ConvertConfig

FG_JSON = {
    "features": [
        {
            "feature_name": "user_id",
            "feature_type": "id_feature",
            "value_type": "String",
            "expression": "user:user_id",
            "default_value": "-1024",
            "combiner": "mean",
            "need_prefix": False,
            "is_multi": False,
        },
        {
            "feature_name": "item_id",
            "feature_type": "id_feature",
            "value_type": "String",
            "expression": "item:item_id",
            "default_value": "-1024",
            "combiner": "mean",
            "need_prefix": False,
            "is_multi": False,
        },
        {
            "feature_name": "user_blue_level",
            "feature_type": "id_feature",
            "value_type": "String",
            "expression": "user:user_blue_level",
            "default_value": "-1024",
            "combiner": "mean",
            "need_prefix": False,
            "is_multi": False,
        },
        {
            "feature_name": "host_price_level",
            "feature_type": "id_feature",
            "value_type": "String",
            "expression": "item:host_price_level",
            "default_value": "-1024",
            "combiner": "mean",
            "need_prefix": False,
            "is_multi": False,
        },
        {
            "feature_name": "user_video_sequence",
            "feature_type": "id_feature",
            "value_type": "String",
            "expression": "user:user_video_sequence",
            "default_value": "-1024",
            "combiner": "mean",
            "need_prefix": False,
            "is_multi": False,
        },
        {
            "feature_name": "item__kv_user_blue_level_exposure_cnt_7d",
            "feature_type": "lookup_feature",
            "value_type": "Double",
            "map": "item:item__kv_user_blue_level_exposure_cnt_7d",
            "key": "user:user_blue_level",
            "needDiscrete": False,
            "needWeighting": False,
            "needKey": False,
            "default_value": "0",
            "combiner": "mean",
            "need_prefix": False,
        },
        {
            "feature_name": "item__kv_user_blue_level_click_focus_cnt_7d",
            "feature_type": "id_feature",
            "value_type": "Double",
            "expression": "item:item__kv_user_blue_level_click_focus_cnt_7d",
            "default_value": "",
            "combiner": "mean",
            "need_prefix": False,
        },
        {
            "feature_name": "item__kv_user_blue_level_click_video_div_exposure_cnt_30d",
            "feature_type": "lookup_feature",
            "value_type": "Double",
            "map": "item:item__kv_user_blue_level_click_video_div_exposure_cnt_30d",
            "key": "user:user_blue_level",
            "needDiscrete": False,
            "needWeighting": False,
            "needKey": False,
            "default_value": "0",
            "combiner": "mean",
            "need_prefix": False,
        },
        {
            "sequence_name": "click_100_seq",
            "sequence_column": "click_100_seq",
            "sequence_length": 100,
            "sequence_delim": ";",
            "attribute_delim": "#",
            "sequence_table": "item",
            "sequence_pk": "user:click_100_seq",
            "features": [
                {
                    "feature_name": "item_id",
                    "feature_type": "id_feature",
                    "value_type": "String",
                    "expression": "item:item_id",
                    "default_value": "-1024",
                    "combiner": "mean",
                    "need_prefix": False,
                    "is_multi": False,
                    "group": "click_100_seq_feature",
                },
                {
                    "feature_name": "ts",
                    "feature_type": "raw_feature",
                    "value_type": "Double",
                    "expression": "user:ts",
                    "default_value": "-1024",
                    "combiner": "mean",
                    "need_prefix": False,
                    "group": "click_100_seq_feature",
                },
            ],
        },
    ],
    "reserves": ["is_click_cover", "is_click_video"],
}

EASYREC_CONFIG = """train_config {
  optimizer_config {
    use_moving_average: false
    adam_asyncw_optimizer {
      weight_decay: 1e-6
      learning_rate {
        constant_learning_rate {
          learning_rate: 0.001
        }
      }
    }
  }
  sync_replicas: false
  save_summary_steps: 1000
  log_step_count_steps: 100
  save_checkpoints_steps: 1000000
  keep_checkpoint_max: 1
}
data_config {
  batch_size: 4096
  label_fields: "is_click_cover"
  label_fields: "is_click_video"
  shuffle: false
  num_epochs: 3
  input_type: OdpsRTPInput
  separator: ""
  selected_cols: "is_click_cover,is_click_video,features"
  input_fields {
    input_name: "is_click_cover"
    input_type: INT32
    default_val: "0"
  }
  input_fields {
    input_name: "is_click_video"
    input_type: INT32
    default_val: "0"
  }
  input_fields {
    input_name: "user_id"
    input_type: STRING
    default_val: "-1024"
  }
  input_fields {
    input_name: "item_id"
    input_type: STRING
    default_val: "-1024"
  }
  input_fields {
    input_name: "user_blue_level"
    input_type: STRING
    default_val: "-1024"
  }
  input_fields {
    input_name: "host_price_level"
    input_type: STRING
    default_val: "-1024"
  }
  input_fields {
    input_name: "user_video_sequence"
    input_type: STRING
    default_val: "-1024"
  }
  input_fields {
    input_name: "item__kv_user_blue_level_exposure_cnt_7d"
    input_type: DOUBLE
    default_val: "0"
  }
  input_fields {
    input_name: "item__kv_user_blue_level_click_focus_cnt_7d"
    input_type: STRING
    default_val: ""
  }
  input_fields {
    input_name: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    input_type: DOUBLE
    default_val: "0"
  }
  input_fields {
    input_name: "click_100_seq__item_id"
    input_type: STRING
    default_val: ""
  }
  input_fields {
    input_name: "click_100_seq__ts"
    input_type: STRING
    default_val: ""
  }
  pai_worker_queue: true
}
feature_configs {
  input_names: "user_id"
  feature_type: IdFeature
  embedding_dim: 4
  hash_bucket_size: 1000
  separator: ""
  combiner: "mean"
}
feature_configs {
  input_names: "item_id"
  feature_type: IdFeature
  embedding_dim: 24
  hash_bucket_size: 1500000
  separator: ""
  combiner: "mean"
}
feature_configs {
  input_names: "user_blue_level"
  feature_type: IdFeature
  embedding_dim: 4
  hash_bucket_size: 140
  separator: ""
  combiner: "mean"
}
feature_configs {
  input_names: "host_price_level"
  feature_type: IdFeature
  embedding_dim: 8
  hash_bucket_size: 180
  separator: ""
  combiner: "mean"
}
feature_configs {
  input_names: "user_video_sequence"
  feature_type: SequenceFeature
  embedding_dim: 24
  hash_bucket_size: 1500000
  separator: ","
  combiner: "mean"
  sub_feature_type: IdFeature
}

feature_configs {
  input_names: "item__kv_user_blue_level_exposure_cnt_7d"
  feature_type: RawFeature
  embedding_dim: 4
  boundaries: 1e-08
  boundaries: 47.00000001
  boundaries: 285.00000001
  boundaries: 672.00000001
  boundaries: 1186.00000001
  boundaries: 1853.00000001
  boundaries: 2716.00000001
  boundaries: 3861.00000001
  boundaries: 5459.00000001
  boundaries: 7817.00000001
  boundaries: 11722.0
  boundaries: 19513.0
  boundaries: 43334.0
  separator: ""
}
feature_configs {
  feature_name: "item__kv_user_blue_level_click_focus_cnt_7d"
  input_names: "item__kv_user_blue_level_click_focus_cnt_7d"
  input_names: "user_blue_level"
  feature_type: LookupFeature
  embedding_dim: 4
  boundaries: 1e-08
  boundaries: 1.00000001
  boundaries: 2.00000001
  boundaries: 5.00000001
  boundaries: 8.00000001
  boundaries: 13.00000001
  boundaries: 19.00000001
  boundaries: 28.00000001
  boundaries: 42.00000001
  boundaries: 67.00000001
  boundaries: 123.00000001
  boundaries: 298.00000001
  separator: ""
}
feature_configs {
  feature_name: "combo_user_blue_level_x_host_price_level"
  input_names: "user_blue_level"
  input_names: "host_price_level"
  feature_type: ComboFeature
  embedding_dim: 4
  hash_bucket_size: 140
}
feature_configs {
  input_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
  feature_type: RawFeature
  separator: ""
}
feature_configs {
  input_names: "click_100_seq__item_id"
  feature_type: SequenceFeature
  separator: ";"
  combiner: "mean"
  sub_feature_type: IdFeature
  embedding_dim: 24
  hash_bucket_size: 1500000
}
feature_configs {
  input_names: "click_100_seq__ts"
  feature_type: SequenceFeature
  separator: ";"
  combiner: "mean"
  sub_feature_type: RawFeature
  embedding_dim: 4
  boundaries: 1e-08
  boundaries: 3.00000001
  boundaries: 25.00000001
  boundaries: 70.00000001
}

model_config {
  model_class: "DBMTL"
  feature_groups {
    group_name: "all"
    feature_names: "user_id"
	feature_names: "item_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "combo_user_blue_level_x_host_price_level"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    wide_deep: DEEP
    sequence_features {
      group_name: "seq_fea_1"
      seq_att_map {
        key: "item_id"
        hist_seq: "user_video_sequence"
      }
      tf_summary: false
      allow_key_search: false
    }
    sequence_features {
      group_name: "seq_fea_2"
      seq_att_map {
        key: "item_id"
        hist_seq: "click_100_seq__item_id"
		aux_hist_seq: "click_100_seq__ts"
      }
      tf_summary: false
      allow_key_search: false
    }
  }
  dbmtl {
    bottom_dnn {
      hidden_units: 1024
      hidden_units: 512
    }
    task_towers {
      tower_name: "is_click_cover"
      label_name: "is_click_cover"
      metrics_set {
        auc {
        }
      }
      loss_type: CLASSIFICATION
      dnn {
        hidden_units: 256
        hidden_units: 128
        hidden_units: 64
        hidden_units: 32
      }
      relation_dnn {
        hidden_units: 32
      }
      weight: 1.0
    }
    task_towers {
      tower_name: "is_click_video"
      label_name: "is_click_video"
      metrics_set {
        auc {
        }
      }
      loss_type: CLASSIFICATION
      dnn {
        hidden_units: 256
        hidden_units: 128
        hidden_units: 64
        hidden_units: 32
      }
      relation_dnn {
        hidden_units: 32
      }
      weight: 2.0
    }
    l2_regularization: 0
  }
}
export_config {
  multi_placeholder: true
}
"""

EASYREC_CONFIG_SIMPLEMULTITASK = """model_config {
  model_class: "SimpleMultiTask"
  feature_groups {
    group_name: "all"
    feature_names: "user_id"
    feature_names: "item_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "combo_user_blue_level_x_host_price_level"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    wide_deep: DEEP
    sequence_features {
      group_name: "seq_fea_1"
      seq_att_map {
        key: "item_id"
        hist_seq: "user_video_sequence"
      }
      tf_summary: false
      allow_key_search: false
    }
    sequence_features {
      group_name: "seq_fea_2"
      seq_att_map {
        key: "item_id"
        hist_seq: "click_100_seq__item_id"
		aux_hist_seq: "click_100_seq__ts"
      }
      tf_summary: false
      allow_key_search: false
    }
  }
  simple_multi_task {
    task_towers {
      tower_name: "is_click_video"
      label_name: "is_click_video"
      dnn {
        hidden_units: [256, 192, 128, 64]
      }
      num_class: 1
      weight: 1.0
      loss_type: CLASSIFICATION
      metrics_set: {
       auc {}
      }
    }
    l2_regularization: 0.0
  }
  embedding_regularization: 0.0
}
export_config {
  multi_placeholder: true
}
"""

EASYREC_CONFIG_MMOE = """model_config {
  model_class: "MMoE"
  feature_groups {
    group_name: "all"
    feature_names: "user_id"
	feature_names: "item_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "combo_user_blue_level_x_host_price_level"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    wide_deep: DEEP
    sequence_features {
      group_name: "seq_fea_1"
      seq_att_map {
        key: "item_id"
        hist_seq: "user_video_sequence"
      }
      tf_summary: false
      allow_key_search: false
    }
    sequence_features {
      group_name: "seq_fea_2"
      seq_att_map {
        key: "item_id"
        hist_seq: "click_100_seq__item_id"
		aux_hist_seq: "click_100_seq__ts"
      }
      tf_summary: false
      allow_key_search: false
    }
  }
  mmoe {
    expert_dnn {
      hidden_units: [256, 192, 128, 64]
    }
    num_expert: 4
    task_towers {
      tower_name: "is_click_video"
      label_name: "is_click_video"
      dnn {
        hidden_units: [256, 192, 128, 64]
      }
      num_class: 1
      weight: 1.0
      loss_type: CLASSIFICATION
      metrics_set: {
       auc {}
      }
    }
    l2_regularization: 1e-06
  }
  embedding_regularization: 5e-05
}
export_config {
  multi_placeholder: true
}
"""

EASYREC_CONFIG_PLE = """model_config {
  model_class: "PLE"
  feature_groups {
    group_name: "all"
    feature_names: "user_id"
	feature_names: "item_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "combo_user_blue_level_x_host_price_level"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    wide_deep: DEEP
    sequence_features {
      group_name: "seq_fea_1"
      seq_att_map {
        key: "item_id"
        hist_seq: "user_video_sequence"
      }
      tf_summary: false
      allow_key_search: false
    }
    sequence_features {
      group_name: "seq_fea_2"
      seq_att_map {
        key: "item_id"
        hist_seq: "click_100_seq__item_id"
		aux_hist_seq: "click_100_seq__ts"
      }
      tf_summary: false
      allow_key_search: false
    }
  }
  ple {
    extraction_networks {
      network_name: "layer1"
      expert_num_per_task: 3
      share_num: 3

      task_expert_net {
        hidden_units: [128, 64, 32]
      }
      share_expert_net {
        hidden_units: [128, 64, 32]
      }
    }
    extraction_networks {
      network_name: "layer2"
      expert_num_per_task: 3
      share_num: 3

      task_expert_net {
        hidden_units: [64, 32, 16]
      }
      share_expert_net {
        hidden_units: [64, 32, 16]
      }
    }
    extraction_networks {
      network_name: "layer3"
      expert_num_per_task: 3
      share_num: 3

      task_expert_net {
        hidden_units: [64, 32, 16]
      }
      share_expert_net {
        hidden_units: [64, 32, 16]
      }
    }

    task_towers {
      tower_name: "is_click_video"
      label_name: "is_click_video"
      loss_type: CLASSIFICATION
      metrics_set: {
        auc {}
      }
      dnn {
        hidden_units: [128, 64, 32]
      }
      weight: 1.0
    }
    l2_regularization: 1e-6
  }
  embedding_regularization: 5e-6
}
export_config {
  multi_placeholder: true
}
"""

EASYREC_CONFIG_DEEPFM = """model_config {
  model_class: "DeepFM"
  feature_groups {
    group_name: "wide"
    feature_names: "user_id"
	feature_names: "item_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "combo_user_blue_level_x_host_price_level"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    wide_deep: WIDE
    sequence_features {
      group_name: "seq_fea_1"
      seq_att_map {
        key: "item_id"
        hist_seq: "user_video_sequence"
      }
      tf_summary: false
      allow_key_search: false
    }
    sequence_features {
      group_name: "seq_fea_2"
      seq_att_map {
        key: "item_id"
        hist_seq: "click_100_seq__item_id"
		aux_hist_seq: "click_100_seq__ts"
      }
      tf_summary: false
      allow_key_search: false
    }
  }
  feature_groups {
    group_name: "fm"
    feature_names: "user_id"
	feature_names: "item_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "combo_user_blue_level_x_host_price_level"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    wide_deep: DEEP
    sequence_features {
      group_name: "seq_fea_1"
      seq_att_map {
        key: "item_id"
        hist_seq: "user_video_sequence"
      }
      tf_summary: false
      allow_key_search: false
    }
    sequence_features {
      group_name: "seq_fea_2"
      seq_att_map {
        key: "item_id"
        hist_seq: "click_100_seq__item_id"
		aux_hist_seq: "click_100_seq__ts"
      }
      tf_summary: false
      allow_key_search: false
    }
  }
  deepfm {
    wide_output_dim: 16

    dnn {
      hidden_units: [128, 64, 32]
    }

    final_dnn {
      hidden_units: [128, 64]
    }
    l2_regularization: 1e-5
  }
  embedding_regularization: 1e-7
}
export_config {
  multi_placeholder: true
}
"""

EASYREC_CONFIG_MULTITOWER = """model_config {
  model_class: "MultiTower"
  feature_groups {
    group_name: "all"
    feature_names: "user_id"
	feature_names: "item_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "combo_user_blue_level_x_host_price_level"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    wide_deep: DEEP
    sequence_features {
      group_name: "seq_fea_1"
      seq_att_map {
        key: "item_id"
        hist_seq: "user_video_sequence"
      }
      tf_summary: false
      allow_key_search: false
    }
    sequence_features {
      group_name: "seq_fea_2"
      seq_att_map {
        key: "item_id"
        hist_seq: "click_100_seq__item_id"
		aux_hist_seq: "click_100_seq__ts"
      }
      tf_summary: false
      allow_key_search: false
    }
  }
  losses {
    loss_type: F1_REWEIGHTED_LOSS
    weight: 1.0
    f1_reweighted_loss {
      f1_beta_square: 1.0
    }
  }
  losses {
    loss_type: PAIR_WISE_LOSS
    weight: 1.0
  }
  multi_tower {
    towers {
      input: "all"
      dnn {
        hidden_units: [256, 128, 96, 64]
      }
    }
    final_dnn {
      hidden_units: [128, 96, 64, 32, 16]
    }
    l2_regularization: 1e-6
  }
  embedding_regularization: 1e-4
}
export_config {
  multi_placeholder: true
}
"""

EASYREC_CONFIG_DSSM = """model_config {
  model_class: "DSSM"
  feature_groups {
    group_name: "user"
    feature_names: "user_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "combo_user_blue_level_x_host_price_level"
    wide_deep: DEEP
    sequence_features {
      group_name: "seq_fea_1"
      seq_att_map {
        key: "item_id"
        hist_seq: "user_video_sequence"
      }
      tf_summary: false
      allow_key_search: false
    }
    sequence_features {
      group_name: "seq_fea_2"
      seq_att_map {
        key: "item_id"
        hist_seq: "click_100_seq__item_id"
		aux_hist_seq: "click_100_seq__ts"
      }
      tf_summary: false
      allow_key_search: false
    }
  }
  feature_groups {
    group_name: "item"
	feature_names: "item_id"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    wide_deep: DEEP
  }
  dssm {
    user_tower {
      id: "user_id"
      dnn {
        hidden_units: [256, 128, 64, 32]
        # dropout_ratio : [0.1, 0.1, 0.1, 0.1]
      }
    }
    item_tower {
      id: "item_id"
      dnn {
        hidden_units: [256, 128, 64, 32]
      }
    }
    simi_func: INNER_PRODUCT
    l2_regularization: 1e-6
  }
  embedding_regularization: 5e-5
}
export_config {
  multi_placeholder: true
}
"""

TRAIN_CONFIG = """train_config {
  sparse_optimizer {
    adam_optimizer {
      lr: 0.001
    }
    constant_learning_rate {
    }
  }
  dense_optimizer {
    adam_optimizer {
      lr: 0.001
    }
    constant_learning_rate {
    }
  }
  num_epochs: 1
  use_tensorboard: false
}
"""

DATA_CONFIG = """data_config {
  batch_size: 4096
  dataset_type: OdpsDataset
  label_fields: "is_click_cover"
  label_fields: "is_click_video"
  num_workers: 8
}
"""

FEATURE_CONFIG = """feature_configs {
  id_feature {
    feature_name: "user_id"
    expression: "user:user_id"
    embedding_dim: 4
    hash_bucket_size: 1000
  }
}
feature_configs {
  id_feature {
    feature_name: "item_id"
    expression: "item:item_id"
    embedding_dim: 24
    hash_bucket_size: 1500000
  }
}
feature_configs {
  id_feature {
    feature_name: "user_blue_level"
    expression: "user:user_blue_level"
    embedding_dim: 4
    hash_bucket_size: 140
  }
}
feature_configs {
  id_feature {
    feature_name: "host_price_level"
    expression: "item:host_price_level"
    embedding_dim: 8
    hash_bucket_size: 180
  }
}
feature_configs {
  sequence_id_feature {
    feature_name: "user_video_sequence"
    expression: "user:user_video_sequence"
    embedding_dim: 24
    hash_bucket_size: 1500000
  }
}
feature_configs {
  lookup_feature {
    feature_name: "item__kv_user_blue_level_exposure_cnt_7d"
    map: "item:item__kv_user_blue_level_exposure_cnt_7d"
    key: "user:user_blue_level"
    embedding_dim: 4
    boundaries: 1e-08
    boundaries: 47.0
    boundaries: 285.0
    boundaries: 672.0
    boundaries: 1186.0
    boundaries: 1853.0
    boundaries: 2716.0
    boundaries: 3861.0
    boundaries: 5459.0
    boundaries: 7817.0
    boundaries: 11722.0
    boundaries: 19513.0
    boundaries: 43334.0
  }
}
feature_configs {
  lookup_feature {
    feature_name: "item__kv_user_blue_level_click_focus_cnt_7d"
    map: "item:item__kv_user_blue_level_click_focus_cnt_7d"
    key: "user:user_blue_level"
    embedding_dim: 4
    boundaries: 1e-08
    boundaries: 1.0
    boundaries: 2.0
    boundaries: 5.0
    boundaries: 8.0
    boundaries: 13.0
    boundaries: 19.0
    boundaries: 28.0
    boundaries: 42.0
    boundaries: 67.0
    boundaries: 123.0
    boundaries: 298.0
  }
}
feature_configs {
  combo_feature {
    feature_name: "combo_user_blue_level_x_host_price_level"
    expression: "user:user_blue_level"
    expression: "item:host_price_level"
    embedding_dim: 4
    hash_bucket_size: 140
  }
}
feature_configs {
  lookup_feature {
    feature_name: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    map: "item:item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    key: "user:user_blue_level"
    embedding_dim: 0
  }
}
feature_configs {
  sequence_feature {
    sequence_name: "click_100_seq"
    sequence_length: 100
    sequence_delim: ";"
    features {
      id_feature {
        feature_name: "item_id"
        expression: "item:item_id"
        embedding_dim: 24
        hash_bucket_size: 1500000
      }
    }
    features {
      raw_feature {
        feature_name: "ts"
        expression: "user:ts"
        embedding_dim: 4
        boundaries: 1e-08
        boundaries: 3.0
        boundaries: 25.0
        boundaries: 70.0
      }
    }
  }
}
"""

FEATURE_CONFIG_NO_FG = """feature_configs {
  id_feature {
    feature_name: "user_id"
    expression: "user:user_id"
    embedding_dim: 4
    hash_bucket_size: 1000
  }
}
feature_configs {
  id_feature {
    feature_name: "item_id"
    expression: "user:item_id"
    embedding_dim: 24
    hash_bucket_size: 1500000
  }
}
feature_configs {
  id_feature {
    feature_name: "user_blue_level"
    expression: "user:user_blue_level"
    embedding_dim: 4
    hash_bucket_size: 140
  }
}
feature_configs {
  id_feature {
    feature_name: "host_price_level"
    expression: "user:host_price_level"
    embedding_dim: 8
    hash_bucket_size: 180
  }
}
feature_configs {
  sequence_id_feature {
    feature_name: "user_video_sequence"
    expression: "user:user_video_sequence"
    sequence_length: 1
    sequence_delim: ","
    embedding_dim: 24
    hash_bucket_size: 1500000
  }
}
feature_configs {
  raw_feature {
    feature_name: "item__kv_user_blue_level_exposure_cnt_7d"
    expression: "user:item__kv_user_blue_level_exposure_cnt_7d"
    embedding_dim: 4
    boundaries: 1e-08
    boundaries: 47.0
    boundaries: 285.0
    boundaries: 672.0
    boundaries: 1186.0
    boundaries: 1853.0
    boundaries: 2716.0
    boundaries: 3861.0
    boundaries: 5459.0
    boundaries: 7817.0
    boundaries: 11722.0
    boundaries: 19513.0
    boundaries: 43334.0
  }
}
feature_configs {
  lookup_feature {
    feature_name: "item__kv_user_blue_level_click_focus_cnt_7d"
    map: "user:item__kv_user_blue_level_click_focus_cnt_7d"
    key: "user:user_blue_level"
    embedding_dim: 4
    boundaries: 1e-08
    boundaries: 1.0
    boundaries: 2.0
    boundaries: 5.0
    boundaries: 8.0
    boundaries: 13.0
    boundaries: 19.0
    boundaries: 28.0
    boundaries: 42.0
    boundaries: 67.0
    boundaries: 123.0
    boundaries: 298.0
  }
}
feature_configs {
  combo_feature {
    feature_name: "combo_user_blue_level_x_host_price_level"
    expression: "user:user_blue_level"
    expression: "user:host_price_level"
    embedding_dim: 4
    hash_bucket_size: 140
  }
}
feature_configs {
  raw_feature {
    feature_name: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    expression: "user:item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
  }
}
feature_configs {
  sequence_id_feature {
    feature_name: "click_100_seq__item_id"
    expression: "user:click_100_seq__item_id"
    sequence_length: 1
    sequence_delim: ";"
    embedding_dim: 24
    hash_bucket_size: 1500000
  }
}
feature_configs {
  sequence_raw_feature {
    feature_name: "click_100_seq__ts"
    expression: "user:click_100_seq__ts"
    sequence_length: 1
    sequence_delim: ";"
    embedding_dim: 4
    boundaries: 1e-08
    boundaries: 3.0
    boundaries: 25.0
    boundaries: 70.0
  }
}
"""


MODEL_CONFIG = """model_config {
  feature_groups {
    group_name: "all"
    feature_names: "user_id"
    feature_names: "item_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "combo_user_blue_level_x_host_price_level"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    group_type: DEEP
    sequence_groups {
      group_name: "seq_fea_1"
      feature_names: "item_id"
      feature_names: "user_video_sequence"
    }
    sequence_groups {
      group_name: "seq_fea_2"
      feature_names: "item_id"
      feature_names: "click_100_seq__item_id"
      feature_names: "click_100_seq__ts"
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_1"
        attn_mlp {
          use_bn: true
        }
      }
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_2"
        attn_mlp {
          use_bn: true
        }
      }
    }
  }
  dbmtl {
    bottom_mlp {
      hidden_units: 1024
      hidden_units: 512
      use_bn: true
    }
    expert_mlp {
      use_bn: true
    }
    num_expert: 0
    task_towers {
      tower_name: "is_click_cover"
      label_name: "is_click_cover"
      metrics {
        auc {
        }
      }
      num_class: 1
      mlp {
        hidden_units: 256
        hidden_units: 128
        hidden_units: 64
        hidden_units: 32
        use_bn: true
      }
      relation_mlp {
        hidden_units: 32
        use_bn: true
      }
    }
    task_towers {
      tower_name: "is_click_video"
      label_name: "is_click_video"
      metrics {
        auc {
        }
      }
      num_class: 1
      mlp {
        hidden_units: 256
        hidden_units: 128
        hidden_units: 64
        hidden_units: 32
        use_bn: true
      }
      relation_mlp {
        hidden_units: 32
        use_bn: true
      }
    }
  }
}
"""

MODEL_CONFIG_SIMPLEMULTITASK = """model_config {
  feature_groups {
    group_name: "all"
    feature_names: "user_id"
    feature_names: "item_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "combo_user_blue_level_x_host_price_level"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    group_type: DEEP
    sequence_groups {
      group_name: "seq_fea_1"
      feature_names: "item_id"
      feature_names: "user_video_sequence"
    }
    sequence_groups {
      group_name: "seq_fea_2"
      feature_names: "item_id"
      feature_names: "click_100_seq__item_id"
      feature_names: "click_100_seq__ts"
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_1"
        attn_mlp {
          use_bn: true
        }
      }
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_2"
        attn_mlp {
          use_bn: true
        }
      }
    }
  }
  simple_multi_task {
    task_towers {
      tower_name: "is_click_video"
      label_name: "is_click_video"
      metrics {
        auc {
        }
      }
      num_class: 1
      mlp {
        hidden_units: 256
        hidden_units: 192
        hidden_units: 128
        hidden_units: 64
        use_bn: true
      }
    }
  }
}
"""

MODEL_CONFIG_MMOE = """model_config {
  feature_groups {
    group_name: "all"
    feature_names: "user_id"
    feature_names: "item_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "combo_user_blue_level_x_host_price_level"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    group_type: DEEP
    sequence_groups {
      group_name: "seq_fea_1"
      feature_names: "item_id"
      feature_names: "user_video_sequence"
    }
    sequence_groups {
      group_name: "seq_fea_2"
      feature_names: "item_id"
      feature_names: "click_100_seq__item_id"
      feature_names: "click_100_seq__ts"
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_1"
        attn_mlp {
          use_bn: true
        }
      }
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_2"
        attn_mlp {
          use_bn: true
        }
      }
    }
  }
  mmoe {
    expert_mlp {
      hidden_units: 256
      hidden_units: 192
      hidden_units: 128
      hidden_units: 64
      use_bn: true
    }
    gate_mlp {
      hidden_units: 256
      hidden_units: 192
      hidden_units: 128
      hidden_units: 64
      use_bn: true
    }
    num_expert: 4
    task_towers {
      tower_name: "is_click_video"
      label_name: "is_click_video"
      metrics {
        auc {
        }
      }
      num_class: 1
      mlp {
        hidden_units: 256
        hidden_units: 192
        hidden_units: 128
        hidden_units: 64
        use_bn: true
      }
    }
  }
}
"""

MODEL_CONFIG_PLE = """model_config {
  feature_groups {
    group_name: "all"
    feature_names: "user_id"
    feature_names: "item_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "combo_user_blue_level_x_host_price_level"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    group_type: DEEP
    sequence_groups {
      group_name: "seq_fea_1"
      feature_names: "item_id"
      feature_names: "user_video_sequence"
    }
    sequence_groups {
      group_name: "seq_fea_2"
      feature_names: "item_id"
      feature_names: "click_100_seq__item_id"
      feature_names: "click_100_seq__ts"
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_1"
        attn_mlp {
          use_bn: true
        }
      }
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_2"
        attn_mlp {
          use_bn: true
        }
      }
    }
  }
  ple {
    task_towers {
      tower_name: "is_click_video"
      label_name: "is_click_video"
      metrics {
        auc {
        }
      }
      num_class: 1
      mlp {
        hidden_units: 128
        hidden_units: 64
        hidden_units: 32
        use_bn: true
      }
    }
  }
}
"""

MODEL_CONFIG_DEEPFM = """model_config {
  feature_groups {
    group_name: "wide"
    feature_names: "user_id"
    feature_names: "item_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "combo_user_blue_level_x_host_price_level"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    group_type: WIDE
    sequence_groups {
      group_name: "seq_fea_1"
      feature_names: "item_id"
      feature_names: "user_video_sequence"
    }
    sequence_groups {
      group_name: "seq_fea_2"
      feature_names: "item_id"
      feature_names: "click_100_seq__item_id"
      feature_names: "click_100_seq__ts"
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_1"
        attn_mlp {
          use_bn: true
        }
      }
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_2"
        attn_mlp {
          use_bn: true
        }
      }
    }
  }
  feature_groups {
    group_name: "fm"
    feature_names: "user_id"
    feature_names: "item_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "combo_user_blue_level_x_host_price_level"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    group_type: DEEP
    sequence_groups {
      group_name: "seq_fea_1"
      feature_names: "item_id"
      feature_names: "user_video_sequence"
    }
    sequence_groups {
      group_name: "seq_fea_2"
      feature_names: "item_id"
      feature_names: "click_100_seq__item_id"
      feature_names: "click_100_seq__ts"
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_1"
        attn_mlp {
          use_bn: true
        }
      }
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_2"
        attn_mlp {
          use_bn: true
        }
      }
    }
  }
  deepfm {
    deep {
      hidden_units: 128
      hidden_units: 64
      hidden_units: 32
      use_bn: true
    }
    final {
      hidden_units: 128
      hidden_units: 64
      use_bn: true
    }
    wide_embedding_dim: 16
  }
}
"""

MODEL_CONFIG_MULTITOWER = """model_config {
  feature_groups {
    group_name: "all"
    feature_names: "user_id"
    feature_names: "item_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "combo_user_blue_level_x_host_price_level"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    group_type: DEEP
    sequence_groups {
      group_name: "seq_fea_1"
      feature_names: "item_id"
      feature_names: "user_video_sequence"
    }
    sequence_groups {
      group_name: "seq_fea_2"
      feature_names: "item_id"
      feature_names: "click_100_seq__item_id"
      feature_names: "click_100_seq__ts"
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_1"
        attn_mlp {
          use_bn: true
        }
      }
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_2"
        attn_mlp {
          use_bn: true
        }
      }
    }
  }
  multi_tower {
    towers {
      input: "all"
      mlp {
        hidden_units: 256
        hidden_units: 128
        hidden_units: 96
        hidden_units: 64
        use_bn: true
      }
    }
    final {
      hidden_units: 128
      hidden_units: 96
      hidden_units: 64
      hidden_units: 32
      hidden_units: 16
      use_bn: true
    }
  }
}
"""

MODEL_CONFIG_DSSM = """model_config {
  feature_groups {
    group_name: "user"
    feature_names: "user_id"
    feature_names: "user_blue_level"
    feature_names: "host_price_level"
    feature_names: "combo_user_blue_level_x_host_price_level"
    group_type: DEEP
    sequence_groups {
      group_name: "seq_fea_1"
      feature_names: "item_id"
      feature_names: "user_video_sequence"
    }
    sequence_groups {
      group_name: "seq_fea_2"
      feature_names: "item_id"
      feature_names: "click_100_seq__item_id"
      feature_names: "click_100_seq__ts"
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_1"
        attn_mlp {
          use_bn: true
        }
      }
    }
    sequence_encoders {
      din_encoder {
        input: "seq_fea_2"
        attn_mlp {
          use_bn: true
        }
      }
    }
  }
  feature_groups {
    group_name: "item"
    feature_names: "item_id"
    feature_names: "item__kv_user_blue_level_exposure_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_focus_cnt_7d"
    feature_names: "item__kv_user_blue_level_click_video_div_exposure_cnt_30d"
    group_type: DEEP
  }
  dssm {
    user_tower {
      input: "user_id"
      mlp {
        hidden_units: 256
        hidden_units: 128
        hidden_units: 64
        hidden_units: 32
        use_bn: true
      }
    }
    item_tower {
      input: "item_id"
      mlp {
        hidden_units: 256
        hidden_units: 128
        hidden_units: 64
        hidden_units: 32
        use_bn: true
      }
    }
    output_dim: 32
  }
}
"""

PYFG_JSON = {
    "features": [
        {
            "feature_name": "user_id",
            "feature_type": "id_feature",
            "value_type": "String",
            "expression": "user:user_id",
            "default_value": "-1024",
            "combiner": "mean",
            "need_prefix": False,
            "is_multi": False,
        },
        {
            "feature_name": "item_id",
            "feature_type": "id_feature",
            "value_type": "String",
            "expression": "item:item_id",
            "default_value": "-1024",
            "combiner": "mean",
            "need_prefix": False,
            "is_multi": False,
        },
        {
            "feature_name": "hotel_price",
            "feature_type": "raw_feature",
            "value_type": "Double",
            "expression": "context:hotel_price",
            "default_value": "-1999999",
            "combiner": "mean",
            "need_prefix": False,
        },
        {
            "feature_name": "item__kv_user_blue_level_exposure_cnt_7d",
            "feature_type": "lookup_feature",
            "value_type": "Double",
            "map": "item:item__kv_user_blue_level_exposure_cnt_7d",
            "key": "user:user_blue_level",
            "needDiscrete": False,
            "needWeighting": False,
            "needKey": False,
            "default_value": "0",
            "combiner": "mean",
            "need_prefix": False,
        },
        {
            "feature_name": "item__kv_user_blue_level_click_focus_cnt_7d",
            "feature_type": "match_feature",
            "value_type": "float",
            "item": "item:item__kv_user_blue_level_click_focus_cnt_7d",
            "user": "user:user_blue_level",
            "category": "ALL",
            "match_type": "hit",
            "need_discrete": False,
        },
        {
            "feature_type": "combo_feature",
            "feature_name": "weekday_level",
            "expression": ["feature:weekday", "user:memberlevel"],
            "need_prefix": False,
            "separator": "\u001d",
            "default_value": "",
            "stub_type": True,
        },
        {
            "feature_name": "user_hotel_distance",
            "feature_type": "custom_feature",
            "operator_name": "BucketOp",
            "operator_lib_file": "libbucket_op.so",
            "expression": ["feature:tmp_hotel_distance"],
            "value_type": "double",
            "default_value": "0",
            "value_dimension": 1,
            "bucket_threshold": "0.001:1000:3000:5000:10000:1000000000",
            "bucket_bins": "3.5:5:4:3:2:1",
        },
        {
            "feature_name": "click_100_seq_filter_bool",
            "feature_type": "expr_feature",
            "expression": "click_100_seq__hotel_price> 0",
            "value_type": "int32",
            "variables": ["feature:click_100_seq__hotel_price"],
            "default_value": "",
            "separator": ";",
            "stub_type": True,
        },
        {
            "feature_name": "click_100_seq__hotel_price_filter",
            "feature_type": "bool_mask_feature",
            "value_type": "double",
            "expression": [
                "feature:click_100_seq__hotel_price",
                "feature:click_100_seq_filter_bool",
            ],
            "separator": ";",
            "stub_type": True,
        },
        {
            "feature_type": "overlap_feature",
            "feature_name": "hit_impr_hotel",
            "query": "item:hotel_id",
            "title": "feature:expr_100_seq__hotel_id_filter",
            "method": "is_contain",
            "separator": ";",
            "normalizer": "",
        },
        {
            "feature_type": "kv_dot_product",
            "feature_name": "uh_tag_hit",
            "query": "user:user_tag2_dict",
            "value_dimension": 1,
            "document": "item:hotel_tag2_dict",
            "separator": ",",
            "default_value": "0",
        },
        {
            "sequence_name": "click_100_seq",
            "sequence_column": "click_100_seq",
            "sequence_length": 100,
            "sequence_delim": ";",
            "attribute_delim": "#",
            "sequence_table": "item",
            "sequence_pk": "user:click_100_seq",
            "features": [
                {
                    "feature_name": "hotel_id",
                    "feature_type": "id_feature",
                    "value_type": "String",
                    "expression": "item:hotel_id",
                    "default_value": "-1024",
                    "combiner": "mean",
                    "need_prefix": True,
                    "value_dimension": 1,
                    "group": "click_100_seq_feature",
                },
                {
                    "feature_name": "hotel_price",
                    "feature_type": "raw_feature",
                    "value_type": "Double",
                    "expression": "user:hotel_price",
                    "default_value": "-1024",
                    "combiner": "mean",
                    "need_prefix": True,
                    "value_dimension": 1,
                    "group": "click_100_seq_feature",
                    "stub_type": True,
                },
                {
                    "feature_name": "ts",
                    "feature_type": "raw_feature",
                    "value_type": "Double",
                    "expression": "user:ts",
                    "default_value": "-1024",
                    "combiner": "mean",
                    "need_prefix": False,
                    "value_dimension": 1,
                    "group": "click_100_seq_feature",
                },
            ],
        },
    ],
    "reserves": ["is_click_cover", "is_click_video"],
}

PYFG_FEATURE_CONFIG = """feature_configs {
  id_feature {
    feature_name: "user_id"
    expression: "user:user_id"
    embedding_dim: 4
    hash_bucket_size: 1000
    pooling: "mean"
    default_value: "-1024"
  }
}
feature_configs {
  id_feature {
    feature_name: "item_id"
    expression: "item:item_id"
    embedding_dim: 24
    hash_bucket_size: 1500000
    pooling: "mean"
    default_value: "-1024"
  }
}
feature_configs {
  raw_feature {
    feature_name: "hotel_price"
    expression: "context:hotel_price"
    default_value: "-1999999"
  }
}
feature_configs {
  lookup_feature {
    feature_name: "item__kv_user_blue_level_exposure_cnt_7d"
    map: "item:item__kv_user_blue_level_exposure_cnt_7d"
    key: "user:user_blue_level"
    embedding_dim: 4
    boundaries: 1e-08
    boundaries: 47.0
    boundaries: 285.0
    boundaries: 672.0
    boundaries: 1186.0
    boundaries: 1853.0
    boundaries: 2716.0
    boundaries: 3861.0
    boundaries: 5459.0
    boundaries: 7817.0
    boundaries: 11722.0
    boundaries: 19513.0
    boundaries: 43334.0
    combiner: "mean"
    default_value: "0"
    need_discrete: false
  }
}
feature_configs {
  match_feature {
    feature_name: "item__kv_user_blue_level_click_focus_cnt_7d"
    nested_map: "user:user_blue_level"
    pkey: "ALL"
    skey: "item:item__kv_user_blue_level_click_focus_cnt_7d"
    embedding_dim: 4
    boundaries: 1e-08
    boundaries: 1.0
    boundaries: 2.0
    boundaries: 5.0
    boundaries: 8.0
    boundaries: 13.0
    boundaries: 19.0
    boundaries: 28.0
    boundaries: 42.0
    boundaries: 67.0
    boundaries: 123.0
    boundaries: 298.0
  }
}
feature_configs {
  combo_feature {
    feature_name: "weekday_level"
    expression: "feature:weekday"
    expression: "user:memberlevel"
    default_value: ""
    separator: ""
    stub_type: true
  }
}
feature_configs {
  custom_feature {
    feature_name: "user_hotel_distance"
    operator_name: "BucketOp"
    operator_lib_file: "libbucket_op.so"
    operator_params {
      fields {
        key: "bucket_bins"
        value {
          string_value: "3.5:5:4:3:2:1"
        }
      }
      fields {
        key: "bucket_threshold"
        value {
          string_value: "0.001:1000:3000:5000:10000:1000000000"
        }
      }
      fields {
        key: "value_type"
        value {
          string_value: "double"
        }
      }
    }
    expression: "feature:tmp_hotel_distance"
    default_value: "0"
    value_dim: 1
  }
}
feature_configs {
  expr_feature {
    feature_name: "click_100_seq_filter_bool"
    expression: "click_100_seq__hotel_price> 0"
    variables: "feature:click_100_seq__hotel_price"
    separator: ";"
    default_value: ""
    stub_type: true
  }
}
feature_configs {
  bool_mask_feature {
    feature_name: "click_100_seq__hotel_price_filter"
    expression: "feature:click_100_seq__hotel_price"
    expression: "feature:click_100_seq_filter_bool"
    separator: ";"
    stub_type: true
  }
}
feature_configs {
  overlap_feature {
    feature_name: "hit_impr_hotel"
    query: "item:hotel_id"
    title: "feature:expr_100_seq__hotel_id_filter"
    method: "is_contain"
    normalizer: ""
    separator: ";"
  }
}
feature_configs {
  kv_dot_product {
    feature_name: "uh_tag_hit"
    query: "user:user_tag2_dict"
    document: "item:hotel_tag2_dict"
    separator: ","
    default_value: "0"
  }
}
feature_configs {
  sequence_feature {
    sequence_name: "click_100_seq"
    sequence_length: 100
    sequence_delim: ";"
    sequence_pk: "user:click_100_seq"
    features {
      id_feature {
        feature_name: "hotel_id"
        expression: "item:hotel_id"
        value_dim: 1
        pooling: "mean"
        default_value: "-1024"
      }
    }
    features {
      raw_feature {
        feature_name: "hotel_price"
        expression: "user:hotel_price"
        value_dim: 1
        default_value: "-1024"
        stub_type: true
      }
    }
    features {
      raw_feature {
        feature_name: "ts"
        expression: "user:ts"
        embedding_dim: 4
        boundaries: 1e-08
        boundaries: 3.0
        boundaries: 25.0
        boundaries: 70.0
        value_dim: 1
        default_value: "-1024"
      }
    }
  }
}
"""


class ConvertConfigTest(unittest.TestCase):
    def setUp(self):
        self.success = False
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_convert_", dir="./tmp")
        self.fg_path = os.path.join(self.test_dir, "fg.json")
        self.easyrec_path = os.path.join(self.test_dir, "easyrec.config")
        self.tzrec_path = os.path.join(self.test_dir, "tzrec.config")
        with open(self.easyrec_path, "w", encoding="utf-8") as f:
            f.write(EASYREC_CONFIG)
        f = open(self.fg_path, "w", encoding="utf-8")
        json.dump(FG_JSON, f)
        f.close()
        self.pyfg_path = os.path.join(self.test_dir, "pyfg.json")
        f = open(self.pyfg_path, "w", encoding="utf-8")
        json.dump(PYFG_JSON, f)
        f.close()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_create_train_config(self):
        convert = ConvertConfig(self.easyrec_path, self.tzrec_path, self.fg_path)
        config = tzrec_pipeline_pb2.EasyRecConfig()
        config = convert._create_train_config(config)
        config_text = text_format.MessageToString(config, as_utf8=True)
        self.assertEqual(config_text, TRAIN_CONFIG)

    def test_create_data_config(self):
        convert = ConvertConfig(self.easyrec_path, self.tzrec_path, self.fg_path)
        config = tzrec_pipeline_pb2.EasyRecConfig()
        config = convert._create_data_config(config)
        config_text = text_format.MessageToString(config, as_utf8=True)
        self.assertEqual(config_text, DATA_CONFIG)

    def test_create_feature_config(self):
        convert = ConvertConfig(self.easyrec_path, self.tzrec_path, self.fg_path)
        config = tzrec_pipeline_pb2.EasyRecConfig()
        config = convert._create_feature_config(config)
        config_text = text_format.MessageToString(config, as_utf8=True)
        self.assertEqual(config_text, FEATURE_CONFIG)

    def test_create_feature_config_no_fg(self):
        convert = ConvertConfig(self.easyrec_path, self.tzrec_path)
        config = tzrec_pipeline_pb2.EasyRecConfig()
        config = convert._create_feature_config_no_fg(config)
        config_text = text_format.MessageToString(config, as_utf8=True)
        self.assertEqual(config_text, FEATURE_CONFIG_NO_FG)

    def test_create_feature_config_use_pyfg(self):
        convert = ConvertConfig(
            self.easyrec_path, self.tzrec_path, self.pyfg_path, use_old_fg=False
        )
        config = tzrec_pipeline_pb2.EasyRecConfig()
        config = convert._create_feature_config_use_pyfg(config)
        config_text = text_format.MessageToString(config, as_utf8=True)
        self.assertEqual(config_text, PYFG_FEATURE_CONFIG)

    def test_create_model_config(self):
        convert = ConvertConfig(self.easyrec_path, self.tzrec_path, self.fg_path)
        config = tzrec_pipeline_pb2.EasyRecConfig()
        config = convert._create_model_config(config)
        config_text = text_format.MessageToString(config, as_utf8=True)
        self.assertEqual(config_text, MODEL_CONFIG)

    def test_create_model_config_simplemultitask(self):
        self.easyrec_simplemultitask_path = os.path.join(
            self.test_dir, "easyrec_simplemultitask.config"
        )
        self.tzrec_simplemultitask_path = os.path.join(
            self.test_dir, "tzrec_simplemultitask.config"
        )
        with open(self.easyrec_simplemultitask_path, "w", encoding="utf-8") as f:
            f.write(EASYREC_CONFIG_SIMPLEMULTITASK)
        convert_simplemultitask = ConvertConfig(
            self.easyrec_simplemultitask_path,
            self.tzrec_simplemultitask_path,
            self.fg_path,
        )
        config = tzrec_pipeline_pb2.EasyRecConfig()
        config = convert_simplemultitask._create_model_config(config)
        config_text = text_format.MessageToString(config, as_utf8=True)
        self.assertEqual(config_text, MODEL_CONFIG_SIMPLEMULTITASK)

    def test_create_model_config_mmoe(self):
        self.easyrec_mmoe_path = os.path.join(self.test_dir, "easyrec_mmoe.config")
        self.tzrec_mmoe_path = os.path.join(self.test_dir, "tzrec_mmoe.config")
        with open(self.easyrec_mmoe_path, "w", encoding="utf-8") as f:
            f.write(EASYREC_CONFIG_MMOE)
        convert_mmoe = ConvertConfig(
            self.easyrec_mmoe_path, self.tzrec_mmoe_path, self.fg_path
        )
        config = tzrec_pipeline_pb2.EasyRecConfig()
        config = convert_mmoe._create_model_config(config)
        config_text = text_format.MessageToString(config, as_utf8=True)
        self.assertEqual(config_text, MODEL_CONFIG_MMOE)

    def test_create_model_config_ple(self):
        self.easyrec_ple_path = os.path.join(self.test_dir, "easyrec_ple.config")
        self.tzrec_ple_path = os.path.join(self.test_dir, "tzrec_ple.config")
        with open(self.easyrec_ple_path, "w", encoding="utf-8") as f:
            f.write(EASYREC_CONFIG_PLE)
        convert_ple = ConvertConfig(
            self.easyrec_ple_path, self.tzrec_ple_path, self.fg_path
        )
        config = tzrec_pipeline_pb2.EasyRecConfig()
        config = convert_ple._create_model_config(config)
        config_text = text_format.MessageToString(config, as_utf8=True)
        self.assertEqual(config_text, MODEL_CONFIG_PLE)

    def test_create_model_config_deepfm(self):
        self.easyrec_deepfm_path = os.path.join(self.test_dir, "easyrec_deepfm.config")
        self.tzrec_deepfm_path = os.path.join(self.test_dir, "tzrec_deepfm.config")
        with open(self.easyrec_deepfm_path, "w", encoding="utf-8") as f:
            f.write(EASYREC_CONFIG_DEEPFM)
        convert_deepfm = ConvertConfig(
            self.easyrec_deepfm_path, self.tzrec_deepfm_path, self.fg_path
        )
        config = tzrec_pipeline_pb2.EasyRecConfig()
        config = convert_deepfm._create_model_config(config)
        config_text = text_format.MessageToString(config, as_utf8=True)
        self.assertEqual(config_text, MODEL_CONFIG_DEEPFM)

    def test_create_model_config_multitower(self):
        self.easyrec_multitower_path = os.path.join(
            self.test_dir, "easyrec_multitower.config"
        )
        self.tzrec_multitower_path = os.path.join(
            self.test_dir, "tzrec_multitower.config"
        )
        with open(self.easyrec_multitower_path, "w", encoding="utf-8") as f:
            f.write(EASYREC_CONFIG_MULTITOWER)
        convert_multitower = ConvertConfig(
            self.easyrec_multitower_path, self.tzrec_multitower_path, self.fg_path
        )
        config = tzrec_pipeline_pb2.EasyRecConfig()
        config = convert_multitower._create_model_config(config)
        config_text = text_format.MessageToString(config, as_utf8=True)
        self.assertEqual(config_text, MODEL_CONFIG_MULTITOWER)

    def test_create_model_config_dssm(self):
        self.easyrec_dssm_path = os.path.join(self.test_dir, "easyrec_dssm.config")
        self.tzrec_dssm_path = os.path.join(self.test_dir, "tzrec_dssm.config")
        with open(self.easyrec_dssm_path, "w", encoding="utf-8") as f:
            f.write(EASYREC_CONFIG_DSSM)
        convert_dssm = ConvertConfig(
            self.easyrec_dssm_path, self.tzrec_dssm_path, self.fg_path
        )
        config = tzrec_pipeline_pb2.EasyRecConfig()
        config = convert_dssm._create_model_config(config)
        config_text = text_format.MessageToString(config, as_utf8=True)
        self.assertEqual(config_text, MODEL_CONFIG_DSSM)

    def _run_env(self):
        return {"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"}


if __name__ == "__main__":
    unittest.main()
