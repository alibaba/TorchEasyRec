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

import unittest

from google.protobuf import text_format

from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.tools.add_feature_info_to_config import AddFeatureInfoToConfig

FEATURE_CONFIG = """
feature_configs {
    id_feature {
        feature_name: "user_id"
        expression: "user:user_id"
        embedding_dim: 4
    }
}
feature_configs {
    id_feature {
        feature_name: "item_id"
        expression: "item:item_id"
        embedding_dim: 4
    }
}
feature_configs {
    id_feature {
        feature_name: "author"
        expression: "item:author"
        embedding_dim: 4
	}
}
feature_configs {
    id_feature {
        feature_name: "day_h"
        expression: "user:day_h"
        embedding_dim: 4
    }
}
feature_configs {
    raw_feature {
        feature_name: "item_cnt"
        expression: "item:item_cnt"
        embedding_dim: 4
    }
}
feature_configs {
    sequence_feature {
        sequence_name: "click_50_seq",
        sequence_length: 50
        sequence_delim: ";"
		features {
			id_feature {
				feature_name: "item_id"
				expression: "item:item_id"
			}
		}
		features {
			id_feature {
				feature_name: "author"
				expression: "item:author"
			}
		}
		features {
			raw_feature {
				feature_name: "ts"
				expression: "user:ts"
			}
		}
	}
}
feature_configs {
    sequence_feature {
        sequence_name: "like_50_seq",
        sequence_length: 50
        sequence_delim: ";"
		features {
			id_feature {
				feature_name: "item_id"
				expression: "item:item_id"
			}
		}
		features {
			id_feature {
				feature_name: "author"
				expression: "item:author"
			}
		}
		features {
			raw_feature {
				feature_name: "ts"
				expression: "user:ts"
				embedding_dim: 2
			}
		}
	}
}
"""

DROPED_FEATURE_CONFIG = """feature_configs {
  id_feature {
    feature_name: "user_id"
    expression: "user:user_id"
    embedding_dim: 4
  }
}
feature_configs {
  id_feature {
    feature_name: "item_id"
    expression: "item:item_id"
    embedding_dim: 4
  }
}
feature_configs {
  id_feature {
    feature_name: "day_h"
    expression: "user:day_h"
    embedding_dim: 4
  }
}
feature_configs {
  raw_feature {
    feature_name: "item_cnt"
    expression: "item:item_cnt"
    embedding_dim: 4
  }
}
feature_configs {
  sequence_feature {
    sequence_name: "click_50_seq"
    sequence_length: 50
    sequence_delim: ";"
    features {
      id_feature {
        feature_name: "item_id"
        expression: "item:item_id"
      }
    }
    features {
      raw_feature {
        feature_name: "ts"
        expression: "user:ts"
      }
    }
  }
}
feature_configs {
  sequence_feature {
    sequence_name: "like_50_seq"
    sequence_length: 50
    sequence_delim: ";"
    features {
      raw_feature {
        feature_name: "ts"
        expression: "user:ts"
        embedding_dim: 2
      }
    }
  }
}
"""

UPDATE_FEATURE_CONFIG = """feature_configs {
  id_feature {
    feature_name: "user_id"
    expression: "user:user_id"
    embedding_dim: 36
    hash_bucket_size: 8278840
  }
}
feature_configs {
  id_feature {
    feature_name: "item_id"
    expression: "item:item_id"
    embedding_dim: 28
    hash_bucket_size: 3848650
  }
}
feature_configs {
  id_feature {
    feature_name: "day_h"
    expression: "user:day_h"
    embedding_dim: 8
    hash_bucket_size: 240
  }
}
feature_configs {
  raw_feature {
    feature_name: "item_cnt"
    expression: "item:item_cnt"
    embedding_dim: 8
    boundaries: -1023.9
    boundaries: 4.01
    boundaries: 461.01
    boundaries: 1057.01
  }
}
feature_configs {
  sequence_feature {
    sequence_name: "click_50_seq"
    sequence_length: 50
    sequence_delim: ";"
    features {
      id_feature {
        feature_name: "item_id"
        expression: "item:item_id"
        embedding_dim: 28
        hash_bucket_size: 3848650
      }
    }
    features {
      raw_feature {
        feature_name: "ts"
        expression: "user:ts"
        embedding_dim: 8
        boundaries: 10
        boundaries: 20
        boundaries: 30
        boundaries: 40
      }
    }
  }
}
feature_configs {
  sequence_feature {
    sequence_name: "like_50_seq"
    sequence_length: 50
    sequence_delim: ";"
    features {
      raw_feature {
        feature_name: "ts"
        expression: "user:ts"
        embedding_dim: 8
        boundaries: 10
        boundaries: 20
        boundaries: 30
        boundaries: 40
      }
    }
  }
}
"""

BEFORE_MODEL_CONFIG = """
model_config {
    feature_groups {
        group_name: "all"
		feature_names: "user_id"
		feature_names: "item_id"
		feature_names: "author"
		feature_names: "day_h"
		feature_names: "item_cnt"
        group_type: DEEP
        sequence_groups {
            group_name: "click_50_seq"
			feature_names: "item_id"
			feature_names: "author"
			feature_names: "click_50_seq__item_id"
			feature_names: "click_50_seq__author"
			feature_names: "click_50_seq__ts"
        }
        sequence_groups {
            group_name: "like_50_seq"
			feature_names: "author"
			feature_names: "like_50_seq__author"
			feature_names: "like_50_seq__ts"
        }
        sequence_encoders {
            din_encoder {
				input: "click_50_seq"
                attn_mlp {
                    hidden_units: [32]
                }
            }
        }
        sequence_encoders {
            din_encoder {
				input: "like_50_seq"
                attn_mlp {
                    hidden_units: [32]
                }
            }
        }
    }
}
"""

AFTER_MODEL_CONFIG = """model_config {
  feature_groups {
    group_name: "all"
    feature_names: "user_id"
    feature_names: "item_id"
    feature_names: "day_h"
    feature_names: "item_cnt"
    group_type: DEEP
    sequence_groups {
      group_name: "click_50_seq"
      feature_names: "item_id"
      feature_names: "click_50_seq__item_id"
      feature_names: "click_50_seq__ts"
    }
    sequence_groups {
      group_name: "like_50_seq"
      feature_names: "like_50_seq__ts"
    }
    sequence_encoders {
      din_encoder {
        input: "click_50_seq"
        attn_mlp {
          hidden_units: 32
        }
      }
    }
    sequence_encoders {
      din_encoder {
        input: "like_50_seq"
        attn_mlp {
          hidden_units: 32
        }
      }
    }
  }
}
"""


class AddFeatureInfoToConfigTest(unittest.TestCase):
    def setUp(self):
        self.add_feature_info = AddFeatureInfoToConfig("", "", "", "", "")

    def test_drop_feature_config(self) -> None:
        config = EasyRecConfig()
        drop_feature_name = [
            "author",
            "click_50_seq__author",
            "like_50_seq__item_id",
            "like_50_seq__author",
        ]
        text_format.Merge(FEATURE_CONFIG, config)
        self.add_feature_info._drop_feature_config(config, drop_feature_name)
        self.assertEqual(str(config), DROPED_FEATURE_CONFIG)

    def test_update_feature_config(self) -> None:
        feature_info_map = {
            "user_id": {"hash_bucket_size": 8278840, "embedding_dim": 36},
            "item_id": {"hash_bucket_size": 3848650, "embedding_dim": 28},
            "day_h": {"hash_bucket_size": 240, "embedding_dim": 8},
            "item_cnt": {
                "boundary": [-1023.9, 4.01, 461.01, 1057.01],
                "embedding_dim": 8,
            },
            "click_50_seq__item_id": {"hash_bucket_size": 3848650, "embedding_dim": 28},
            "click_50_seq__ts": {"boundary": [10, 20, 30, 40], "embedding_dim": 8},
            "like_50_seq__ts": {"boundary": [10, 20, 30, 40], "embedding_dim": 8},
        }
        config = EasyRecConfig()
        text_format.Merge(DROPED_FEATURE_CONFIG, config)
        general_feature = self.add_feature_info._update_feature_config(
            config, feature_info_map
        )
        expend_general_feature = ["user_id", "item_id", "day_h", "item_cnt"]
        self.assertEqual(str(config), UPDATE_FEATURE_CONFIG)
        self.assertEqual(general_feature, expend_general_feature)

    def test_update_feature_group(self) -> None:
        config = EasyRecConfig()
        drop_feature_name = [
            "author",
            "click_50_seq__author",
            "like_50_seq__item_id",
            "like_50_seq__author",
        ]
        text_format.Merge(BEFORE_MODEL_CONFIG, config)
        self.add_feature_info._update_feature_group(config, drop_feature_name)
        self.assertEqual(str(config), AFTER_MODEL_CONFIG)


if __name__ == "__main__":
    unittest.main()
