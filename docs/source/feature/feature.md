# 特征

TorchEasyRec多种类型的特征，包括IdFeature、RawFeature、ComboFeature、LookupFeature、MatchFeature、ExprFeature、OverlapFeature、TokenizeFeature、SequenceFeature。

**共用配置**

- **feature_name**: 特征名/特征输出名

- **embedding_dim**: 特征嵌入维度，计算方法可以参考: `embedding_dim=8+x^{0.25}`，`embedding_dim` 需要为4的倍数, 其中，x 为不同特征取值的个数

- **embedding_name**: 特征嵌入名，默认不需要设置，如需两个特征共享嵌入(embedding)参数，可将这两个特征的嵌入名设置成相同

- **pooling**: 多值特征嵌入池化方式，默认为`sum`，可以选`sum`/`mean`

- **init_fn**: 特征嵌入初始化方式，默认不需要设置，如需自定义，可以设置任意的torch内置初始化函数，如`nn.init.uniform_,a=-0.01,b=0.01`

- **default_value**: 特征默认值。如果默认值为""，则没有默认值，后续模型中对于空特征的嵌入为零向量。注意: 该默认值为`bucketize`前的默认值。`bucketize`的配置包括`num_buckets`/`hash_bucket_size`/`vocab_list`/`vocab_dict`/`vocab_file`/`boundaries`

- **separator**: FG在输入为string类型时的多值分隔符，默认为`\x1d`。更建议用数组（ARRAY）类型来表示多值，训练和推理性能更好

- **fg_encoded_default_value**: FG编码后的数据的默认值，当fg_mode=FG_NONE并且不是用pai-fg编码数据时，可以设置该参数填充空值

- **trainable**: Embedding Variable是否可训练，默认为true

- **stub_type**: 是否只作为FG的中间结果，不作为FG的输出特征，默认为false。注意: 不能在fg_mode=FG_NORMAL模式下使用。

- **data_type**: 训练的EmbeddingTable的数据类型，支持FP32和FP16，默认为FP32。

## IdFeature: 类别型特征

类别型特征，例如手机品牌、item_id、user_id、年龄段、星座等，一般在表里面存储的类型一般是string、bigint、array\<string>或array\<bigint>。可支持多值Id特征

```
feature_configs {
    id_feature {
        feature_name: "uid"
        # fg_mode=FG_NONE 数据已经被FG编号好的情况下，expression可以不写
        expression: "user:uid"
        embedding_dim: 32
        hash_bucket_size: 100000
    }
}
feature_configs {
    id_feature {
        feature_name: "month"
        expression: "user:month"
        embedding_dim: 32
        num_buckets: 12
        default_value: "0"
    }
}
feature_configs {
    id_feature {
        feature_name: "cate"
        expression: "item:cate"
        embedding_dim: 32
        vocab_list: ["1", "2", "3", "4", "5", "6", "7"]
    }
}
feature_configs {
    id_feature {
        feature_name: "cate"
        expression: "item:cate"
        embedding_dim: 32
        vocab_dict: [{key:"a" value:2}, {key:"b" value:3}, {key:"c" value:2}]
    }
feature_configs {
    id_feature {
        feature_name: "cate"
        expression: "item:cate"
        embedding_dim: 32
        zch: {
            zch_size: 1000000
            eviction_interval: 2
            lfu {}
        }
    }
}
```

- **expression**: 特征FG所依赖的字段来源，由两部分组成`input_side`:`input_name`

  - `input_side`一共支持五种 \[`user`, `item`, `context`, `feature`, `const`\]
    - `user`: 用户侧特征输入，线上推理时从请求中传入
    - `item`: 物品侧特征输入，线上推理时会从实时缓存在内存中的特征表里获取
    - `context`: 由上下文产生物品侧特征输入，线上推理时从请求中传入，如`recall_name`等
    - `feature`: 来自其他特征FG的输出，如下述`lookup_age_feat`的输入`age_binning`来自于RawFeature `age`的分箱结果
    - `const`: 输入为常量
  - `input_name`为来源字段的实际名称

- **hash_bucket_size**: hash bucket的大小。为减少hash冲突，建议设置
  `hash_bucket_size  = number_ids*ratio, ratio in [5,10]`

  - 如果需要跟tensorflow的hash保持一致，可以设置环境变量 USE_FARM_HASH_TO_BUCKETIZE=true

- **num_buckets**: buckets数量, 仅仅当输入是integer类型时，可以使用num_buckets

- **vocab_list**: 指定词表，适合取值比较少可以枚举的特征，如星期，月份，星座等

  - 未设置**default_bucketize_value**时: default_bucketize_value=1, **编号需会从2开始**，编码0预留给默认值(default_value)，编码1预留给超出词表的词
    - 注意：如果default_value本身在vocab_list中，那么default_value的编码不为0，将会是vocab_list中的编码
  - 设置**default_bucketize_value**时: vocab_list 用户完全自主控制

- **vocab_dict**: 指定字典形式词表，适合多个词需要编码到同一个编号情况

  - 未设置**default_bucketize_value**时: default_bucketize_value=1, **编号需要从2开始**，编码0预留给默认值(default_value)，编码1预留给超出词表的词
    - 注意：如果default_value本身在vocab_dict中，vocab_dict中的等于default_value的元素的索引值将会被更新为0
  - 设置**default_bucketize_value**时: vocab_dict 用户完全自主控制

- **vocab_file**: 指定词表或字典形式词表的文件路径，适合取值比较多兵可以枚举的特征，编码未预留，必须设置**default_bucketize_value**参数

  - 词表形式：一行一个词
  - 字典词表形式：一行一个词和编号，词和编号间用空格分隔

- **zch**: 零冲突hash，可设置Id的准入和驱逐策略，详见[文档](../zch.md)

- **weighted**: 是否为带权重的Id特征，输入形式为`k1:v1\x1dk2:v2`

- **value_dim**: 默认值是0，可以设置1，value_dim=0时支持多值ID输出

- **default_bucketize_value**: （可选）指定超出词表的词的编码。当配置了default_bucketize_value时，vocab_list和vocab_dict将不会预留编码给默认值和超出词表的词，用户可完全自主控制vocab_list或vocab_dict

- NOTE: hash_bucket_size, num_buckets, vocab_list, 只能指定其中之一，不能同时指定

## RawFeature: 数值型特征

```
feature_configs {
    raw_feature {
        feature_name: "ctr"
        expression: "item:ctr"
    }
}
```

- **expression**: 特征FG所依赖的字段来源，由两部分组成`input_side`:`input_name`，`input_side`可以取值为\[`user`, `item`, `context`, `feature`, `const`\]，`input_name`为来源字段的名称

- **normalizer**: 指定连续值特征的变换方式，支持4种，默认不变换

  - log10

  ```
  配置例子: method=log10,threshold=1e-10,default=-10
  计算公式: x = x > threshold ? log10(x) : default;
  ```

  - zscore

  ```
  配置例子: method=zscore,mean=0.0,standard_deviation=10.0
  计算公式: x = (x - mean) / standard_deviation
  ```

  - minmax

  ```
  配置例子: method=minmax,min=2.1,max=2.2
  计算公式: x = (x - min) / (max - min)
  ```

  - expression，表达式配置方式详见**ExprFeature**

  ```
  配置例子: method=expression,expr=sign(x)
  计算公式: 可以配置任意的函数或表达式，变量名固定为x，代表表达式的输入
  ```

连续值类特征可以先使用分箱组件+进行离散化，可以进行等频/等距/自动离散化，变成离散值。分箱组件使用方法见:
[机器学习组件](https://help.aliyun.com/zh/pai/user-guide/binning)

```
feature_configs {
    raw_feature {
        feature_name: "ctr"
        expression: "item:ctr"
        boundaries: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        embedding_dim: 8
    }
}
feature_configs {
    raw_feature {
        feature_name: "price"
        expression: "item:price"
        embedding_dim: 8
        mlp {}
    }
}
feature_configs {
    raw_feature {
        feature_name: "price"
        expression: "item:price"
        embedding_dim: 8
        autodis {
           num_channels: 3
           temperature: 0.1
           keep_prob: 0.8
        }
    }
}
```

- **boundaries**: 分箱/分桶的边界值，通过一个数组来设置。
- **mlp**: 由一层MLP变换特征到`embedding_dim`维度
- **autodis**: 由AutoDis模块变换特征到`embedding_dim`维度，详见[AutoDis文档](../autodis.md)

Embedding特征: 支持string类型如`"0.1|0.2|0.3|0.4"`；支持ARRAY<float>类型如`[0.1,0.2,0.3,0.4]`（建议，性能更好），配置方式如下

```
feature_configs: {
    raw_feature {
        feature_name: "pic_emb"
        expression: "item:pic_emb"
        feature_type: RawFeature
        separator: "|"
        value_dim: 4
    }
}
```

- **separator**: FG多值分隔符，默认为`\x1d`
- **value_dim**: 默认值为1， 指定Embedding特征的输入维度

## ComboFeature: 组合特征

对输入的离散值进行组合（即笛卡尔积）, 如age + cate:

```
feature_configs: {
    combo_feature {
        feature_name: "combo_age_cate"
        expression: ["user:age", "item:cate"]
        embedding_dim: 16
        hash_bucket_size: 1000
    }
}
```

- expression: 特征FG所依赖组合字段的来源，数量 >= 2
- 其余配置同IdFeature，NOTE: ComboFeature不包含`num_buckets`配置

## LookupFeature: 字典查询特征

`lookup_feature`依赖`map`和`key`两个字段从一组kv中匹配特征值。生成特征时，使用key在map字段所持有的kv对中进行匹配，获取最终的特征。

`map`是一个多值的kv map，支持string类型，如`"k1:v1\x1dk2:v2"`，其中`:`为kv分割符，`\x1d`为固定的多值分隔符；也支持MAP\<string,bigint>/MAP\<bigint,bigint>等MAP类型，如`{"k1":v1,"k2":v2}`（建议，性能更好）

`key`是一个多值的id，多值分隔符可以由**separator**指定，默认为`\x1d`。

```
feature_configs: {
    lookup_feature {
        feature_name: "user_cate_cnt"
        map: "user:kv_cate_cnt"
        key: "item:cate"
        embedding_dim: 16
        boundaries: [0, 1, 2, 3, 4]
    }
}
```

- **map**: 特征FG所依赖map字段的来源
- **key**: 特征FG所依赖key字段的来源
- **combiner**: 如果key为多值，可以设置combiner来对查找的值进行聚合，默认为`sum`，支持`sum`/`mean`/`min`/`max`
- **need_discrete**: 查到的值是否为离散值，默认为false
- **need_key**: 查到的值是否拼接key作为前缀，默认为false

如果Map的值为离散值 或 `need_key=true`，可设置:

- **value_dim**: 默认值是1，可以设置0，value_dim=0时支持多值ID输出
- 其余配置同IdFeature

如果Map的值为连续值，可设置:

- **value_dim**: 默认值是1，连续值输出维度
- 其余配置同RawFeature

## MatchFeature: 主从键字典查询特征

`match_feature`依赖`nested_map`和`pkey`和`skey`三个字段从kkv中匹配到特征值。`nested_map`是一个多值的kkv map，如`pk1^sk1:0.2,sk2:0.3,sk3:0.5|pk2^sk4:0.1`，`:`为内层kv分割符，`,`为内层多值分隔符，`^`为外层kv分割符，`|`为外层KV分隔符，分隔符不可以指定。生成特征时，使用`pkey`作为主键`skey`作为子健在`nested_map`字段所持有的kkv对中进行匹配，获取最终的特征。

```
feature_configs: {
    match_feature {
        feature_name: "user_cate_brand_cnt"
        nested_map: "user:kkv_cate_brand_cnt"
        pkey: "item:cate"
        skey: "item:brand"
        embedding_dim: 16
        boundaries: [0, 1, 2, 3, 4]
    }
}
```

- **nested_map**: 特征FG所依赖nested_map字段的来源
- **pkey**: 特征FG所依赖主键字段的来源。可以设置为ALL，将匹配所有pkey下面的指定的skey的值
- **skey**: 特征FG所依赖子键字段的来源。可以设置为ALL，将指定pkey下面所有skey的值
- **combiner**: 如果key为多值，可以设置combiner来对查找的值进行聚合，支持`sum`/`mean`/`min`/`max`
- **need_discrete**: 查到的值是否为离散值，默认为false
- **show_pkey**: 查到的值是否拼接pkey作为前缀，默认为false
- **show_skey**: 查到的值是否拼接skey作为前缀，默认为false

如果Map的值为离散值 或 `show_pkey=true` 或 `show_skey=true`，可设置:

- **value_dim**: 默认值是1，可以设置0，value_dim=0时支持多值ID输出
- 其余配置同IdFeature

如果Map的值为连续值，可设置:

- **value_dim**: 目前只支持value_dim=1
- 其余配置同RawFeature

## ExprFeature: 表达式特征

对数值型特征进行运算，如判断当前用户年龄是否>18，用户年龄是否符合物品年龄需求等。

```
feature_configs: {
    expr_feature {
        feature_name: "combo_age_cate"
        variables: ["user:u_age", "item:i_age"]
        expression: "u_age == i_age ? 1 : 0"
        embedding_dim: 8
        boundaries: [0.5]
    }
}
```

- **variables**: 特征FG所依赖表达式中的字段的来源

- **expression**: 表达式本身

- **value_dim**: 默认值是0，value_dim=0时支持多值ID输出

- **内置函数**: 详见[表达式文档](https://help.aliyun.com/zh/airec/what-is-pai-rec/user-guide/built-in-feature-operator?#1d09c2da3aajb)

  | 函数名      | 参数数量 | 解释                                                                    |
  | ----------- | -------- | ----------------------------------------------------------------------- |
  | rnd         | 0        | Generate a random number between 0 and 1                                |
  | sin         | 1        | sine function                                                           |
  | cos         | 1        | cosine function                                                         |
  | tan         | 1        | tangens function                                                        |
  | asin        | 1        | arcus sine function                                                     |
  | acos        | 1        | arcus cosine function                                                   |
  | atan        | 1        | arcus tangens function                                                  |
  | sinh        | 1        | hyperbolic sine function                                                |
  | cosh        | 1        | hyperbolic cosine                                                       |
  | tanh        | 1        | hyperbolic tangens function                                             |
  | asinh       | 1        | hyperbolic arcus sine function                                          |
  | acosh       | 1        | hyperbolic arcus tangens function                                       |
  | atanh       | 1        | hyperbolic arcur tangens function                                       |
  | log2        | 1        | logarithm to the base 2                                                 |
  | log10       | 1        | logarithm to the base 10                                                |
  | log         | 1        | logarithm to base e (2.71828...)                                        |
  | ln          | 1        | logarithm to base e (2.71828...)                                        |
  | exp         | 1        | e raised to the power of x                                              |
  | sqrt        | 1        | square root of a value                                                  |
  | sign        | 1        | sign function -1 if x\<0; 1 if x>0                                      |
  | abs         | 1        | absolute value                                                          |
  | rint        | 1        | round to nearest integer                                                |
  | floor       | 1        | 向下取整                                                                |
  | ceil        | 1        | 向上取整                                                                |
  | trunc       | 1        | 截断取整（直接去掉小数部分）                                            |
  | round       | 1        | 四舍五入，总是使用"远离零"的舍入方式（round half away from zero）       |
  | roundp      | 2        | 自定义精度取整函数, e.g. roundp(3.14159,2)=3.14                         |
  | sigmoid     | 1        | sigmoid function                                                        |
  | sphere_dist | 4        | sphere distance between two gps points, args(lng1, lat1, lng2, lat2)    |
  | haversine   | 4        | haversine distance between two gps points, args(lng1, lat1, lng2, lat2) |
  | sigmoid     | 1        | sigmoid function                                                        |
  | min         | var.     | min of all arguments                                                    |
  | max         | var.     | max of all arguments                                                    |
  | sum         | var.     | sum of all arguments                                                    |
  | avg         | var.     | mean value of all arguments                                             |

备注：上述内置函数支持批量计算和广播机制

- **内置向量函数**:

  | 函数名       | 参数数量 | 解释                                                  |
  | ------------ | -------- | ----------------------------------------------------- |
  | len          | 1        | the length of a vector                                |
  | l2_norm      | 1        | l2 normalize of a vector                              |
  | squared_norm | 1        | squared normalize of a vector                         |
  | dot          | 2        | dot product of two vectors                            |
  | euclid_dist  | 2        | euclidean distance between two vectors                |
  | std_dev      | 1        | standard deviation of a vector, divide n              |
  | pop_std_dev  | 1        | population standard deviation of a vector, divide n-1 |
  | variance     | 1        | sample variance of a vector, divide n                 |
  | pop_variance | 1        | population variance of a vector, divide n-1           |
  | reduce_min   | 1        | reduce min of a vector                                |
  | reduce_max   | 1        | reduce max of a vector                                |
  | reduce_sum   | 1        | reduce sum of a vector                                |
  | reduce_mean  | 1        | reduce mean of a vector                               |
  | reduce_prod  | 1        | reduce product of a vector                            |

备注：当表达式包含上述内置向量函数时，非向量函数参数的其他变量只能是单值类型(scalar)。

- **内置二元操作符**:

  | 操作符 | 描述                      | 优先级 |
  | ------ | ------------------------- | ------ |
  | \|\|   | logical or                | 1      |
  | &&     | logical and               | 2      |
  | \|     | bitwise or                | 3      |
  | &      | bitwise and               | 4      |
  | \<=    | less or equal             | 5      |
  | >=     | greater or equal          | 5      |
  | !=     | not equal                 | 5      |
  | ==     | equal                     | 5      |
  | >      | greater than              | 5      |
  | \<     | less than                 | 5      |
  | +      | addition                  | 6      |
  | -      | subtraction               | 6      |
  | \*     | multiplication            | 7      |
  | /      | division                  | 7      |
  | ^      | raise x to the power of y | 8      |

- **内置三元操作符**

  | 操作符 | 描述                  | 优先级           |
  | ------ | --------------------- | ---------------- |
  | ?:     | if then else operator | C++ style syntax |

- **内置常量**

  | 操作符 | 描述                 | 优先级                     |
  | ------ | -------------------- | -------------------------- |
  | \_pi   | The one and only pi. | 3.141592653589793238462643 |
  | \_e    | Euler's number.      | 2.718281828459045235360287 |

- 其余配置同RawFeature

## OverlapFeature: 重合匹配特征

`overlap_feature`会计算`query`和`title`两个字段字词重合比例，`query`和`title`中字词的分割符默认为`\x1d`，可以用多值分隔符由**separator**指定。

```
feature_configs: {
    overlap_feature {
        feature_name: "user_cate_cnt"
        query: "user:query"
        title: "item:title"
        method: "title_common_ratio"
        embedding_dim: 8
        boundaries: [0.2, 0.5, 0.8]
    }
}
```

- **query**: 特征FG所依赖query字段的来源

- **title**: 特征FG所依赖title字段的来源

- **method**: 重合计算方式，可选 query_common_ratio | title_common_ratio | is_contain | is_equal

  | 方式                | 描述                                                        | 备注                                                          |
  | ------------------- | ----------------------------------------------------------- | ------------------------------------------------------------- |
  | query_common_ratio  | 计算query与title间重复term数占query中term比例               | 取值为[0,1]                                                   |
  | title_common_ratio  | 计算query与title间重复term数占title中term比例               | 取值为[0,1]                                                   |
  | is_contain          | 计算query是否全部包含在title中，保持顺序                    | 0表示未包含，1表示包含                                        |
  | is_equal            | 计算query是否与title完全相同                                | 0表示不完全相同，1表示完全相同                                |
  | index_of            | 计算query作为整体第一次出现在title中的位置                  | 没有出现返回-1.0                                              |
  | proximity_min_cover | 计算query term在title中的邻近度                             | 取值为[0, length(title)], 0表示存在不能匹配的term             |
  | proximity_min_dist  | 计算query term在title中的邻近度 (minimum pairwise distance) | 取值为[0, length(title)+1], length(title)+1表示没有匹配的term |
  | proximity_max_dist  | 计算query term在title中的邻近度 (maximum pairwise distance) | 取值为[0, length(title)+1], length(title)+1表示没有匹配的term |
  | proximity_avg_dist  | 计算query term在title中的邻近度 (average pairwise distance) | 取值为[0, length(title)+1], length(title)+1表示没有匹配的term |

- 其余配置同RawFeature

## TokenizeFeature: 分词特征

`tokenize_feature` 对输入字符串分词，返回分词之后的词id。支持tokenize-cpp的分词词典文件。

```
feature_configs: {
    tokenize_feature {
        feature_name: "title_token"
        expression: "item:title"
        vocab_file: "tokenizer.json"
        embedding_dim: 8
        text_normalizer: {
            norm_options: [TEXT_LOWER2UPPER, TEXT_SBC2DBC, TEXT_CHT2CHS, TEXT_FILTER]
        }
    }
}
```

- **expression**: 特征FG所依赖分词字段的来源

- **vocab_file**: 分词字典，完全兼容 https://github.com/mlc-ai/tokenizers-cpp 库的分词文件

- **tokenizer_type**: 分词类型，支持bpe、sentencepiece，默认为bpe

- **text_normalizer**: 可选，是否对文本进行归一化

  - **stop_char_file**: 停用词表路径，默认为系统内置，详见[stop_char](https://tzrec.oss-cn-beijing.aliyuncs.com/third_party/stop_char)
  - **norm_options**: 归一化选项，默认为TEXT_LOWER2UPPER, TEXT_SBC2DBC, TEXT_CHT2CHS, TEXT_FILTER

  | 方式              | 描述                   |
  | ----------------- | ---------------------- |
  | TEXT_LOWER2UPPER  | 小写转换成大写         |
  | TEXT_UPPER2LOWER  | 大写转换成小写         |
  | TEXT_SBC2DBC      | 全角到半角             |
  | TEXT_CHT2CHS      | 繁体到简体             |
  | TEXT_FILTER       | 去除特殊符号           |
  | TEXT_SPLITCHRS    | 中文拆成单字(空格分隔) |
  | TEXT_REMOVE_SPACE | 去除空格               |

## KvDotProduct: KeyValue点积特征

计算两个key-value索引的向量的点积，或两个集合的交集的大小。

```
feature_configs: {
    kv_dot_product {
        feature_name: "query_doc_sim"
        query: "user:query"
        document: "item:title"
        separator: "|"
        default_value: "0"
    }
}
```

- **query**: 描述该feature所依赖的query字段来源

- **document**: 描述该feature所依赖的document字段来源

示例

| query              | document           | output |
| ------------------ | ------------------ | ------ |
| "a:0.5\|b:0.5"     | "d:0.5\|b:0.5"     | 0.25   |
| ["a:0.5", "b:0.5"] | ["d:0.5", "b:0.5"] | 0.25   |
| {"a":0.5, "b":0.5} | {"d":0.5, "b":0.5} | 0.25   |
| ["a:0.5", "b:0.5"] | {"d":0.5, "b":0.5} | 0.25   |
| ["a", "b", "c"]    | ["a", "b", "d"]    | 2.0    |
| ["a", "b", "c"]    | "a\|b\|d"          | 2.0    |
| ["a", "b", "c"]    | {"a":0.5, "b":0.5} | 1.0    |

- **kv_delimiter**: 可选项，指定输入特征中的kv对之间的分隔符，默认为 ":"，只能是单个符号

- 其余配置同RawFeature

## BoolMaskFeature

通过布尔值过滤元素，类似tf.boolean_mask(tensor, mask).

```
feature_configs: {
    bool_mask_feature {
        feature_name: "query_doc_sim"
        expression: ["user:click_items", "item:is_valid"]
        separator: ","
    }
}
```

- **expression**:该feature所依赖的字段来源,第一个字段表示字段的取值, 第二个字段表示对第一个字段的取值进行Mask
- 其余配置同IdFeature

示例

| 输入            | mask                    | 未进行bucketize的输出 |
| --------------- | ----------------------- | --------------------- |
| "123,456,90,80" | "true,false,true,false" | ["123", "90"]         |
| "123,456,90,80" | [1, 0, 1, 0]            | ["123", "90"]         |
| [1, 2, 3, 4]    | [1, 0, 1, 0]            | [1, 3]                |
| [1, 2, 3, 4]    | "true,false,true,false" | [1, 3]                |

## CustomFeature: 自定义特征

自定义特征，自定义方式参考[自定义算子文档](https://help.aliyun.com/zh/airec/what-is-pai-rec/user-guide/custom-feature-operator)

```
feature_configs: {
    custom_feature {
        feature_name: "edit_distance"
        operator_name: "EditDistance"
        operator_lib_file: "pyfg/lib/libedit_distance.so"
        expression: ["user:query", "item:title"]
        operator_params {
            fields {
                key: "encoding"
                value {
                    string_value: "utf-8"
                }
            }
        }
    }
}
```

- operator_name: 特征算子注册的名字，建议与实现的类名保持一致

- operator_lib_file: 指定特征算子动态库文件的路径，必须以.so结尾。如果是`pyfg/lib/`开头的路径，则为pyfg官方自定义so

- expression: 特征FG所依赖组合字段的来源

- 其余配置如果是类别型特征同IdFeature，如果是数值型特征同RawFeature

| 算子名称     | 算子功能 | 算子动态库                   | 算子参数                                                                         |
| ------------ | -------- | ---------------------------- | -------------------------------------------------------------------------------- |
| EditDistance | 编辑距离 | pyfg/lib/libedit_distance.so | • encoding: 输入文本的编码，可选：utf-8, latin，默认值为latin                    |
| RegexReplace | 正则替换 | pyfg/lib/libregex_replace.so | • regex_patten: 正则表达式，匹配的文本片段将会被替换 <br>• replacement: 替换文本 |
|              |          |                              |                                                                                  |

## SequenceFeature：序列特征

序列特征分为**分组序列特征**和**普通序列特征**两种

### 分组序列特征

分组序列特征的子序列格式一般为`XX;XX;XX`，如用户点击的Item的序列特征为`item_id1;item_id2;item_id3`，其中`;`为序列分隔符，也支持ARRAY类型，详建下文。

分组序列特征支持使用物品或行为的属性构建一组子序列，如类目序列`cate1;cate2;cate1`、品牌序列`brand1;brand2;brand1`、行为时间序列`ts1;ts2;ts3`等。一条样本中，同一分组的子序列的长度需要保持相同。

分组序列特征在线上模型服务时比其他序列特征更加高效，只需传递`sequence_pk`，线上模型服务从物品特征内存Cache中关联出物品属性子特征的序列，无需从请求中传递。

分组序列特征支持前文中**所有子特征类型**。

```
# 分组序列特征
feature_configs: {
    sequence_feature {
        sequence_name: "click_seq"
        sequence_length: 50
        sequence_delim: ";"
        sequence_pk: "user:click_seq_pk"
        features {
            id_feature {
                feature_name: "item_id"
                expression: "item:iid"
                embedding_dim: 32
                hash_bucket_size: 100000
            }
        }
        features {
            id_feature {
                feature_name: "cate"
                expression: "item:cate"
                embedding_dim: 32
                hash_bucket_size: 1000
            }
        }
        features {
            raw_feature {
                feature_name: "ts"
                expression: "user:ts"
            }
        }
        features {
            custom_feature {
                feature_name: "seq_expr"
                operator_name: "SeqExpr"
                operator_lib_file: "pyfg/lib/libseq_expr.so"
                expression: ["user:ulng", "user:ulat", "item:ilng", "item:ilat"]
                operator_params {
                    key: "formula"
                    string_value: "spherical_distance"
                }
            }
        }
        features {
            lookup_feature {
                feature_name: "user_cate_cnt"
                map: "user:kv_cate_cnt"
                key: "item:cate"
                embedding_dim: 16
                boundaries: [0, 1, 2, 3, 4]
            }
        }
        features {
            lookup_feature {
                feature_name: "user_search_cate_cnt"
                map: "user:kv_search_cate_cnt"
                key: "user:searched_cate"
                sequence_fields: ["searched_cate"]
                embedding_dim: 16
                boundaries: [0, 1, 2, 3, 4]
            }
        }
    }
}
```

- **sequence_name**: 序列特征名
- **sequence_length**: 序列特征最大长度
- **sequence_delim**: 序列特征分隔符
- **sequence_pk**: 序列特征主键，一般为ItemId列表，主要用于线上模型服务，线上模型服务会使用该ItemId列表从物品特征内存Cache中关联出物品属性子特征的序列，无需从请求中传递。而行为属性相关子序列（如行为时间序列`ts1;ts2;ts3`）跟用户相关，则仍需从请求从传递。
- **features**: 序列特征子特征，配置同对应的子特征类型的配置
  - **feature_name**: 子特征特征名，完整的子特征名应拼接上`${sequence_name}__`前缀，以上述配置中`item_id`子特征为例，子特征名列名应为`click_seq__item_id`
  - **expression**: 特征FG所依赖子特征字段来源名，由两部分组成`input_side`:`input_name`。
    - 在线上模型服务中，如果子特征的`input_side`为`item`，子序列无需从请求中传递；如果子特征的`input_side`为`user`，子序列需要从请求中传递。
  - **sequence_fields**: (可选) 指定每个序列子特征的序列类型的输入字段名，序列类型的字段在输入样本数据中列名应拼接上`${sequence_name}__`前缀
    - 在不指定sequence_fields的情况下:
      - 对于只有一个输入字段的特征算子（如: IdFeature，RawFeature，TokenizeFeature等），`input_side != feature`的输入字段默认是序列类型
        - 以上述配置中`item_id`子特征为例，`item:iid`对应的输入样本数据中列名应为`click_seq__iid`
      - 对于输入字段大于一个的特征算子（如: LookupFeature，ComboFeature等），`input_side != item`的输入字段默认是序列类型
        - 以上述配置中`user_cate_cnt`子特征为例，`item:cate`对应的输入样本数据中列名应为`click_seq__cate`，`user:kv_cate_cnt`对应的输入样本数据中列名应为`kv_cate_cnt`
    - 在指定sequence_fields的情况下：只有指定的字段认为是序列类型
      - 以上述配置中`user_search_cate_cnt`子特征为例，`user:searched_cate`对应的输入样本数据中列名应为`click_seq__searched_cate`，`user:kv_search_cate_cnt`对应的输入样本数据中列名应为`kv_search_cate_cnt`
  - 其中当特征值为离散值时（如IdFeature，ComboFeature等），value_dim的默认值与非序列的版本不同
    - **value_dim**: 默认值是1，可以设置0，value_dim=0时支持多值ID输出
- 序列类型的输入支持的类型:
  - 支持string类型`item_id1;item_id2;item_id3`， 其中`;`为序列分隔符；
  - 支持array\<string>或array\<bigint>类型为`[item_id1,item_id2,item_id3]`（建议，性能更好）；
  - 支持多值序列array\<array\<string>>或array\<array\<bigint>>类型，多值序列需设置value_dim=0，通常情况下训练推理性能比单值序列差一些。
  - 支持array\<float>为`[price1,price2,price3]`或者array\<array\<float>>类型为`[[emb11,emb12],[emb21,emb22]]`（建议，性能更好）。

### 普通序列特征

```
# 普通特征
feature_configs: {
    sequence_id_feature {
        feature_name: "click_itemid_seq"
        sequence_length: 50
        sequence_delim: ";"
        expression: "user:click_iid_seq"
        embedding_dim: 32
        hash_bucket_size: 100000
    }
}
feature_configs: {
    sequence_raw_feature {
        feature_name: "click_price_seq"
        sequence_length: 50
        sequence_delim: ";"
        expression: "user:click_price_seq"
    }
}
feature_configs: {
    sequence_custom_feature {
        feature_name: "seq_expr_1"
        operator_name: "SeqExpr"
        operator_lib_file: "pyfg/lib/libseq_expr.so"
        expression: ["user:cur_time", "item:clk_time_seq"]
        operator_params {
            fields {
                key: "formula"
                value {
                    string_value: "cur_time-clk_time_seq"
                }
            }
        }
    }
}
feature_configs: {
    sequence_custom_feature {
        feature_name: "seq_expr_2"
        operator_name: "SeqExpr"
        operator_lib_file: "pyfg/lib/libseq_expr.so"
        expression: ["user:ulng", "user:ulat", "item:ilng", "item:ilat"]
        operator_params {
            fields {
                key: "formula"
                value {
                    string_value: "spherical_distance"
                }
            }
        }
    }
}
```

- 配置同非序列版本，特征类型带`sequence_`前缀
- 其中当特征值为离散值时（如IdFeature，ComboFeature等），value_dim的默认值与非序列的版本不同
  - **value_dim**: 默认值是1，可以设置0，value_dim=0时支持多值ID输出
