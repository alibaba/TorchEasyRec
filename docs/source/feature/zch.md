# 零冲突Hash Embedding

零冲突Hash (Zero Collision Hash, zch) 是特征Id化的一种方式，它相比设置`hash_bucket_size`的方式能减少hash冲突，相比设置`vocab_dict`和`vocab_list`的方式能更灵活动态地进行id的准入和驱逐。零冲突Hash常用于user id，item id，combo feature等超大id枚举数的特征配置中。

以id_feature的配置为例，零冲突Hash只需在id_feature新增一个zch的配置字段

```
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

- **zch_size**: 零冲突Hash的Bucket大小，Id数超过后会根据Id的驱逐策略进行淘汰

- **eviction_interval**: Id准入和驱逐策略执行的频率（训练步数间隔）

- **eviction_policy**: 驱逐策略，可选`lfu`，`lru`，`distance_lfu`，详见下文驱逐策略

- **threshold_filtering_func**: 准入策略lambda函数，默认为全部准入，详见下文准入策略

## 驱逐策略

### LFU_EvictionPolicy

驱逐最小出现次数的Id
id_score = access_cnt

```
lfu {}
```

### LRU_EvictionPolicy

驱逐最早出现的Id
id_score = 1 / pow((current_iter - last_access_iter), decay_exponent)

```
lru {
    decay_exponent: 1.0
}
```

### DistanceLFU_EvictionPolicy

综合出现次数和出现时间综合根据综合驱逐id_score较小的Id
id_score = access_cnt / pow((current_iter - last_access_iter), decay_exponent)

```
distance_lfu {
    decay_exponent: 1.0
}
```

## 准入策略

准入策略需设置一个lambda函数表达式，函数输入输出应符合如下格式

- 输入：一个1维的IntTensor表示最近`eviction_interval`个batch中每个id的出现次数
- 输出：一个1维的BoolTensor表示保留的id位置 和 一个float值表示id出现次数的阈值

函数可支持直接用torch的tensor库来撰写，样例如下：

```
zch: {
    zch_size: 1000000
    eviction_interval: 2
    lfu {}
    threshold_filtering_func: "lambda x: (x > 10, 10)"
}
```

函数也可以支持调用内置函数：`dynamic_threshold_filter`, `average_threshold_filter` 和 `probabilistic_threshold_filter`，样例如下：

```
zch: {
    zch_size: 1000000
    eviction_interval: 2
    lfu {}
    threshold_filtering_func: "lambda x: dynamic_threshold_filter(x, 1.0)"
}
```

相关内置函数的实现细节如下：

```python
@torch.no_grad()
def dynamic_threshold_filter(
    id_counts: torch.Tensor,
    threshold_skew_multiplier: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Threshold is total_count / num_ids * threshold_skew_multiplier. An id is
    added if its count is strictly greater than the threshold.
    """

    num_ids = id_counts.numel()
    total_count = id_counts.sum()

    BASE_THRESHOLD = 1 / num_ids
    threshold_mass = BASE_THRESHOLD * threshold_skew_multiplier

    threshold = threshold_mass * total_count
    threshold_mask = id_counts > threshold

    return threshold_mask, threshold


@torch.no_grad()
def average_threshold_filter(
    id_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Threshold is average of id_counts. An id is added if its count is strictly
    greater than the mean.
    """
    if id_counts.dtype != torch.float:
        id_counts = id_counts.float()
    threshold = id_counts.mean()
    threshold_mask = id_counts > threshold

    return threshold_mask, threshold


@torch.no_grad()
def probabilistic_threshold_filter(
    id_counts: torch.Tensor,
    per_id_probability: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Each id has probability per_id_probability of being added. For example,
    if per_id_probability is 0.01 and an id appears 100 times, then it has a 60%
    of being added. More precisely, the id score is 1 - (1 - per_id_probability) ^ id_count,
    and for a randomly generated threshold, the id score is the chance of it being added.
    """
    probability = torch.full_like(id_counts, 1 - per_id_probability, dtype=torch.float)
    id_scores = 1 - torch.pow(probability, id_counts)

    threshold: torch.Tensor = torch.rand(id_counts.size(), device=id_counts.device)
    threshold_mask = id_scores > threshold

    return threshold_mask, threshold
```
