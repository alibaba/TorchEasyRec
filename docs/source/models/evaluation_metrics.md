# 评估指标

## 简介

在模型训练与评估过程中，选择合适的评估指标对于衡量模型性能至关重要。本文档基于 `tzrec` 中定义的评估指标配置，详细介绍各类评估指标及其参数含义。

---

## AUC

**AUC（Area Under the ROC Curve）** 是二分类任务中最常用的评估指标之一，表示模型区分正负样本的能力。其值介于 0 到 1 之间，数值越大说明模型性能越好。

| 字段名      | 类型   | 是否可选 | 描述                                                                                  |
|-------------|--------|----------|-------------------------------------------------------------------------------------|
| thresholds  | uint32 | optional | 计算 AUC 时使用的阈值数量，默认为 200。<br>更高的阈值数可以提升精度，但会增加计算开销。<br>当正负样本比例过大时，建议调大 `thresholds`。 |

## AUC计算
```SQL
pai -name=evaluate -project=algo_public
    -DoutputMetricTableName=output_metric_table
    -DoutputDetailTableName=output_detail_table
    -DinputTableName=input_data_table
    -DlabelColName=label
    -DscoreColName=score;
```
---

## MulticlassAUC

**多分类 AUC（Multiclass AUC）** 扩展了传统 AUC 到多类别场景，适用于标签数量大于 2 的分类任务。支持多种平均方式来综合各类别 AUC 值。

| 字段名     | 类型   | 是否可选 | 描述 |
|------------|--------|----------|------|
| thresholds | uint32 | optional | 计算每个类别 AUC 时使用的阈值数量，默认为 200。 |
| average    | string | optional | 多类别的聚合方式，默认为 `'macro'`。<br>可选值：<br>- `'macro'`: 对每个类别的 AUC 取简单平均<br>- `'weighted'`: 按各类别的样本支持度（support）加权平均 |

> ⚠️ 注意：该指标通常通过“一对多”（One-vs-Rest）策略为每个类别单独计算 AUC。

---

## Accuracy

**准确率（Accuracy）** 表示预测正确的样本占总样本的比例，适用于分类任务。当类别数大于 1 时，支持 Top-K 准确率。

| 字段名     | 类型   | 是否可选 | 描述 |
|------------|--------|----------|------|
| threshold  | float  | optional | 概率阈值，用于将预测概率转换为类别标签，默认为 0.5。<br>仅对二分类有效。 |
| top_k      | uint32 | optional | Top-K 准确率中的 K 值，当 `num_class > 1` 时生效，默认为 1。 |

---

## RecallAtK

**召回率@K（Recall@K）** 衡量在前 K 个推荐结果中有多少比例的相关项目被成功召回。常用于推荐系统和信息检索任务。

| 字段名 | 类型   | 是否可选 | 描述 |
|--------|--------|----------|------|
| top_k  | uint32 | optional | 排名前 K 的位置，默认为 5。<br>例如 Recall@5 表示只考虑前 5 个预测项是否包含真实正样本。 |

> ✅ 公式：  
> Recall@K = (Top-K 中正样本的数量) / (正样本总数)

---

## MeanAbsoluteError

**平均绝对误差（MAE）** 是回归任务中的常用指标，表示预测值与真实值之间绝对误差的平均值。

| 字段名 | 类型 | 是否可选 | 描述 |
|--------|------|----------|------|
| -      | -    | -        | 无配置参数，直接计算所有样本的平均绝对误差。 |


> ✅ 公式：  
> MAE = (1/n) × Σ|y_i - ŷ_i|，即预测值与真实值之间绝对误差的平均值
---

## MeanSquaredError

**均方误差（MSE）** 衡量预测值与真实值之间的平方误差的平均值，对异常值更敏感，广泛用于回归任务。

| 字段名 | 类型 | 是否可选 | 描述 |
|--------|------|----------|------|
| -      | -    | -        | 无配置参数，直接计算所有样本的均方误差。 |


> ✅ 公式：  
> MSE = (1/n) × Σ(y_i - ŷ_i)²，即预测值与真实值之间平方误差的平均值

---

## GroupedAUC

**分组 AUC（Grouped AUC）** 是一种扩展的 AUC 指标，用于评估在特定分组维度下的排序能力。常用于推荐系统中按用户、物品或其他业务维度进行分组评估。

| 字段名       | 类型   | 是否必填 | 描述 |
|--------------|--------|----------|------|
| grouping_key | string | required | 用于分组的字段名称（如 `"user_id"`），必须提供。分组字段需要再feature_configs中，一般可以设置较大的num_buckets或者hash_bucket_size |

---

## XAUC

**分组 XAUC** 是 GroupedAUC 的变体，用于评估在连续值目标的分组维度下的排序能力

| 字段名         | 类型   | 是否可选 | 描述 |
|----------------|--------|----------|------|
| sample_ratio   | float  | optional | 随机采样比例，用于减少计算量，默认为 `1e-3`（即 0.1%）。 |
| max_pairs      | uint64 | optional | 最大正负样本对数限制，控制计算规模。 |
| in_batch       | bool   | optional | 是否仅在当前 batch 内计算配对，默认为 `false`。<br>`true` 表示只和同 batch 用户做比较；`false` 支持全局比较。 |

> 🔍 说明：XAUC 通过构建正负样本对（一个用户的一个正样本 vs 其他用户的负样本）来评估跨用户排序性能。

---

## GroupedXAUC

**交叉分组 AUC（Grouped XAUC）** 是 GroupedAUC 的变体，用于评估在连续值目标的分组维度下的排序能力。

| 字段名                  | 类型   | 是否可选 | 描述 |
|-------------------------|--------|----------|------|
| grouping_key            | string | required | 分组键，指定用于划分群体的字段。 |
| max_pairs_per_group     | uint64 | optional | 每个分组中用于计算的最大正负样本对数，默认为 100。<br>用于控制计算复杂度和稳定性。 |

---

## Group AUC 计算

若训练时显存不足，可通过 SQL 计算 GAUC：

```sql
SELECT AVG(gauc * pos_cnt) / SUM(pos_cnt) AS gauc
FROM (
    SELECT 
        group_name,
        (rank_pos - pos_cnt * (pos_cnt + 1) / 2.0) / (pos_cnt * neg_cnt) AS gauc,
        pos_cnt
    FROM (
        SELECT 
            group_name,
            SUM(CASE WHEN label = 1 THEN rn ELSE 0 END) AS rank_pos,
            SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS pos_cnt,
            SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS neg_cnt
        FROM (
            SELECT 
                group_name, 
                label, 
                ROW_NUMBER() OVER (PARTITION BY group_name ORDER BY probs ASC) AS rn
            FROM your_table
        ) t1
        GROUP BY group_name
        HAVING pos_cnt > 0 AND neg_cnt > 0  -- 确保每组有正负样本
    ) t2
) t3;
