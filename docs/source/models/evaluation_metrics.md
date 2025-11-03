# 评估指标

## 简介

在模型训练与评估过程中，选择合适的评估指标对于衡量模型性能至关重要。本文档基于 `tzrec` 中定义的评估指标配置，详细介绍各类评估指标及其参数含义。

---

## AUC

**AUC（Area Under the ROC Curve）** 是二分类任务中最常用的评估指标之一，表示模型区分正负样本的能力。其值介于 0 到 1 之间，数值越大说明模型性能越好。

| 字段名      | 类型   | 是否可选 | 描述                                                                                  |
|-------------|--------|----------|-------------------------------------------------------------------------------------|
| thresholds  | uint32 | optional | 计算 AUC 时使用的阈值数量，默认为 200。<br>更高的阈值数可以提升精度，但会增加计算开销。<br>当正负样本比例过大时，建议调大 `thresholds`。 |

---

## Accuracy

**准确率（Accuracy）** 表示预测正确的样本占总样本的比例，适用于分类任务。当类别数大于 1 时，支持 Top-K 准确率。

| 字段名     | 类型   | 是否可选 | 描述 |
|------------|--------|----------|------|
| threshold  | float  | optional | 概率阈值，用于将预测概率转换为类别标签，默认为 0.5。<br>仅对二分类有效。 |
| top_k      | uint32 | optional | Top-K 准确率中的 K 值，当 `num_class > 1` 时生效，默认为 1。 |

---

## GroupedAUC

**分组 AUC（Grouped AUC）** 是一种扩展的 AUC 指标，用于评估在特定分组维度下的排序能力。常用于推荐系统中按用户、物品或其他业务维度进行分组评估。

| 字段名       | 类型   | 是否必填 | 描述 |
|--------------|--------|----------|------|
| grouping_key | string | required | 用于分组的字段名称（如 `"user_id"`），必须提供。 |

---

## GroupedXAUC

**交叉分组 AUC（Grouped XAUC）** 是 GroupedAUC 的变体，通常用于衡量跨群体之间的排序公平性或差异性，例如比较不同用户群体间的推荐效果。

| 字段名                  | 类型   | 是否可选 | 描述 |
|-------------------------|--------|----------|------|
| grouping_key            | string | required | 分组键，指定用于划分群体的字段。 |
| max_pairs_per_group     | uint64 | optional | 每个分组中用于计算的最大正负样本对数，默认为 100。<br>用于控制计算复杂度和稳定性。 |

---

## Group AUC 计算

若训练时显存不足，可通过 SQL 计算 GAUC：

```sql
select group_name, (rank_pos - pos_cnt * (pos_cnt+1) / 2) / (pos_cnt * neg_cnt) as gauc
from (
    select group_name,
           sum(if(label=1, rn, 0)) as rank_pos,
           sum(if(label=1, 1, 0)) as pos_cnt,
           sum(if(label=0, 1, 0)) as neg_cnt
    from (
        select group_name, label, 
               rank() over(partition by group_name order by probs asc) as rn
        from your_table
    )
    group by group_name
);
