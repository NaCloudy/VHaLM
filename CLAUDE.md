# VHaLM: From Visual Moments to Haiku

深度学习课程项目（UW-Madison STAT 453, 2025 Fall, Group 3），构建多图生俳句的两阶段算法流程。

## 项目概述

给定3张语义相关的图片，生成符合5-7-5音节结构的英文俳句。

**两阶段流程：**
1. **Stage 1（多图描述）**：冻结 CLIP ViT-B/32 提取图像特征 + 轻量 multi-token attention 融合 + LoRA 微调 T5 生成统一描述
2. **Stage 2（俳句转写）**：基于 prompt 工程 + 确定性音节校验 + LLM-as-judge reranking，无需训练

**数据集：** 从 MS COCO (~9,000张) 通过 CLIP kNN 图挖掘三角团，构建 2,099 个语义一致的图片三元组；用 GPT-4o-mini 融合 captions 生成伪标签。

**关键结果：**
- Stage 1：相比 T5-small baseline +0.031 METEOR
- Stage 2：20样本 pilot 中 70% 满足严格 5-7-5 结构

## 目录结构

```
data/          # 数据（COCO子集、特征、分组）
src/           # 源码
  model1/      # Stage 1：多图描述模型
experiment/    # 实验记录与评估结果
models/        # 模型权重
reports/       # 课程报告（proposal / midway / final）
```

## 状态

仓库已重构完成，适合作为个人主页项目展示。后续进行小优化。
