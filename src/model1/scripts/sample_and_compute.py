import json
import random
import numpy as np
from collections import defaultdict

# 设置随机种子
random.seed(453)
np.random.seed(453)

# 读取原始JSON文件
input_file = 'best_model_test_samples_20251124_143737.json'
output_file = 'final_model1_sample120.json'

print("读取文件...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

original_samples = data['samples']
print(f"原始样本数: {len(original_samples)}")

# 随机抽样120条
print("随机抽样120条样本...")
sampled_samples = random.sample(original_samples, 120)
print(f"抽样后样本数: {len(sampled_samples)}")

# 重新计算overall_metrics
print("重新计算overall_metrics...")
metrics_sum = defaultdict(float)
metrics_count = 0

for sample in sampled_samples:
    metrics = sample.get('metrics', {})
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            metrics_sum[metric_name] += metric_value
    metrics_count += 1

# 计算平均值
overall_metrics = {}
for metric_name, total_value in metrics_sum.items():
    overall_metrics[metric_name] = total_value / metrics_count

print("\n重新计算的overall_metrics:")
for metric_name, value in overall_metrics.items():
    print(f"  {metric_name}: {value:.4f}")

# 构建新的数据结构
new_data = {'overall_metrics': overall_metrics, 'num_samples': len(sampled_samples), 'samples': sampled_samples}

# 保存为JSON文件
print(f"\n保存到 {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print(f"✅ 完成！已保存 {len(sampled_samples)} 条样本到 {output_file}")
print(f"原始测试集大小: {len(original_samples)}")
print(f"抽样后大小: {len(sampled_samples)}")
print(f"抽样比例: {len(sampled_samples)/len(original_samples):.2%}")
