import json
import csv
import os

# 读取JSON文件
json_path = 'final_model1_sample120.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 准备CSV数据
csv_data = []
for sample in data['samples']:
    row = {
        'group_id': sample['group_id'],
        'image_ids': '|'.join(sample['image_ids']),  # 用|分隔多个图片ID
        'reference': sample['reference'],
        'prediction': sample['prediction'],
        'tags': '|'.join(sample['tags']),  # 用|分隔多个标签
        'METEOR': sample['metrics']['METEOR'],
        'CIDEr': sample['metrics']['CIDEr'],
        'BLEU-1': sample['metrics']['BLEU-1'],
        'BLEU-2': sample['metrics']['BLEU-2'],
        'BLEU-3': sample['metrics']['BLEU-3'],
        'BLEU-4': sample['metrics']['BLEU-4']
    }
    csv_data.append(row)

# 写入CSV文件
csv_path = 'final_model1_sample120.csv'
fieldnames = ['group_id', 'image_ids', 'reference', 'prediction', 'tags', 'METEOR', 'CIDEr', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']

with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:  # 使用utf-8-sig确保Excel正确显示中文
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_data)

print(f"转换完成！共处理 {len(csv_data)} 条记录")
print(f"CSV文件已保存至: {csv_path}")
