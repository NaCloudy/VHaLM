import torch
import clip
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="ViT-L/14", choices=["ViT-B/32", "ViT-L/14"],
                    help="CLIP model variant (ViT-L/14 gives 768-dim features, ViT-B/32 gives 512-dim)")
parser.add_argument("--image_dir", default="coco_subset/images")
parser.add_argument("--caption_json", default="coco_subset/captions_subset.json")
parser.add_argument("--features_path", default="image_features.npy")
parser.add_argument("--ids_path", default="image_ids.json")
args = parser.parse_args()

# 路径配置
image_dir = args.image_dir
caption_json = args.caption_json
features_path = args.features_path
ids_path = args.ids_path

# 1. 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(args.model, device=device)
print(f"Loaded CLIP {args.model}, feature dim: {model.visual.output_dim}")

# 2. 读取 captions_subset.json 中的顺序
with open(caption_json, 'r') as f:
    data = json.load(f)

# 获取按顺序排列的 img_id
img_ids = [item["img_id"] for item in data]

# 3. 提取图像特征
features = []
valid_ids = []

for img_id in tqdm(img_ids, desc="Extracting image features"):
    img_path = os.path.join(image_dir, img_id)
    if not os.path.exists(img_path):
        print(f"⚠️ 图片不存在: {img_id}")
        continue
    try:
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image)
            feat /= feat.norm(dim=-1, keepdim=True)
        features.append(feat.cpu().numpy())
        valid_ids.append(img_id)
    except Exception as e:
        print(f"⚠️ 无法处理 {img_id}: {e}")

# 4. 保存结果
features = np.concatenate(features, axis=0)
np.save(features_path, features)

with open(ids_path, 'w') as f:
    json.dump(valid_ids, f, indent=2)

print(f"✅ 已保存 {len(valid_ids)} 张图片的特征到 {features_path}")
print(f"✅ 对应的 img_id 已保存到 {ids_path}")
