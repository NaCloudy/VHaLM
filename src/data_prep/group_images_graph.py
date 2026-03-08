import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import random

# === ⚙️ 配置区域：在这里调节“主题”的宽严程度 ===
CONFIG = {
    # 基础设置
    "features_path": "image_features.npy",
    "ids_path": "image_ids.json",
    "captions_path": "coco_subset/captions_subset.json",
    "output_path": "image_groups_curriculum.json",
    
    "K_neighbors": 50,  # 用于计算邻域重叠的 K 值
    
    # 🎯 难度分级定义
    # Easy: 强相关，几乎就是同类东西，邻居结构高度重叠
    "easy": {
        "sim_range": (0.60, 0.95),  # 相似度区间
        "overlap_thr": 0.20,        # 邻居重叠率 (Jaccard index proxy)
        "target_ratio": 0.5         # 希望 Easy 组占总组数的比例
    },
    
    # Medium: 同一主题但有变化，邻居重叠要求降低
    "medium": {
        "sim_range": (0.30, 0.65),
        "overlap_thr": 0.05,
        "target_ratio": 0.5         # 剩下的尽量填满 Medium
    }
}

# === Step 1. 加载与预处理 ===
print("🔹 加载数据与特征...")
features = np.load(CONFIG["features_path"])

# L2 归一化，确保 dot product = cosine similarity
norms = np.linalg.norm(features, axis=1, keepdims=True)
features = features / (norms + 1e-10)

with open(CONFIG["ids_path"], "r") as f:
    img_ids = json.load(f)
with open(CONFIG["captions_path"], "r") as f:
    captions_data = json.load(f)
caption_map = {item["img_id"]: item["captions"] for item in captions_data}

n_images = len(features)
print(f"✅ 图片总数: {n_images}")

# === Step 2. 构建 kNN 图 (语义拓扑) ===
print(f"🔹 计算 Top-{CONFIG['K_neighbors']} 近邻图...")
# 使用 sklearn 的暴力或树搜索，metric='cosine' 返回的是 distance (1 - sim)
nbrs = NearestNeighbors(n_neighbors=CONFIG['K_neighbors'], metric='cosine', algorithm='auto', n_jobs=-1)
nbrs.fit(features)
distances, indices = nbrs.kneighbors(features)

# 将 distance 转回 similarity
similarities = 1 - distances

# 预计算每个节点的邻居集合 (用于快速计算 overlap)
print("🔹 构建邻居集合索引...")
neighbor_sets = [set(indices[i]) for i in range(n_images)]

def get_neighborhood_overlap(idx_a, idx_b):
    """计算两个节点的邻域重叠度 (Intersection / K)"""
    set_a = neighbor_sets[idx_a]
    set_b = neighbor_sets[idx_b]
    # 这里用简单的交集除以 K，也可以用 Jaccard: len(intersection) / len(union)
    return len(set_a.intersection(set_b)) / CONFIG['K_neighbors']

def check_pair_validity(idx_a, idx_b, criteria):
    """检查两个节点是否满足特定的难度标准"""
    # 1. 检查是否在 Top-K 邻居里 (如果不互为邻居，说明相似度可能太低或排不到前 K)
    # 为了简单，我们直接算点积，不依赖是否在 indices 里，这样更准确
    sim = np.dot(features[idx_a], features[idx_b])
    
    min_s, max_s = criteria['sim_range']
    if not (min_s <= sim <= max_s):
        return False, sim
    
    # 2. 检查邻域结构
    overlap = get_neighborhood_overlap(idx_a, idx_b)
    if overlap < criteria['overlap_thr']:
        return False, sim
        
    return True, sim

# === Step 3. 贪心分组策略 ===
used_mask = np.zeros(n_images, dtype=bool)
final_groups = []

# 估算目标组数 (每组3张)
total_target_groups = n_images // 3

def run_grouping_pass(phase_name, config_key, max_groups):
    print(f"\n🚀 开始 {phase_name} 阶段分组...")
    params = CONFIG[config_key]
    count = 0
    
    # 随机打乱锚点顺序，避免聚类偏差
    available_anchors = np.where(~used_mask)[0]
    np.random.shuffle(available_anchors)
    
    pbar = tqdm(total=max_groups)
    
    for anchor_idx in available_anchors:
        if count >= max_groups:
            break
        if used_mask[anchor_idx]:
            continue
            
        # 获取该锚点的候选邻居 (排除已用的)
        # 我们只在 Top-K 里找，这天然过滤了无关图片，提升速度
        candidates = []
        anchor_sims = similarities[anchor_idx] # Top-K sims
        anchor_ind = indices[anchor_idx]       # Top-K indices
        
        # 筛选第一轮：Anchor 与 Candidate 的关系
        valid_candidates = []
        for k in range(1, CONFIG['K_neighbors']): # 跳过自己(0)
            cand_idx = anchor_ind[k]
            if used_mask[cand_idx]:
                continue
            
            # 检查 Anchor-Candidate 是否满足当前难度标准
            sim = anchor_sims[k]
            overlap = get_neighborhood_overlap(anchor_idx, cand_idx)
            
            if (params['sim_range'][0] <= sim <= params['sim_range'][1]) and \
               (overlap >= params['overlap_thr']):
                valid_candidates.append(cand_idx)
        
        # 尝试在 valid_candidates 里找一对 (B, C)，使得 B-C 也满足条件（三角形约束）
        # 为了增加 Storytelling，我们尽量找 valid_candidates 里 sim 差异适中的
        found_group = False
        
        # 简单的双重循环找三角形 (因为 candidate 数量很少，通常 < 20，所以很快)
        # 也可以加 heuristic：优先选离 anchor 远一点的，增加多样性
        n_cand = len(valid_candidates)
        for i in range(n_cand):
            if found_group: break
            b_idx = valid_candidates[i]
            
            for j in range(i + 1, n_cand):
                c_idx = valid_candidates[j]
                
                # 检查 B 和 C 的关系
                # 注意：B和C可能不互为Top-K，所以必须手动算 dot product
                is_valid, bc_sim = check_pair_validity(b_idx, c_idx, params)
                
                if is_valid:
                    # ✅ 找到符合条件的三元组 (Anchor, B, C)
                    used_mask[anchor_idx] = True
                    used_mask[b_idx] = True
                    used_mask[c_idx] = True
                    
                    # 记录组信息
                    group_imgs = []
                    for idx in [anchor_idx, b_idx, c_idx]:
                        img_id = img_ids[idx]
                        group_imgs.append({
                            "img_id": img_id,
                            "captions": caption_map.get(img_id, [])
                        })
                    
                    # 计算组内平均相似度用于记录
                    # sim(A,B)已经在上面算过，近似取 indices 里的
                    # 这里为了严谨重新算一下平均
                    avg_sim = (np.dot(features[anchor_idx], features[b_idx]) + 
                               np.dot(features[anchor_idx], features[c_idx]) + 
                               bc_sim) / 3.0
                    
                    final_groups.append({
                        "group_id": len(final_groups),
                        "difficulty": config_key,
                        "avg_similarity": float(avg_sim),
                        "images": group_imgs
                    })
                    
                    count += 1
                    pbar.update(1)
                    found_group = True
                    break
    pbar.close()
    print(f"✅ {phase_name} 阶段完成，生成了 {count} 组。")

# === 执行分组流程 ===

# 1. Easy 阶段：严格筛选，只要高质量
target_easy = int(total_target_groups * CONFIG["easy"]["target_ratio"])
run_grouping_pass("Easy (High Consistency)", "easy", target_easy)

# 2. Medium 阶段：放宽标准，处理剩下的
# 剩下的所有名额都可以给 Medium，甚至可以稍微超发一点以利用数据
remaining_quota = total_target_groups - len(final_groups)
# 如果剩下很多图，可以允许 Medium 多跑一点，只要能组成合法的组
run_grouping_pass("Medium (Storytelling)", "medium", n_images) 

# === Step 4. 统计与保存 ===
print(f"\n📊 统计结果:")
print(f"总图片数: {n_images}")
print(f"成功分组数: {len(final_groups)} (消耗 {len(final_groups)*3} 张图)")
print(f"未分组图片: {n_images - len(final_groups)*3} (视为噪声或孤立点丢弃)")

easy_count = sum(1 for g in final_groups if g['difficulty'] == 'easy')
medium_count = sum(1 for g in final_groups if g['difficulty'] == 'medium')
print(f"Easy 组: {easy_count}")
print(f"Medium 组: {medium_count}")

with open(CONFIG["output_path"], "w", encoding="utf-8") as f:
    json.dump(final_groups, f, indent=2, ensure_ascii=False)

print(f"📁 结果已保存至: {CONFIG['output_path']}")