import json
import random

# 加载原始合并后的数据
with open("Merged_SafetyBench.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 转换为列表并打乱顺序
all_items = list(data.values())
random.shuffle(all_items)

# 取前 218 项
sampled_items = all_items[:218]

# 重新编号为 0, 1, 2, ...
sampled_dict = {str(i): item for i, item in enumerate(sampled_items)}

# 保存为新文件
with open("SafetyBench.json", "w", encoding="utf-8") as f:
    json.dump(sampled_dict, f, ensure_ascii=False, indent=2)

print(f"Saved 218 randomly sampled entries to Merged_SafetyBench_sampled.json")
