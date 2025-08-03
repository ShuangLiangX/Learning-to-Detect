import json
import random
a = "advbench-safe"
# 读取原始JSON文件
with open(f'{a}.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
random.seed(42)
# 确保数据项数量超过100
if len(data) < 200:
    raise ValueError("数据集必须包含超过100项")

# 获取数据项的键列表
keys = list(data.keys())

# 随机抽取100个键
selected_keys = random.sample(keys, 200)
# print(len(selected_keys))
# print(selected_keys)
# exit(0)

# 分割数据为训练集和测试集
train_keys = selected_keys[:100]
test_keys = selected_keys[100:]
print(len(train_keys))
print(len(test_keys))
train_data = {key: data[key] for key in train_keys}
test_data = {key: data[key] for key in test_keys}

# 将训练集保存为train.json
with open(f'advbench-train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

# 将测试集保存为test.json
with open(f'advbench-test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)


# 读取原始JSON文件
with open(f'GQA.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
random.seed(42)

# 获取数据项的键列表
keys = list(data.keys())

# 随机抽取100个键
selected_keys = random.sample(keys, 100)



train_data = {key: data[key] for key in selected_keys}
test_data = {key: data[key] for key in keys if key not in selected_keys}

# 将训练集保存为train.json
with open(f'GQA-train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)
with open(f'GQA-test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)
