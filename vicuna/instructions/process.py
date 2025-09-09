import json
import random
random.seed(42)

with open(f'advbench.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for i in list(data.keys()):
    if data[i]["safety"] == "unsafe":
        del data[i]

keys = list(data.keys())
selected_keys = random.sample(keys, 200)

train_keys = selected_keys[:100]
test_keys = selected_keys[100:]
train_data = {key: data[key] for key in train_keys}
test_data = {key: data[key] for key in test_keys}

with open(f'advbench-train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)
with open(f'advbench-test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)


with open(f'GQA.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

keys = list(data.keys())
selected_keys = random.sample(keys, 100)
train_data = {key: data[key] for key in selected_keys}
test_data = {key: data[key] for key in keys if key not in selected_keys}

with open(f'GQA-train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)
with open(f'GQA-test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)
