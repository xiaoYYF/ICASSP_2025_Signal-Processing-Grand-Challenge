import json
import os
import random

# 加载数据集
data_path = "/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/data/Task1.json"
with open(data_path, 'r') as f:
    data = json.load(f)

# 设置划分比例
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

# 打乱数据
random.shuffle(data)

# 计算各个子集的大小
total_len = len(data)
train_size = int(total_len * train_ratio)
validation_size = int(total_len * validation_ratio)

# 划分数据集
train_data = data[:train_size]
validation_data = data[train_size:train_size + validation_size]
test_data = data[train_size + validation_size:]

# 创建目标文件夹
output_dir = "/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/data/Task1"
os.makedirs(output_dir, exist_ok=True)

# 保存划分后的数据集
train_json_path = os.path.join(output_dir, "train.json")
validation_json_path = os.path.join(output_dir, "validation.json")
test_json_path = os.path.join(output_dir, "test.json")

with open(train_json_path, 'w') as f:
    json.dump(train_data, f, indent=4)

with open(validation_json_path, 'w') as f:
    json.dump(validation_data, f, indent=4)

with open(test_json_path, 'w') as f:
    json.dump(test_data, f, indent=4)

print("数据集划分完成并保存至以下路径：")
print("训练集:", train_json_path)
print("验证集:", validation_json_path)
print("测试集:", test_json_path)
