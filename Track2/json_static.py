import json
import numpy as np
from tqdm import tqdm

# 读取原始 JSON 文件
# with open('/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/data/Task2/train.json', 'r') as f:
#     data = json.load(f)
#
# processed_data = []
#
# # 使用 tqdm 来显示进度条
# for item in tqdm(data, desc="Processing data"):
#     # 获取原始数据
#     accel = item['accel']  # accel 是 n 个元素，每个元素包含 [ax, ay, az]
#     labels = item['labels']
#     ts = item['ts']
#
#     # 转换为 NumPy 数组，便于后续处理
#     accel = np.array(accel)  # 形状为 (n, 3)，表示 n 个样本，每个样本包含 3 轴数据
#     labels = np.array(labels)
#     ts = np.array(ts)
#
#     # 将 accel 重复到长度 280000，并保证每个元素仍然包含 [ax, ay, az]
#     num_repeat = 280000 // len(accel)
#     accel = np.tile(accel, (num_repeat, 1))[:280000]  # 形状变为 (280000, 3)
#
#     # 将 labels 和 ts 重复到长度 280000
#     labels = np.tile(labels, num_repeat)[:280000]
#     ts = np.tile(ts, num_repeat)[:280000]
#
#     # 切割成 100 个长度为 2800 的条目
#     for i in range(100):
#         start_idx = i * 2800
#         end_idx = start_idx + 2800
#
#         new_item = {
#             'subject': item['subject'],
#             'accel': accel[start_idx:end_idx].tolist(),  # 每段的形状为 (2800, 3)
#             'labels': labels[start_idx:end_idx].tolist(),
#             'ts': ts[start_idx:end_idx].tolist()
#         }
#         processed_data.append(new_item)
#
# # 保存到新的 JSON 文件中
# with open('/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/data/Task2/train_new.json', 'w') as f:
#     json.dump(processed_data, f, indent=4)
# with open('/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/data/Task2/train_new.json', 'r') as f:
#     data = json.load(f)  # 加载 JSON 数据
# print(len(data))
# import json
# import numpy as np
#
# # 输入和输出 JSON 文件路径
# input_file = "/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/data/Task2/train_new.json"  # 输入文件
# output_file = "/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/data/Task2/train_v2.json"  # 输出文件
#
# # 读取 JSON 文件
# with open(input_file, 'r') as f:
#     data = json.load(f)
#
# # 初始化一个空列表，用于存储有效数据
# filtered_data = []
#
# # 遍历 JSON 数据
# for item in data:
#     try:
#         # 将 accel 转换为 NumPy 数组
#         accel = np.array(item['accel'])
#
#         # 检查 accel 的形状是否为 (2800, 3)
#         if accel.shape == (2800, 3):
#             filtered_data.append(item)  # 如果符合要求，保留该条目
#         else:
#             print(f"Skipping item with invalid shape: {accel.shape}")
#     except Exception as e:
#         print(f"Error processing item: {e}")
#         continue  # 如果有任何异常，跳过该条目
#
# # 将过滤后的数据写入新的 JSON 文件
# with open(output_file, 'w') as f:
#     json.dump(filtered_data, f, indent=4)
#
# print(f"Filtering complete. Valid entries saved to {output_file}.")
# with open('/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/data/Task2/train_v2.json', 'r') as f:
#     data = json.load(f)  # 加载 JSON 数据
# print(len(data))


import json
import random

# 输入和输出文件路径
input_file = "/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/data/Task2/train_v4.json"  # 输入原始文件
output_train_file = "/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/data/Task2/train_v5.json"  # 训练集输出文件
output_val_file = "/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/data/Task2/evalv2.json"  # 验证集输出文件

# 读取 JSON 文件
with open(input_file, 'r') as f:
    data = json.load(f)

# 检查数据量是否足够
if len(data) < 400:
    raise ValueError("Not enough data to create a validation set of 400 entries.")

# 随机抽取 400 条作为验证集
validation_set = random.sample(data, 400)

# 从原始数据中删除这 400 条
training_set = [item for item in data if item not in validation_set]

# 保存验证集到新的 JSON 文件
with open(output_val_file, 'w') as f:
    json.dump(validation_set, f, indent=4)

# 保存训练集到新的 JSON 文件
with open(output_train_file, 'w') as f:
    json.dump(training_set, f, indent=4)

print(f"Validation set with 400 entries saved to {output_val_file}.")
print(f"Training set with remaining entries saved to {output_train_file}.")
