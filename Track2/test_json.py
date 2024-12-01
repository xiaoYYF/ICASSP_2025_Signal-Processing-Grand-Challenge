import json


# 读取原始json文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# 保存为新的json文件
def save_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


# 切割数据的函数
def split_data(entry, chunk_size=2800):
    ts = entry['ts']
    accel = entry['accel']
    label = entry['labels']
    new_entries = []

    # 按照chunk_size进行切割
    for start in range(0, len(ts), chunk_size):
        end = start + chunk_size
        new_ts = ts[start:end]
        new_accel = accel[start:end]
        new_label = label[start:end]
        # 如果最后一段不足chunk_size，进行填充
        if len(new_ts) < chunk_size:
            repeat_count = chunk_size - len(new_ts)
            new_ts += new_ts[-1:] * repeat_count
            new_accel += new_accel[-1:] * repeat_count
            new_label += new_label[-1:] * repeat_count

        # 生成新的数据条目
        new_entry = {
            'subject': entry['subject'],
            'ts': new_ts,
            'accel': new_accel,
            'labels': new_label
        }
        new_entries.append(new_entry)

    return new_entries
def check_and_count_entries(data, chunk_size=2800):
    correct_length_count = 0
    incorrect_length_count = 0
    for entry in data:
        if len(entry['ts']) == chunk_size and len(entry['accel']) == chunk_size:
            correct_length_count += 1
        else:
            incorrect_length_count += 1
    print(f"总条目数: {len(data)}")
    print(f"长度为{chunk_size}的条目数: {correct_length_count}")
    print(f"长度不为{chunk_size}的条目数: {incorrect_length_count}")

# 主函数
if __name__ == "__main__":
    # 替换为你的原始json文件路径
    input_file = '/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/data/Task2/train.json'
    output_file = '/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/data/Task2/train_v4.json'

    # 加载数据
    data = load_json(input_file)

    # 存储新数据的列表
    new_data = []

    # 对每一条数据进行切割
    for entry in data:
        new_data.extend(split_data(entry))

    # 保存为新的json文件
    save_json(new_data, output_file)
    print(f"处理完成，新的数据已保存到 {output_file}")
    check_and_count_entries(new_data)