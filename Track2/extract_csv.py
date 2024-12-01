import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('submission.csv')

# 删除 subject 和 timestamp 重复的行，保留第一个出现的行
df_unique = df.drop_duplicates(subset=['subject', 'timestamp'], keep='first')

# 重新编号 id，让它从 0 开始自增
df_unique.reset_index(drop=True, inplace=True)
df_unique['id'] = df_unique.index

# 保存为新的 CSV 文件
df_unique.to_csv('submission_new.csv', index=False)

print(f"处理完成，生成的文件为 'submission_new.csv'，总行数为：{len(df_unique)}")
