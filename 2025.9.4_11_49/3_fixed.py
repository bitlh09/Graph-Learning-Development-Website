import os
from spektral.datasets import citation
from spektral.data import Dataset

# 设置下载目录 - 使用实例方式而不是类属性
download_dir = os.path.join(".", "data")  # ./data 文件夹

# 下载并加载 Citeseer 数据集
# 注意：这里不直接设置 Dataset.download_dir
dataset = citation.Citation(name="citeseer")

# 打印信息
print(f"数据已下载到: {download_dir}/citeseer")
print("样本数量:", len(dataset))