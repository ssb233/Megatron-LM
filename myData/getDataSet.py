from datasets import load_dataset
import json
from tqdm import tqdm

# -------- 配置 --------
DATASET_NAME = "Elriggs/openwebtext-100k"  # Hugging Face 数据集名
SPLIT = "train"                             # 使用的数据集拆分
OUTPUT_FILE = "openwebtext_100k.jsonl"     # 输出 JSONL 文件
TEXT_FIELD = "text"                         # 数据集中文本字段名称

# -------- 加载数据集 --------
dataset = load_dataset(DATASET_NAME, split=SPLIT)
print(f"数据集总条目数: {len(dataset)}")

# -------- 转换为 JSONL --------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in tqdm(dataset):
        # 只保留文本字段
        text = item.get(TEXT_FIELD)
        if not text:
            continue
        json_line = json.dumps({"text": text}, ensure_ascii=False)
        f.write(json_line + "\n")

print(f"已生成 JSONL 文件: {OUTPUT_FILE}")
