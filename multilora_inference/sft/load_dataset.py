import json
import datasets

def extract_data(speaker: str, from_role: str, value_content: str, conversation: list):
    for message in conversation:
        if message[from_role] == speaker:
            return message[value_content]
    return None

def load_infinity_instruct():
    data = datasets.load_dataset("BAAI/Infinity-Instruct", "0625", split="train")
    data_en = data.filter(lambda row: row["langdetect"] == "en")
    data_en = data_en.map(lambda row: {"instruction": extract_data("human", "from", "value", row["conversations"]), "output": extract_data("gpt", "from", "value", row["conversations"])})
    data_en = data_en.filter(lambda row: row["instruction"] is not None and row["output"] is not None)
    data_en = data_en.select_columns(["instruction", "output"])
    data_en = data_en.to_pandas()
    return data_en


def load_dataset(dataset_name: str):
    if dataset_name == "infinity_instruct":
        return load_infinity_instruct()
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

if __name__ == "__main__":
    data = load_dataset("infinity_instruct")
    print(data.head())