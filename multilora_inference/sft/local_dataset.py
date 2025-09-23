import datasets
from sklearn.model_selection import train_test_split

def extract_data(speaker: str, from_role: str, value_content: str, conversation: list):
    for message in conversation:
        if message[from_role] == speaker:
            return message[value_content]
    return None

def load_infinity_instruct():
    data = datasets.load_dataset("BAAI/Infinity-Instruct", "0625", split="train")
    data = data.filter(lambda row: row["langdetect"] == "en")
    data = data.map(lambda row: {"instruction": extract_data("human", "from", "value", row["conversations"]), "output": extract_data("gpt", "from", "value", row["conversations"])})
    data = data.filter(lambda row: row["instruction"] is not None and row["output"] is not None)
    data = data.select_columns(["instruction", "output"])
    data = data.to_pandas()
    return data

def load_numina_math():
    data = datasets.load_dataset("AI-MO/NuminaMath-CoT", split="train")
    data = data.filter(lambda row: row["source"] in ("synthetic_amc", "synthetic_math"))
    data = data.map(lambda row: {"instruction": row["problem"], "output": row["solution"]})
    data = data.filter(lambda row: row["instruction"] is not None and row["output"] is not None)
    data = data.select_columns(["instruction", "output"])
    data = data.to_pandas()
    return data

def load_opc_sft_educational():
    data = datasets.load_dataset("OpenCoder-LLM/opc-sft-stage2", "educational_instruct", split="train")
    data = data.map(lambda row: {"instruction": row["instruction"], "output": row["output"]})
    data = data.filter(lambda row: row["instruction"] is not None and row["output"] is not None)
    data = data.select_columns(["instruction", "output"])
    data = data.to_pandas()
    return data

def load_opc_evol_instruct():
    data = datasets.load_dataset("OpenCoder-LLM/opc-sft-stage2", "educational_instruct", split="train")
    data = data.map(lambda row: {"instruction": row["instruction"], "output": row["output"]})
    data = data.filter(lambda row: row["instruction"] is not None and row["output"] is not None)
    data = data.select_columns(["instruction", "output"])
    data = data.to_pandas()
    return data

def load_text_to_sql():
    data = datasets.load_dataset("gretelai/synthetic_text_to_sql", split="train")
    data = data.map(lambda row: {"instruction": row["sql_prompt"] + "\n\nSQL CONTEXT:\n\n" + row["sql_context"], "output": row["sql"]})
    data = data.filter(lambda row: row["instruction"] is not None and row["output"] is not None)
    data = data.select_columns(["instruction", "output"])
    data = data.to_pandas()
    return data

def load_ifeval_like_data():
    data = datasets.load_dataset("argilla/ifeval-like-data", "filtered", split="train")
    data = data.map(lambda row: {"instruction": row["prompt"], "output": row["response"]})
    data = data.filter(lambda row: row["instruction"] is not None and row["output"] is not None)
    data = data.select_columns(["instruction", "output"])
    data = data.to_pandas()
    return data

def load_multilingual_cohere_aya():
    data = datasets.load_dataset("CohereLabs/aya_dataset",split="train")
    data = data.map(lambda row: {"instruction": row["inputs"], "output": row["targets"]})
    data = data.filter(lambda row: row["language"] in ("French", "Tamil", "Telugu", "Spanish", "Italian", "Portuguese", "Urdu", "Marathi", "Greek", "Dutch"))
    data = data.filter(lambda row: row["instruction"] is not None and row["output"] is not None)
    data = data.select_columns(["instruction", "output", "language"])
    data = data.to_pandas()
    return data

def load_dataset(dataset_name: str):
    if dataset_name == "infinity_instruct":
        data = load_infinity_instruct()
    elif dataset_name == "numina_math":
        data = load_numina_math()
    elif dataset_name == "opc_sft_educational":
        data = load_opc_sft_educational()
    elif dataset_name == "opc_evol_instruct":
        data = load_opc_evol_instruct()
    elif dataset_name == "text_to_sql":
        data = load_text_to_sql()
    elif dataset_name == "ifeval_like_data":
        data = load_ifeval_like_data()
    elif dataset_name == "multilingual_cohere_aya":
        data = load_multilingual_cohere_aya()
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    data_train, data_test = train_test_split(data, test_size=1000, random_state=42)
    return data_train, data_test

if __name__ == "__main__":
    data, _ = load_dataset("infinity_instruct")
    print(data.head())
    
    data, _ = load_dataset("numina_math")
    print(data.head())
    
    data, _ = load_dataset("opc_sft_educational")
    print(data.head())
    
    data, _ = load_dataset("opc_evol_instruct")
    print(data.head())
    
    data, _ = load_dataset("text_to_sql")
    print(data.head())
    
    data, _ = load_dataset("ifeval_like_data")
    print(data.head())
    
    data, _ = load_dataset("multilingual_cohere_aya")
    print(data.head())