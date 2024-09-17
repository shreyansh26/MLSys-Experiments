HF_TOKEN=abcd \
python convert_csv_to_jsonl.py \
	--input_csv_path train_json.csv \
	--output_jsonl_path train_json.jsonl \
	--instruct true

HF_TOKEN=abcd \
python convert_csv_to_jsonl.py \
        --input_csv_path val_json.csv \
        --output_jsonl_path val_json.jsonl \
        --instruct true
