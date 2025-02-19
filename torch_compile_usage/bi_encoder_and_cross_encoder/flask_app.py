from flask import Flask, request, jsonify
import time
import re
import torch
from transformers import AutoTokenizer, AutoModel
import random
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

# Torch and inductor config
torch.backends.cudnn.benchmark = True
torch._dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision('high')
torch._inductor.config.triton.cudagraph_trees = False  # disable cudagraphs

random.seed(42)
device = "cuda"

# Mean Pooling - takes attention mask into account
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_models():
    # Load model and tokenizer from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    biencoder = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
    biencoder.eval()
    return biencoder, tokenizer

def resub_utterance(utterance: str) -> str:
    cleaned = re.sub("<.*?>+", "", utterance)
    cleaned = re.sub(r"\[([^\]]*)\]", r"\1", cleaned)
    cleaned = re.sub(r"\d*", "", cleaned)
    translation_table = str.maketrans({key: " " for key in """\n!"#$%&\()*+,./:;<=>?@[\\]^_`{|}~"""})
    return cleaned.translate(translation_table).strip()

def get_bienc_pred(model, tokenizer, sentences, batch_size=64):
    cleaned_sentences = [resub_utterance(s) for s in sentences]
    encoded_input = tokenizer(
        cleaned_sentences,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors='pt'
    ).to(device)
    
    embeddings = []
    for i in range(0, len(cleaned_sentences), batch_size):
        with torch.no_grad():
            
            model_output = model(
                input_ids=encoded_input.input_ids[i:i+batch_size],
                attention_mask=encoded_input.attention_mask[i:i+batch_size]
            )
            sentence_embeddings = mean_pooling(model_output, encoded_input.attention_mask[i:i+batch_size])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            embeddings.append(sentence_embeddings)
    return torch.cat(embeddings, dim=0)

def basic_load_models():
    biencoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    return biencoder, tokenizer

def basic_get_bienc_pred(biencoder, tokenizer, sentences, batch_size=64):
    cleaned_sentences = [resub_utterance(s) for s in sentences]
    embeddings = biencoder.encode(cleaned_sentences, batch_size=batch_size, convert_to_tensor=True)
    return embeddings

app = Flask(__name__)

# Load the model when the module is imported.
print("Loading models...")
biencoder, tokenizer = load_models()
print("Models loaded")

@app.route('/')
def index():
    # global biencoder
    # print("Compiling model")
    # biencoder.forward = torch.compile(biencoder.forward, mode="max-autotune")
    # dummy_input = torch.randint(0, 100, (32, 64)).to(device)
    # dummy_mask = torch.ones((32, 64), dtype=torch.long).to(device)
    # _ = biencoder(input_ids=dummy_input, attention_mask=dummy_mask)
    # return "Model compiled"
    print("Model not compiled")
    return "Model not compiled"

@app.route('/predict', methods=['POST'])
def predict():
    print("Predicting...")
    data = request.get_json()
    if not data or 'sentences' not in data:
        return jsonify({"error": "Missing 'sentences' in request"}), 400
    
    sentences = data['sentences']
    if not isinstance(sentences, list):
        return jsonify({"error": "'sentences' should be a list"}), 400

    start = time.time()
    embeddings = get_bienc_pred(biencoder, tokenizer, sentences)
    end = time.time()

    result = {
        "inference_time": end - start,
        "embeddings": embeddings.cpu().tolist()
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=8090)