import time
import re
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import random
import torch.nn.functional as F
import numpy as np

random.seed(42)

device = "cuda"

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_models():
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    biencoder = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)

    biencoder.eval()

    return biencoder, tokenizer

def resub_utterance(utterance: str) -> str:
    """
    Clean transcripts to remove numbers and redacted info
    """
    cleaned_utterance = re.sub("<.*?>+", "", utterance)
    cleaned_utterance = re.sub(r"\[([^\]]*)\]", r"\1", cleaned_utterance)
    cleaned_utterance = re.sub(r"\d*", "", cleaned_utterance)
    translation_table = str.maketrans({key: " " for key in """\n!"#$%&\()*+,./:;<=>?@[\\]^_`{|}~"""})
    cleaned_utterance = cleaned_utterance.translate(translation_table)
    return cleaned_utterance.strip()

def get_bienc_pred(biencoder, tokenizer, sentences, batch_size=32, show_progress=False):
    cleaned_sentences = [resub_utterance(s) for s in sentences]
    encoded_input = tokenizer(cleaned_sentences, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)

    embeddings = []
    for i in range(0, len(cleaned_sentences), batch_size):          
        with torch.no_grad():
            # Compute token embeddings
            model_output = biencoder(input_ids=encoded_input.input_ids[i:i+batch_size], attention_mask=encoded_input.attention_mask[i:i+batch_size])

            # Perform pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input.attention_mask[i:i+batch_size])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            embeddings.append(sentence_embeddings)
    
    return torch.cat(embeddings, dim=0)

def basic_load_models():
    biencoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    return biencoder, tokenizer

def basic_get_bienc_pred(biencoder, tokenizer, sentences, batch_size=32):
    cleaned_sentences = [resub_utterance(s) for s in sentences]
    embeddings = biencoder.encode(cleaned_sentences, batch_size=batch_size, convert_to_tensor=True)
    return embeddings

def main():
    st = False
    if st:
        biencoder, tokenizer = basic_load_models()
    else:
        biencoder, tokenizer = load_models()

    sentences = [
        f"Sentence A {i}" + "example1 " * random.randint(128, 256)
        for i in range(4096)
    ]
    print(tokenizer(sentences[0], return_tensors="pt").input_ids.shape)
    print(tokenizer(sentences[1], return_tensors="pt").input_ids.shape)

    print("Warming up...")
    if st:
        _ = basic_get_bienc_pred(biencoder, tokenizer, sentences[:128], batch_size=32)
    else:
        _ = get_bienc_pred(biencoder, tokenizer, sentences[:128], batch_size=32)

    print("Starting benchmark on full dataset...")
    start = time.time()
    if st:
        embeddings = basic_get_bienc_pred(biencoder, tokenizer, sentences, batch_size=32)
    else:
        embeddings = get_bienc_pred(biencoder, tokenizer, sentences, batch_size=32)
    end = time.time()
    print(f"Inference time: {end - start:.4f} seconds")

    print("\nSample scores from biencoder_1:")
    print(embeddings.shape)
    print(embeddings[0].shape)

    # Move embeddings to CPU and convert to numpy format
    embeddings_cpu = embeddings.cpu().numpy()
    
    # Save embeddings to a file
    if st:
        np.save("outputs/basic_simple_embeddings.npy", embeddings_cpu)
    else:
        np.save("outputs/simple_embeddings.npy", embeddings_cpu)

if __name__ == "__main__":
    main() 