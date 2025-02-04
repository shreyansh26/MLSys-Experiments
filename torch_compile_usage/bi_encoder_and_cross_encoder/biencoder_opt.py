import time
import re
import torch
from transformers import AutoTokenizer, AutoModel
import random
import torch.nn.functional as F
import numpy as np

torch.backends.cudnn.benchmark = True
torch._dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision('high')
torch._inductor.config.triton.cudagraph_trees = False
# torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

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

    print("Compiling models")
    biencoder.forward = torch.compile(biencoder.forward, mode="max-autotune")

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

def get_bienc_pred(biencoder, tokenizer, sentences, batch_size=32):
    cleaned_sentences = [resub_utterance(s) for s in sentences]
    encoded_input = tokenizer(cleaned_sentences, padding="max_length", truncation=True, max_length=256, return_tensors='pt').to(device)

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

def main():
    biencoder, tokenizer = load_models()

    sentences = [
        f"Sentence A {i}" + "example1 " * random.randint(128, 256)
        for i in range(4096)
    ]
    print(tokenizer(sentences[0], return_tensors="pt").input_ids.shape)
    print(tokenizer(sentences[1], return_tensors="pt").input_ids.shape)

    print("Warming up...")
    _ = get_bienc_pred(biencoder, tokenizer, sentences[:128], batch_size=32)

    print("Starting benchmark on full dataset...")
    start = time.time()
    embeddings = get_bienc_pred(biencoder, tokenizer, sentences, batch_size=32)
    end = time.time()
    print(f"Inference time: {end - start:.4f} seconds")

    print("\nSample scores from biencoder:")
    print(embeddings.shape)
    print(embeddings[0].shape)

    # Move embeddings to CPU and convert to numpy format
    embeddings_cpu = embeddings.cpu().numpy()
    
    # Save embeddings to a file
    np.save("outputs/opt_embeddings.npy", embeddings_cpu)

if __name__ == "__main__":
    main() 