import time
import re
import torch
from packaging import version
from sentence_transformers.cross_encoder import CrossEncoder as CrossEncoderST
from CrossEncoder import CrossEncoder
from transformers import AutoTokenizer
import random
import numpy as np

torch.backends.cudnn.benchmark = True
torch._dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision('high')
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

random.seed(42)
device = "cuda"

def load_models():
    """
    Load two CrossEncoder models on GPU, set them to evaluation mode,
    and (if PyTorch >= 2.0) optionally wrap them with torch.compile for speed.
    """
    cross_encoder_1 = CrossEncoder("stsb-roberta-base", device=device, max_length=512)
    # cross_encoder_1.model.to(torch.bfloat16)
    cross_encoder_2 = CrossEncoderST("jina-reranker-v2-base-multilingual", device=device, trust_remote_code=True, max_length=512)
    # cross_encoder_2.model.to(torch.bfloat16)

    cross_encoder_1.model.eval()
    cross_encoder_2.model.eval()

    print("Compiling models")
    cross_encoder_1.model = torch.compile(cross_encoder_1.model, mode="max-autotune", dynamic=True)
    # cross_encoder_2.model = torch.compile(cross_encoder_2.model, mode="max-autotune", dynamic=True)

    return cross_encoder_1, cross_encoder_2

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

def get_cross_pred(cross_encoder_1, cross_encoder_2, pairs, batch_size=32, show_progress=False):
    """
    Runs inference using the two cross-encoders.  
    - show_progress=False disables the progress bar for faster benchmarks
    - Uses autocast(float16) + no_grad for speed.
    """
    cleaned_pairs = [(resub_utterance(p[0]), resub_utterance(p[1])) for p in pairs]

    with torch.no_grad():
        # Optional mixed precision
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        scores1 = cross_encoder_1.predict(cleaned_pairs, batch_size=batch_size, show_progress_bar=show_progress)
        scores2 = cross_encoder_2.predict(cleaned_pairs, batch_size=batch_size, show_progress_bar=show_progress)

    return scores1, scores2

def main():
    cross_encoder_1, cross_encoder_2 = load_models()

    sentence_pairs = [
        (f"Sentence A {i}" + "example1 " * random.randint(115, 128), f"Sentence B {i}" + "example2 " * random.randint(115, 128))
        for i in range(4096)
    ]
    # tokenizer = AutoTokenizer.from_pretrained("stsb-roberta-base")
    # print(tokenizer(sentence_pairs[0][0], sentence_pairs[0][1], return_tensors="pt").input_ids.shape)
    # print(tokenizer(sentence_pairs[1][0], sentence_pairs[1][1], return_tensors="pt").input_ids.shape)

    print("Warming up...")
    _ = get_cross_pred(cross_encoder_1, cross_encoder_2, sentence_pairs[:128], batch_size=32, show_progress=False)
    _ = get_cross_pred(cross_encoder_1, cross_encoder_2, sentence_pairs[:128], batch_size=32, show_progress=False)

    print("Starting benchmark on full dataset...")
    start = time.time()
    scores1, scores2 = get_cross_pred(cross_encoder_1, cross_encoder_2, sentence_pairs, batch_size=32, show_progress=False)
    end = time.time()
    print(f"Inference time: {end - start:.4f} seconds")

    print("\nSample scores from cross_encoder_1:")
    for i, score in enumerate(scores1[:10]):
        print(f"  Pair {i} Score = {score:.4f}")

    print("\nSample scores from cross_encoder_2:")
    for i, score in enumerate(scores2[:10]):
        print(f"  Pair {i} Score = {score:.4f}")

    # Save scores to a file
    np.save("outputs/opt_cross_encoder_scores_1.npy", scores1)
    np.save("outputs/opt_cross_encoder_scores_2.npy", scores2)

if __name__ == "__main__":
    main()