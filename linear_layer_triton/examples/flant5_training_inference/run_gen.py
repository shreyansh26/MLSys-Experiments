import torch
import time
from transformers import AutoTokenizer

import argparse

parser = argparse.ArgumentParser(description='FlanT5 generation')
parser.add_argument('--mode', type=str, default='pure_torch', choices=['pure_torch', 'triton_linear', 'triton_linear_with_flash_attention'], help='Model mode')

args = parser.parse_args()

if args.mode == 'pure_torch':
    from transformers import T5ForConditionalGeneration
elif args.mode == 'triton_linear':
    from modeling_t5_triton import T5ForConditionalGeneration
elif args.mode == 'triton_linear_with_flash_attention':
    from modeling_t5_triton_flash_attn import T5ForConditionalGeneration

NUM_ITERS = 10
MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=1024)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

model = model.to("cuda")
model.half()
model.eval()

text = '''It was in the spring of the year 1894 that all London was interested,
     and the fashionable world dismayed, by the murder of the Honourable
     Ronald Adair under most unusual and inexplicable circumstances. The
     public has already learned those particulars of the crime which came
     out in the police investigation; but a good deal was suppressed upon
     that occasion, since the case for the prosecution was so
     overwhelmingly strong that it was not necessary to bring forward all
     the facts. Only now, at the end of nearly ten years, am I allowed to
     supply those missing links which make up the whole of that remarkable
     chain. The crime was of interest in itself, but that interest was as
     nothing to me compared to the inconceivable sequel, which afforded me
     the greatest shock and surprise of any event in my adventurous life.
     Even now, after this long interval, I find myself thrilling as I
     think of it, and feeling once more that sudden flood of joy,
     amazement, and incredulity which utterly submerged my mind. Let me
     say to that public which has shown some interest in those glimpses
     which I have occasionally given them of the thoughts and actions of a
     very remarkable man that they are not to blame me if I have not
     shared my knowledge with them, for I should have considered it my
     first duty to have done so had I not been barred by a positive
     prohibition from his own lips, which was only withdrawn upon the
     third of last month.

     It can be imagined that my close intimacy with Sherlock Holmes had
     interested me deeply in crime, and that after his disappearance I
     never failed to read with care the various problems which came before
     the public, and I even attempted more than once for my own private
     satisfaction to employ his methods in their solution, though with
     indifferent success. There was none, however, which appealed to me
     like this tragedy of Ronald Adair. As I read the evidence at the
     inquest, which led up to a verdict of wilful murder against some
     person or persons unknown, I realized more clearly than I had ever
     done the loss which the community had sustained by the death of
     Sherlock Holmes. There were points about this strange business which
     would, I was sure, have specially appealed to him, and the efforts of
     the police would have been supplemented, or more probably
     anticipated, by the trained observation and the alert mind of the
     first criminal agent in Europe. All day as I drove upon my round I
     turned over the case in my mind, and found no explanation which
     appeared to me to be adequate. At the risk of telling a twice-told
     tale I will recapitulate the facts as they were known to the public
     at the conclusion of the inquest.

     The Honourable Ronald Adair was the second son of the Earl of
     Maynooth, at that time Governor of one of the Australian Colonies.
     Adair's mother had returned from Australia to undergo the operation
     for cataract, and she, her son Ronald, and her daughter Hilda were
     living together at 427, Park Lane. The youth moved in the best
     society, had, so far as was known, no enemies, and no particular
     vices. He had been engaged to Miss Edith Woodley, of Carstairs, but
     the engagement had been broken off by mutual consent some months
     before, and there was no sign that it had left any very profound
     feeling behind it. For the rest the man's life moved in a narrow and
     conventional circle, for his habits were quiet and his nature
     unemotional. Yet it was upon this easy-going young aristocrat that
     death came in most strange and unexpected form between the hours of
     ten and eleven-twenty on the night of March 30, 1894.

     Ronald Adair was fond of cards, playing continually, but never for
     such stakes as would hurt him. He was a member of the Baldwin, the
     Cavendish, and the Bagatelle card clubs. It was shown that after
     dinner on the day of his death he had played a rubber of whist at the
     latter club. He had also played there in the afternoon. The evidence
     of those who had played with him--Mr. Murray, Sir John Hardy, and
     Colonel Moran--showed that the game was whist, and that there was a
     fairly equal fall of the cards. Adair might have lost five pounds,
     but not more. His fortune was a considerable one, and such a loss
     could not in any way affect him. He had played nearly every day at
     one club or other, but he was a cautious player, and usually rose a
     winner. It came out in evidence that in partnership with Colonel
     Moran he had actually won as much as four hundred and twenty pounds
     in a sitting some weeks before from Godfrey Milner and Lord Balmoral.
     So much for his recent history, as it came out at the inquest.'''

# input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you.", return_tensors="pt", padding='max_length', max_length=512).input_ids.to("cuda")  # Batch size 1
input_ids = tokenizer([f"summarize: {text}"]*10, return_tensors="pt", padding='max_length', max_length=1024, truncation=True).input_ids.to("cuda")  # Batch size 10

print(input_ids.shape)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("Starting compilation and warmup!")
for _ in range(NUM_ITERS):
    outputs = model.generate(input_ids, max_new_tokens=200)

start.record()
outputs = model.generate(input_ids, max_new_tokens=200)
end.record()
torch.cuda.synchronize()

# avg_time = 0
# for _ in range(NUM_ITERS):
#     start = time.perf_counter()
#     outputs = model.generate(input_ids, max_new_tokens=200)
#     end = time.perf_counter()
#     avg_time += (end - start)
#     torch.cuda.synchronize()

print(f"Time taken - {start.elapsed_time(end)}ms")
print(f"Output - {tokenizer.decode(outputs[0], skip_special_tokens=True)}")