from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from hf_token import HF_TOKEN
import time 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prefix-caching', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

convo = '''customer: Name(s) of item(s) affected: Iced caramel macchiato [PERSON] gum Share details: Iced caramel macchiato was spilled and leaking a mess everywhere as well as the gum was missing. Issue with food temperature (too hot/cold): falseDidn&#39;t like taste: falsePortion size too small: falsePackaging damaged, couldn&#39;t eat: falsePackaging damaged, could eat: falseDrink spilled: trueFood was stale/spoiled: [LOCATION] was overcooked/burnt: falseFood safety issue: false
customer: Share a photo of the item:
assistant: Hi JAY (JAQUAN),Were sorry to hear that you were unhappy with the quality of the food you received. After taking a look at your account, I can confirm weve already issued a refund for this order.Thank you for your patience.
customer: yes i seen you only refunded for the gum
customer: however my initial problem was with my drink.
customer: as you can see in the picture there is coffe spilled all in the bag and was drinking all through my apartment.
customer: About a quarter of the drink was spilled.
assistant: Hi [PERSON] (jaquan),Thank you for reaching out. I'm [PERSON] from the support team and I'm here to help you.We apologize for the inconvenience caused as your received order was spilled.Upon checking on our end, I'm pleased to inform you that we have already processed a refund of $3.52 for the reported item, including tax and service fee. The refund was processed on Feb 7, 2024. Your refund will reflect in 3-5 business days.Thank you so much for your time and patience in sharing your concern with us. Thanks for using the app.
customer: Yes I received a refund of the 3$ HOWEVER in addition to the spilled drink I ALSO was missing my gum. Which was about 3$. So how much did I receive for the spilled drink and how much for my missing gum.
assistant: Hi [PERSON] (jaquan),Thanks for being a member. I'm [PERSON] from the support team and I'm here to help youWe are sorry for the inconvenience caused and understand your concern regarding the missing item.Since it wouldn't be fair on your part to pay for something you didn't receive, I'm glad to inform you that we've successfully processed a refund of $3.18 for the incorrect or missing, including tax and service fee. Your refund will reflect in 3-5 business days.We hope to offer a better experience on your future orders.Thanks for being a loyal member. It was indeed a pleasure assisting you. Wish you a wonderful day ahead.
customer: Are you not understanding.
customer: I am missing my GUM first thing. Which was about 3$!! In ADDITION my drink was SPILLED.
customer: So how am I receiving 3.18 refund for BOTH items
customer: That 3.18 is just for the gum!!
customer: What about the spilled drink
customer: Can you break it down and explain to me why I received 3.18 for BOTH OF THE ITEMS. And how much of the 3.18 is for EACH item.
customer: Ok. So I just checked and I see 2 refunds for about 3$. So why arent you saying that correctly. This is completely HORRIBLE service and HORRIBLE communication upon your end. Continuously I have support people reply with the SAME message and not address my issues correctly. Ive asked to be told which is for what and have been given basic messages not addressing what Im Fully saying. I hope the next representative can see this and show it to a manager to better understand your customers.
assistant: Hi [PERSON] (jaquan),Thank you for reaching out. I'm [PERSON] from the support team and I'm here to help you.We are sorry to hear that your order was missing some items and also a spilled drink. This is not the experience we expect you to have.Upon checking on our end, I'm pleased to inform you that we have already processed a total refund of $6.7 for the reported item, including tax and service fee. The refund was processed on Feb 8, 2024, Your refund will reflect in 3-5 business days.Thank you so much for your time and patience in sharing your concern with us. Thanks for using the app.
customer: Name(s) of item(s) affected: Iced caramel macchiato [PERSON] gum Share details: Iced caramel macchiato was spilled and leaking a mess everywhere as well as the gum was missing. Issue with food temperature (too hot/cold): falseDidn&#39;t like taste: falsePortion size too small: falsePackaging damaged, couldn&#39;t eat: falsePackaging damaged, could eat: falseDrink spilled: trueFood was stale/spoiled: [LOCATION] was overcooked/burnt: falseFood safety issue: false
customer: Share a photo of the item:
assistant: Hi JAY (JAQUAN),Were sorry to hear that you were unhappy with the quality of the food you received. After taking a look at your account, I can confirm weve already issued a refund for this order.Thank you for your patience.
customer: yes i seen you only refunded for the gum
customer: however my initial problem was with my drink.
customer: as you can see in the picture there is coffe spilled all in the bag and was drinking all through my apartment.
customer: About a quarter of the drink was spilled.
assistant: Hi [PERSON] (jaquan),Thank you for reaching out. I'm [PERSON] from the support team and I'm here to help you.We apologize for the inconvenience caused as your received order was spilled.Upon checking on our end, I'm pleased to inform you that we have already processed a refund of $3.52 for the reported item, including tax and service fee. The refund was processed on Feb 7, 2024. Your refund will reflect in 3-5 business days.Thank you so much for your time and patience in sharing your concern with us. Thanks for using the app.
customer: Yes I received a refund of the 3$ HOWEVER in addition to the spilled drink I ALSO was missing my gum. Which was about 3$. So how much did I receive for the spilled drink and how much for my missing gum.
assistant: Hi [PERSON] (jaquan),Thanks for being a member. I'm [PERSON] from the support team and I'm here to help youWe are sorry for the inconvenience caused and understand your concern regarding the missing item.Since it wouldn't be fair on your part to pay for something you didn't receive, I'm glad to inform you that we've successfully processed a refund of $3.18 for the incorrect or missing, including tax and service fee. Your refund will reflect in 3-5 business days.We hope to offer a better experience on your future orders.Thanks for being a loyal member. It was indeed a pleasure assisting you. Wish you a wonderful day ahead.
customer: Are you not understanding.
customer: I am missing my GUM first thing. Which was about 3$!! In ADDITION my drink was SPILLED.
customer: So how am I receiving 3.18 refund for BOTH items
customer: That 3.18 is just for the gum!!
customer: What about the spilled drink
customer: Can you break it down and explain to me why I received 3.18 for BOTH OF THE ITEMS. And how much of the 3.18 is for EACH item.
customer: Ok. So I just checked and I see 2 refunds for about 3$. So why arent you saying that correctly. This is completely HORRIBLE service and HORRIBLE communication upon your end. Continuously I have support people reply with the SAME message and not address my issues correctly. Ive asked to be told which is for what and have been given basic messages not addressing what Im Fully saying. I hope the next representative can see this and show it to a manager to better understand your customers.
assistant: Hi [PERSON] (jaquan),Thank you for reaching out. I'm [PERSON] from the support team and I'm here to help you.We are sorry to hear that your order was missing some items and also a spilled drink. This is not the experience we expect you to have.Upon checking on our end, I'm pleased to inform you that we have already processed a total refund of $6.7 for the reported item, including tax and service fee. The refund was processed on Feb 8, 2024, Your refund will reflect in 3-5 business days.Thank you so much for your time and patience in sharing your concern with us. Thanks for using the app.
customer: Name(s) of item(s) affected: Iced caramel macchiato [PERSON] gum Share details: Iced caramel macchiato was spilled and leaking a mess everywhere as well as the gum was missing. Issue with food temperature (too hot/cold): falseDidn&#39;t like taste: falsePortion size too small: falsePackaging damaged, couldn&#39;t eat: falsePackaging damaged, could eat: falseDrink spilled: trueFood was stale/spoiled: [LOCATION] was overcooked/burnt: falseFood safety issue: false
customer: Share a photo of the item:
assistant: Hi JAY (JAQUAN),Were sorry to hear that you were unhappy with the quality of the food you received. After taking a look at your account, I can confirm weve already issued a refund for this order.Thank you for your patience.
customer: yes i seen you only refunded for the gum
customer: however my initial problem was with my drink.
customer: as you can see in the picture there is coffe spilled all in the bag and was drinking all through my apartment.
customer: About a quarter of the drink was spilled.
assistant: Hi [PERSON] (jaquan),Thank you for reaching out. I'm [PERSON] from the support team and I'm here to help you.We apologize for the inconvenience caused as your received order was spilled.Upon checking on our end, I'm pleased to inform you that we have already processed a refund of $3.52 for the reported item, including tax and service fee. The refund was processed on Feb 7, 2024. Your refund will reflect in 3-5 business days.Thank you so much for your time and patience in sharing your concern with us. Thanks for using the app.
customer: Yes I received a refund of the 3$ HOWEVER in addition to the spilled drink I ALSO was missing my gum. Which was about 3$. So how much did I receive for the spilled drink and how much for my missing gum.
assistant: Hi [PERSON] (jaquan),Thanks for being a member. I'm [PERSON] from the support team and I'm here to help youWe are sorry for the inconvenience caused and understand your concern regarding the missing item.Since it wouldn't be fair on your part to pay for something you didn't receive, I'm glad to inform you that we've successfully processed a refund of $3.18 for the incorrect or missing, including tax and service fee. Your refund will reflect in 3-5 business days.We hope to offer a better experience on your future orders.Thanks for being a loyal member. It was indeed a pleasure assisting you. Wish you a wonderful day ahead.
customer: Are you not understanding.
customer: I am missing my GUM first thing. Which was about 3$!! In ADDITION my drink was SPILLED.
customer: So how am I receiving 3.18 refund for BOTH items
customer: That 3.18 is just for the gum!!
customer: What about the spilled drink
customer: Can you break it down and explain to me why I received 3.18 for BOTH OF THE ITEMS. And how much of the 3.18 is for EACH item.
customer: Ok. So I just checked and I see 2 refunds for about 3$. So why arent you saying that correctly. This is completely HORRIBLE service and HORRIBLE communication upon your end. Continuously I have support people reply with the SAME message and not address my issues correctly. Ive asked to be told which is for what and have been given basic messages not addressing what Im Fully saying. I hope the next representative can see this and show it to a manager to better understand your customers.
assistant: Hi [PERSON] (jaquan),Thank you for reaching out. I'm [PERSON] from the support team and I'm here to help you.We are sorry to hear that your order was missing some items and also a spilled drink. This is not the experience we expect you to have.Upon checking on our end, I'm pleased to inform you that we have already processed a total refund of $6.7 for the reported item, including tax and service fee. The refund was processed on Feb 8, 2024, Your refund will reflect in 3-5 business days.Thank you so much for your time and patience in sharing your concern with us. Thanks for using the app.'''


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

def craft_chat_message(text):
    messages = [
        {
            "role": "system",
            "content": "Given a conversation, answer the question that follows."
        },
        {"role": "user", "content": text}
    ]
    return messages

questions = [
    "Write a summary of the above conversation?",
    "Did the assistant solve the customer problem?",
]

NUM_SAMPLES = 1000
BATCH_SIZE = 16
TOTAL_SAMPLES = [] 

for _ in range(NUM_SAMPLES//len(questions)):
    for question in questions:
        messages = craft_chat_message(convo + "\n\n" + question)
        formatted_chat_message = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        TOTAL_SAMPLES.append(formatted_chat_message)

print(len(TOTAL_SAMPLES))

outputs = []
tot_time = 0

if args.prefix_caching:
    model = LLM(model=MODEL_NAME, gpu_memory_utilization=0.9, enable_prefix_caching=True)
    print("Prefx caching enabled!")
else:
    model = LLM(model=MODEL_NAME, gpu_memory_utilization=0.9)
    print("No prefix caching!")

sampling_params = SamplingParams(temperature=0, max_tokens=300)

for _ in range(0, NUM_SAMPLES, BATCH_SIZE):
    samples = TOTAL_SAMPLES[_: _+BATCH_SIZE]
    start_time = time.time()
    llm_outputs = model.generate(samples, sampling_params)
    end_time = time.time()
    tot_time += (end_time - start_time)
    output_texts = [o.outputs[0].text for o in llm_outputs]
    outputs.extend(output_texts)

print(len(outputs))
print(outputs[0:len(questions)])
print(f"Time per request: {tot_time/NUM_SAMPLES:.3f}s")
print(f"Number of requests in one second: {1/(tot_time/NUM_SAMPLES):.3f}")

# Batch Size - 16 (fixed)
# 900 token prefix
# Time per request: 0.124s
# Number of requests in one second: 8.055

# Prefix Caching 
# 900 token prefix
# Batch Size - 16 (fixed)
# Time per request: 0.093s
# Number of requests in one second: 10.776

# Batch Size - 16 (fixed)
# 3000 token prefix
# Time per request: 0.216s
# Number of requests in one second: 4.620

# Prefix Caching 
# 900 token prefix
# Batch Size - 16 (fixed)
# Time per request: 0.143s
# Number of requests in one second: 6.977