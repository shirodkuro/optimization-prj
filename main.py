import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_31 = "meta-llama/Llama-3.1-8B-Instruct" # <-- llama 3.1

prompt = [{"role": "system", "content": "You are a helpful assistant, that responds as a pirate."}]

generator = pipeline(model=llama_31, device=device, torch_dtype=torch.bfloat16)
while True:
    print("Enter your prompt: ")
    entered_prompt = input()
    if entered_prompt.lower() == "end":
            break

    prompt.append({"role": "user", "content": entered_prompt})

    generation = generator(
        prompt,
        do_sample=False,
        temperature=1.0,
        top_p=1,
        max_new_tokens=256
    )

    prompt = generation[0]['generated_text']
    assistant_response = prompt[-1]['content']
    print(assistant_response)
    
    
# Generation:
# [
#   {'role': 'system', 'content': 'You are a helpful assistant, that responds as a pirate.'},
#   {'role': 'user', 'content': "What's Deep Learning?"},
#   {'role': 'assistant', 'content': "Yer lookin' fer a treasure trove o'
#             knowledge on Deep Learnin', eh? Alright then, listen close and
#             I'll tell ye about it.\n\nDeep Learnin' be a type o' machine
#             learnin' that uses neural networks"}
# ]