import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_31 = "meta-llama/Llama-3.1-8B-Instruct"  # <-- llama 3.1

tokenizer = AutoTokenizer.from_pretrained(llama_31)
model = AutoModelForCausalLM.from_pretrained(
    llama_31,
    torch_dtype=torch.bfloat16,
).to(device)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

prompt = [{"role": "system", "content": "You are a helpful assistant, that responds as a pirate."}]

while True:
    print("Enter your prompt: ")
    entered_prompt = input()
    if entered_prompt.lower() == "end":
        break

    prompt.append({"role": "user", "content": entered_prompt})


    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.no_grad():
        generation = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id,
        )


    new_tokens = generation[0][inputs["input_ids"].shape[-1]:]
    assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    prompt.append({"role": "assistant", "content": assistant_response})
    print(assistant_response)