import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from huggingface_hub import login

token = "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"
login(token=token)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    dtype=torch.bfloat16,
    device_map="mps",
    attn_implementation="sdpa"
)


synthetic_dataset = []

# Per ora lavoro su un singolo batch, poi va esteso facendo un for su batch disgiunti del dataset

batch = ["Generate a text similar to: I like spaghetti", "Generate a text similar to: I am italian", "Generate a text similar to: My name is Antonio"]
input_ids_batch = []
for prompt in batch:
    input_ids_batch.append(tokenizer(prompt, return_tensors="pt").to(model.device))

t = 0

# riga 6
while t < 10:
    # riga 7
    synthetic_prompt = []

    # TODO: manca la gestione di EOS

    # riga 9
    Z = []
    for input_ids in input_ids_batch:
        generation_output = model.generate(**input_ids, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)
        logits = generation_output.logits[0]
        Z.append(logits)
    Z = torch.stack(Z).to(model.device)
    Z = Z.squeeze()

    # salto righe 10-12

    # riga 13
    TAU = 1
    # TODO: clipping - capire se il max è su tutti gli elementi o sugli elementi dello stesso vettore
    z_mean = torch.mean(Z, dim=0)
    z_soft = torch.softmax(z_mean / TAU, dim=0)

    sampled_token = torch.multinomial(z_soft, num_samples=1)
    t = t + 1

    # ora sostanzialmente faccio la concatenazione
    for input_ids in input_ids_batch:
        input_ids["input_ids"] = torch.cat([input_ids["input_ids"],
                                            torch.full((input_ids["input_ids"].size(0), 1), sampled_token[0]).to(
                                                model.device)], dim=1)
        input_ids["attention_mask"] = torch.cat([input_ids["attention_mask"],
                                                 torch.ones(input_ids["attention_mask"].size(0), 1,
                                                            dtype=input_ids["attention_mask"].dtype).to(
                                                     model.device)], dim=1)

tanto_sono_tutti_uguali = tokenizer.decode(input_ids_batch[0]['input_ids'][0], skip_special_tokens=True)
print(tanto_sono_tutti_uguali)