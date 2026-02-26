import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os

token = "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"
if token:
    login(token=token)
else:
    print("Attenzione: Token Hugging Face non fornito. L'accesso a modelli privati potrebbe fallire.")

MODEL_NAME = "google/gemma-3-1b-it"

get_gemma_prompt = lambda private_text: f"<bos><start_of_turn>user\nGenerate exactly one sentence similar to: {private_text}\n<end_of_turn>\n<start_of_turn>model\nOutput:"

try:
    print("Caricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer pronto.")
    print("Caricamento modello...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,  # Utilizza bfloat16 per efficienza su hardware compatibile (es. GPU moderne)
        device_map="mps",      # Target GPU Apple Silicon; usa "cuda" per NVIDIA, "cpu" per CPU
        attn_implementation="sdpa" # Meccanismo di attenzione ottimizzato (Scaled Dot Product Attention)
    )
    print("Modello pronto.")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        print("Avviso: ID del token EOS non trovato per questo tokenizer. La generazione potrebbe non fermarsi naturalmente.")
except Exception as e:
    print(f"Errore durante il caricamento del modello o del tokenizer: {e}")
    print("Assicurati di avere accesso al modello su Hugging Face e che il tuo ambiente sia configurato correttamente.")


def generate_shared_sequence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int = 50,
    tau: float = 1.0,
    eos_token_id: int = None
) -> list[str]:
    if not prompts:
        return []

    input_data_batch = []
    for prompt in prompts:
        encoded_input = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_data_batch.append(encoded_input)

    generated_tokens_count = 0
    print(f"\nInizio della generazione condivisa per {len(prompts)} prompt...")
    print(f"Max nuovi token: {max_new_tokens}, ID Token EOS: {eos_token_id}")

    while generated_tokens_count < max_new_tokens:
        step_logits_list = []

        for input_data in input_data_batch:
            with torch.no_grad():
                generation_output = model.generate(**input_data, max_new_tokens=1, return_dict_in_generate=True,
                                                   output_logits=True, do_sample=False, num_beams=1)
                logits_for_next_token = generation_output.logits[0].squeeze(0)
                step_logits_list.append(logits_for_next_token)

        stacked_logits = torch.stack(step_logits_list, dim=0)

        mean_logits = torch.mean(stacked_logits, dim=0)

        # sampled_token_tensor = torch.argmax(mean_logits)
        # sampled_token_id = sampled_token_tensor.item()

        probabilities = torch.softmax(mean_logits / tau, dim=0)

        sampled_token_tensor = torch.multinomial(probabilities, num_samples=1)
        sampled_token_id = sampled_token_tensor.item()

        if eos_token_id is not None and (sampled_token_id == eos_token_id or sampled_token_id == 106):
            print(f"Token EOS ({eos_token_id}) generato. Interruzione della generazione.")
            break

        for i in range(len(input_data_batch)):
            current_input_ids = input_data_batch[i]["input_ids"]
            current_attention_mask = input_data_batch[i]["attention_mask"]

            input_data_batch[i]["input_ids"] = torch.cat(
                [current_input_ids, sampled_token_tensor.view(1, 1).to(model.device)], dim=1
            )

            input_data_batch[i]["attention_mask"] = torch.cat(
                [current_attention_mask, torch.ones(1, 1, dtype=torch.long).to(model.device)], dim=1
            )

        generated_tokens_count += 1

    decoded_outputs = [
        tokenizer.decode(data["input_ids"][0], skip_special_tokens=True)
        for data in input_data_batch
    ]

    return decoded_outputs


if __name__ == "__main__":

    initial_batch_prompts = [
        get_gemma_prompt("I am not  a computer scientist"),
        get_gemma_prompt("I am Italian"),
        get_gemma_prompt("Loving pizza!!!")
    ]

    generated_texts = generate_shared_sequence(
        model=model,
        tokenizer=tokenizer,
        prompts=initial_batch_prompts,
        max_new_tokens=50, # Genera fino a 20 nuovi token per una dimostrazione più rapida
        eos_token_id=eos_token_id
    )

    print("\n--- Testi Generati (Strategia del Token Condiviso) ---")
    for i, text in enumerate(generated_texts):
        # print(f"Prompt {i+1}: {initial_batch_prompts[i]}")
        print(f"{text}\n")

