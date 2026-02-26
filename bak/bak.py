def generate_synthetic_data_1(args):

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    text_generator = TextGenerator(args.model_name)


    # setting dataloader,tokenizer, model
    dataloader = get_dataloader(args.input_file, args.batch_size, False, args.num_workers)

    # Tokenizer loading and setting
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = 'left'

    # Model loading and setting
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation="sdpa"
    )

    model.eval()


    # compute public logits
    public_logits = compute_public_logits(public_prompt, model, tokenizer)


    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
        batch_with_prompt = [f"<bos><start_of_turn>user\nGenerate exactly one sentence similar to: {s}\n<end_of_turn>\n<start_of_turn>model\nOutput:" for s in batch]

        encoded = tokenizer(
            batch_with_prompt,
            return_tensors="pt",
            padding='longest'
        )

        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        for t in tqdm(range(args.max_token), desc=f"Batch {i+1}/{len(dataloader)} - Generating tokens"):

            # genero un token per ogni prompt
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_logits=True,
                    eos_token_id=tokenizer.eos_token_id
                )

            # logits dell'ultimo token per ciascun prompt o [:, -1, :] -> se dovessi aumentare max new tokens
            logits = generation_output.logits[0] # shape [batch_size, vocab_size]

            # Media dei logits su tutti i prompt
            logits_mean = torch.mean(logits, dim=0, keepdim=True)  # shape [1, vocab_size]

            if random.uniform(0, 1) < 0:
                probs = torch.softmax(public_logits / args.tau_p, dim=-1)
            else:
                # softmax con temperatura
                probs = torch.softmax(logits_mean / args.tau, dim=-1) # controllare se dim=-1

            #  campiona un token (stesso token per tutti i prompt)
            sampled_token = torch.multinomial(probs, num_samples=1) # shape [1, 1]

            # print(f"Step {t}: sampled_token={sampled_token.item()}, P(EOS)={probs[0, tokenizer.eos_token_id].item()}")

            # faccio break anche se trovo il ., non solo EOS (alcuni modelli non ce l'hanno)
            if sampled_token.item() == tokenizer.eos_token_id or sampled_token.item() == 106:
                tqdm.write(
                    f"\n[Batch {i + 1}] Stop token '{tokenizer.decode([sampled_token.item()])}' (ID: {sampled_token.item()}) generated after {t + 1} token.\n")
                break

            # concateno il token generato a tutti i prompt
            input_ids = torch.cat([input_ids, sampled_token.repeat(input_ids.size(0), 1)], dim=1)

            # aggiorno l'attention mask
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0)), 1, dtype=attention_mask.dtype).to(model.device)], dim=1)

        prompt = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
        generated = tokenizer.decode(input_ids[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)

        print(prompt)
        print(generated + "\n")

        with open(args.output_file, "a", encoding="utf-8") as f:
            f.write(generated.strip() + "\n")


# non generando un token speciale come fine o genero fino al max e tronco a <end_of_turn>
# oppure mi fermo al punto, come faccio ora

def compute_public_logits(prompt, model, tokenizer):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding='longest'
    )

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    generation_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_logits=True,
        eos_token_id=tokenizer.eos_token_id
    )

    return generation_output.logits[0]


public_prompt = "Generate a short phrase."