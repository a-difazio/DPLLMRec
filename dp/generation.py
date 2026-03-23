from tqdm import tqdm
import torch
import random
from numpy.random import laplace
from utils import clip_logits, distance
from dataloader import get_dataloader
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == 'mps' else torch.float16


class TextGenerator:
    """
    Class for generating synthetic text from an input prompt.
    """
    def __init__(self, model_name, tokenizer_padding_side='left'):
        self.model_name = model_name
        self.tokenizer_padding_side = tokenizer_padding_side

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        Loads model and tokenizer
        """
        print(f"Loading model and tokenizer {self.model_name} on {DEVICE} with dtype {DTYPE}.")

        # Tokenizer loading and setting
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = self.tokenizer_padding_side

        # Model loading and setting
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            dtype=DTYPE,
            device_map=DEVICE,
            attn_implementation='sdpa'
        )

        self.model.eval()
        print(f"Loading complete.")

    def _prepare_batch(self, batch):

        batch_with_prompt = [
            (f"<bos><start_of_turn>user\nGenerate exactly one sentence similar to: {s}\n"
             f"<end_of_turn>\n<start_of_turn>model\nOutput:") for s in batch]

        encoded = self.tokenizer(
            batch_with_prompt,
            return_tensors="pt",
            padding='longest'
        )

        return encoded

    def generate(self, batch, args, batch_idx, total_batches):

        encoded_private_prompts = self._prepare_batch(batch)
        input_ids_private = encoded_private_prompts["input_ids"].to(self.model.device)
        attention_mask_private = encoded_private_prompts["attention_mask"].to(self.model.device)

        encoded_public_prompts = self.tokenizer(
            args.public_prompt,
            return_tensors="pt",
            padding='longest',
        ).to(self.model.device)

        input_ids_public = encoded_public_prompts["input_ids"].to(self.model.device)
        attention_mask_public = encoded_public_prompts["attention_mask"].to(self.model.device)

        noisy_threshold = args.theta + torch.tensor(laplace(0, args.epsilon),
                                                    device=self.model.device, dtype=self.model.dtype)

        t = 0

        # for t in tqdm(range(args.max_token), desc=f"Batch {batch_idx + 1}/{total_batches} - Generating tokens"):
        pbar = tqdm(total=args.max_private_token, desc=f"Batch {batch_idx + 1}/{total_batches} - Generating tokens")
        while t < args.max_private_token:

            with torch.no_grad():
                generation_output_private = self.model.generate(
                    input_ids=input_ids_private,
                    attention_mask=attention_mask_private,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_logits=True,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            logits_private = generation_output_private.logits[0]

            with torch.no_grad():
                generation_output_public = self.model.generate(
                    input_ids=input_ids_public,
                    attention_mask=attention_mask_public,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_logits=True,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            logits_public = generation_output_public.logits[0].squeeze(0)

            noisy_distance = distance(logits_private, logits_public) + torch.tensor(laplace(0, 2*args.epsilon),
                                                            device=self.model.device, dtype=self.model.dtype)

            if noisy_distance >= noisy_threshold:

                clipped_logits = clip_logits(logits_private, args.clipping_bound)
                mean_logits = clipped_logits.mean(dim=0, keepdim=True)

                probs = torch.softmax(mean_logits / args.tau_private, dim=-1)

                noisy_threshold = args.theta + torch.tensor(laplace(0, args.epsilon),
                                                            device=self.model.device, dtype=self.model.dtype)
                t += 1
                pbar.update(1)

            else:
                probs = torch.softmax(logits_public / args.tau_public, dim=-1)

            #  Sampling of one token (the same for all prompts)
            sampled_token = torch.multinomial(probs, num_samples=1)  # shape [1, 1]

            # Break if EOS
            if sampled_token.item() == self.tokenizer.eos_token_id or sampled_token.item() == 106:
                tqdm.write(
                    f"\n[Batch {batch_idx + 1}] Stop token '{self.tokenizer.decode([sampled_token.item()])}' "
                    f"(ID: {sampled_token.item()}) generated after {t + 1} token.\n")
                break

            # Concatenate the sampled token
            input_ids_private = torch.cat([input_ids_private, sampled_token.repeat(input_ids_private.size(0), 1)], dim=1)

            # Update the attention mask
            attention_mask_private = torch.cat(
                [attention_mask_private, torch.ones((attention_mask_private.size(0)), 1, dtype=attention_mask_private.dtype).to(self.model.device)],
                dim=1)

            # Concatenate the sampled token
            input_ids_public = torch.cat([input_ids_public, sampled_token.repeat(input_ids_public.size(0), 1)], dim=1)

            # Update the attention mask
            attention_mask_public = torch.cat(
                [attention_mask_public, torch.ones((attention_mask_public.size(0)), 1, dtype=attention_mask_public.dtype).to(self.model.device)],
                dim=1)

        prompt = self.tokenizer.decode(encoded_private_prompts["input_ids"][0], skip_special_tokens=True)
        generated = self.tokenizer.decode(input_ids_private[0, encoded_private_prompts["input_ids"].shape[1]:], skip_special_tokens=True)

        print(f"Original Batch Prompt Example: {prompt}")
        print(f"Generated (Batch {batch_idx + 1}): {generated}\n")

        return generated


def generate_synthetic_data(args):
    """
    Generate synthetic text from an input prompt dataset
    """

    # setting random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # initialize model and tokenizer
    text_generator = TextGenerator(args.model_name)

    # setting dataloader
    dataloader = get_dataloader(args.input_file, args.batch_size, False, args.num_workers)
    total_batches = len(dataloader)

    # generation and result saving
    with open(args.output_file, "a", encoding="utf-8") as f:
        for idx, batch in tqdm(enumerate(dataloader), total=total_batches, desc="Processing batches"):
            generated_text = text_generator.generate(batch, args, idx, total_batches)
            f.write(generated_text + "\n")
            f.flush()
