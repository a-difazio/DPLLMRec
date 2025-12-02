import os
import argparse
from collections import Counter
from tqdm import tqdm
from model import SASRec
from utils import *
from types import SimpleNamespace


def setup_generation(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Directory {args.dataset_dir} not found.")

    args_path = os.path.join(args.dataset_dir, 'args.txt')

    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.txt not found in {args_path} directory.")

    model_path = os.path.join(args.dataset_dir, args.model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}.")

    args_dict = {}
    with open(args_path, 'r') as f:
        for line in f:
            line = line.strip()
            arg, val = line.split(',')
            if val.isdigit():
                val = int(val)
            elif val.replace('.', '', 1).isdigit():
                val = float(val)
            args_dict[arg] = val
    f.close()

    merged = {**args_dict, **vars(args)}
    args = SimpleNamespace(**merged)

    return args, model_path


def top_p(probs, p):
    # take elements that sums to p

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)

    mask = cum <= p

    mask[mask.sum()] = True

    filtered = torch.zeros_like(probs)
    filtered[sorted_idx[mask]] = probs[sorted_idx[mask]]

    filtered = filtered / filtered.sum()

    return filtered


def top_k(probs, k):
    # take top k elements
    topk_vals, topk_idx = torch.topk(probs, k)

    filtered = torch.zeros_like(probs)
    filtered[topk_idx] = topk_vals

    filtered = filtered / filtered.sum()

    return filtered


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='ml-1m_original')
    parser.add_argument('--model_file', required=True)
    parser.add_argument('--temperature', type=float, default=1.2)
    parser.add_argument('--penalty', type=float, default=0.95)
    parser.add_argument('--topp', type=float, default=0.9)
    parser.add_argument('--topk', type=float, default=50)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args, model_path = setup_generation(args)

    device = torch.device(args.device)

    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    model = SASRec(usernum, itemnum, args)
    model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    items_indices = np.array(range(1, itemnum+1))

    output_path = os.path.join(args.dataset_dir, 'generated.txt')

    output = open(output_path, 'w')
    output.write(f'user,item\n')
    print(f'Starting generation.')
    print(f'Saving result in: {output_path}')

    with torch.no_grad():
        for user, seq in tqdm(user_train.items()):
            print(f"User {user}\n")
            #if user == 4:
                #break

            prompt = seq[:]
            prompt_len = len(prompt)
            generated_sequence = []
            gen_len = min(prompt_len, args.maxlen)

            for _ in range(gen_len):

                # padding
                seq_input = np.zeros([args.maxlen], dtype=np.int32)
                cut = prompt[-args.maxlen:]
                seq_input[-len(cut):] = cut

                # predict (output dimension [1, num_items])
                logits = model.predict(np.array([user]), np.array([seq_input]), items_indices)
                logits = logits[0]

                # sample
                probs = torch.softmax(logits / args.temperature, dim=-1)

                # anti-repetition penalty
                for item, count in Counter(generated_sequence).items():
                    probs[item - 1] *= (args.penalty ** count)

                probs = probs / probs.sum()


                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                print("\nentropy: ", entropy.mean().item())

                pmax, idx = probs.max(dim=-1)
                print("max:", pmax.item(), "item:", idx.item())

                top_vals, top_idx = torch.topk(probs, k=10, dim=-1)
                for i in range(10):
                    print(f"{i + 1}: item={top_idx[i].item()}  p={top_vals[i].item():.4f}")

                next_item_idx = torch.multinomial(probs, num_samples=1).item()
                next_item = items_indices[next_item_idx].item()

                generated_sequence.append(next_item)
                prompt.append(next_item)
                output.write(f'{user},{next_item}\n')

            print(f'Original {seq}')
            print(f'Generated {generated_sequence}')

    output.close()
    print(f'Generation completed.')
