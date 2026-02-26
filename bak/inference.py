import os
import torch
import argparse
import numpy as np
import sys

from model import SASRec
from utils import data_partition, build_index, evaluate, evaluate_valid


def load_args_from_file(args_file_path):
    # Crea un oggetto Namespace vuoto
    loaded_args = argparse.Namespace()
    with open(args_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value_str = line.split('=', 1) # Assumendo il formato key=value
            # Tenta di convertire il valore al tipo corretto
            try:
                if value_str.lower() == 'true':
                    value = True
                elif value_str.lower() == 'false':
                    value = False
                elif '.' in value_str:
                    value = float(value_str)
                else:
                    value = int(value_str)
            except ValueError:
                value = value_str # Se non è numero o booleano, lascialo come stringa

            setattr(loaded_args, key, value)
    return loaded_args

def main_inference():
    parser = argparse.ArgumentParser(description="Inferenza con modello SASRec")
    parser.add_argument('--model_dir', required=True, help="Directory contenente il modello e args.txt (es. ml-1m_output_test_run_lr_001)")
    # Aggiungi qui eventuali argomenti specifici per l'inferenza (es. --user_id, --num_recommendations)
    inference_args = parser.parse_args()

    # Percorsi dei file
    args_file_path = os.path.join(inference_args.model_dir, 'args.txt')
    model_checkpoint_path = os.path.join(inference_args.model_dir, 'SASRec.epoch=X.lr=Y.pth') # Devi sapere il nome del checkpoint esatto

    # 1. Carica gli argomenti di training
    train_args = load_args_from_file(args_file_path)
    print(f"Loaded training arguments: {train_args}")

    # Qui dovresti caricare usernum e itemnum usati durante il training
    # Di solito, questi sono anche salvati o derivabili dal dataset.
    # Per semplicità, li passiamo come esempio, ma idealmente dovrebbero provenire
    # dal contesto del dataset o dal file di configurazione.
    # Ad esempio, potresti aggiungere usernum e itemnum al tuo args.txt
    usernum = getattr(train_args, 'usernum', 1000) # Fallback se non è nel file
    itemnum = getattr(train_args, 'itemnum', 5000) # Fallback se non è nel file


    # 2. Inizializza il modello con gli argomenti di training
    model = SASRec(usernum, itemnum, train_args) # Assicurati che il costruttore di SASRec accetti l'oggetto args
    model.to(train_args.device)

    # 3. Carica i pesi del modello
    if not os.path.exists(model_checkpoint_path):
        print(f"Error: Model checkpoint not found at {model_checkpoint_path}. Please specify the correct filename.")
        # Potresti aggiungere logica per trovare l'ultimo checkpoint o un checkpoint specifico
        return

    model.load_state_dict(torch.load(model_checkpoint_path, map_location=torch.device(train_args.device)))
    model.eval() # Metti il modello in modalità valutazione

    print(f"Model loaded successfully from {model_checkpoint_path}")

    # Ora puoi usare 'model' per fare inferenza
    # Esempio:
    # user_id = inference_args.user_id # Se hai aggiunto un argomento per l'utente
    # recommendations = model.predict(user_id, ...)
    # print(f"Recommendations for user {user_id}: {recommendations}")

if __name__ == '__main__':
    main_inference()


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--state_dict_path', required=True, type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='mps', type=str)
parser.add_argument('--norm_first', action='store_true', default=False)
parser.add_argument('--topk', default=10, type=int)

args = parser.parse_args()

if __name__ == '__main__':

    u2i_index, i2u_index = build_index(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = data_partition(args.dataset)

    model = SASRec(usernum, itemnum, args).to(args.device)

    # Carica i pesi addestrati
    try:
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        print(f"Model successfully loaded from {args.state_dict_path}")
    except Exception as e:
        print(f"Error loading model from {args.state_dict_path}: {e}")
        sys.exit(1)  # Esci se il caricamento del modello fallisce

    model.eval()  # Imposta il modello in modalità valutazione (disabilita dropout, batchnorm, ecc.)

    print("\n--- Valutazione completa sul set di test ---")
    # Le funzioni `evaluate` e `evaluate_valid` provengono da utils.py
    t_test = evaluate(model, [user_train, user_valid, user_test, usernum, itemnum], args)
    print('Test (NDCG@%d: %.4f, HR@%d: %.4f)' % (args.topk, t_test[0], args.topk, t_test[1]))

    print("\n--- Esempio di Inferenza per un utente specifico ---")
    # Troviamo un utente di esempio con una storia di training valida
    example_user_id = -1
    for u in user_train.keys():
        if len(user_train[u]) > 0:
            example_user_id = u
            break

    if example_user_id == -1:
        print("Nessun utente con dati di training trovato per l'inferenza di esempio.")
        sys.exit(0)

    print(f"Usando ID utente di esempio: {example_user_id}")

    # Costruisci la sequenza storica dell'utente per la predizione
    # Include gli elementi di train e, se presente, l'elemento di validazione come parte della storia.
    full_sequence_for_prediction = list(user_train[example_user_id])
    if len(user_valid[example_user_id]) > 0:
        full_sequence_for_prediction.append(user_valid[example_user_id][0])

    # Pad o tronca la sequenza a maxlen
    seq = np.zeros([args.maxlen], dtype=np.int32)
    start_index = max(0, len(full_sequence_for_prediction) - args.maxlen)
    seq[args.maxlen - len(full_sequence_for_prediction[start_index:]):] = full_sequence_for_prediction[start_index:]

    print(f"Storia utente {example_user_id} (ultimi {args.maxlen} elementi): {seq[np.where(seq != 0)]}")

    # Genera gli item candidati per la predizione: tutti gli item eccetto il padding (0)
    candidate_items = np.arange(1, itemnum + 1)

    with torch.no_grad():  # Non c'è bisogno di calcolare i gradienti durante l'inferenza
        # Il modello si aspetta dimensioni batch, quindi wrappa utente singolo/sequenza in liste/array
        user_ids_batch = np.array([example_user_id])
        seq_batch = np.array([seq])
        item_indices_batch = candidate_items  # Tutti gli item come candidati

        # Predici i logits per tutti gli item candidati
        logits = model.predict(user_ids_batch, seq_batch, item_indices_batch)

        # Ordina gli item per logits in ordine decrescente
        sorted_item_indices = torch.argsort(logits[0], descending=True).cpu().numpy()

        # Mappa agli ID item reali
        recommended_item_ids = candidate_items[sorted_item_indices]

        print(f"\nTop {args.topk} raccomandazioni per l'utente {example_user_id}:")
        # Prepara un set di item già visti per contrassegnarli
        already_seen_items = set(full_sequence_for_prediction)

        for i in range(args.topk):
            item_id = recommended_item_ids[i]
            seen_status = "(visto)" if item_id in already_seen_items else ""
            print(f"{i + 1}. Item {item_id} {seen_status}")

    print("\n--- Esempio di Generazione Autoregressiva (Predizione del prossimo item) ---")
    # Questo dimostra come predire il *singolo prossimo* item più probabile
    # e poi aggiungerlo alla sequenza per predire il successivo.

    current_seq_for_autoregressive = list(user_train[example_user_id][-5:])  # Usa gli ultimi 5 item per iniziare
    if len(user_valid[example_user_id]) > 0:
        current_seq_for_autoregressive.append(user_valid[example_user_id][0])  # Aggiungi anche l'item di validazione

    print(f"Sequenza di partenza autoregressiva: {current_seq_for_autoregressive}")

    num_generations = 3  # Quanti item aggiuntivi vogliamo generare
    for _ in range(num_generations):
        seq_ar = np.zeros([args.maxlen], dtype=np.int32)
        start_index_ar = max(0, len(current_seq_for_autoregressive) - args.maxlen)
        seq_ar[args.maxlen - len(current_seq_for_autoregressive[start_index_ar:]):] = \
            current_seq_for_autoregressive[start_index_ar:]

        # Predici il prossimo item usando tutti gli item possibili come candidati
        logits_ar = model.predict(np.array([example_user_id]), np.array([seq_ar]), candidate_items)
        next_item_id = candidate_items[torch.argmax(logits_ar[0]).item()]  # Prendi l'item con il logit più alto

        current_seq_for_autoregressive.append(next_item_id)
        if len(current_seq_for_autoregressive) > args.maxlen:
            current_seq_for_autoregressive.pop(0)  # Mantieni la lunghezza della sequenza entro maxlen

        print(f"  Predetto il prossimo item: {next_item_id}. Nuova sequenza: {current_seq_for_autoregressive}")

    print("\nInferenza completata.")