import pandas as pd
import pickle


users_to_print = [1, 2]

def print_sequence(user, sequence, movies, id2item):
    print(f"User {user}")

    for sasrec_id in sequence:
        real_id = id2item[sasrec_id]
        try:
            title = movies.loc[real_id]['title']
            genre = movies.loc[real_id]['genres']
            print(f"{sasrec_id:<8} | {real_id:<8} | {title[:40]:<40} | {genre}")
        except:
            print(f"{sasrec_id:<8} | {real_id:<8} | Title Not Found")


movies = pd.read_csv('../ml-1m/movies.dat', sep='::', names=['id', 'title', 'genres'], engine='python', encoding='latin-1')
movies.set_index('id', inplace=True)

with open('../ml-1m/item_mapping.pkl', 'rb') as f:
    id2item = pickle.load(f)

orig_dataset = pd.read_csv('../ml-1m/ml-1m.txt', sep=" ", names=['user', 'item'])
gen_dataset = pd.read_csv('../SASRec/ml-1m_mapping/generated_no_repetition.txt', sep=',', names=['user', 'item'])

orig_dict = orig_dataset.groupby('user')['item'].apply(list).to_dict()
gen_dict = gen_dataset.groupby('user')['item'].apply(list).to_dict()

common_users = sorted(gen_dict.keys() & orig_dict.keys())

for user in users_to_print:
    if user in common_users:
        print("\n--- Original ---\n")
        print_sequence(user, orig_dict[user], movies, id2item)

        print("\n--- Generated ---\n")
        print_sequence(user, gen_dict[user], movies, id2item)

