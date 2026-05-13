import pandas as pd


train = pd.read_csv('amazon_music_tsts/amazon_music.tsv', header=None, sep='\t')
test = pd.read_csv('amazon_music_tsts/amazon_music_test.tsv', header=None, sep='\t')

train_seq = train.groupby(0)[1].apply(list)
test_seq = test.groupby(0)[1].apply(list)

full_seq = train_seq.combine(test_seq, lambda x, y: x + y)

df = full_seq.explode().reset_index()

df.to_csv('amazon_music_tsts/amazon_music_tsts.tsv', header=False, index=False, sep='\t')