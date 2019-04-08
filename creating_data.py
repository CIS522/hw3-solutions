import pandas as pd
import math

all_data_df = pd.read_csv('./original_data/sarcasm/train-balanced-sarcasm.csv')
num_data = len(all_data_df)  # 1010826 over one million sentences

shuffled_data_df = all_data_df[['label', 'comment', 'parent_comment']].sample(frac=1).reset_index(drop=True)
print(shuffled_data_df['label'].head())
print(shuffled_data_df['comment'].head())

train_df = shuffled_data_df.loc[:math.floor(num_data * 0.7), ]
test_df = shuffled_data_df.loc[math.floor(num_data * 0.7):, ].reset_index(drop=True)

train_df.to_csv('./data/reddit_train.csv')
test_df.to_csv('./data/reddit_test.csv')
