# benchmark: 72%
# https://www.kaggle.com/kashnitsky/a4-demo-sarcasm-detection-with-logit-solution

import torch
import torchtext
import torchtext.data as data
import spacy
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim

tick = time.time()
print("hello")

spacy_en = spacy.load('en')

print("loaded spacy")
print(time.time() - tick)

spacy.prefer_gpu()


# python -m spacy download en


def tokenizer(text):  # create a tokenizer function
    # print("-START-")
    # print(text)
    # print([tok.text for tok in spacy_en.tokenizer(text)])
    # print("-END-")
    return [tok.text for tok in spacy_en.tokenizer(text)]


TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)

print("created fields")
print(time.time() - tick)

train, test = data.TabularDataset.splits(
    path='./data/', train='reddit_train.csv',
    test='reddit_test.csv', format='csv', skip_header=True,
    fields=[('index', None), ('label', LABEL), ('comment', TEXT), ('parent_comment', TEXT)])  # TODO: replace baby with reddit

print(train.__dict__['examples'][0].__dict__)

print("split train, test")
print(time.time() - tick)

TEXT.build_vocab(train, vectors="glove.6B.100d")
vocab = TEXT.vocab

print("build GloVE vocab")
print(time.time() - tick)

train_iter, test_iter = data.Iterator.splits(
    (train, test), sort_key=lambda x: len(x.comment),
    batch_sizes=(32, 256, 256), device=-1)

print("made iterators")
print(time.time() - tick)


class CNN1d(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()

        self.embed = nn.Embedding(len(vocab), embedding_dim)  # we used GLove 100-dim
        self.embed.weight.data.copy_(vocab.vectors)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]

        text = text.permute(1, 0)

        # text = [batch size, sent len]

        embedded = self.embed(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)

        # embedded = [batch size, emb dim, sent len]

        conved = [F.relu(conv(embedded)) for conv in self.convs]

        # conv_n = [batch size, n_filters, sent len - filter_sizes[n] - 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


cnn = CNN1d(100, 10, [5, 5, 5], 1, 0.5)

print("made CNN")
print(time.time() - tick)

optimizer = optim.Adam(cnn.parameters())

criterion = nn.BCEWithLogitsLoss()
#
# model = model.to(device)
# criterion = criterion.to(device)

batch_num = 0
for example in train_iter:
    batch_num += 1
    if batch_num % 100 == 0:
        print('Batch: ', batch_num, ', ', time.time() - tick)
    label = example.label
    comment = example.comment
    parent_comment = example.parent_comment

    prediction = torch.squeeze(cnn(comment), 1)
    optimizer.zero_grad()
    loss = criterion(prediction.float(), label.float())
    loss.backward()
    optimizer.step()
    # print(comment)

correct = 0
total = 0
with torch.no_grad():
    for example in test_iter:
        label = example.label
        comment = example.comment
        parent_comment = example.parent_comment
        # images, labels = images.to(device), labels.to(device)

        outputs = cnn(comment)
        prediction = torch.squeeze(cnn(comment), 1)
        total += label.size(0)
        correct += (torch.round(prediction.float()) == label.float()).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

print("trained CNN")
print(time.time() - tick)