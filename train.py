"""
NLP preprocessing pipeline
concepts:
    Tokenization (splits sentence into tokens based on algorithm),
    lower + stemming (chop of ends of the words to get root word)
    exclude punctuation characters
    bag of words
"""

import json
import torch
# import string
import numpy as np
import torch.nn as nn
from model import NNModel
from utils import bag_of_words, stem, tokenize
from torch.utils.data import Dataset, DataLoader

with open('intents.json', 'r') as file:
    intents = json.load(file) # load json data as a dictionary

words = []
tags = []
A = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pat in intent['patterns']:
        word = tokenize(pat)
        words.extend(word)
        A.append((word, tag))

ignore = ['?','.',',','!']
words = sorted(set(stem(word) for word in words if word not in ignore))
tags = sorted(set(tags))

X_train, y_train = [], []

for pat, tag in A:
    bag = bag_of_words(pat, words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
# y_train = torch.tensor(y_train, dtype=torch.long) # The target should be a LongTensor using nn.CrossEntropyLoss (or nn.NLLLoss), since it is used to index the output logit (or log probability) for the current target class  (note the indexing in x[class])

# hyperparameters
batch_size = 8
hidden_size = 8
num_epochs = 1000
learning_rate = 0.001
output_size = len(tags)
input_size = len(X_train[0])

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()

# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
device = torch.device('cpu')
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
model = NNModel(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
print(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, )

for epoch in range(1, num_epochs + 1):
    for wds, labels in train_loader:
        wds = wds.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        # forward
        output = model(wds)
        loss = criterion(output, labels)
        # back propogation and optimise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}/{num_epochs}, loss={loss.item():.4f}')

print(f'Final loss : {loss.item():.4f}')

data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'words': words,
    'tags': tags
}

FILE = 'data.pth'
torch.save(data, FILE)
print(F'Complete, file saved to {FILE}')