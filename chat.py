import random
import json
import torch
from model import Model
from utils import bag_of_words, tokenize

device = torch.device('cpu')

with open('intents.json', 'r') as file:
    intents = json.load(file)

FILE = 'data.pth'
data = torch.load(FILE)
input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
words = data['words']
tags = data['tags']
model_state = data['model_state']

model = Model(input_size, output_size, hidden_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'Eren'
print("let's chat. Type quit to exit.")

while True:
    sentence = input("You : ")
    if sentence == 'quit':
        break
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, prediction = torch.max(output, dim=1)
    tag = tags[prediction.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][prediction.item()]
    if prob.item() > 0.70:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name} : I do not understand")