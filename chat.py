import streamlit as st
import random
import json
import torch
from model import NNModel
from utils import bag_of_words, tokenize
from streamlit_chat import message

st.title("BOTHEAD")
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

model = NNModel(input_size, output_size, hidden_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'BOTHEAD'

st.write("let's chat. Type quit to exit.")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
i = 0

get_data = lambda i: st.text_input(label='You: ', key=str(i))

def getResponse(sentence):
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
                return (f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        return f"{bot_name}: I do not understand"
user_input = get_data(i)

if user_input:
    if user_input == 'quit':
        for intent in intents['intents']:
            if 'goodbye' == intent['tag']:
                message(f"{bot_name}: {random.choice(intent['responses'])}")
                break
        st.session_state['generated'] = []
    else:
        out = getResponse(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(out)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i)+'_')
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
