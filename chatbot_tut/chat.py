import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words,tokenize,stem
with open("index.json","r") as f:
    intents=json.load(f)
#checking for GPU Support
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#simply if cuda is available for GPU support then we use it, otherwise we assign it to the cpu
FILE="data.pth"
data=torch.load(FILE)
#EXTRACTION OF VALUES FROM DATA variable  in which data is stored as a dictionary.
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval() # FOR EVALUATION OF MODEL

bot_name="Bot-pyrates"
print(f"hi I am chatbot named {bot_name},...... type 'quit' to exit")
while True:
    sentence = input("You : ")
    if sentence=="quit":
        break
    sentence=tokenize(sentence)
    X=bag_of_words(sentence, all_words)
    X=X.reshape(1,X.shape[0])
    X= torch.from_numpy(X).to(device)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)   #doubt
    tag=tags[predicted.item()]
    print(tag+ " this is tag ")
    probs = torch.softmax(output, dim=1)      #softmax function applied for probability
    prob = probs[0][predicted.item()]        #doubt
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print("I do not understand...")
                
                    


