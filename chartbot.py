import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize
import json
import numpy as np

# -------------------------------
# 1. NLTK Download
# -------------------------------
nltk.download('punkt')

# -------------------------------
# 2. Load Dataset
# -------------------------------
with open('dataset.json', 'r') as f:
    data = json.load(f)

all_words = []
xy = []

for item in data:
    question = item['question']
    answer = item['answer']
    tokens = word_tokenize(question.lower())
    all_words.extend(tokens)
    xy.append((tokens, answer))

all_words = sorted(set(all_words))

# -------------------------------
# 3. Helper function for bag-of-words
# -------------------------------
def bag_of_words(tokenized_sentence, words):
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1
    return bag

# -------------------------------
# 4. Neural Network Model
# -------------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# -------------------------------
# 5. Prepare training data
# -------------------------------
X_train = []
y_train = []

for (pattern_sentence, answer) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(answer)

X_train = np.array(X_train)
answers_list = list(sorted(set(y_train)))
answer_to_idx = {a: i for i, a in enumerate(answers_list)}
idx_to_answer = {i: a for a, i in answer_to_idx.items()}
y_train = np.array([answer_to_idx[a] for a in y_train])

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()

# -------------------------------
# 6. Training the model
# -------------------------------
input_size = len(all_words)
hidden_size = 16
output_size = len(answers_list)
learning_rate = 0.001
num_epochs = 500

model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# -------------------------------
# 7. Save trained model and metadata
# -------------------------------
torch.save(model.state_dict(), "model.pth")
np.save("words.npy", all_words)
with open("answers.json", "w") as f:
    json.dump(idx_to_answer, f)

# -------------------------------
# 8. Chat function
# -------------------------------
def chat():
    print("Start chatting! Type 'quit' to exit")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break
        sentence_tokens = word_tokenize(sentence.lower())
        bag = bag_of_words(sentence_tokens, all_words)
        bag = torch.from_numpy(bag).float()
        output = model(bag)
        _, predicted_idx = torch.max(output, dim=0)
        answer = idx_to_answer[predicted_idx.item()]
        print("Bot:", answer)

chat()
