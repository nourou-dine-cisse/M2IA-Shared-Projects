import argparse
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from gensim.models import KeyedVectors
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="lstm", type=str, help="lstm or cnn")
parser.add_argument("--embeddings", default="random", type=str, help="embeddings of TP1 (.bin file) or random")
parser.add_argument("--train", required=True, type=str, help="training file (conll format)")
parser.add_argument("--valid", required=True, type=str, help="validation file (conll format)")
parser.add_argument("--test", required=True, type=str, help="test file (conll format)")
parser.add_argument("--epochs", default=20, type=int, help="number of epochs")
parser.add_argument("--results", default="results_lstm_cnn.csv", type=str, help="file where the scores are written")
args = parser.parse_args()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

embedding_size = 100
hidden_size = 128
number_of_filters = 100
dropout = 0.5
batch_size = 32
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_conll(filename):
    sentences = []
    labels = []
    sentence = []
    sentence_labels = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            columns = line.split()
            if len(columns) == 5:
                # lowercase because the embeddings of TP1 were trained on lowercased text
                sentence.append(columns[1].lower())
                # seqeval only recognizes the prefixes B- and I- in uppercase
                sentence_labels.append(columns[4].upper())
            else:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    labels.append(sentence_labels)
                sentence = []
                sentence_labels = []
    if len(sentence) > 0:
        sentences.append(sentence)
        labels.append(sentence_labels)
    return sentences, labels


def build_vocabulary(all_sentences):
    word2id = {"<PAD>": 0}
    for sentence in all_sentences:
        for word in sentence:
            if word not in word2id:
                word2id[word] = len(word2id)
    return word2id


def build_label_vocabulary(all_labels):
    label2id = {}
    for sentence_labels in all_labels:
        for label in sentence_labels:
            if label not in label2id:
                label2id[label] = len(label2id)
    id2label = {}
    for label in label2id:
        id2label[label2id[label]] = label
    return label2id, id2label


def build_embedding_matrix(word2id, embeddings_file):
    matrix = np.random.normal(0, 0.1, (len(word2id), embedding_size))
    matrix[0] = np.zeros(embedding_size)
    found = 0
    if embeddings_file != "random":
        vectors = KeyedVectors.load_word2vec_format(embeddings_file, binary=True)
        for word in word2id:
            if word in vectors.key_to_index:
                matrix[word2id[word]] = vectors[word]
                found = found + 1
    print("Words with a vector from TP1:", found, "/", len(word2id) - 1)
    return torch.tensor(matrix, dtype=torch.float)


def make_batches(sentences, labels, word2id, label2id, shuffle):
    order = list(range(len(sentences)))
    if shuffle:
        random.shuffle(order)
    batches = []
    for start in range(0, len(order), batch_size):
        batch_indices = order[start:start + batch_size]
        lengths = []
        for i in batch_indices:
            lengths.append(len(sentences[i]))
        max_length = max(lengths)
        # the padding positions get the label -100, which is ignored by the loss
        words = np.zeros((len(batch_indices), max_length), dtype=np.int64)
        tags = np.full((len(batch_indices), max_length), -100, dtype=np.int64)
        for row in range(len(batch_indices)):
            i = batch_indices[row]
            for position in range(len(sentences[i])):
                words[row, position] = word2id[sentences[i][position]]
                tags[row, position] = label2id[labels[i][position]]
        batches.append((torch.tensor(words), torch.tensor(tags), torch.tensor(lengths)))
    return batches


class NerLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, number_of_labels, dropout):
        super(NerLSTM, self).__init__()
        self.name = "lstm"
        # freeze=True: the vectors of TP1 are not modified, so the scores only depend on them
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size, number_of_labels)

    def forward(self, words, lengths):
        x = self.embedding(words)
        # packing prevents the LSTM from reading the padding at the end of the short sentences
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=words.shape[1])
        x = self.dropout(x)
        x = self.fc(x)
        return x


class NerCNN(nn.Module):
    def __init__(self, embedding_matrix, number_of_filters, number_of_labels, dropout):
        super(NerCNN, self).__init__()
        self.name = "cnn"
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)
        # odd kernel sizes with padding = size // 2 keep one output vector per word
        self.conv3 = nn.Conv1d(embedding_matrix.shape[1], number_of_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embedding_matrix.shape[1], number_of_filters, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * number_of_filters, number_of_labels)

    def forward(self, words):
        x = self.embedding(words)
        # Conv1d expects the shape (batch, channels, length)
        x = x.transpose(1, 2)
        x3 = torch.relu(self.conv3(x))
        x5 = torch.relu(self.conv5(x))
        x = torch.cat([x3, x5], dim=1)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def predict(model, sentences, labels):
    batches = make_batches(sentences, labels, word2id, label2id, shuffle=False)
    true_labels = []
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for words, tags, lengths in batches:
            words = words.to(device)
            if model.name == "lstm":
                scores = model(words, lengths)
            else:
                scores = model(words)
            best = scores.argmax(dim=2).cpu()
            for row in range(words.shape[0]):
                true_sentence = []
                predicted_sentence = []
                for position in range(lengths[row]):
                    true_sentence.append(id2label[tags[row, position].item()])
                    predicted_sentence.append(id2label[best[row, position].item()])
                true_labels.append(true_sentence)
                predicted_labels.append(predicted_sentence)
    return true_labels, predicted_labels


train_sentences, train_labels = read_conll(args.train)
valid_sentences, valid_labels = read_conll(args.valid)
test_sentences, test_labels = read_conll(args.test)
print("Sentences: train", len(train_sentences), "| valid", len(valid_sentences), "| test", len(test_sentences))

word2id = build_vocabulary(train_sentences + valid_sentences + test_sentences)
label2id, id2label = build_label_vocabulary(train_labels + valid_labels + test_labels)
number_of_labels = len(label2id)
print("Vocabulary size:", len(word2id) - 1, "| number of labels:", number_of_labels)

embedding_matrix = build_embedding_matrix(word2id, args.embeddings)

if args.model == "lstm":
    model = NerLSTM(embedding_matrix, hidden_size, number_of_labels, dropout)
else:
    model = NerCNN(embedding_matrix, number_of_filters, number_of_labels, dropout)
model = model.to(device)
print(model)

optimizer = Adam(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss(ignore_index=-100)

best_f1 = -1
best_epoch = 0
best_state = None

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    train_batches = make_batches(train_sentences, train_labels, word2id, label2id, shuffle=True)
    for words, tags, lengths in train_batches:
        words = words.to(device)
        tags = tags.to(device)
        optimizer.zero_grad()
        if model.name == "lstm":
            scores = model(words, lengths)
        else:
            scores = model(words)
        loss = loss_function(scores.reshape(-1, number_of_labels), tags.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()

    valid_true, valid_predicted = predict(model, valid_sentences, valid_labels)
    valid_f1 = f1_score(valid_true, valid_predicted)
    print(f"Epoch {epoch + 1}/{args.epochs} | train loss {total_loss / len(train_batches):.4f} | valid F1 {valid_f1:.4f}")

    # we keep the model of the epoch with the best F1 on the validation set
    if valid_f1 > best_f1:
        best_f1 = valid_f1
        best_epoch = epoch + 1
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
test_true, test_predicted = predict(model, test_sentences, test_labels)
precision = precision_score(test_true, test_predicted)
recall = recall_score(test_true, test_predicted)
f1 = f1_score(test_true, test_predicted)

print("Best epoch on the validation set:", best_epoch)
print(f"Test | precision {precision:.4f} | recall {recall:.4f} | F1 {f1:.4f}")
print(classification_report(test_true, test_predicted, digits=4))

embeddings_name = args.embeddings.split("/")[-1].replace(".bin", "")
dataset_name = args.train.split("/")[-1]
with open(args.results, "a") as f:
    f.write(f"{dataset_name},{args.model},{embeddings_name},{best_epoch},{precision:.4f},{recall:.4f},{f1:.4f}\n")
