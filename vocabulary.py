import spacy
from collections import Counter
import json

spacy_eng = spacy.load('en_core_web_sm')

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
      
    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        try:
          return [token.text.lower() for token in spacy_eng.tokenizer(text)]
        except:
            print(text)

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in self.tokenize(text)
        ]

    def decode(self, tokens):
        words = [self.itos[token] for token in tokens if token != self.stoi["<PAD>"]]
        return words

    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.itos, f, indent=4)

    def load_vocab(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.itos = {int(k): v for k, v in data.items()}
        self.stoi = {v: k for k, v in self.itos.items()}
        
        return data