from os import listdir
from collections import Counter
from string import punctuation
import string
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import torch
from torch import nn


# Cleaning reviews for embedding

# filter out words in reviews that are not in the vocab file
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# load the vocabulary
vocab_filename = '/Dataset/vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# turn a doc into clean tokens
def clean_doc1(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens


# load all docs in a directory
def process_docs1(directory, vocab, is_trian):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc1(doc, vocab)
		# add to list
		documents.append(tokens)
	return documents
 
# load all training reviews
positive_docs = process_docs1('/Dataset/txt_sentoken/neg', vocab, True)
negative_docs = process_docs1('/Dataset/txt_sentoken/pos', vocab, True)
train_docs = negative_docs + positive_docs

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)


# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)


# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])



## Building model

TXtrain = torch.from_numpy(Xtrain)
Tytrain = torch.from_numpy(ytrain)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
      super(NeuralNetwork, self).__init__()
      self.embd = nn.Embedding(vocab_size, 100, max_norm=True)
      self.conv = nn.Conv1d(100, 32, 8)
      self.pool = nn.MaxPool1d(2)
      self.dens1 = nn.Linear(32, 10)
      self.dens2 = nn.Linear(10, 1)
      self.act1 = nn.ReLU()
      self.act2 = nn.Sigmoid()

    def forward(self, x):
      x= self.embd(x)
      x = torch.permute(x, (1, 0))
      x = self.act1(self.conv(x))
      x = self.pool(x)
      x = x.view(x.size(0), -1)
      x = torch.permute(x, (1, 0))
      x = self.act1(self.dens1(x))
      x = self.act2(self.dens2(x))
      return x[0,0]

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.BCELoss()
optimizer1 = torch.optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=5, gamma=0.1)

def train(model, loss_fn, optimizer):
  nEpochs = 50
  model.train()
  model.to(device)
  for epoch in range (0,nEpochs):
    for i in range (0, TXtrain.shape[0]):
      model = model.train()
      X = TXtrain[i,:].to(device)
      Y = Tytrain[i].to(device)
      optimizer.zero_grad()
      pred = model(X)
      Y = Y.to(torch.float32)
      loss = loss_fn(pred,Y)
      # Backpropagation
      loss.backward()
      optimizer.step()
      print('[%d] train loss: %f ' % (epoch+1, loss))
    lr_scheduler.step()

train(model, loss_fn, optimizer1)

