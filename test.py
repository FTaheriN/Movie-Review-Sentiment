"""## Test"""

# Test data
# load all test reviews
positive_docs = process_docs1('/Dataset/txt_sentoken/pos', vocab, False)
negative_docs = process_docs1('/Dataset/txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])


TXtest = torch.from_numpy(Xtest)
Tytest = torch.from_numpy(ytest)


def test(model, loss_fn):
  for i in range (0, TXtest.shape[0]):
    model = model.eval()
    with torch.no_grad():
      X = TXtest[i,:].to(device)
      Y = Tytest[i].to(device)
      pred = model(X)
      Y = Y.to(torch.float32)
      loss = loss_fn(pred,Y)
      print('train loss: %f ' % (loss))

test(model,loss_fn)