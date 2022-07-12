# Movie-Review-Sentiment
A basic program to train a model and classify movie reviews as positive or negative based on the Keras and Pytorch libraries. The 
[IMDB Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/review_polarity.tar.gz) is used.
The dataset consists of 1000 positive and 1000 negative reviews. The first 800 reviews of each class is used for training and the last 200 for testing.

First you must run the datasetpreperation file in order to make a list of vocabulary that is used in the trainig dataset.
The movieReviewSentiment file contains a part to clean the train data and also to convert words to number sequnces to be used in the embedding layer of the model.
The model is then built and the trainig process strats for 50 epochs.

You can add the content of test.py at the end of movieReviewSentiment.py and test the performance of the model. 
Make sure that in order to convert test data to number sequnces, you must use the same tokenizer used in training data. 
