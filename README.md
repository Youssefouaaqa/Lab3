# Lab3

## Part 1: Classification Task:
### Scraping
We used for scraping the BeautifulSoup library for retrieving the articles from [Wikipedia](https://ar.wikipedia.org/). The topic chosen is `الحرب العالمية الاولى`, we took 500 articles on the subject, some of which have no relation to said topic. The link that was scraped first is [Search results](https://ar.wikipedia.org/w/index.php?limit=500&offset=0&profile=default&search=%D8%A7%D9%84%D8%AD%D8%B1%D8%A8+%D8%A7%D9%84%D8%B9%D8%A7%D9%84%D9%85%D9%8A%D8%A9+%D8%A7%D9%84%D8%A3%D9%88%D9%84%D9%89&title=%D8%AE%D8%A7%D8%B5:%D8%A8%D8%AD%D8%AB&ns0=1) for the topic found using the search bar of Wikipedia.\
The scoring was done manually, to give accurate relevancy scores to each article, this method ensures that the scores are actually reasonable. However, it is quite a tedious task to score these articles, other methods can be used such as word embedding and cosine similarity to automate this task.
### NLP Pipeline
#### Cleaning
During the cleaning process, we removed the Arabic stop words, punctuation, and numbers. We also removed the Arabic diacritics. and then stored the data into a JSON file.
#### Stemming
For the stemming part, we used the `ArabicLightStemmer` from the `tashaphyne` library. We stored the stemmed data into a JSON file.
#### Lemmatization
For the lemmatization part, we used the `lemmatizer` from the `qalsadi` library. We stored the lemmatized data into a JSON file.
### Embeddings
For the embeddings part, we used the `gensim` library to load a Arabic pretrained GloVe model. We then used the embeddings to get the average of the embeddings of each word in the article. We stored the embeddings in a CSV file.
### Language Modeling
For the language mpdeling part we used the `pytorch` library to test out multiple models, such as RNN, Bidirectional RNN, GRU, and LSTM. We used the embeddings to train the model.
The architecture of all models is:
- Input layer of 256 neurones.
- 1 hidden layer of 256 neurones.
- 100 epochs.
- A learning rate of 0.01.
### Evaluation
The RMSE for all models is basically the same, coming down to 2.73 and a loss is 7.49, this shows that the models are overfitting, which is expected since the dataset is quite small. and the embeddings might not accurately represent the articles. We can try to use a larger dataset and a more accurate embedding model to get better results. Unfortunately we couldn't continue working on the embedding and finetuning the model due to time and computational constraints.
## Part 2: Transformer (Text generation):
### Data
The data used for this part is a dataset found in kaggle, The dataset is a dataset for fake news, we only used the column text for this part, and we used the first 1000 rows of the dataset.
### Model
For the model, we used the `transformers` library from huggingface, we used the `GPT2` model to generate text. We used the `GPT2LMHeadModel` model to generate text. We used the `GPT2Tokenizer` to tokenize the text. the model is `gpt2-medium`.
### Fine Tuning
We fine-tuned the model on the dataset, for a 100 epochs, a batch size of 16 and a learning rate of $3*10^{-5}$. We used the `AdamW` optimizer. after fine-tuning the model we saved it to a file. the sum loss is 2992.61, due to computational constraints we couldn't fine-tune the model for more epochs.
### Evaluation
We used the final model to generate text, and the results were reasonable, the model was able to generate text that is coherent and somewhat relevant to the topic. The model can be improved by fine-tuning it for more epochs and using a larger dataset. However, sometimes the text genreated was only pointing out to the same thing over and over again, such as telling to us to go to the article detailes in the followig link, this can be due to the small dataset and the lack of diversity in the data.
**Example of generated text:**
```text
news:Just got off the phone with Rep. John Lewis (D-GA) who was brutally beaten by a white supremacist in Charlottesville, VA. Rep. Lewis is a civil rights icon, and I am heartbroken that he has to endure this. He is a true American hero.  John Lewis (@repjohnlewis) August 14, 2017Featured image via Chip Somodevilla / Getty images<|endoftext|> 

news:You can see the full video below:Featured image via screengrab<|endoftext|> 
```
