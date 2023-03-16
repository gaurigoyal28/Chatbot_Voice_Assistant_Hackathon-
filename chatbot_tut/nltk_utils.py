import nltk
import numpy as np
nltk.download("punkt")# punkt is an nltk package which has pre trained tokenizer which makes  .word_tokenizer function work ,is downloaded once and can be used anytime afterwards.
from nltk.stem.porter import PorterStemmer #from different types of stemmers, portersetmmer is chosen.
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    stemmer=PorterStemmer()
    return stemmer.stem(word.lower())#first lower the word then stemmerize it
def bag_of_words(tokenized_sentence,all_words):
    tokenized_sentence=[stem(w) for w in tokenized_sentence]
    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx , w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0
    return bag

a="how are you" #checking on some examples
print(a)
print(tokenize(a))
words=["organize","Organize","Organ"]
stemmed_words=[stem(i) for i in words]#list comprehension to take one word at a time and pass onto stem function
print(stemmed_words)