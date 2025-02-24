#Libraries and datasets being used
import pandas as pd
import re
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv('/Users/srivalli_nalla/Desktop/SE/spread sheet for DIB.csv')
print(data.head())

# Converting ratings into sentiment labels (1 = Positive, 0 = Negative)
data['label'] = data['rating'].apply(lambda x: 1 if x == 5 else 0)
print(data['label'].value_counts())

#cleaning the text data- punctuation, stopwords, and lowercase the text to prepare for analysis 
nltk.download('punkt')
from tqdm import tqdm

def preprocess_text(text_data): 
    preprocessed_text = [] 
    for sentence in tqdm(text_data): 
        sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
        preprocessed_text.append(' '.join(token.lower() 
                                        for token in nltk.word_tokenize(sentence) 
                                        if token.lower() not in stopwords.words('english'))) 
    return preprocessed_text

# Apply function to dataset
data['review'] = preprocess_text(data['review'].values)
print(data.head())
