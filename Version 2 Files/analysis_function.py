from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def analyze_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    return 'Positive' if polarity > 0.1 else 'Negative' if polarity < -0.1 else 'Neutral', polarity

def generate_wordcloud(text_series):
    text = ' '.join(text_series.dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def analyze_numeric_ratings(df, column_name):
    results = {
        'average': df[column_name].mean(),
        'median': df[column_name].median(),
        'min': df[column_name].min(),
        'max': df[column_name].max(),
        'distribution': df[column_name].value_counts().sort_index()
    }
    return results

def get_sample_data(df, column_name, analysis_type, n_samples=5):
    if analysis_type == 'numeric':
        return df[[column_name, 'rating']].sample(min(n_samples, len(df)))
    else:
        return df[[column_name, 'sentiment', 'polarity']].sample(min(n_samples, len(df)))