import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_pie_chart(df):
    """Generate a pie chart showing sentiment distribution (for text analysis)."""
    counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Sentiment Distribution")
    fig.text(0.5, 0.01, "Caption: This pie chart shows the percentage distribution of sentiments from the text data.", 
    ha="center", fontsize=10)
    return fig

def generate_sentiment_pie_chart(df):
    """Generate a pie chart showing sentiment distribution based on numeric ratings."""
    sentiment_counts = df['rating_sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Sentiment Distribution")
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.05,
    "Caption: This pie chart represents the percentage of numeric ratings grouped into sentiment categories derived from employees' ratings.",
     ha="center", fontsize=10)
    return fig

def generate_scatter_plot(df):
    """Generate a scatter plot of polarity values (for text analysis) with axes switched."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df['polarity'], range(len(df)), color='blue')
    ax.set_title("Polarity Scatter Plot")
    ax.set_xlabel("Polarity")
    ax.set_ylabel("Data Inputs")
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.05,
     "Caption: This scatter plot shows each review's sentiment polarity where each point's horizontal position represents its order and vertical position shows its polarity score.",
      ha="center", fontsize=10)
    return fig

def generate_sentiment_scatter_plot(df):
    """Generate a scatter plot for numeric ratings
       x-axis: rating (1-5), y-axis: Inputs, colored by sentiment.
    """
    colors = {"Negative": "red", "Neutral": "gray", "Positive": "green"}
    df_filtered = df[df['numeric'].between(1, 5)]
    fig, ax = plt.subplots(figsize=(6, 4))
    x_values = df_filtered['numeric']
    y_values = df_filtered.index
    ax.scatter(x_values, y_values, c=df_filtered['rating_sentiment'].map(colors), s=20)
    ax.set_title("Scores Scatter Plot")
    ax.set_xlabel("Score")
    ax.set_ylabel("Data Inputs")
    ax.set_xticks([1, 2, 3, 4, 5])
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.05,
    "Caption: This scatter plot displays numeric ratings (on the x-axis) for each data input (ordered on the y-axis), with colors indicating the assigned sentiment.",
     ha="center", fontsize=10)
    return fig

def generate_wordcloud(text, title):
    """Generate a word cloud figure from the provided text."""
    if not text.strip():
        text = "No data available."
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    fig.text(0.5, 0.01,
    "Caption: The word cloud visualizes the most frequent words from the text, with larger words representing higher frequencies.",
    ha="center", fontsize=10)
    return fig

def generate_bar_chart_text(df):
    """Generate a bar chart showing sentiment distribution for text analysis."""
    counts = df['sentiment'].value_counts()
    color_mapping = {"Positive": "green", "Neutral": "yellow", "Negative": "red"}
    colors = [color_mapping.get(sentiment, "blue") for sentiment in counts.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Text Sentiment Distribution - Bar Chart")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Data Inputs")
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.05,
    "Caption: This bar chart shows the number of text reviews in each sentiment category (Positive, Neutral, Negative), with different colors representing each category.",
    ha="center", fontsize=10)
    return fig

def generate_bar_chart_numeric(df):
    """Generate a bar chart showing sentiment distribution based on numeric ratings."""
    counts = df['rating_sentiment'].value_counts()
    color_mapping = {"Positive": "green", "Neutral": "yellow", "Negative": "red"}
    colors = [color_mapping.get(sentiment, "blue") for sentiment in counts.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Numeric Sentiment Distribution - Bar Chart")
    ax.set_xlabel("Rating Sentiment")
    ax.set_ylabel("Data inputs")
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.05,
    "Caption: This bar chart illustrates how many numeric ratings fall into each sentiment category, with colors used to differentiate between Positive, Neutral, and Negative ratings.",
    ha="center", fontsize=10)
    return fig

