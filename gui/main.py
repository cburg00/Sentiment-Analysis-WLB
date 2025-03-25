import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datasets import load_dataset
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import customtkinter as ctk

nltk.download('vader_lexicon')

ds = load_dataset("kmrmanish/Employees_Reviews_Dataset")
df = ds['train'].to_pandas()

class SentimentAnalysis:
    def __init__(self, df):
        self.df = df
        self.sia = SentimentIntensityAnalyzer()

    def aggregate_text(self, row):
        texts = []
        for col in ["work_life_balance", "work_satisfaction", "Likes", "Dislikes"]:
            if col in row and pd.notnull(row[col]):
                texts.append(str(row[col]))
        return " ".join(texts)

    def analyze(self):
        sentiments = {"positive": 0, "negative": 0, "neutral": 0}
        for _, row in self.df.iterrows():
            agg_text = self.aggregate_text(row)
            score = self.sia.polarity_scores(agg_text)['compound']
            if score >= 0.05:
                sentiments["positive"] += 1
            elif score <= -0.05:
                sentiments["negative"] += 1
            else:
                sentiments["neutral"] += 1

        total = sum(sentiments.values())
        percentages = {k: (v / total * 100) if total > 0 else 0 for k, v in sentiments.items()}
        return percentages

    def generate_wordcloud(self, column):
        text = " ".join(str(t) for t in self.df[column] if pd.notnull(t))
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        return wc

sentiment_analyzer = SentimentAnalysis(df)

def create_sentiment_figure():
    percentages = sentiment_analyzer.analyze()
    labels = list(percentages.keys())
    sizes = list(percentages.values())
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title("Sentiment Analysis", fontsize=16)
    ax.axis('equal')
    return fig

def create_likes_wc_figure():
    wc = sentiment_analyzer.generate_wordcloud("Likes")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Likes Word Cloud", fontsize=16)
    return fig

def create_dislikes_wc_figure():
    wc = sentiment_analyzer.generate_wordcloud("Dislikes")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Dislikes Word Cloud", fontsize=16)
    return fig

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

app = ctk.CTk()
app.geometry("1200x800")
app.title("Employee Sentiment Analysis")

message_label = ctk.CTkLabel(
    app, 
    text="Thank you for using our services! Here are your results", 
    font=("Helvetica", 20)
)
message_label.pack(pady=(20, 10))

header_label = ctk.CTkLabel(
    app, 
    text="Employee Sentiment Analysis", 
    font=("Helvetica", 28, "bold")
)
header_label.pack(pady=(10, 10))

tabview = ctk.CTkTabview(app, width=1100, height=700)
tabview.pack(padx=20, pady=20)

tabview.add("Sentiment Analysis")
tabview.add("Likes Word Cloud")
tabview.add("Dislikes Word Cloud")

sentiment_fig = create_sentiment_figure()
sentiment_canvas = FigureCanvasTkAgg(sentiment_fig, master=tabview.tab("Sentiment Analysis"))
sentiment_canvas.draw()
sentiment_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

likes_fig = create_likes_wc_figure()
likes_canvas = FigureCanvasTkAgg(likes_fig, master=tabview.tab("Likes Word Cloud"))
likes_canvas.draw()
likes_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

dislikes_fig = create_dislikes_wc_figure()
dislikes_canvas = FigureCanvasTkAgg(dislikes_fig, master=tabview.tab("Dislikes Word Cloud"))
dislikes_canvas.draw()
dislikes_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

app.mainloop()
