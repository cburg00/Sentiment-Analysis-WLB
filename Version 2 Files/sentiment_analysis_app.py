import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler

# Ensure necessary NLTK data is available
nltk.download('stopwords')
nltk.download('punkt')

# Helper to determine base path when frozen
if getattr(sys, 'frozen', False):
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(__file__)

# Analysis functions

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob and return (sentiment, polarity)."""
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        sentiment = 'Positive'
    elif polarity < -0.1:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment, polarity


def analyze_numeric_ratings(df, col):
    """Calculate basic statistics and distribution for numeric data."""
    results = {
        'average': df[col].mean(),
        'median': df[col].median(),
        'min': df[col].min(),
        'max': df[col].max(),
        'distribution': df[col].value_counts().sort_index()
    }
    return results


def scale_numbers(col):
    scaler = MinMaxScaler(feature_range=(-1,1))
    is_numeric = pd.api.types.is_numeric_dtype(col)
    if is_numeric:
        scaled = scaler.fit_transform(col.values.reshape(-1,1))
        return pd.Series(scaled.flatten(), index=col.index)
    return None


def generate_predefined_summary_numeric(positive_pct, neutral_pct, negative_pct, average_rating):
    summary = "Work-Life Balance Analysis:\n"
    summary += f"- Positive Responses: {positive_pct:.1f}%\n"
    summary += f"- Neutral Responses: {neutral_pct:.1f}%\n"
    summary += f"- Negative Responses: {negative_pct:.1f}%\n"
    summary += f"- Average Score: {average_rating:.2f}/5\n\n"
    if average_rating >= 4.5:
        summary += (
            "Excellent overall sentiment regarding work-life balance.\n\n"
            "Recommendations:\n"
            "- Maintain current work-life balance initiatives.\n\n"
            "- Continue regular employee satisfaction checks."
        )
    elif average_rating >= 4.0:
        summary += (
            "Very good sentiment overall, with employees generally satisfied.\n\n"
            "Recommendations:\n"
            "- Gather feedback to pinpoint minor improvements.\n"
            "- Keep open communication channels."
        )
    elif average_rating >= 3.5:
        summary += (
            "Good sentiment overall, though some areas need improvement.\n\n"
            "Recommendations:\n"
            "- Investigate causes behind neutral/negative responses.\n"
            "- Offer more flexible scheduling options."
        )
    elif average_rating >= 3.0:
        summary += (
            "Moderate sentiment indicates mixed experiences among employees.\n\n"
            "Recommendations:\n"
            "- Introduce structured work-life balance programs (e.g., wellness initiatives).\n"
            "- Increase flexibility and clarity on available support."
        )
    elif average_rating >= 2.5:
        summary += (
            "Below average sentiment suggests significant concerns with work-life balance.\n\n"
            "Recommendations:\n"
            "- Conduct surveys to identify stressors.\n"
            "- Implement flexible hours, mental health days, and stress management workshops."
        )
    else:
        summary += (
            "Poor sentiment demonstrates severe dissatisfaction.\n\n"
            "Immediate Recommendations:\n"
            "- Hold urgent employee forums to discuss pain points.\n"
            "- Develop comprehensive policies with substantial flexibility and wellness support."
        )
    if negative_pct > 30:
        summary += "\n\nNote: A high proportion of negative responses indicates widespread dissatisfaction that should be urgently addressed."
    return summary


def generate_predefined_summary_text(sentiment_counts, avg_polarity):
    summary = "Work-Life Balance Analysis:\n"
    summary += f"- Sentiment Breakdown: {sentiment_counts}\n"
    summary += f"- Average Sentiment Polarity: {avg_polarity:.2f}\n\n"
    positive = sentiment_counts.get('Positive', 0)
    neutral = sentiment_counts.get('Neutral', 0)
    negative = sentiment_counts.get('Negative', 0)
    total = positive + neutral + negative
    if total:
        negative_pct = (negative / total) * 100
    else:
        negative_pct = 0
    if avg_polarity >= 0.5:
        summary += (
            "Highly positive sentiment indicates employees feel very supported.\n\n"
            "Recommendations:\n"
            "- Maintain current positive practices and gather regular feedback."
        )
    elif avg_polarity >= 0.2:
        summary += (
            "Overall positive sentiment with minor issues.\n\n"
            "Recommendations:\n"
            "- Explore common neutral/negative themes and improve flexibility or wellness programs."
        )
    elif avg_polarity >= 0.0:
        summary += (
            "Neutral sentiment suggests mixed experiences.\n\n"
            "Recommendations:\n"
            "- Increase dialogue and introduce clear flexible working and wellness policies."
        )
    elif avg_polarity >= -0.2:
        summary += (
            "Negative sentiment indicates growing dissatisfaction.\n\n"
            "Recommendations:\n"
            "- Conduct detailed feedback sessions and prioritize flexible schedules and mental health resources."
        )
    else:
        summary += (
            "Very negative sentiment highlights critical issues.\n\n"
            "Urgent Recommendations:\n"
            "- Immediately address employee concerns with comprehensive changes and increased support."
        )
    if negative_pct > 30:
        summary += "\n\nNote: Over 30% negative responses indicate deep-rooted dissatisfaction that must be addressed promptly."
    return summary

# Visualization functions

def generate_pie_chart(df):
    counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Sentiment Distribution")
    fig.text(0.5, 0.01, "Caption: This pie chart shows the percentage distribution of sentiments from the text data.", ha="center", fontsize=10)
    return fig


def generate_sentiment_pie_chart(df):
    counts = df['rating_sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Sentiment Distribution")
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.05, "Caption: This pie chart represents the percentage of numeric ratings grouped into sentiment categories.", ha="center", fontsize=10)
    return fig


def generate_scatter_plot(df):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df['polarity'], range(len(df)))
    ax.set_title("Polarity Scatter Plot")
    ax.set_xlabel("Polarity")
    ax.set_ylabel("Data Inputs")
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.05, "Caption: Each point represents a review's polarity score.", ha="center", fontsize=10)
    return fig


def generate_sentiment_scatter_plot(df):
    colors = df['rating_sentiment'].map({'Negative':'red','Neutral':'gray','Positive':'green'})
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df['numeric'], df.index, c=colors, s=20)
    ax.set_title("Scores Scatter Plot")
    ax.set_xlabel("Score")
    ax.set_ylabel("Data Inputs")
    ax.set_xticks([1,2,3,4,5])
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.05, "Caption: Numeric ratings plotted with color indicating sentiment.", ha="center", fontsize=10)
    return fig


def generate_wordcloud(text, title):
    if not text.strip():
        text = "No data available."
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    fig.text(0.5, 0.01, "Caption: Word cloud of text frequency.", ha="center", fontsize=10)
    return fig


def generate_bar_chart_text(df):
    counts = df['sentiment'].value_counts()
    colors = [ 'green' if s=='Positive' else 'yellow' if s=='Neutral' else 'red' for s in counts.index ]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Text Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.05, "Caption: Number of text reviews per sentiment.", ha="center", fontsize=10)
    return fig


def generate_bar_chart_numeric(df):
    counts = df['rating_sentiment'].value_counts()
    colors = [ 'green' if s=='Positive' else 'yellow' if s=='Neutral' else 'red' for s in counts.index ]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Numeric Sentiment Distribution")
    ax.set_xlabel("Rating Sentiment")
    ax.set_ylabel("Count")
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.05, "Caption: Number of numeric ratings per sentiment.", ha="center", fontsize=10)
    return fig

# Main Application
class SentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Analysis App")
        self.geometry("900x700")
        self.data = None
        self.word_cloud_message = None
        self.create_control_panel()

    def create_control_panel(self):
        frame = ttk.Frame(self)
        frame.pack(fill='x', padx=10, pady=10)
        self.data_source_var = tk.StringVar(value="pre")
        tk.Label(frame, text="Data Source:").grid(row=0, column=0, sticky="w")
        tk.Radiobutton(frame, text="Pre-imported", variable=self.data_source_var, value="pre").grid(row=0, column=1, padx=5)
        tk.Radiobutton(frame, text="CSV File", variable=self.data_source_var, value="csv").grid(row=0, column=2, padx=5)
        tk.Radiobutton(frame, text="Manual Input", variable=self.data_source_var, value="manual").grid(row=0, column=3, padx=5)
        ttk.Button(frame, text="Load Data", command=self.load_data).grid(row=1, column=0, pady=5, sticky="w")
        tk.Label(frame, text="Manual Input (one entry per line):").grid(row=2, column=0, columnspan=4, sticky="w")
        self.manual_text = tk.Text(frame, height=5, width=60)
        self.manual_text.grid(row=3, column=0, columnspan=4, pady=5)
        placeholder = "I love my job\nI hate my job"
        self.add_placeholder(self.manual_text, placeholder)
        self.manual_text.bind("<FocusIn>", lambda e: self.remove_placeholder(self.manual_text, placeholder))
        self.manual_text.bind("<FocusOut>", lambda e: self.restore_placeholder(self.manual_text, placeholder))
        tk.Label(frame, text="Select Column for Analysis:").grid(row=4, column=0, sticky="w")
        self.column_combobox = ttk.Combobox(frame, state="readonly")
        self.column_combobox.grid(row=4, column=1, columnspan=3, sticky="we", padx=5)
        self.analyze_btn = ttk.Button(frame, text="Analyze", command=self.perform_analysis)
        self.analyze_btn.grid(row=5, column=0, columnspan=4, pady=10)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

    def load_data(self):
        src = self.data_source_var.get()
        if src == "pre":
            sample = {'text_reviews': ["I love working here!","This job is terrible, I hate it","It's not too bad working here, but it could be better","Absolutely fantastic working here!","Worst job ever!","I love my job","My boss is wonderful and makes my job easier","The best place to work"],
                      'work_life_balance': [4,2,5,1,2,1,3,2]}
            self.data = pd.DataFrame(sample)
            messagebox.showinfo("Info", "Pre-imported sample data loaded.")
        elif src == "csv":
            initial = getattr(sys, 'frozen', False) and os.path.dirname(sys.executable) or os.getcwd()
            file = filedialog.askopenfilename(initialdir=initial, filetypes=[("CSV files","*.csv")])
            if file:
                try:
                    self.data = pd.read_csv(file)
                    messagebox.showinfo("Info", f"CSV data loaded from {file}.")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load CSV: {e}")
                    return
        else:
            text = self.manual_text.get("1.0","end-1c").strip()
            if text:
                self.data = pd.DataFrame({"text": text.splitlines()})
                messagebox.showinfo("Info","Manual input data loaded.")
            else:
                messagebox.showwarning("Warning","No manual input provided.")
                return
        self.update_column_options()

    def update_column_options(self):
        if self.data is not None:
            cols = list(self.data.columns)
            self.column_combobox['values'] = cols
            if cols:
                self.column_combobox.current(0)

    def perform_analysis(self):
        col = self.column_combobox.get()
        if not col or col not in self.data.columns:
            messagebox.showerror("Error","Selected column not found.")
            return
        series = self.data[col].astype(str)
        is_pct = series.str.contains('%').mean() > 0.8
        if is_pct:
            nums = series.str.rstrip('%').astype(float)/100.0
        else:
            nums = pd.to_numeric(self.data[col], errors='coerce')
        if nums.notnull().mean() > 0.8:
            self.data['numeric'] = nums
            self.data['scaled'] = scale_numbers(nums)
            self.data['rating_sentiment'] = self.data['scaled'].apply(lambda x: 'Negative' if x<=-0.1 else 'Neutral' if x<=0.1 else 'Positive')
            fig1 = generate_sentiment_pie_chart(self.data)
            fig2 = generate_sentiment_scatter_plot(self.data)
            fig3 = generate_bar_chart_numeric(self.data)
            self.word_cloud_message = "Data analyzed was numerical, so no word cloud was generated."
            counts = self.data['rating_sentiment'].value_counts()
            total = counts.sum()
            pos_pct = counts.get('Positive',0)/total*100
            neu_pct = counts.get('Neutral',0)/total*100
            neg_pct = counts.get('Negative',0)/total*100
            avg = self.data['numeric'].mean()
            summary = generate_predefined_summary_numeric(pos_pct, neu_pct, neg_pct, avg)
        else:
            self.data['sentiment'], self.data['polarity'] = zip(*self.data[col].apply(analyze_sentiment))
            fig1 = generate_pie_chart(self.data)
            fig2 = generate_scatter_plot(self.data)
            fig3 = generate_bar_chart_text(self.data)
            pos_txt = ' '.join(self.data[self.data['sentiment']=='Positive'][col].dropna().astype(str))
            neg_txt = ' '.join(self.data[self.data['sentiment']=='Negative'][col].dropna().astype(str))
            self.fig_pos_wc = generate_wordcloud(pos_txt, "Positive Word Cloud")
            self.fig_neg_wc = generate_wordcloud(neg_txt, "Negative Word Cloud")
            self.word_cloud_message = None
            counts = self.data['sentiment'].value_counts().to_dict()
            avg_pol = self.data['polarity'].mean()
            summary = generate_predefined_summary_text(counts, avg_pol)
        self.display_results(fig1, fig2, fig3, summary)

    def display_results(self, fig1, fig2, fig3, summary):
        for child in self.notebook.winfo_children():
            child.destroy()
        # Pie
        f1 = ttk.Frame(self.notebook)
        self.notebook.add(f1, text="Pie Chart")
        c1 = FigureCanvasTkAgg(fig1, master=f1)
        c1.draw(); c1.get_tk_widget().pack(fill='both',expand=True)
        plt.close(fig1)
        # Scatter
        f2 = ttk.Frame(self.notebook)
        self.notebook.add(f2, text="Scatter Plot")
        c2 = FigureCanvasTkAgg(fig2, master=f2)
        c2.draw(); c2.get_tk_widget().pack(fill='both',expand=True)
        plt.close(fig2)
        # Bar
        f3 = ttk.Frame(self.notebook)
        self.notebook.add(f3, text="Bar Chart")
        c3 = FigureCanvasTkAgg(fig3, master=f3)
        c3.draw(); c3.get_tk_widget().pack(fill='both',expand=True)
        plt.close(fig3)
        # Word Clouds or message
        if self.word_cloud_message:
            fw = ttk.Frame(self.notebook)
            self.notebook.add(fw, text="Word Cloud")
            tk.Label(fw, text=self.word_cloud_message, font=("Arial",14)).pack(expand=True, fill='both', padx=10, pady=10)
        else:
            fp = ttk.Frame(self.notebook)
            self.notebook.add(fp, text="Positive Word Cloud")
            cp = FigureCanvasTkAgg(self.fig_pos_wc, master=fp)
            cp.draw(); cp.get_tk_widget().pack(fill='both',expand=True)
            plt.close(self.fig_pos_wc)
            fn = ttk.Frame(self.notebook)
            self.notebook.add(fn, text="Negative Word Cloud")
            cn = FigureCanvasTkAgg(self.fig_neg_wc, master=fn)
            cn.draw(); cn.get_tk_widget().pack(fill='both',expand=True)
            plt.close(self.fig_neg_wc)
        # Summary
        fs = ttk.Frame(self.notebook)
        self.notebook.add(fs, text="Summary")
        tw = tk.Text(fs, wrap='word', font=("Arial",12))
        tw.insert(tk.END, summary)
        tw.config(state="disabled")
        tw.pack(fill='both', expand=True)

    def add_placeholder(self, widget, text):
        widget.insert("1.0", text)
        widget.tag_config("placeholder", foreground="gray")
        widget.tag_add("placeholder", "1.0", "end")

    def remove_placeholder(self, widget, text):
        if widget.get("1.0","end-1c") == text:
            widget.delete("1.0","end")

    def restore_placeholder(self, widget, text):
        if not widget.get("1.0","end-1c").strip():
            self.add_placeholder(widget, text)

if __name__ == "__main__":
    app = SentimentApp()
    app.mainloop()
