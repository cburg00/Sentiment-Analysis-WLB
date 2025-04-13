import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler

# ==================== Analysis Functions ====================
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

def generate_predefined_summary_numeric(positive_pct, neutral_pct, negative_pct, average_rating):
    summary = "Work-Life Balance Analysis:\n"
    summary += f"- Positive Responses: {positive_pct:.1f}%\n"
    summary += f"- Neutral Responses: {neutral_pct:.1f}%\n"
    summary += f"- Negative Responses: {negative_pct:.1f}%\n"
    summary += f"- Average Score: {average_rating:.2f}/5\n\n"

    if average_rating >= 4.5:
        summary += ("Excellent overall sentiment regarding work-life balance. Employees feel highly supported and balanced.\n\n"
                    "Recommendations:\n"
                    "- Maintain current work-life balance initiatives.\n\n"
                    "- Continue regular employee satisfaction checks.")
    elif average_rating >= 4.0:
        summary += ("Very good sentiment overall, with employees generally satisfied.\n\n"
                    "Recommendations:\n"
                    "- Gather feedback to pinpoint minor improvements.\n"
                    "- Keep open communication channels.")
    elif average_rating >= 3.5:
        summary += ("Good sentiment overall, though some areas need improvement.\n\n"
                    "Recommendations:\n"
                    "- Investigate causes behind neutral/negative responses.\n"
                    "- Offer more flexible scheduling options.")
    elif average_rating >= 3.0:
        summary += ("Moderate sentiment indicates mixed experiences among employees.\n\n"
                    "Recommendations:\n"
                    "- Introduce structured work-life balance programs (e.g., wellness initiatives).\n"
                    "- Increase flexibility and clarity on available support.")
    elif average_rating >= 2.5:
        summary += ("Below average sentiment suggests significant concerns with work-life balance.\n\n"
                    "Recommendations:\n"
                    "- Conduct surveys to identify stressors.\n"
                    "- Implement flexible hours, mental health days, and stress management workshops.")
    else:
        summary += ("Poor sentiment demonstrates severe dissatisfaction.\n\n"
                    "Immediate Recommendations:\n"
                    "- Hold urgent employee forums to discuss pain points.\n"
                    "- Develop comprehensive policies with substantial flexibility and wellness support.")

    # If negative responses are high, add a note.
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
    negative_pct = (negative / total) * 100 if total else 0

    if avg_polarity >= 0.5:
        summary += ("Highly positive sentiment indicates employees feel very supported.\n\n"
                    "Recommendations:\n"
                    "- Maintain current positive practices and gather regular feedback.")
    elif avg_polarity >= 0.2:
        summary += ("Overall positive sentiment with minor issues.\n\n"
                    "Recommendations:\n"
                    "- Explore common neutral/negative themes and improve flexibility or wellness programs.")
    elif avg_polarity >= 0.0:
        summary += ("Neutral sentiment suggests mixed experiences.\n\n"
                    "Recommendations:\n"
                    "- Increase dialogue and introduce clear flexible working and wellness policies.")
    elif avg_polarity >= -0.2:
        summary += ("Negative sentiment indicates growing dissatisfaction.\n\n"
                    "Recommendations:\n"
                    "- Conduct detailed feedback sessions and prioritize flexible schedules and mental health resources.")
    else:
        summary += ("Very negative sentiment highlights critical issues.\n\n"
                    "Urgent Recommendations:\n"
                    "- Immediately address employee concerns with comprehensive changes and increased support.")
    if negative_pct > 30:
        summary += "\n\nNote: Over 30% negative responses indicate deep-rooted dissatisfaction that must be addressed promptly."
    return summary

def scale_numbers(col):
    scaler = MinMaxScaler(feature_range=(-1,1))
    is_numeric = pd.api.types.is_numeric_dtype(col)
    if is_numeric:
        col_scaled_values = scaler.fit_transform(col.values.reshape(-1,1))
        col_scaled = pd.DataFrame(col_scaled_values, columns=["data_scaled"], index=col.index)
        return col_scaled
    return None

# ==================== Visualization Functions ====================
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

# ==================== Main Application ====================
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
        placeholder_text = "I love my job\nI hate my job"
        self.add_placeholder(self.manual_text, placeholder_text)
        
        self.manual_text.bind("<FocusIn>", lambda event: self.remove_placeholder(self.manual_text, placeholder_text, event))
        self.manual_text.bind("<FocusOut>", lambda event: self.restore_placeholder(self.manual_text, placeholder_text, event))
        
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
            sample_data = {
                'text_reviews': [
                    "I love working here!",
                    "This job is terrible, I hate it",
                    "It's not too bad working here, but it could be better",
                    "Absolutely fantastic working here!",
                    "Worst job ever!",
                    "I love my job",
                    "My boss is wonderful and makes my job easier",
                    "The best place to work"
                ],
                'work_life_balance': [4, 2, 5, 1, 2, 1, 3, 2]
            }
            self.data = pd.DataFrame(sample_data)
            messagebox.showinfo("Info", "Pre-imported sample data loaded.")
        elif src == "csv":
            file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file:
                try:
                    self.data = pd.read_csv(file)
                    messagebox.showinfo("Info", f"CSV data loaded from {file}.")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load CSV: {e}")
                    return
        elif src == "manual":
            text_input = self.manual_text.get("1.0", "end-1c").strip()
            if text_input:
                lines = text_input.splitlines()
                self.data = pd.DataFrame({"text": lines})
                messagebox.showinfo("Info", "Manual input data loaded.")
            else:
                messagebox.showwarning("Warning", "No manual input provided.")
                return
        self.update_column_options()
    
    def update_column_options(self):
        if self.data is not None:
            cols = list(self.data.columns)
            self.column_combobox['values'] = cols
            if cols:
                self.column_combobox.current(0)
    
    def perform_analysis(self):
        selected_col = self.column_combobox.get()
        if not selected_col or selected_col not in self.data.columns:
            messagebox.showerror("Error", "Selected column not found.")
            return
        
        col_series = self.data[selected_col].astype(str)
        #more than 80% of values are percents
        is_percent = col_series.str.contains('%').mean() > 0.8

        #checks if value is percent. converts to float if true
        if is_percent:
            numeric_vals = col_series.str.rstrip('%').astype(float) / 100.0
        else:
            numeric_vals = pd.to_numeric(self.data[selected_col], errors='coerce')

        numeric_ratio = numeric_vals.notnull().mean()
        
        if numeric_ratio > 0.8:
            self.data['numeric'] = numeric_vals
            #scales data between -1,1
            self.data['scaled'] = scale_numbers(numeric_vals)
            self.data['rating_sentiment'] = self.data['scaled'].apply(
                lambda x: "Negative" if x <= -0.1 else "Neutral" if x <= 0.1 else "Positive"
            )
            
            fig_pie = generate_sentiment_pie_chart(self.data)
            fig_scatter = generate_sentiment_scatter_plot(self.data)
            fig_bar = generate_bar_chart_numeric(self.data)
            self.word_cloud_message = "Data analyzed was numerical, so no word cloud was generated."
            
            sentiment_counts = self.data['rating_sentiment'].value_counts()
            total = sentiment_counts.sum()
            positive_pct = (sentiment_counts.get("Positive", 0) / total) * 100
            neutral_pct = (sentiment_counts.get("Neutral", 0) / total) * 100
            negative_pct = (sentiment_counts.get("Negative", 0) / total) * 100
            avg_rating = self.data['numeric'].mean()
            summary = generate_predefined_summary_numeric(positive_pct, neutral_pct, negative_pct, avg_rating)
        else:
            self.data['sentiment'], self.data['polarity'] = zip(*self.data[selected_col].apply(analyze_sentiment))
            fig_pie = generate_pie_chart(self.data)
            fig_scatter = generate_scatter_plot(self.data)
            fig_bar = generate_bar_chart_text(self.data)
            pos_text = ' '.join(self.data[self.data['sentiment'] == 'Positive'][selected_col].dropna().astype(str))
            neg_text = ' '.join(self.data[self.data['sentiment'] == 'Negative'][selected_col].dropna().astype(str))
            self.fig_pos_wc = generate_wordcloud(pos_text, "Positive Word Cloud")
            self.fig_neg_wc = generate_wordcloud(neg_text, "Negative Word Cloud")
            self.word_cloud_message = None
            sentiment_counts = self.data['sentiment'].value_counts().to_dict()
            avg_polarity = self.data['polarity'].mean()
            summary = generate_predefined_summary_text(sentiment_counts, avg_polarity)
        
        self.display_results(fig_pie, fig_scatter, fig_bar, summary)
    
    def display_results(self, fig_pie, fig_scatter, fig_bar, summary):
        for child in self.notebook.winfo_children():
            child.destroy()
        
        # Pie Chart Tab.
        frame_pie = ttk.Frame(self.notebook)
        self.notebook.add(frame_pie, text="Pie Chart")
        canvas1 = FigureCanvasTkAgg(fig_pie, master=frame_pie)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig_pie)
        
        # Scatter Plot Tab.
        frame_scatter = ttk.Frame(self.notebook)
        self.notebook.add(frame_scatter, text="Scatter Plot")
        canvas2 = FigureCanvasTkAgg(fig_scatter, master=frame_scatter)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig_scatter)

        # Bar Chart Tab.
        frame_bar = ttk.Frame(self.notebook)
        self.notebook.add(frame_bar, text="Bar Chart")
        canvas_bar = FigureCanvasTkAgg(fig_bar, master=frame_bar)
        canvas_bar.draw()
        canvas_bar.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig_bar)
        
        # Word Cloud Tab(s) or Message.
        if self.word_cloud_message:
            frame_wc = ttk.Frame(self.notebook)
            self.notebook.add(frame_wc, text="Word Cloud")
            tk.Label(frame_wc, text=self.word_cloud_message, font=("Arial", 14)).pack(expand=True, fill='both', padx=10, pady=10)
        else:
            frame_pos = ttk.Frame(self.notebook)
            self.notebook.add(frame_pos, text="Positive Word Cloud")
            canvas_pos = FigureCanvasTkAgg(self.fig_pos_wc, master=frame_pos)
            canvas_pos.draw()
            canvas_pos.get_tk_widget().pack(fill='both', expand=True)
            plt.close(self.fig_pos_wc)
            
            frame_neg = ttk.Frame(self.notebook)
            self.notebook.add(frame_neg, text="Negative Word Cloud")
            canvas_neg = FigureCanvasTkAgg(self.fig_neg_wc, master=frame_neg)
            canvas_neg.draw()
            canvas_neg.get_tk_widget().pack(fill='both', expand=True)
            plt.close(self.fig_neg_wc)
        
        # Summary Tab.
        frame_sum = ttk.Frame(self.notebook)
        self.notebook.add(frame_sum, text="Summary")
        text_widget = tk.Text(frame_sum, wrap='word', font=("Arial", 12))
        text_widget.insert(tk.END, summary)
        text_widget.config(state="disabled")
        text_widget.pack(fill='both', expand=True)

    def add_placeholder(self, manual_text, placeholder_text):
        manual_text.insert("1.0", placeholder_text)
        manual_text.tag_add("placeholder", "1.0", "end")
        manual_text.tag_config("placeholder", foreground="gray")
        
    def remove_placeholder(self, manual_text, placeholder_text, event=None):
        current_text = manual_text.get("1.0", "end-1c")
        if current_text == placeholder_text:
            manual_text.delete("1.0", "end")
            
    def restore_placeholder(self, manual_text, placeholder_text, event=None):
        current_text = manual_text.get("1.0", "end-1c").strip()
        if not current_text:
            self.add_placeholder(manual_text, placeholder_text)

if __name__ == "__main__":
    app = SentimentApp()
    app.mainloop()