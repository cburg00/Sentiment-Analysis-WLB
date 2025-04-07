import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from analysis_function import (analyze_sentiment, analyze_numeric_ratings,
                               generate_predefined_summary_numeric, generate_predefined_summary_text)
from data_visuals import (generate_pie_chart, generate_sentiment_pie_chart, generate_scatter_plot,
                          generate_sentiment_scatter_plot, generate_wordcloud,generate_bar_chart_text, generate_bar_chart_numeric)

class SentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Analysis App")
        self.geometry("900x700")
        self.data = None
        self.word_cloud_message = Nonea
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
            text_input = self.manual_text.get("1.0", tk.END).strip()
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
        
        numeric_vals = pd.to_numeric(self.data[selected_col], errors='coerce')
        numeric_ratio = numeric_vals.notnull().mean()
        
        if numeric_ratio > 0.8:
            self.data['numeric'] = numeric_vals
            self.data['rating_sentiment'] = self.data['numeric'].apply(
                lambda x: "Negative" if x <= 2 else ("Neutral" if x == 3 else "Positive")
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
    
    def display_results(self, fig_pie, fig_scatter, fig_bar,summary):
        for child in self.notebook.winfo_children():
            child.destroy()
        
        # Pie Chart Tab.
        frame_pie = ttk.Frame(self.notebook)
        self.notebook.add(frame_pie, text="Pie Chart")
        canvas1 = FigureCanvasTkAgg(fig_pie, master=frame_pie)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill='both', expand=True)
        
        # Scatter Plot Tab.
        frame_scatter = ttk.Frame(self.notebook)
        self.notebook.add(frame_scatter, text="Scatter Plot")
        canvas2 = FigureCanvasTkAgg(fig_scatter, master=frame_scatter)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill='both', expand=True)

         # Bar Chart Tab.
        frame_bar = ttk.Frame(self.notebook)
        self.notebook.add(frame_bar, text="Bar Chart")
        canvas_bar = FigureCanvasTkAgg(fig_bar, master=frame_bar)
        canvas_bar.draw()
        canvas_bar.get_tk_widget().pack(fill='both', expand=True)
        
        
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
            
            frame_neg = ttk.Frame(self.notebook)
            self.notebook.add(frame_neg, text="Negative Word Cloud")
            canvas_neg = FigureCanvasTkAgg(self.fig_neg_wc, master=frame_neg)
            canvas_neg.draw()
            canvas_neg.get_tk_widget().pack(fill='both', expand=True)
        
        # Summary Tab.
        frame_sum = ttk.Frame(self.notebook)
        self.notebook.add(frame_sum, text="Summary")
        text_widget = tk.Text(frame_sum, wrap='word', font=("Arial", 12))
        text_widget.insert(tk.END, summary)
        text_widget.config(state="disabled")
        text_widget.pack(fill='both', expand=True)

if __name__ == "__main__":
    app = SentimentApp()
    app.mainloop()
