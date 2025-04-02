import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_function import *
from utils import *
from config import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from wordcloud import WordCloud

class WorkLifeBalanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Work-Life Balance Analysis Dashboard")
        self.geometry("900x600")
        ctk.set_appearance_mode("dark")
        
        self.df = None
        self.selected_column = None
        
        self.create_widgets()
    
    def create_widgets(self):
        self.menu_frame = ctk.CTkFrame(self)
        self.content_frame = ctk.CTkFrame(self)
        self.label = ctk.CTkLabel(self.menu_frame, text="Input Options", font=("Arial", 16))
        self.input_method = ctk.CTkComboBox(self.menu_frame, values=["Text Input", "CSV Upload", "Pre-loaded Dataset"], command=self.handle_input)
        
        self.menu_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, ipadx=5)
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.label.pack(pady=10)
        self.input_method.pack(pady=10)

        self.column_selector = None
        self.analyze_column_btn = None
        self.chart_tabview = None
    
    def handle_input(self, choice):
        if choice == "CSV Upload":
            self.upload_csv()
        elif choice == "Pre-loaded Dataset":
            self.load_preloaded_data()
    
    def upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            self.show_column_selector()
    
    def load_preloaded_data(self):
        self.df = load_employee_reviews()
        self.show_column_selector()
    
    def show_column_selector(self):
        if self.df is not None:
            if self.column_selector:
                self.column_selector.destroy()
            if self.analyze_column_btn:
                self.analyze_column_btn.destroy()
            
            self.column_selector = ctk.CTkComboBox(self.menu_frame, values=list(self.df.columns))
            self.column_selector.pack(pady=10)
            
            self.analyze_column_btn = ctk.CTkButton(self.menu_frame, text="Analyze Column", command=self.analyze_column)
            self.analyze_column_btn.pack(pady=5)
    
    def analyze_column(self):
        self.selected_column = self.column_selector.get()
        if self.selected_column:
            self.df['sentiment'] = self.df[self.selected_column].apply(lambda x: analyze_sentiment(str(x))[0])
            self.df['polarity'] = self.df[self.selected_column].apply(lambda x: analyze_sentiment(str(x))[1])
            self.create_chart_tabview()
    
    def create_chart_tabview(self):
        if self.chart_tabview:
            self.chart_tabview.destroy()
        
        self.chart_tabview = ctk.CTkTabview(self.content_frame)
        self.chart_tabview.pack(fill=tk.BOTH, expand=True)
        
        self.pie_tab = self.chart_tabview.add("Sentiment Pie Chart")
        self.wordcloud_tab = self.chart_tabview.add("Word Cloud")
        
        self.display_pie_chart()
        self.display_wordcloud()
    
    def display_pie_chart(self):
        sentiment_counts = self.df['sentiment'].value_counts()
        labels = sentiment_counts.index.tolist()
        sizes = sentiment_counts.values.tolist()
        colors = ['green', 'red', 'blue']
        
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        
        canvas = FigureCanvasTkAgg(fig, master=self.pie_tab)
        canvas.get_tk_widget().pack()
        canvas.draw()
    
    def display_wordcloud(self):
        text = ' '.join(self.df[self.selected_column].dropna().astype(str))
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        
        canvas = FigureCanvasTkAgg(fig, master=self.wordcloud_tab)
        canvas.get_tk_widget().pack()
        canvas.draw()
    
    def run(self):
        self.mainloop()

if __name__ == "__main__":
    app = WorkLifeBalanceApp()
    app.run()