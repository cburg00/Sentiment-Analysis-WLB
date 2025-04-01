import streamlit as st
import pandas as pd
from analysis_function import *
from utils import *
from config import *

def init_app():
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
    st.title("Work-Life Balance Analysis Dashboard")
    st.markdown("Analyze both text sentiment and numeric ratings about work-life balance")

def sidebar_controls():
    with st.sidebar:
        st.header("Input Options")
        input_method = st.radio("Choose input method:", 
                              ("Text Input", "CSV Upload", "Pre-loaded Dataset"))
        
        if input_method == "CSV Upload":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state['df'] = df
                st.session_state['available_columns'] = list(df.columns)
        
        elif input_method == "Pre-loaded Dataset":
            if st.button("Load Employee Reviews Dataset"):
                with st.spinner('Loading dataset...'):
                    df = load_employee_reviews()
                    st.session_state['df'] = df
                    st.session_state['available_columns'] = list(df.columns)
                    st.success("Dataset loaded successfully!")
        
        return input_method

def text_input_analysis():
    user_input = st.text_area("Enter work-life balance feedback:", height=150)
    if st.button("Analyze Text") and user_input.strip():
        sentiment, polarity = analyze_sentiment(user_input)
        col1, col2 = st.columns(2)
        with col1: st.metric("Sentiment", sentiment)
        with col2: st.metric("Polarity Score", f"{polarity:.2f}")

def dataset_analysis(input_method):
    if (input_method in ["CSV Upload", "Pre-loaded Dataset"]) and 'df' in st.session_state:
        df = st.session_state['df']
        st.subheader("Analysis Setup")
        selected_column = st.selectbox(
            "Select the column to analyze:",
            st.session_state['available_columns'],
            key='column_selector'
        )
        
        is_numeric = is_numeric_column(df[selected_column])
        analysis_type = 'numeric' if is_numeric else 'text'
        
        if st.button(f"Analyze as {'Numeric Ratings' if is_numeric else 'Text Sentiment'}"):
            with st.spinner('Analyzing...'):
                st.session_state['analysis_df'] = df.copy()
                st.session_state['analysis_type'] = analysis_type
                
                if analysis_type == 'numeric':
                    st.session_state['analysis_df']['rating'] = df[selected_column].apply(convert_to_numeric)
                else:
                    st.session_state['analysis_df']['sentiment'] = df[selected_column].apply(lambda x: analyze_sentiment(x)[0])
                    st.session_state['analysis_df']['polarity'] = df[selected_column].apply(lambda x: analyze_sentiment(x)[1])
        
        if 'analysis_df' in st.session_state:
            display_analysis_results(selected_column)

def display_analysis_results(selected_column):
    analysis_df = st.session_state['analysis_df']
    st.write(f"Analyzing {len(analysis_df)} records from column: '{selected_column}'")
    
    if st.session_state['analysis_type'] == 'numeric':
        display_numeric_analysis(analysis_df, selected_column)
    else:
        display_text_analysis(analysis_df, selected_column)

def display_numeric_analysis(df, column_name):
    st.subheader("Numeric Ratings Analysis")
    results = analyze_numeric_ratings(df, 'rating')
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Rating", f"{results['average']:.2f}")
        st.metric("Median Rating", f"{results['median']:.2f}")
    
    with col2:
        st.metric("Minimum Rating", results['min'])
        st.metric("Maximum Rating", results['max'])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['rating'], bins=20, kde=True, ax=ax)
    ax.set_xlabel("Rating Value")
    st.pyplot(fig)
    
    st.subheader("Rating Distribution")
    st.bar_chart(results['distribution'])

def display_text_analysis(df, column_name):
    st.subheader("Text Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Positive %", f"{100*(df['sentiment'] == 'Positive').mean():.1f}%")
        st.metric("Negative %", f"{100*(df['sentiment'] == 'Negative').mean():.1f}%")
    
    with col2:
        st.metric("Average Polarity", f"{df['polarity'].mean():.2f}")
        st.metric("Neutral %", f"{100*(df['sentiment'] == 'Neutral').mean():.1f}%")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red', 'blue'], ax=ax1)
    ax1.set_ylabel('')
    sns.histplot(df['polarity'], bins=20, kde=True, ax=ax2)
    ax2.set_xlabel("Sentiment Polarity (-1 to 1)")
    st.pyplot(fig)
    
    st.subheader("Word Cloud")
    st.pyplot(generate_wordcloud(df[column_name]))
    
def display_analysis_results(selected_column):
    analysis_df = st.session_state['analysis_df']
    st.write(f"Analyzing {len(analysis_df)} records from column: '{selected_column}'")
    
    if st.session_state['analysis_type'] == 'numeric':
        display_numeric_analysis(analysis_df, selected_column)
    else:
        display_text_analysis(analysis_df, selected_column)
    
    st.subheader("Sample Data Preview")
    sample_size = st.slider("Number of samples to show", 1, 20, 5)
    sample_data = get_sample_data(
        analysis_df, 
        selected_column, 
        st.session_state['analysis_type'],
        sample_size
    )
    st.dataframe(sample_data)

def main():
    init_app()
    input_method = sidebar_controls()
    
    if input_method == "Text Input":
        text_input_analysis()
    else:
        dataset_analysis(input_method)
    
    st.markdown("---")
    st.markdown("""
    *Analysis types:*
    - *Text Sentiment: Uses NLP to analyze emotional tone (-1 to 1 polarity)*
    - *Numeric Ratings: Analyzes score-based ratings (e.g., 1-5 stars)*
    """)

if __name__ == "__main__":
    main()