import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler

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
