import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

def load_dataset(file_path):

    """
    Load the dataset into a Pandas DataFrame.

    """

    return pd.read_csv(file_path)


def summarize_emotion_scores(df,type):

    """
    Group by emotion label and calculate summary statistics for emotional scores.
    
    """

    grouped = df.groupby("emotion_label")["emotion_score"].describe()
    print(f"Summary Statistics for Emotional Scores by Emotion Label for  {type} dataset:")
    print(grouped)


def summarize_dataset(df,type):

    """
    Print summary statistics and basic information about the dataset.

    """

    print(f"Dataset Info for {type} dataset:")
    print(df.info())
    print(f"\nDataset Head for {type} dataset:")
    print(df.head())
    print(f"\nSummary Statistics for {type} dataset:")
    print(df.describe(include='all'))

def check_missing_values(df,type):

    """
    Check for missing values in the dataset.

    """

    print(f"\nMissing Values for {type} dataset:")
    print(df.isnull().sum())

def check_duplicates(df,type):

    """
    Check for duplicate rows in the dataset.

    """

    print(f"\nDuplicate Rows for {type} dataset:")
    print(df.duplicated().sum())

def analyze_categorical_features(df, categorical_columns,type):

    """
    Analyze and plot the distribution of categorical features.

    """

    for column in categorical_columns:
        print(f"\nValue Counts for {column} for {type} dataset:")
        print(df[column].value_counts())
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=column, order=df[column].value_counts().index)
        plt.title(f"Distribution of {column} for {type} dataset")
        plt.xticks(rotation=45)
        plt.show()

def plot_emotion_scores(df,type):

    """
    Visualize the distribution of emotional scores by emotion label.
    
    """

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="emotion_label", y="emotion_score", palette="Set2")
    plt.title(f"Boxplot of Emotional Scores by Emotion Label for {type} dataset")
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x="emotion_label", y="emotion_score", palette="Set3", inner="point")
    plt.title(f"Violin Plot of Emotional Scores by Emotion Label for {type} dataset")
    plt.xticks(rotation=45)
    plt.show()

def kde_plot_emotion_scores(df,type):

    """
    Plot KDE of emotional scores for each emotion label.
    
    """

    plt.figure(figsize=(12, 6))
    for emotion in df["emotion_label"].unique():
        subset = df[df["emotion_label"] == emotion]
        sns.kdeplot(subset["emotion_score"], label=emotion, fill=True, alpha=0.5,warn_singular=False)

    plt.title(f"KDE Plot of Emotional Scores by Emotion Label for {type} dataset")
    plt.xlabel("Emotion Score")
    plt.ylabel("Density")
    plt.legend(title="Emotion Label")
    plt.show()

def analyze_text_column(df, column_name, type):

    """
    Perform EDA on the text column of the dataset.
    :param df: DataFrame containing the dataset.
    :param column_name: The name of the text column to analyze.
    :param type: Dataset type for labeling in plots and prints.

    """
    
    nltk.download("punkt", quiet=True)

    # Word and sentence length calculations
    df["word_count"] = df[column_name].apply(lambda x: len(word_tokenize(str(x))))
    df["sentence_count"] = df[column_name].apply(lambda x: len(sent_tokenize(str(x))))

    # Summary statistics
    print(f"\nSummary Statistics for {column_name} column in {type} dataset:")
    print(df[["word_count", "sentence_count"]].describe())

    min_word_length=8

    # Tokenize and filter words by length
    all_words = [
        word.lower() for text in df[column_name].dropna() for word in word_tokenize(text)
        if len(word) >= min_word_length
    ]
    
    # Count word frequencies
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(30)

    # Print the most common words
    print(f"\nMost Common Words (Length >= {min_word_length}) in {column_name} for {type} dataset:")
    for word, count in common_words:
        print(f"{word}: {count}")

    # Plot the most common words
    common_words_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])
    plt.figure(figsize=(10, 5))
    sns.barplot(data=common_words_df, x="Frequency", y="Word", palette="viridis")
    plt.title(f"Most Common Words (Length >= {min_word_length}) in {column_name} for {type} dataset")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.show()

def analyze_frequency_ranges(audio_path):

    """
    Analyze the frequency ranges of a given audio file.
    :param audio_path: Path to the audio file.
    :return: Dictionary with min, max, and mean frequencies.

    """

    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Handle stereo by averaging channels if necessary
        if y.ndim == 2:  # Stereo
            y = np.mean(y, axis=0)
        
        # Compute FFT
        fft = np.fft.fft(y)
        magnitude = np.abs(fft)
        frequencies = np.fft.fftfreq(len(magnitude), d=1/sr)

        # Keep only the positive frequencies
        positive_freqs = frequencies[:len(frequencies)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]

        # Frequency statistics
        min_freq = positive_freqs[np.argmax(positive_magnitude > 0)]  # First non-zero frequency
        max_freq = positive_freqs[np.argmax(positive_magnitude[::-1] > 0)]  # Last non-zero frequency
        mean_freq = np.average(positive_freqs, weights=positive_magnitude)  # Weighted average frequency

        return {"min_freq": min_freq, "max_freq": max_freq, "mean_freq": mean_freq}

    except Exception as e:
        print(f"Error analyzing frequency for {audio_path}: {e}")
        return {"min_freq": None, "max_freq": None, "mean_freq": None}

def analyze_audio_column(df, column_name, dataset_type):

    """
    Perform frequency range analysis for the audio column in the dataset.
    :param df: DataFrame containing the dataset.
    :param column_name: The name of the audio column to analyze.
    :param dataset_type: Dataset type for labeling in plots and prints.

    """

    frequency_data = []

    print(f"\nAnalyzing frequency ranges for {column_name} in {dataset_type} dataset...")
    for audio_path in df[column_name]:
        if os.path.exists(audio_path):
            freq_analysis = analyze_frequency_ranges(audio_path)
            frequency_data.append(freq_analysis)
        else:
            frequency_data.append({"min_freq": None, "max_freq": None, "mean_freq": None})

    # Add frequency data to DataFrame
    freq_df = pd.DataFrame(frequency_data)
    df["min_freq"] = freq_df["min_freq"]
    df["max_freq"] = freq_df["max_freq"]
    df["mean_freq"] = freq_df["mean_freq"]

    # Summary statistics
    print("\nFrequency Range Summary Statistics:")
    print(df[["min_freq", "max_freq", "mean_freq"]].describe())

    # Plot mean frequencies
    plt.figure(figsize=(10, 5))
    plt.hist(df["mean_freq"].dropna(), bins=20, color="skyblue", edgecolor="black")
    plt.title(f"Histogram of Mean Frequencies in {dataset_type} Dataset")
    plt.xlabel("Mean Frequency (Hz)")
    plt.ylabel("Count")
    plt.grid(axis="y")
    plt.show()


def perform_eda_text(file_path):

    """
    Perform EDA on the conversation dataset.
    
    """

    # Load dataset
    df = load_dataset(file_path)

    # Summarize dataset
    summarize_dataset(df,"text")

    # Check for missing values
    check_missing_values(df,"text")

    # Check for duplicate rows
    check_duplicates(df,"text")

    # Define categorical and numerical columns
    categorical_columns = ["type", "human_type", "emotion_label"]

    # Analyze categorical features
    analyze_categorical_features(df, categorical_columns,"text")

    # Analyze emotional score features
    summarize_emotion_scores(df,"text")

    # Plot emotional scores
    plot_emotion_scores(df,"text")

    # Plot KDE of emotional scores
    kde_plot_emotion_scores(df,"text")

    # Analyze the text column
    analyze_text_column(df, "text", "text")

def perform_eda_audio(file_path):

    """
    Perform EDA on the audio dataset.
    
    """

    # Load dataset
    df = load_dataset(file_path)

    # Summarize dataset
    summarize_dataset(df,"audio")

    # Check for missing values
    check_missing_values(df,"audio")

    # Check for duplicate rows
    check_duplicates(df,"audio")

    # Define categorical and numerical columns
    categorical_columns = ['emotion', 'type', 'human_type']

    # Analyze categorical features
    analyze_categorical_features(df, categorical_columns,"audio")

    # Analyze audio column
    analyze_audio_column(df, "audio_path", "audio")


def main():

    # Path to the dataset
    text_file_path = "conversation_text.csv"
    audio_file_path = "conversation_audio.csv"

    # Perform EDA on the conversation text dataset
    perform_eda_text(text_file_path)

    # Perform EDA on the conversation audio dataset

    perform_eda_audio(audio_file_path)

if __name__ == "__main__":
    main()