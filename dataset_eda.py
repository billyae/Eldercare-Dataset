import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# KDE plot for emotional scores
def kde_plot_emotion_scores(df,type):

    """
    Plot KDE of emotional scores for each emotion label.
    
    """

    plt.figure(figsize=(12, 6))
    for emotion in df["emotion_label"].unique():
        subset = df[df["emotion_label"] == emotion]
        sns.kdeplot(subset["emotion_score"], label=emotion, fill=True, alpha=0.5)

    plt.title(f"KDE Plot of Emotional Scores by Emotion Label for {type} dataset")
    plt.xlabel("Emotion Score")
    plt.ylabel("Density")
    plt.legend(title="Emotion Label")
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

    # Define categorical and numerical columns
    categorical_columns = ["type", "human_type", "emotion_label"]

    numerical_columns = ["emotion_score"]

    # Analyze categorical features
    analyze_categorical_features(df, categorical_columns,"text")

    # Analyze emotional score features
    summarize_emotion_scores(df,"text")

    # Plot emotional scores
    plot_emotion_scores(df,"text")

    # Plot KDE of emotional scores
    kde_plot_emotion_scores(df,"text")

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

    # Define categorical and numerical columns
    categorical_columns = ['emotion', 'type', 'human_type']

    # Analyze categorical features
    analyze_categorical_features(df, categorical_columns,"audio")


def main():

    # Path to the dataset
    text_file_path = "conversation_text.csv"
    audio_file_path = "audio.csv"

    # Perform EDA on the conversation text dataset
    perform_eda_text(text_file_path)

    # Perform EDA on the conversation audio dataset

    perform_eda_audio(audio_file_path)

if __name__ == "__main__":
    main()