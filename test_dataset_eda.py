import pytest
import pandas as pd
import os
from io import StringIO
from dataset_eda import (
    load_dataset,
    summarize_emotion_scores,
    summarize_dataset,
    check_missing_values,
    check_duplicates,
    analyze_categorical_features,
    plot_emotion_scores,
    kde_plot_emotion_scores,
    analyze_text_column,
    analyze_audio_column
)

@pytest.fixture
def sample_data():

    """
    Fixture to create a sample dataset for testing.

    """
    
    return pd.read_csv("conversation_text_test.csv")

@pytest.fixture
def sample_audio_data():
    """
    Fixture to create a sample audio dataset for testing.
    """
 
    return pd.read_csv("conversation_audio_test.csv")

def test_load_dataset(sample_data):
    """
    Test loading dataset from a CSV file.
    """
    temp_file = "test_dataset.csv"
    sample_data.to_csv(temp_file, index=False)
    loaded_df = load_dataset(temp_file)
    os.remove(temp_file)

    # Assert that the loaded DataFrame matches the sample DataFrame
    pd.testing.assert_frame_equal(sample_data, loaded_df)

def test_summarize_emotion_scores(sample_data, capsys):
    """
    Test summarizing emotion scores by emotion label.
    """
    summarize_emotion_scores(sample_data, "text")
    captured = capsys.readouterr()
    assert "neutral" in captured.out


def test_summarize_dataset(sample_data, capsys):
    """
    Test dataset summary statistics.
    """
    summarize_dataset(sample_data, "text")
    captured = capsys.readouterr()
    assert "Dataset Info" in captured.out
    assert "conversation_id" in captured.out
    assert "emotion_score" in captured.out

def test_check_missing_values(sample_data, capsys):
    """
    Test for missing values in the dataset.
    """
    check_missing_values(sample_data, "text")
    captured = capsys.readouterr()
    assert "Missing Values" in captured.out
    assert "0" in captured.out  # No missing values in the dataset

def test_check_duplicates(sample_data, capsys):
    """
    Test for duplicate rows in the dataset.
    """
    check_duplicates(sample_data, "text")
    captured = capsys.readouterr()
    assert "Duplicate Rows" in captured.out
    assert "0" in captured.out  # No duplicates in the dataset

def test_analyze_categorical_features(sample_data):
    """
    Test analysis of categorical features.
    """
    categorical_columns = ["type", "human_type", "emotion_label"]
    analyze_categorical_features(sample_data, categorical_columns, "text")
    # Ensure the function runs without errors (visual validation is needed for plots)

def test_plot_emotion_scores(sample_data):
    """
    Test plotting emotional scores by emotion label.
    """
    plot_emotion_scores(sample_data, "text")
    # Ensure the function runs without errors (visual validation is needed for plots)

def test_kde_plot_emotion_scores(sample_data):
    """
    Test KDE plot of emotional scores.
    """
    kde_plot_emotion_scores(sample_data, "text")
    # Ensure the function runs without errors (visual validation is needed for plots)

def test_analyze_text_column(sample_data):
    """
    Test the text column analysis function.
    """
    try:
        analyze_text_column(sample_data, "text", "text")
    except ValueError as e:
        assert "empty sequence" not in str(e)

def test_analyze_audio_column(sample_audio_data):

    """
    Test the audio column analysis function.

    """

    analyze_audio_column(sample_audio_data, "audio_path", "audio")
    assert "min_freq" in sample_audio_data.columns
    assert "max_freq" in sample_audio_data.columns
    assert "mean_freq" in sample_audio_data.columns