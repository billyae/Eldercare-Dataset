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
def sample_text_data():

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

def test_load_dataset(sample_text_data):

    """
    Test loading dataset from a CSV file.
    params:sample_text_data: Sample dataset to test loading from CSV file.

    """

    temp_file = "test_dataset.csv"
    sample_text_data.to_csv(temp_file, index=False)
    loaded_df = load_dataset(temp_file)
    os.remove(temp_file)

    # Assert that the loaded DataFrame matches the sample DataFrame
    pd.testing.assert_frame_equal(sample_text_data, loaded_df)

def test_summarize_emotion_scores(sample_text_data, capsys):
    
    """
    Test summarizing emotion scores by emotion label.
    params:sample_text_data: Sample dataset to test summarizing emotion scores.
    params:capsys: Pytest fixture to capture stdout and stderr.

    """

    summarize_emotion_scores(sample_text_data, "text")
    captured = capsys.readouterr()
    assert "neutral" in captured.out


def test_summarize_dataset(sample_text_data, capsys):

    """
    Test dataset summary statistics.
    params:sample_text_data: Sample dataset to test summarizing.
    params:capsys: Pytest fixture to capture stdout and stderr.

    """

    summarize_dataset(sample_text_data, "text")
    captured = capsys.readouterr()
    assert "Dataset Info" in captured.out
    assert "conversation_id" in captured.out
    assert "emotion_score" in captured.out

def test_check_missing_values(sample_text_data, capsys):

    """
    Test for missing values in the dataset.
    params:sample_text_data: Sample dataset to test missing values.
    params:capsys: Pytest fixture to capture stdout and stderr

    """

    check_missing_values(sample_text_data, "text")
    captured = capsys.readouterr()
    assert "Missing Values" in captured.out
    assert "0" in captured.out  # No missing values in the dataset

def test_check_duplicates(sample_text_data, capsys):

    """
    Test for duplicate rows in the dataset.
    params:sample_text_data: Sample dataset to test for duplicates.
    params:capsys: Pytest fixture to capture stdout and stderr

    """

    check_duplicates(sample_text_data, "text")
    captured = capsys.readouterr()
    assert "Duplicate Rows" in captured.out
    assert "0" in captured.out  # No duplicates in the dataset

def test_analyze_categorical_features(sample_text_data):

    """
    Test analysis of categorical features.
    params:sample_text_data: Sample dataset to test categorical feature analysis.

    """

    categorical_columns = ["type", "human_type", "emotion_label"]
    analyze_categorical_features(sample_text_data, categorical_columns, "text")
    # Ensure the function runs without errors (visual validation is needed for plots)

def test_analyze_text_column(sample_text_data):
    
    """
    Test the text column analysis function.
    params:sample_text_data: Sample dataset to test text column

    """

    try:
        analyze_text_column(sample_text_data, "text", "text")
    except ValueError as e:
        assert "empty sequence" not in str(e)

def test_analyze_audio_column(sample_audio_data):

    """
    Test the audio column analysis function.
    params:sample_audio_data: Sample dataset to test audio

    """

    analyze_audio_column(sample_audio_data, "audio_path", "audio")
    assert "min_freq" in sample_audio_data.columns
    assert "max_freq" in sample_audio_data.columns
    assert "mean_freq" in sample_audio_data.columns