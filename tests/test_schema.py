import pandas as pd
from src.data import makeDataSet
from src.config import All_Cols, age_min, age_max, gender_values, age_group_values, label_values

def test_schema_columns_present_and_ordered():
    df = makeDataSet(seed=1)
    assert set(df.columns) == set(All_Cols)

def test_basic_value_rules():
    df = makeDataSet(seed=1)
    assert df["age"].between(age_min, age_max - 1).all()
    assert set(df["gender"].unique()).issubset(set(gender_values))
    assert set(df["age_group"].unique()).issubset(set(age_group_values))
    assert set(df["label"].unique()).issubset(set(label_values))