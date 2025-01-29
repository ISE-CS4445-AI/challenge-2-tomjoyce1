# DO NOT CHANGE THIS FILE! If any code is changed, the instructor will be notified on Github classroom's assignment dashboard.

# Common imports
import pandas as pd
import numpy as np
import os

# Importing student's solution
import challenge_2_export  # the python file created after nbconvert

def test_mcqfunction_1():
    # Test MCQ function 1's answer
    q1_ans = os.environ.get("C2_MCQF_1", "")
    assert q1_ans, "No C2_MCQF_1 found in environment!"
    
    if challenge_2_export.answer_q1() != q1_ans:
        raise AssertionError("Wrong answer for MCQ 1!")

def test_mcqfunction_2():
    # Test MCQ function 2's answer
    q2_ans = os.environ.get("C2_MCQF_2", "")
    assert q2_ans, "No C2_MCQF_2 found in environment!"

    if challenge_2_export.answer_q2() != q2_ans:
        raise AssertionError("Wrong answer for MCQ 2!")

def test_mcqfunction_3():
    # Test MCQ function 3's answer
    q3_ans = os.environ.get("C2_MCQF_3", "")
    assert q3_ans, "No C2_MCQF_3 found in environment!"

    if challenge_2_export.answer_q3() != q3_ans:
        raise AssertionError("Wrong answer for MCQ 3!")

def test_mcqfunction_4():
    # Test MCQ function 4's answer
    q4_ans = os.environ.get("C2_MCQF_4", "")
    assert q4_ans, "No C2_MCQF_4 found in environment!"

    if challenge_2_export.answer_q4() != q4_ans:
        raise AssertionError("Wrong answer for MCQ 4!")
    
def test_mcqfunction_5():
    # Test MCQ function 5's answer
    q5_ans = os.environ.get("C2_MCQF_5", "")
    assert q5_ans, "No C2_MCQF_5 found in environment!"

    ans_for_q5, why = challenge_2_export.answer_q5()
    if ans_for_q5 != q5_ans:
        raise AssertionError("Wrong answer for MCQ 5!")
    assert why, "No explanation found for MCQ 5!"

def test_preprocess_no_na():
    """
    Check that after running preprocess_final(df),
    essential columns have no missing values.
    """
    test_df = challenge_2_export.load_data('titanic.csv')
    
    # check total missing values before preprocessing
    na_count = challenge_2_export.number_missing_values(test_df).sum()
    total_na = os.environ.get("C2_TOTAL_NA", "")
    assert total_na, "No C2_TOTAL_NA found in environment!"
    assert na_count == int(total_na), f"Missing value count before preprocess_final not equal to expected value!"

    # no missing values should be present after preprocessing
    df_proc = challenge_2_export.preprocess_final(test_df)
    na_count = df_proc.isnull().sum().sum()
    assert na_count == 0, f"Found {na_count} missing values after preprocess_final!"
    
def test_columns_dropped():
    """
    Check that certain columns (Cabin, Ticket, Name, PassengerId) are gone.
    """
    test_df = challenge_2_export.load_data('titanic.csv')
    df_proc = challenge_2_export.preprocess_final(test_df)
    
    # columns that must NOT appear
    forbidden_cols = os.environ.get("C2_FORBIDDEN_COLS", "")
    assert forbidden_cols, "No C2_FORBIDDEN_COLS found in environment!"

    for col in forbidden_cols.split(','):
        assert col not in df_proc.columns, f"Column {col} was not dropped!"

def test_sex_embarked_encoded():
    """
    Check 'Sex' is numeric 0/1,
    and 'Embarked' if present => one-hot or numeric columns
    """
    test_df = challenge_2_export.load_data('titanic.csv')
    df_proc = challenge_2_export.preprocess_final(test_df)
    
    # check sex
    assert df_proc['Sex'].dtype in [np.int64, np.float64], "Sex not numeric after encoding!"
    # check embarked is not present or if present, it's numeric columns
    embarked_cols = [col for col in df_proc.columns if 'Embarked' in col]
    for c in embarked_cols:
        assert df_proc[c].dtype in [np.int64, np.float64, np.bool], f"Embarked col {c} not numeric!"
    
def test_new_feature_created():
    """
    Check that at least 1 new feature is present.
    E.g. 'FamilySize' or something else not in original set.
    We'll guess 'FamilySize' but flexible if we see something else.
    """
    test_df = challenge_2_export.load_data('titanic.csv')
    df_proc = challenge_2_export.preprocess_final(test_df)
    
    original_cols = ['Survived','Sex','Age','SibSp','Parch','Fare','Embarked']
    # after dropping columns, we expect some new col not in original
    # e.g. 'FamilySize' or something
    new_cols = set(df_proc.columns) - set(original_cols)
    # remove any one-hot columns from Embarked
    new_cols = {col for col in new_cols if not col.startswith('Embarked_')}
    
    assert len(new_cols) >= 1, "No new feature found! Must create at least 1 new feature."
