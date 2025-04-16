import pandas as pd

def preprocess(df):
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
    df['appointment_day'] = pd.to_datetime(df['appointment_day'])
    df['days_between'] = (df['appointment_day'] - df['scheduled_day']).dt.days

    df['no_show'] = df['no_show'].map({'No': 0, 'Yes': 1})
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})

    df = df.dropna()
    return df
