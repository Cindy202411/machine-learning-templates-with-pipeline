from xgboost import XGBClassifier

def build_model(params):
    return XGBClassifier(**params)
