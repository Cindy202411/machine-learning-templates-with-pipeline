import sys
import importlib
from sklearn.model_selection import train_test_split
from config import load_config
from data_loader import load_data
from preprocess import preprocess
from evaluate import evaluate_model

def run():
    config = load_config()
    df = load_data(config['sqlite_path'], config['table_name'])
    df = preprocess(df)

    X = df[config['features']]
    y = df[config['target']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_state']
    )

    if config["model"] == "xgboost":
        model = importlib.import_module("models.xgboost_model").build_model(config["xgboost_params"])
    else:
        model = importlib.import_module(f"models.{config['model']}_model").build_model()

    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    run()

