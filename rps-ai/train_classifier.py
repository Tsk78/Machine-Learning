import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train():
    df = pd.read_csv("rps_dataset.csv")

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Training Accuracy: {acc * 100:.2f}%")

    with open("rps_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("Classifier saved to 'rps_classifier.pkl'!")


if __name__ == "__main__":
    train()