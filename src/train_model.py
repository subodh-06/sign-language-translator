import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_CSV = os.path.join("data", "sign_data.csv")
MODEL_PATH = os.path.join("models", "sign_model.pkl")
LABEL_ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

def main():
    if not os.path.exists(DATA_CSV):
        print(f"Data file not found: {DATA_CSV}")
        return

    os.makedirs("models", exist_ok=True)

    # Load data
    df = pd.read_csv(DATA_CSV)

    # Separate features and labels
    X = df.drop("label", axis=1).values
    y = df["label"].values

    # Encode labels A-Z -> 0-25
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split into train & test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Create and train model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Evaluate
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save model and label encoder
    joblib.dump(knn, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Label encoder saved to {LABEL_ENCODER_PATH}")

if __name__ == "__main__":
    main()
