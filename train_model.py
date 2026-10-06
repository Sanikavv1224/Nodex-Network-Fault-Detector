"""
train_model.py
----------------
JOB: Train a Decision Tree Classifier that learns:
     "given wifi_signal, router_ping, external_ping -> what's the fault?"

 - Training ranges cover what a REAL laptop/phone-hotspot produces
   (a healthy hotspot often has 60-120 ms external ping).
 - Timeouts (999) are part of the training data, because monitor.py logs them.
 - Prints an evaluation (accuracy + confusion matrix) and saves it to
   model_evaluation.txt, so the report can include real numbers.

HONEST NOTE: the data is SYNTHETIC. Accuracy on a held-out part of the same
synthetic generator is optimistic - it shows the tree learned the rules we
wrote, not that it is perfect on real networks. Real-world testing is the true test.
"""

import random
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

TIMEOUT = 999
LABELS = ["Normal", "WiFi", "Router", "ISP"]


def maybe_timeout(value, chance):
    """With probability `chance`, the ping got NO reply (999 sentinel)."""
    return TIMEOUT if random.random() < chance else value


def generate_sample(label):
    """One row of (wifi_signal, router_ping, external_ping) for a given label."""
    if label == "Normal":
        # healthy: good signal, router fast, internet 20-140 ms
        return [random.randint(60, 100), random.uniform(1, 45), random.uniform(20, 140)]

    if label == "WiFi":
        # weak signal is the root cause; pings are bad or lost because of it
        return [random.randint(5, 45),
                maybe_timeout(random.uniform(20, 400), 0.15),
                maybe_timeout(random.uniform(50, 500), 0.25)]

    if label == "Router":
        # signal OK, but the router itself is slow or not replying
        return [random.randint(55, 100),
                maybe_timeout(random.uniform(100, 400), 0.25),
                maybe_timeout(random.uniform(30, 400), 0.15)]

    if label == "ISP":
        # signal OK, router fast, internet beyond it slow or not replying
        return [random.randint(55, 100),
                random.uniform(1, 45),
                maybe_timeout(random.uniform(170, 500), 0.30)]


def build_training_data(samples_per_class=400):
    rows = []
    for label in LABELS:
        for _ in range(samples_per_class):
            wifi, router, external = generate_sample(label)
            rows.append({"wifi_signal": wifi, "router_ping": router,
                         "external_ping": external, "label": label})
    return pd.DataFrame(rows)


def train_and_save_model(path="model.joblib"):
    random.seed(42)
    df = build_training_data()
    X = df[["wifi_signal", "router_ping", "external_ping"]]
    y = df["label"]

    # Hold back 20% to test on data the tree has never seen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y)

    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = pd.DataFrame(confusion_matrix(y_test, pred, labels=LABELS),
                      index=[f"true {l}" for l in LABELS],
                      columns=[f"pred {l}" for l in LABELS])
    report = classification_report(y_test, pred, labels=LABELS, digits=3)

    text = (f"Held-out test accuracy (synthetic data): {acc:.1%}\n\n"
            f"Confusion matrix:\n{cm}\n\n{report}\n"
            "Note: tested on data from the same synthetic generator, so this is\n"
            "optimistic. See real-world hotspot tests for true performance.\n")
    print(text)
    with open("model_evaluation.txt", "w") as f:
        f.write(text)

    # Final model uses ALL the data
    model.fit(X, y)
    joblib.dump(model, path)
    print(f"Model trained on {len(df)} samples and saved to {path}")
    return model


if __name__ == "__main__":
    train_and_save_model()