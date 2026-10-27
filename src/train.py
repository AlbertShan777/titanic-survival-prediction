"""Evaluate the original experiments and generate a Kaggle submission."""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from .titanic import FEATURES, add_features, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    train = add_features(pd.read_csv(args.data_dir / "train.csv"))
    test = add_features(pd.read_csv(args.data_dir / "test.csv"))
    X, y = train[FEATURES], train["Survived"]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    results = {}
    for name in ["baseline_logistic", "logistic", "random_forest", "tuned_random_forest", "xgboost"]:
        model = build_model("logistic" if name == "baseline_logistic" else name,
                            include_title=name != "baseline_logistic")
        model.fit(X_train, y_train)
        results[name] = accuracy_score(y_valid, model.predict(X_valid))
    model = build_model()
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    results["logistic_cv"] = {"scores": scores.tolist(), "mean": float(scores.mean()),
                              "std": float(scores.std())}
    model.fit(X, y)
    predictions = model.predict(test[FEATURES])
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
    assert len(submission) == len(test) and submission["Survived"].isin([0, 1]).all()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_dir / "submission.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
