"""Reusable feature engineering and model factories extracted from Albert's notebook."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL = ["Sex", "Embarked", "Title"]
FEATURES = NUMERIC + CATEGORICAL


def add_features(frame):
    frame = frame.copy()
    frame["FamilySize"] = frame["SibSp"] + frame["Parch"] + 1
    frame["IsAlone"] = (frame["FamilySize"] == 1).astype(int)
    frame["Title"] = frame["Name"].str.extract(r",\s*([^.]*)\.")
    frame["Title"] = frame["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    common_titles = ["Mr", "Miss", "Mrs", "Master"]
    frame["Title"] = frame["Title"].apply(lambda x: x if x in common_titles else "Rare")
    return frame


def build_model(kind="logistic", include_title=True):
    categorical = CATEGORICAL if include_title else ["Sex", "Embarked"]
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    if kind == "logistic":
        classifier = LogisticRegression(max_iter=1000)
    elif kind == "random_forest":
        classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    elif kind == "tuned_random_forest":
        classifier = RandomForestClassifier(n_estimators=300, max_depth=5,
            min_samples_split=5, min_samples_leaf=2, random_state=42)
    elif kind == "xgboost":
        from xgboost import XGBClassifier
        classifier = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss")
    else:
        raise ValueError(f"Unknown model: {kind}")
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
