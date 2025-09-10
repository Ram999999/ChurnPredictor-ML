import os
import json
import joblib
import pickle
import bz2
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	roc_auc_score,
	average_precision_score,
	precision_recall_curve,
	
)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from lifelines import CoxPHFitter

import shap

DATA_PATH = "Telco-Customer-Churn.csv"
MODEL_PATH = "model.pkl"
SURVIVAL_MODEL_PATH = "survivemodel.pkl"
EXPLAINER_PATH = "explainer.bz2"

APP_COLUMNS = [
	'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
	'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
	'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
	'InternetService_Fiber optic', 'InternetService_No', 'Contract_One year', 'Contract_Two year',
	'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

CATEGORICAL_YESNO = ['Partner', 'Dependents', 'PhoneService', 'OnlineSecurity', 'OnlineBackup',
	'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']

# These columns appear as strings with special values like 'No phone service' or 'No internet service'
DEPENDENT_ON_INT = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']


def _coerce_yes_no(series: pd.Series) -> pd.Series:
	return series.map({'Yes': 1, 'No': 0}).astype('float')


def _preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
	"""
	Transform raw Telco churn dataframe into the exact schema expected by the Flask app.
	Returns (X, y) where X has columns == APP_COLUMNS.
	"""
	data = df.copy()

	# Clean TotalCharges (some rows are blank)
	data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

	# Map gender Female->1, Male->0 to match app's select values
	data['gender'] = data['gender'].map({'Male': 0, 'Female': 1}).astype('float')

	# Ensure numeric types
	data['SeniorCitizen'] = data['SeniorCitizen'].astype('float')
	data['tenure'] = pd.to_numeric(data['tenure'], errors='coerce').astype('float')
	data['MonthlyCharges'] = pd.to_numeric(data['MonthlyCharges'], errors='coerce').astype('float')

	# Binary encode obvious Yes/No
	for col in CATEGORICAL_YESNO:
		data[col] = data[col].replace({'No internet service': 'No', 'No phone service': 'No'})
		data[col] = _coerce_yes_no(data[col])

	# PhoneService and MultipleLines special handling
	data['PhoneService'] = data['PhoneService'].map({'Yes': 1, 'No': 0}).astype('float')
	# MultipleLines: 'No phone service' -> 0; 'Yes'->1; 'No'->0
	data['MultipleLines'] = data['MultipleLines'].replace({'No phone service': 'No'})
	data['MultipleLines'] = data['MultipleLines'].map({'Yes': 1, 'No': 0}).astype('float')

	# InternetService dummies: base is DSL
	data['InternetService'] = data['InternetService'].fillna('DSL')
	data['InternetService_Fiber optic'] = (data['InternetService'] == 'Fiber optic').astype('float')
	data['InternetService_No'] = (data['InternetService'] == 'No').astype('float')
	# When no internet, dependent services must be zero
	no_internet_mask = data['InternetService'] == 'No'
	for col in DEPENDENT_ON_INT:
		data.loc[no_internet_mask, col] = 0.0

	# Contract dummies: base is Month-to-month
	data['Contract_One year'] = (data['Contract'] == 'One year').astype('float')
	data['Contract_Two year'] = (data['Contract'] == 'Two year').astype('float')

	# PaymentMethod dummies: base is Bank transfer (automatic)
	data['PaymentMethod_Credit card (automatic)'] = (data['PaymentMethod'] == 'Credit card (automatic)').astype('float')
	data['PaymentMethod_Electronic check'] = (data['PaymentMethod'] == 'Electronic check').astype('float')
	data['PaymentMethod_Mailed check'] = (data['PaymentMethod'] == 'Mailed check').astype('float')

	# Fill TotalCharges if missing
	missing_tc = data['TotalCharges'].isna()
	data.loc[missing_tc, 'TotalCharges'] = data.loc[missing_tc, 'MonthlyCharges'] * data.loc[missing_tc, 'tenure']

	# Assemble X in the app's expected column order
	X = data.reindex(columns=APP_COLUMNS)

	# Target: Churn Yes->1, No->0
	y = data['Churn'].map({'Yes': 1, 'No': 0}).astype('int')

	# Final clean
	X = X.fillna(0.0).astype('float')
	return X, y


def load_data(path: str = DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
	df = pd.read_csv(path)
	X, y = _preprocess_dataframe(df)
	return X, y


def evaluate_threshold_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
	pred = (y_proba >= threshold).astype(int)
	metrics = {
		"accuracy": accuracy_score(y_true, pred),
		"precision": precision_score(y_true, pred, zero_division=0),
		"recall": recall_score(y_true, pred, zero_division=0),
		"f1": f1_score(y_true, pred, zero_division=0),
		"roc_auc": roc_auc_score(y_true, y_proba),
		"pr_auc": average_precision_score(y_true, y_proba),
		"threshold": threshold,
	}
	return metrics


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, preference: str = "precision") -> float:
	precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
	thresholds = np.append(thresholds, 1.0)
	if preference == "precision":
		best_idx = np.argmax(precisions)
	else:
		f1s = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
		best_idx = np.argmax(f1s)
	return float(thresholds[best_idx])


def train_models() -> Dict[str, Any]:
	X, y = load_data()
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	# Pipelines
	log_reg = Pipeline(steps=[
		("scaler", StandardScaler(with_mean=False)),
		("clf", LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')),
	])

	rf = RandomForestClassifier(
		n_estimators=400,
		max_depth=None,
		min_samples_split=4,
		min_samples_leaf=2,
		random_state=42,
		class_weight='balanced_subsample',
	)

	# Grid on RF for a bit more rigor
	rf_grid = {
		"n_estimators": [300, 500],
		"max_depth": [None, 12, 18],
		"min_samples_split": [2, 4],
		"min_samples_leaf": [1, 2],
	}

	rf_search = GridSearchCV(
		estimator=rf,
		param_grid=rf_grid,
		scoring='average_precision',
		cv=3,
		n_jobs=-1,
		verbose=0,
	)

	# Fit models
	log_reg.fit(X_train, y_train)
	rf_search.fit(X_train, y_train)
	rf_best = rf_search.best_estimator_

	# Evaluate
	results = {}
	for name, model in [("log_reg", log_reg), ("random_forest", rf_best)]:
		proba = model.predict_proba(X_test)[:, 1]
		best_thr = find_best_threshold(y_test.values, proba, preference="precision")
		metrics = evaluate_threshold_metrics(y_test.values, proba, best_thr)
		metrics["best_threshold"] = best_thr
		results[name] = metrics

	# Select best by PR-AUC, tie-break on precision
	best_name = max(results.keys(), key=lambda k: (results[k]["pr_auc"], results[k]["precision"]))
	best_model = log_reg if best_name == "log_reg" else rf_best

	# Persist classifier
	with open(MODEL_PATH, "wb") as f:
		pickle.dump(best_model, f)

	# Train survival model (Cox PH) using same features plus duration/event
	cox_df = X.copy()
	cox_df['tenure'] = X['tenure']
	cox_df['event'] = y.astype(int)
	cph = CoxPHFitter(penalizer=0.1)
	# lifelines requires a DataFrame without perfectly collinear columns; small ridge via penalizer helps
	cph.fit(cox_df, duration_col='tenure', event_col='event', show_progress=False)
	with open(SURVIVAL_MODEL_PATH, "wb") as f:
		pickle.dump(cph, f)

	# Create SHAP explainer on a small background sample to keep size low
	background = X_train.sample(min(500, len(X_train)), random_state=42)
	try:
		explainer = shap.Explainer(best_model, background)
	except Exception:
		# fallback to Kernel for non-supported models (slower but fine for single predictions)
		explainer = shap.KernelExplainer(best_model.predict_proba, background)

	with bz2.BZ2File(EXPLAINER_PATH, 'w') as f:
		joblib.dump(explainer, f)

	artifact_report = {
		"selected_model": best_name,
		"metrics": results[best_name],
		"all_metrics": results,
	}
	return artifact_report


if __name__ == "__main__":
	report = train_models()
	print(json.dumps(report, indent=2))
