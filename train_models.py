import pandas as pd
import numpy as np
import pickle
import os
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')


class ModelTrainer:
    """Refined trainer for multiple classification models"""

    def __init__(self, data_path='data/breast_cancer.csv', model_dir='models'):
        self.data_path = Path(data_path)
        self.model_dir = Path(model_dir)
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()

        # Data placeholders
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None

        # Ensure directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load, split, and scale the dataset"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        df = pd.read_csv(self.data_path)
        X = df.drop('target', axis=1)
        y = df['target']

        # Stratified split to maintain class balance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale and save scaler
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        self._save_object(self.scaler, 'scaler.pkl')

        print(f"--- Data Summary ---")
        print(f"Features: {self.X_train.shape[1]} | Train: {len(self.X_train)} | Test: {len(self.X_test)}")
        print(f"Class Dist: {np.bincount(self.y_train)}\n")

    def train_models(self):
        """Initialize and train all classifiers using a dictionary-driven approach"""
        model_definitions = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
            'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
        }

        print(f"{'=' * 20} Training Phase {'=' * 20}")
        for name, model in model_definitions.items():
            print(f"Training: {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model

            # Save using slugified name
            file_name = f"{name.lower().replace(' ', '_')}.pkl"
            self._save_object(model, file_name)

        print("Done!\n")

    def evaluate_models(self):
        """Evaluate models and compile a performance report"""
        print(f"{'=' * 20} Evaluation Phase {'=' * 20}")

        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]

            self.results[name] = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'AUC': roc_auc_score(self.y_test, y_prob),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1': f1_score(self.y_test, y_pred),
                'MCC': matthews_corrcoef(self.y_test, y_pred)
            }
            print(f"✓ {name} evaluated.")

        self._save_object(self.results, 'results.pkl')
        return self.results

    def get_summary_df(self):
        """Returns a clean DataFrame of all results"""
        df = pd.DataFrame.from_dict(self.results, orient='index')
        return df.reset_index().rename(columns={'index': 'Model'}).round(4)

    def _save_object(self, obj, filename):
        """Helper to serialize objects"""
        with open(self.model_dir / filename, 'wb') as f:
            pickle.dump(obj, f)


def main():
    # Setup paths
    trainer = ModelTrainer(data_path='data/breast_cancer.csv')

    # Execute Pipeline
    try:
        trainer.load_data()
        trainer.train_models()
        trainer.evaluate_models()

        # Display and Save Final Results
        results_df = trainer.get_summary_df()
        print(f"\n{'=' * 20} Final Summary {'=' * 20}")
        print(results_df.to_string(index=False))

        csv_path = trainer.model_dir / 'evaluation_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"\nFinal report saved to: {csv_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()