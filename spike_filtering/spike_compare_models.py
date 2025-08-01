from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from .spike_visualize import variance_explained, get_feature_matrix
import numpy as np
import pandas as pd
def get_top_feature_indices(mi_vals, n):
    """Return indices of the top n features by MI value."""
    return np.argsort(mi_vals)[-n:][::-1]

def get_pca_components(X, n_components):
    """Return the first n_components of PCA-transformed X."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def get_feature_combinations(X, mi_vals, n_top5=5, n_top3=3):
    """Return a dict of all 6 feature combinations."""
    idx_top5 = get_top_feature_indices(mi_vals, n_top5)
    idx_top3 = get_top_feature_indices(mi_vals, n_top3)
    combos = {}
    # Raw features
    combos['all'] = X
    # Raw top features
    combos['top5'] = X[:, idx_top5]
    combos['top3'] = X[:, idx_top3]
    # PCA on top features
    combos['pca2_top5'] = get_pca_components(X[:, idx_top5], 2)
    combos['pca3_top5'] = get_pca_components(X[:, idx_top5], 3)
    combos['pca2_top3'] = get_pca_components(X[:, idx_top3], 2)
    combos['pca3_top3'] = get_pca_components(X[:, idx_top3], 3)
    return combos, idx_top5, idx_top3

def get_models(random_state=42):
    """Return a dict of model name to model instance."""
    return {
        'GradientBoosting': GradientBoostingClassifier(random_state=random_state),
        'RandomForest': RandomForestClassifier(random_state=random_state),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=random_state)
    }

def evaluate_model(model : RandomForestClassifier , X_train, X_test, y_train, y_test, feature_names=None):
    """Train and evaluate a model, returning accuracy and ROC-AUC."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    if feature_names is not None and importances is not None:
        importance_dict = dict(zip(feature_names, importances))
    else:
        importance_dict = importances
    return acc, auc, report, importance_dict

def preprocess_data(df : pd.DataFrame):
    feature_matrix = get_feature_matrix(df, 'raw_z_features')
    labels = df['label'].values
    f_valse, mi_vals = variance_explained(feature_matrix, labels)
    return feature_matrix, labels, mi_vals

def compare_models(df : pd.DataFrame):
    """Run all models on all feature combinations and print results."""
    X, y, mi_vals = preprocess_data(df)
    feature_names = eval(df.iloc[0]['feature_names']) if 'feature_names' in df.columns else None
    combos, idx_top5, idx_top3 = get_feature_combinations(X, mi_vals)
    models = get_models()
    results = {}
    X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    for combo_name, X_combo in combos.items():
        # Determine feature names for this combo
        if combo_name == 'top5':
            fn = [feature_names[i] for i in idx_top5] if feature_names is not None else None
        elif combo_name == 'top3':
            fn = [feature_names[i] for i in idx_top3] if feature_names is not None else None
        elif combo_name == 'all':
            fn = feature_names
        else:
            fn = [f"PC{i+1}" for i in range(X_combo.shape[1])]
            # PCA features: name as PC1, PC2, ...
        X_train, X_test = train_test_split(X_combo, test_size=0.2, random_state=42, stratify=y)
        results[combo_name] = {}
        for model_name, model in models.items():
            acc, auc, report,importances = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names=fn)
            results[combo_name][model_name] = {'accuracy': acc, 'roc_auc': auc, 'report': report, 'improtances': importances}
            print(f"\n[{model_name} | {combo_name}] Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}")
            if importances is not None:
                print("Feature importances:")
                if isinstance(importances, dict):
                    for k, v in sorted(importances.items(), key=lambda x: -x[1]):
                        print(f"  {k:20s}: {v:.3f}")
                else:
                    print(importances)
    # Optionally, return results for further analysis
    return results

# Example usage in your main_plot or after MI calculation:
# f_vals, mi_vals = variance_explained(X, y)
# results = compare_models(X, y, mi_vals, feature_names)