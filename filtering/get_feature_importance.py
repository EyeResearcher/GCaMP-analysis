def export_feature_importance(model_path="roi_classifier_model.pkl",
                              label_csv_path="roi_labels.csv",
                              output_csv_path="ranked_feature_importance.csv"):
    """
    Loads a trained model and label data, extracts feature importances,
    and saves the ranked list to a CSV.
    """
    from joblib import load
    import pandas as pd
    from pathlib import Path
    full_path = Path(input("What is the path to the model you want to check?"))
    # Load model and data
    model = load(full_path / model_path)
    df = pd.read_csv(full_path / label_csv_path)

    # Extract feature columns
    feature_columns = df.columns.difference(["label", "source_file"])
    importances = model.feature_importances_
    print(f"Feature importances: {importances}")
    # Rank and save
    ranked_features = sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)
    ranked_df = pd.DataFrame(ranked_features, columns=["Feature", "Importance"])
    ranked_df.to_csv(full_path / output_csv_path, index=False)
    print(f"Feature importances saved to: {output_csv_path}")
    
export_feature_importance()