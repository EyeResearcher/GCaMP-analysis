"""Compare ROI classifier performance with different normalization strategies."""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from roi_classifier.feature_extraction import prepare_roi_training_data
from roi_classifier.train import train_roi_classifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_normalizations():
    """
    Compare MinMax vs DeltaF/F normalization for ROI classifier.
    
    Extracts features with both strategies and trains separate models.
    """
    # Paths
    labels_path = Path('training_data/roi__filtering/roi_labels.csv')
    
    if not labels_path.exists():
        logger.error(f"Labels file not found: {labels_path}")
        return
    
    logger.info("="*80)
    logger.info("ROI Classifier Normalization Comparison")
    logger.info("="*80)
    
    # Test both normalization strategies
    strategies = ['minmax', 'deltaf']
    results = {}
    
    for norm in strategies:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {norm.upper()} normalization")
        logger.info(f"{'='*80}\n")
        
        # Extract features with this normalization
        features_path = Path(f'training_data/roi__filtering/roi_features_{norm}.csv')
        logger.info(f"Extracting features with {norm} normalization...")
        features_df = prepare_roi_training_data(
            labels_path, 
            output_path=features_path,
            normalization=norm
        )
        
        logger.info(f"\nFeature statistics for {norm}:")
        print(features_df[['derivative_skew', 'spike_prom_mean']].describe())
        
        # Train classifier
        model_path = Path(f'roi_classifier/models/roi_classifier_{norm}.pkl')
        logger.info(f"\nTraining classifier with {norm} normalization...")
        train_roi_classifier(
            features_path,
            model_path,
            test_size=0.15,  # Use 85% for training
            normalization=norm
        )
        
        # Store results
        from joblib import load
        model_dict = load(model_path)
        results[norm] = model_dict['performance']
        
        logger.info(f"\nCompleted {norm} normalization")
    
    # Compare results
    logger.info("\n" + "="*80)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*80)
    
    comparison = []
    for norm in strategies:
        perf = results[norm]
        comparison.append({
            'Normalization': norm.upper(),
            'Test Accuracy': f"{perf['test_accuracy']:.3f}",
            'Train Accuracy': f"{perf['train_accuracy']:.3f}",
            'CV Mean': f"{perf['cv_mean']:.3f}",
            'CV Std': f"{perf['cv_std']:.3f}",
            'Good Precision': f"{perf['classification_report']['1']['precision']:.3f}",
            'Good Recall': f"{perf['classification_report']['1']['recall']:.3f}",
            'Good F1': f"{perf['classification_report']['1']['f1-score']:.3f}"
        })
    
    comp_df = pd.DataFrame(comparison)
    print("\n" + comp_df.to_string(index=False))
    
    # Recommendations
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS")
    logger.info("="*80)
    
    minmax_acc = results['minmax']['test_accuracy']
    deltaf_acc = results['deltaf']['test_accuracy']
    
    if minmax_acc > deltaf_acc + 0.02:
        logger.info("✓ MinMax normalization performs better (+2% accuracy)")
        logger.info("  → Use: roi_filtering.normalization = 'minmax' in config")
        logger.info("  → Model: roi_classifier/models/roi_classifier_minmax.pkl")
    elif deltaf_acc > minmax_acc + 0.02:
        logger.info("✓ DeltaF/F normalization performs better (+2% accuracy)")
        logger.info("  → Use: roi_filtering.normalization = 'deltaf' in config")
        logger.info("  → Model: roi_classifier/models/roi_classifier_deltaf.pkl")
    else:
        logger.info("≈ Both normalizations perform similarly")
        logger.info("  → MinMax is simpler and recommended as default")
        logger.info("  → Use: roi_filtering.normalization = 'minmax' in config")
    
    logger.info("\n" + "="*80)
    logger.info("Next steps:")
    logger.info("1. Update config/pipeline_config.yaml with chosen normalization")
    logger.info("2. Copy the corresponding model to roi_classifier/models/roi_classifier.pkl")
    logger.info("3. Run pipeline with new configuration")
    logger.info("="*80 + "\n")

if __name__ == '__main__':
    compare_normalizations()
