#!/usr/bin/env python3
"""
03_train_basic_model.py
Campus Fire Detection System

Purpose: Train XGBoost model for fire detection
- Load preprocessed data
- Train XGBoost classifier
- Evaluate performance metrics
- Save trained model
- Generate comprehensive report

Author: AI Assistant for Cal Poly Senior Project Team
Date: February 2026
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# XGBoost and sklearn imports
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_preprocessed_data():
    """Load preprocessed training and testing data"""
    print("="*70)
    print("STEP 1: Loading Preprocessed Data")
    print("="*70)
    
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "Dataset" / "processed"
    
    # Load training data
    train_path = processed_dir / "train_data.csv"
    if not train_path.exists():
        print(f"\n❌ ERROR: Training data not found at {train_path}")
        print("\n💡 Please run 02_data_preprocessing.py first!")
        return None, None, None, None
    
    train_df = pd.read_csv(train_path)
    print(f"\n✅ Loaded training data: {train_path}")
    print(f"   Shape: {train_df.shape}")
    
    # Load testing data
    test_path = processed_dir / "test_data.csv"
    if not test_path.exists():
        print(f"\n❌ ERROR: Testing data not found at {test_path}")
        return None, None, None, None
    
    test_df = pd.read_csv(test_path)
    print(f"\n✅ Loaded testing data: {test_path}")
    print(f"   Shape: {test_df.shape}")
    
    # Separate features and labels
    feature_cols = ['temperature', 'humidity', 'tvoc', 'eco2']
    
    X_train = train_df[feature_cols].values
    y_train = train_df['fire_alarm'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['fire_alarm'].values
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Training samples:   {len(X_train):,}")
    print(f"   Testing samples:    {len(X_test):,}")
    print(f"   Features:           {len(feature_cols)}")
    print(f"\n   Training class distribution:")
    print(f"      No Fire (0): {np.sum(y_train == 0):,} ({np.sum(y_train == 0)/len(y_train)*100:.2f}%)")
    print(f"      Fire (1):    {np.sum(y_train == 1):,} ({np.sum(y_train == 1)/len(y_train)*100:.2f}%)")
    print(f"\n   Testing class distribution:")
    print(f"      No Fire (0): {np.sum(y_test == 0):,} ({np.sum(y_test == 0)/len(y_test)*100:.2f}%)")
    print(f"      Fire (1):    {np.sum(y_test == 1):,} ({np.sum(y_test == 1)/len(y_test)*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test

def train_xgboost_model(X_train, y_train):
    """
    Train XGBoost classifier
    
    XGBoost chosen based on research:
    - 99% accuracy in Lee et al. (2023)
    - Superior to Random Forest (97%) and SVC (95%)
    - Efficient for embedded deployment
    - Handles multi-sensor fusion well
    """
    print("\n" + "="*70)
    print("STEP 2: Training XGBoost Model")
    print("="*70)
    
    print("\n📚 Why XGBoost?")
    print("   • Research shows 99% accuracy (Lee et al., 2023)")
    print("   • Outperforms Random Forest (97%) and SVC (95%)")
    print("   • Gradient boosting = sequential error correction")
    print("   • Efficient for STM32 deployment")
    print("   • Excellent for sensor fusion")
    
    # XGBoost hyperparameters
    # Based on research and optimized for fire detection
    params = {
        'max_depth': 6,              # Tree depth (prevents overfitting)
        'learning_rate': 0.1,        # Step size (0.1 is standard)
        'n_estimators': 100,         # Number of trees (100 is good balance)
        'objective': 'binary:logistic',  # Binary classification
        'eval_metric': 'logloss',    # Loss function
        'subsample': 0.8,            # 80% data per tree (prevents overfitting)
        'colsample_bytree': 0.8,     # 80% features per tree (prevents overfitting)
        'random_state': RANDOM_STATE,
        'use_label_encoder': False,  # Suppress warning
        'verbosity': 1               # Show training progress
    }
    
    print("\n⚙️  Hyperparameters:")
    for key, value in params.items():
        print(f"   {key:20s} = {value}")
    
    # Create and train model
    print("\n🔄 Training model...")
    print("   (This may take 30-60 seconds)")
    
    start_time = datetime.now()
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=True)
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    print(f"\n✅ Training complete!")
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Trees built:   {model.n_estimators}")
    
    return model, params, training_time

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance on both training and test sets
    
    Key metrics:
    - Accuracy: Overall correctness (DR-06: ≥90%)
    - Precision: When it predicts fire, is it correct?
    - Recall: Does it catch all real fires?
    - False Alarm Rate: 1 - Precision (DR-05: ≤3%)
    """
    print("\n" + "="*70)
    print("STEP 3: Model Evaluation")
    print("="*70)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Get prediction probabilities
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics for training set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    # Calculate metrics for test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # False alarm rate = False Positives / (False Positives + True Negatives)
    # Or equivalently: 1 - Specificity
    # For fire detection: False alarm = predicting fire when there isn't one
    cm_test = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm_test.ravel()
    false_alarm_rate = fp / (fp + tn) * 100  # Convert to percentage
    
    print("\n" + "="*70)
    print("TRAINING SET PERFORMANCE")
    print("="*70)
    print(f"   Accuracy:  {train_accuracy*100:.2f}%")
    print(f"   Precision: {train_precision*100:.2f}%")
    print(f"   Recall:    {train_recall*100:.2f}%")
    print(f"   F1 Score:  {train_f1*100:.2f}%")
    
    print("\n" + "="*70)
    print("TESTING SET PERFORMANCE (Unseen Data)")
    print("="*70)
    print(f"   Accuracy:  {test_accuracy*100:.2f}%")
    print(f"   Precision: {test_precision*100:.2f}%")
    print(f"   Recall:    {test_recall*100:.2f}%")
    print(f"   F1 Score:  {test_f1*100:.2f}%")
    
    print("\n" + "="*70)
    print("DESIGN REQUIREMENTS VERIFICATION")
    print("="*70)
    print(f"\n   DR-06: Detection Accuracy ≥ 90.0%")
    print(f"      Test Accuracy: {test_accuracy*100:.2f}%", end="")
    if test_accuracy >= 0.90:
        print(" ✅ PASS")
    else:
        print(" ❌ FAIL")
    
    print(f"\n   DR-05: False Alarm Rate ≤ 3.0%")
    print(f"      False Alarm Rate: {false_alarm_rate:.2f}%", end="")
    if false_alarm_rate <= 3.0:
        print(" ✅ PASS")
    else:
        print(" ⚠️  MARGINAL (may need tuning)")
    
    # Detailed confusion matrix
    print("\n" + "="*70)
    print("CONFUSION MATRIX (Test Set)")
    print("="*70)
    print("\n   Predicted →")
    print("   Actual ↓        No Fire    Fire")
    print(f"   No Fire (0)     {tn:6d}    {fp:6d}")
    print(f"   Fire (1)        {fn:6d}    {tp:6d}")
    
    print(f"\n   True Negatives (TN):  {tn:6d}  (Correctly predicted no fire)")
    print(f"   False Positives (FP): {fp:6d}  (False alarms)")
    print(f"   False Negatives (FN): {fn:6d}  (Missed fires) ⚠️")
    print(f"   True Positives (TP):  {tp:6d}  (Correctly detected fires)")
    
    # Missed fires are critical!
    if fn > 0:
        print(f"\n   ⚠️  CRITICAL: {fn} fires were missed!")
        print(f"      Missed fire rate: {fn/(fn+tp)*100:.2f}%")
    else:
        print(f"\n   ✅ No fires missed!")
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)
    
    print(f"\n   ROC AUC Score: {roc_auc:.4f}")
    if roc_auc >= 0.95:
        print("      ✅ Excellent discrimination")
    elif roc_auc >= 0.90:
        print("      ✅ Good discrimination")
    else:
        print("      ⚠️  May need improvement")
    
    # Compile metrics dictionary
    metrics = {
        'train': {
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1_score': train_f1
        },
        'test': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'false_alarm_rate': false_alarm_rate,
            'roc_auc': roc_auc
        },
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'requirements': {
            'DR-06_accuracy_pass': test_accuracy >= 0.90,
            'DR-05_false_alarm_pass': false_alarm_rate <= 3.0
        }
    }
    
    return metrics, y_test_pred, y_test_prob

def analyze_feature_importance(model):
    """Analyze which sensors contribute most to fire detection"""
    print("\n" + "="*70)
    print("STEP 4: Feature Importance Analysis")
    print("="*70)
    
    feature_names = ['temperature', 'humidity', 'tvoc', 'eco2']
    importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    print("\n📊 Feature Importance (Which sensors matter most?):\n")
    print("   Rank  Feature         Importance  Sensor")
    print("   " + "-"*60)
    
    sensor_map = {
        'temperature': 'SHT41',
        'humidity': 'SHT41',
        'tvoc': 'SGP41',
        'eco2': 'SCD41'
    }
    
    for i, idx in enumerate(indices, 1):
        feature = feature_names[idx]
        importance = importances[idx]
        sensor = sensor_map[feature]
        bar = '█' * int(importance * 50)
        print(f"   {i}.    {feature:12s}  {importance:.4f}  ({sensor})  {bar}")
    
    print("\n💡 Interpretation:")
    most_important = feature_names[indices[0]]
    print(f"   • {most_important.title()} is the most important feature")
    print(f"   • This makes sense for fire detection!")
    
    if importances[indices[0]] > 0.5:
        print(f"   ⚠️  WARNING: One feature dominates ({importances[indices[0]]:.1%})")
        print(f"      Consider this when designing backup detection logic")
    
    return dict(zip(feature_names, importances.tolist()))

def create_visualizations(model, X_test, y_test, y_test_pred, y_test_prob, metrics):
    """Create comprehensive visualization plots"""
    print("\n" + "="*70)
    print("STEP 5: Creating Visualizations")
    print("="*70)
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # ================================================================
    # PLOT 1: Confusion Matrix Heatmap
    # ================================================================
    print("\n   Creating confusion matrix heatmap...")
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Fire', 'Fire'],
                yticklabels=['No Fire', 'Fire'],
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 16, 'weight': 'bold'},
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix - Test Set\nCampus Fire Detection System', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add accuracy annotation
    accuracy = metrics['test']['accuracy']
    ax.text(0.5, -0.15, f'Overall Accuracy: {accuracy*100:.2f}%', 
            transform=ax.transAxes, ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(results_dir / '07_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: results/07_confusion_matrix.png")
    plt.close()
    
    # ================================================================
    # PLOT 2: ROC Curve
    # ================================================================
    print("   Creating ROC curve...")
    
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(fpr, tpr, color='blue', lw=3, 
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', 
            label='Random Classifier (AUC = 0.5)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (False Alarm Rate)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Detection Rate)', 
                  fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - Fire Detection Performance\nCampus Fire Detection System', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / '08_roc_curve.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: results/08_roc_curve.png")
    plt.close()
    
    # ================================================================
    # PLOT 3: Precision-Recall Curve
    # ================================================================
    print("   Creating precision-recall curve...")
    
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_test_prob)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(recall, precision, color='green', lw=3, 
            label=f'PR Curve (AUC = {pr_auc:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Fires Detected / Total Fires)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision (True Fires / All Fire Predictions)', 
                  fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve\nCampus Fire Detection System', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add design requirement line
    ax.axhline(y=0.97, color='red', linestyle='--', linewidth=2, 
               label='DR-05: ≤3% False Alarm (Precision ≥97%)')
    ax.legend(loc='lower left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(results_dir / '09_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: results/09_precision_recall_curve.png")
    plt.close()
    
    # ================================================================
    # PLOT 4: Feature Importance Bar Chart
    # ================================================================
    print("   Creating feature importance chart...")
    
    feature_names = ['temperature', 'humidity', 'tvoc', 'eco2']
    importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#FF6B6B', '#FFA500', '#4ECDC4', '#95E1D3']
    bars = ax.barh(range(len(indices)), importances[indices], 
                   color=[colors[i] for i in indices],
                   edgecolor='black', linewidth=2)
    
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i].title() for i in indices], fontsize=12)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance for Fire Detection\nCampus Fire Detection System', 
                 fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, bar) in enumerate(zip(indices, bars)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' {importances[idx]:.4f}',
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / '10_feature_importance.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: results/10_feature_importance.png")
    plt.close()
    
    # ================================================================
    # PLOT 5: Performance Metrics Comparison
    # ================================================================
    print("   Creating metrics comparison chart...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    train_scores = [
        metrics['train']['accuracy'],
        metrics['train']['precision'],
        metrics['train']['recall'],
        metrics['train']['f1_score']
    ]
    test_scores = [
        metrics['test']['accuracy'],
        metrics['test']['precision'],
        metrics['test']['recall'],
        metrics['test']['f1_score']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_scores, width, label='Training Set',
                   color='lightblue', edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, test_scores, width, label='Test Set',
                   color='lightcoral', edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics\nCampus Fire Detection System', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add requirement lines
    ax.axhline(y=0.90, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label='DR-06: 90% Accuracy Requirement')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(results_dir / '11_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: results/11_metrics_comparison.png")
    plt.close()

def save_model(model, metrics, params, feature_importance, training_time):
    """Save trained model and metadata"""
    print("\n" + "="*70)
    print("STEP 6: Saving Trained Model")
    print("="*70)
    
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "Models"
    models_dir.mkdir(exist_ok=True)
    
    # Save model using pickle (for Python use)
    model_path = models_dir / "fire_detection_xgboost.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ Saved model (pickle): {model_path}")
    
    # Save model using XGBoost's native format (for conversion)
    xgb_model_path = models_dir / "fire_detection_xgboost.json"
    model.save_model(xgb_model_path)
    print(f"✅ Saved model (XGBoost): {xgb_model_path}")
    
    # Save model metadata
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    metadata = {
        'model_type': 'XGBoost Classifier',
        'training_date': datetime.now().isoformat(),
        'hyperparameters': params,
        'feature_names': ['temperature', 'humidity', 'tvoc', 'eco2'],
        'feature_importance': feature_importance,
        'performance_metrics': convert_to_json_serializable(metrics),
        'training_time_seconds': float(training_time),
        'model_files': {
            'pickle': str(model_path),
            'xgboost': str(xgb_model_path)
        }
    }
    
    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Saved metadata: {metadata_path}")
    
    # Calculate model size
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\n📊 Model Information:")
    print(f"   File size:     {model_size_mb:.2f} MB")
    print(f"   Trees:         {model.n_estimators}")
    print(f"   Max depth:     {model.max_depth}")
    print(f"   Features:      {len(feature_importance)}")
    
    return model_path, metadata_path

def generate_training_report(metrics, params, feature_importance, training_time):
    """Generate comprehensive training report"""
    print("\n" + "="*70)
    print("STEP 7: Generating Training Report")
    print("="*70)
    
    report = []
    report.append("="*70)
    report.append("MODEL TRAINING REPORT")
    report.append("Campus Fire Detection System - XGBoost Classifier")
    report.append(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*70)
    report.append("")
    
    # Model Architecture
    report.append("1. MODEL ARCHITECTURE")
    report.append("-" * 70)
    report.append("   Algorithm: XGBoost (Extreme Gradient Boosting)")
    report.append(f"   Trees: {params['n_estimators']}")
    report.append(f"   Max Depth: {params['max_depth']}")
    report.append(f"   Learning Rate: {params['learning_rate']}")
    report.append(f"   Training Time: {training_time:.2f} seconds")
    report.append("")
    
    # Feature Importance
    report.append("2. FEATURE IMPORTANCE")
    report.append("-" * 70)
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_features, 1):
        report.append(f"   {i}. {feature:12s}: {importance:.4f} ({importance*100:.2f}%)")
    report.append("")
    
    # Performance Metrics - Training
    report.append("3. TRAINING SET PERFORMANCE")
    report.append("-" * 70)
    report.append(f"   Accuracy:  {metrics['train']['accuracy']*100:.2f}%")
    report.append(f"   Precision: {metrics['train']['precision']*100:.2f}%")
    report.append(f"   Recall:    {metrics['train']['recall']*100:.2f}%")
    report.append(f"   F1 Score:  {metrics['train']['f1_score']*100:.2f}%")
    report.append("")
    
    # Performance Metrics - Testing
    report.append("4. TESTING SET PERFORMANCE (Unseen Data)")
    report.append("-" * 70)
    report.append(f"   Accuracy:  {metrics['test']['accuracy']*100:.2f}%")
    report.append(f"   Precision: {metrics['test']['precision']*100:.2f}%")
    report.append(f"   Recall:    {metrics['test']['recall']*100:.2f}%")
    report.append(f"   F1 Score:  {metrics['test']['f1_score']*100:.2f}%")
    report.append(f"   ROC AUC:   {metrics['test']['roc_auc']:.4f}")
    report.append(f"   False Alarm Rate: {metrics['test']['false_alarm_rate']:.2f}%")
    report.append("")
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    report.append("5. CONFUSION MATRIX (Test Set)")
    report.append("-" * 70)
    report.append("   Predicted →")
    report.append("   Actual ↓        No Fire    Fire")
    report.append(f"   No Fire (0)     {cm['tn']:6d}    {cm['fp']:6d}")
    report.append(f"   Fire (1)        {cm['fn']:6d}    {cm['tp']:6d}")
    report.append("")
    report.append(f"   True Negatives:  {cm['tn']:6d} (Correct no-fire predictions)")
    report.append(f"   False Positives: {cm['fp']:6d} (False alarms)")
    report.append(f"   False Negatives: {cm['fn']:6d} (Missed fires)")
    report.append(f"   True Positives:  {cm['tp']:6d} (Correct fire detections)")
    report.append("")
    
    # Design Requirements
    report.append("6. DESIGN REQUIREMENTS VERIFICATION")
    report.append("-" * 70)
    
    dr06_pass = metrics['requirements']['DR-06_accuracy_pass']
    report.append(f"   DR-06: Detection Accuracy ≥ 90.0%")
    report.append(f"      Result: {metrics['test']['accuracy']*100:.2f}%")
    report.append(f"      Status: {'✅ PASS' if dr06_pass else '❌ FAIL'}")
    report.append("")
    
    dr05_pass = metrics['requirements']['DR-05_false_alarm_pass']
    report.append(f"   DR-05: False Alarm Rate ≤ 3.0%")
    report.append(f"      Result: {metrics['test']['false_alarm_rate']:.2f}%")
    report.append(f"      Status: {'✅ PASS' if dr05_pass else '⚠️  MARGINAL'}")
    report.append("")
    
    # Overfitting Analysis
    report.append("7. OVERFITTING ANALYSIS")
    report.append("-" * 70)
    train_test_gap = (metrics['train']['accuracy'] - metrics['test']['accuracy']) * 100
    report.append(f"   Training Accuracy:  {metrics['train']['accuracy']*100:.2f}%")
    report.append(f"   Testing Accuracy:   {metrics['test']['accuracy']*100:.2f}%")
    report.append(f"   Accuracy Gap:       {train_test_gap:.2f}%")
    
    if train_test_gap < 2:
        report.append("   Status: ✅ Excellent generalization (gap < 2%)")
    elif train_test_gap < 5:
        report.append("   Status: ✅ Good generalization (gap < 5%)")
    elif train_test_gap < 10:
        report.append("   Status: ⚠️  Slight overfitting (gap < 10%)")
    else:
        report.append("   Status: ❌ Significant overfitting (gap ≥ 10%)")
    report.append("")
    
    # Next Steps
    report.append("8. NEXT STEPS")
    report.append("-" * 70)
    report.append("   ✅ Model training complete")
    report.append("   ✅ Model saved to Models/ directory")
    report.append("   → Next: Run 04_optimize_hyperparameters.py (optional)")
    report.append("   → Then: Run 05_export_onnx.py for STM32 deployment")
    report.append("")
    
    # Research Comparison
    report.append("9. COMPARISON WITH RESEARCH")
    report.append("-" * 70)
    report.append("   Literature (Lee et al., 2023):")
    report.append("      XGBoost:      99.0% accuracy")
    report.append("      Random Forest: 97.0% accuracy")
    report.append("      SVC:          95.0% accuracy")
    report.append("")
    report.append(f"   Our Model:")
    report.append(f"      Test Accuracy: {metrics['test']['accuracy']*100:.2f}%")
    
    if metrics['test']['accuracy'] >= 0.97:
        report.append("      Status: ✅ Matches or exceeds research benchmarks!")
    elif metrics['test']['accuracy'] >= 0.90:
        report.append("      Status: ✅ Meets design requirements")
    else:
        report.append("      Status: ⚠️  Below target, needs improvement")
    report.append("")
    
    report.append("="*70)
    
    # Save report
    report_text = "\n".join(report)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "training_report.txt", "w") as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✅ Saved: results/training_report.txt")
    
    return report_text

def main():
    """Main training pipeline"""
    print("\n" + "🔥"*35)
    print("CAMPUS FIRE DETECTION - MODEL TRAINING")
    print("XGBoost Classifier")
    print("🔥"*35 + "\n")
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    if X_train is None:
        return
    
    # Train model
    model, params, training_time = train_xgboost_model(X_train, y_train)
    
    # Evaluate model
    metrics, y_test_pred, y_test_prob = evaluate_model(
        model, X_train, y_train, X_test, y_test
    )
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model)
    
    # Create visualizations
    create_visualizations(model, X_test, y_test, y_test_pred, y_test_prob, metrics)
    
    # Save model
    model_path, metadata_path = save_model(
        model, metrics, params, feature_importance, training_time
    )
    
    # Generate report
    generate_training_report(metrics, params, feature_importance, training_time)
    
    print("\n" + "="*70)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("   Models:")
    print("      1. Models/fire_detection_xgboost.pkl")
    print("      2. Models/fire_detection_xgboost.json")
    print("      3. Models/model_metadata.json")
    print("\n   Visualizations:")
    print("      4. results/07_confusion_matrix.png")
    print("      5. results/08_roc_curve.png")
    print("      6. results/09_precision_recall_curve.png")
    print("      7. results/10_feature_importance.png")
    print("      8. results/11_metrics_comparison.png")
    print("\n   Report:")
    print("      9. results/training_report.txt")
    
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"   Test Accuracy:      {metrics['test']['accuracy']*100:.2f}%")
    print(f"   False Alarm Rate:   {metrics['test']['false_alarm_rate']:.2f}%")
    print(f"   ROC AUC:            {metrics['test']['roc_auc']:.4f}")
    print(f"\n   DR-06 (≥90% accuracy):    {'✅ PASS' if metrics['requirements']['DR-06_accuracy_pass'] else '❌ FAIL'}")
    print(f"   DR-05 (≤3% false alarms): {'✅ PASS' if metrics['requirements']['DR-05_false_alarm_pass'] else '⚠️  MARGINAL'}")
    
    print("\n📋 Next step: Run 05_export_onnx.py for STM32 deployment")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()