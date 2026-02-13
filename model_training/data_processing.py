#!/usr/bin/env python3
"""
data_preprocessing.py
Campus Fire Detection System

Purpose: Preprocess dataset for training
- Handle class imbalance
- Train/test split
- Generate scaler parameters (CRITICAL for firmware!)
- Save processed datasets
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_dataset():
    """Load the smoke detection dataset"""
    print("="*70)
    print("STEP 1: Loading Dataset")
    print("="*70)
    
    # Path relative to project root
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "Dataset" / "raw" / "smoke_detection_iot.csv"
    
    if not data_path.exists():
        print(f"\n❌ ERROR: Dataset not found at {data_path}")
        print("\n💡 Please download the dataset first:")
        print("   cd ../Dataset/raw/")
        print("   kaggle datasets download -d deepcontractor/smoke-detection-dataset")
        print("   unzip smoke-detection-dataset.zip")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Samples: {len(df):,}")
    
    return df

def select_features(df):
    """Select features that match our hardware sensors"""
    print("\n" + "="*70)
    print("STEP 2: Feature Selection")
    print("="*70)
    
    # Features our hardware can measure
    feature_columns = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]']
    target_column = 'Fire Alarm'
    
    # Create clean dataset
    df_clean = df[feature_columns + [target_column]].copy()
    
    # Rename for cleaner code
    df_clean.columns = ['temperature', 'humidity', 'tvoc', 'eco2', 'fire_alarm']
    
    print("\n✅ Selected features:")
    print("   1. temperature  → SHT41")
    print("   2. humidity     → SHT41")
    print("   3. tvoc         → SGP41")
    print("   4. eco2         → SCD41")
    print(f"\n📊 Selected dataset shape: {df_clean.shape}")
    
    # Check for missing values
    missing = df_clean.isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️  Warning: Missing values detected!")
        print(missing[missing > 0])
        print("   Dropping rows with missing values...")
        df_clean = df_clean.dropna()
        print(f"   New shape: {df_clean.shape}")
    else:
        print("   ✅ No missing values")
    
    return df_clean

def analyze_class_balance(df):
    """Analyze and report class distribution"""
    print("\n" + "="*70)
    print("STEP 3: Class Balance Analysis")
    print("="*70)
    
    class_counts = df['fire_alarm'].value_counts()
    class_pct = df['fire_alarm'].value_counts(normalize=True) * 100
    
    print("\n📊 Current class distribution:")
    print(f"   No Fire (0): {class_counts[0]:6,} samples ({class_pct[0]:5.2f}%)")
    print(f"   Fire (1):    {class_counts[1]:6,} samples ({class_pct[1]:5.2f}%)")
    
    imbalance_ratio = class_counts[1] / class_counts[0]
    print(f"\n📈 Imbalance ratio (Fire/No-Fire): {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 2.0 or imbalance_ratio < 0.5:
        print("   ⚠️  Significant class imbalance detected!")
        print("   → Will apply SMOTE to balance classes")
        return True
    else:
        print("   ✅ Classes are reasonably balanced")
        return False

def create_train_test_split(df, test_size=0.2):
    """Split dataset into training and testing sets"""
    print("\n" + "="*70)
    print("STEP 4: Train/Test Split")
    print("="*70)
    
    # Separate features and target
    X = df[['temperature', 'humidity', 'tvoc', 'eco2']].values
    y = df['fire_alarm'].values
    
    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class distribution
    )
    
    print(f"\n✅ Split complete:")
    print(f"   Test size: {test_size*100:.0f}%")
    print(f"\n   Training set:")
    print(f"      Total:   {len(X_train):6,} samples")
    print(f"      No Fire: {np.sum(y_train == 0):6,} samples ({np.sum(y_train == 0)/len(y_train)*100:5.2f}%)")
    print(f"      Fire:    {np.sum(y_train == 1):6,} samples ({np.sum(y_train == 1)/len(y_train)*100:5.2f}%)")
    print(f"\n   Testing set:")
    print(f"      Total:   {len(X_test):6,} samples")
    print(f"      No Fire: {np.sum(y_test == 0):6,} samples ({np.sum(y_test == 0)/len(y_test)*100:5.2f}%)")
    print(f"      Fire:    {np.sum(y_test == 1):6,} samples ({np.sum(y_test == 1)/len(y_test)*100:5.2f}%)")
    
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    """Apply SMOTE to balance training set"""
    print("\n" + "="*70)
    print("STEP 5: SMOTE Resampling (Class Balancing)")
    print("="*70)
    
    print("\n📊 Before SMOTE:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"   Class {label}: {count:6,} samples")
    
    # Apply SMOTE
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print("\n📊 After SMOTE:")
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"   Class {label}: {count:6,} samples")
    
    print(f"\n✅ Training set balanced!")
    print(f"   Original size:  {len(X_train):6,}")
    print(f"   Balanced size:  {len(X_train_balanced):6,}")
    print(f"   Samples added:  {len(X_train_balanced) - len(X_train):6,}")
    
    return X_train_balanced, y_train_balanced

def fit_scaler(X_train):
    """
    Fit MinMax scaler on training data
    
    CRITICAL: These scaler parameters MUST be copied to STM32 firmware!
    """
    print("\n" + "="*70)
    print("STEP 6: Fitting MinMax Scaler")
    print("="*70)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    
    feature_names = ['temperature', 'humidity', 'tvoc', 'eco2']
    
    print("\n📊 Scaler Parameters (Min-Max for each feature):")
    print("-" * 70)
    
    scaler_params = {
        'features': feature_names,
        'min_values': scaler.data_min_.tolist(),
        'max_values': scaler.data_max_.tolist(),
        'feature_range': [0, 1]
    }
    
    for i, feature in enumerate(feature_names):
        print(f"   {feature:12s}:  min = {scaler.data_min_[i]:10.6f}  |  max = {scaler.data_max_[i]:10.6f}")
    
    print("\n" + "="*70)
    print("⚠️  CRITICAL: COPY THESE VALUES TO FIRMWARE!")
    print("="*70)
    print("\n📋 Firmware Integration:")
    print("   File: 05_Firmware/Core/Src/fire_detection_app.c")
    print("   Location: scaler_params struct")
    print("\n   Example C code:")
    print("   static const ScalerParams_t scaler = {")
    for i, feature in enumerate(feature_names):
        min_val = scaler.data_min_[i]
        max_val = scaler.data_max_[i]
        print(f"       .{feature:12s}_min = {min_val:.6f}f,")
        print(f"       .{feature:12s}_max = {max_val:.6f}f,")
    print("   };")
    print("="*70)
    
    return scaler, scaler_params

def transform_data(scaler, X_train, X_test):
    """Apply scaler transformation to train and test sets"""
    print("\n" + "="*70)
    print("STEP 7: Normalizing Data")
    print("="*70)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n✅ Data normalized to [0, 1] range")
    print(f"\n📊 Training set (scaled):")
    print(f"   Shape: {X_train_scaled.shape}")
    print(f"   Min:   {X_train_scaled.min():.6f}")
    print(f"   Max:   {X_train_scaled.max():.6f}")
    print(f"   Mean:  {X_train_scaled.mean():.6f}")
    
    print(f"\n📊 Testing set (scaled):")
    print(f"   Shape: {X_test_scaled.shape}")
    print(f"   Min:   {X_test_scaled.min():.6f}")
    print(f"   Max:   {X_test_scaled.max():.6f}")
    print(f"   Mean:  {X_test_scaled.mean():.6f}")
    
    return X_train_scaled, X_test_scaled

def visualize_distributions(X_train_original, X_train_scaled, y_train):
    """Visualize before/after normalization"""
    print("\n" + "="*70)
    print("STEP 8: Visualization")
    print("="*70)
    
    feature_names = ['temperature', 'humidity', 'tvoc', 'eco2']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Feature Distributions: Before vs After Normalization', 
                 fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(feature_names):
        # Before normalization
        ax1 = axes[0, i]
        ax1.hist(X_train_original[:, i], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title(f'{feature.title()}\n(Original)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # After normalization
        ax2 = axes[1, i]
        ax2.hist(X_train_scaled[:, i], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title(f'{feature.title()}\n(Normalized [0,1])', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.set_xlim([0, 1])
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    project_root = Path(__file__).resolve().parent.parent
    plt.savefig(project_root / results_dir / "05_normalization_comparison.png", dpi=300, bbox_inches='tight')
    print("\n✅ Saved: results/05_normalization_comparison.png")
    plt.close()
    
    # Class distribution after SMOTE
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    unique, counts = np.unique(y_train, return_counts=True)
    axes[0].bar(unique, counts, color=['green', 'red'], edgecolor='black', linewidth=2)
    axes[0].set_title('Class Distribution After SMOTE', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['No Fire', 'Fire'])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (label, count) in enumerate(zip(unique, counts)):
        axes[0].text(label, count, f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts, labels=['No Fire', 'Fire'], autopct='%1.1f%%',
                colors=['green', 'red'], startangle=90)
    axes[1].set_title('Class Balance After SMOTE', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / "06_class_balance_after_smote.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: results/06_class_balance_after_smote.png")
    plt.close()

def save_processed_data(X_train, X_test, y_train, y_test, scaler_params):
    """Save processed datasets and scaler parameters"""
    print("\n" + "="*70)
    print("STEP 9: Saving Processed Data")
    print("="*70)
    
    # Create processed directory
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "Dataset" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    feature_names = ['temperature', 'humidity', 'tvoc', 'eco2']
    
    # Save training data
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['fire_alarm'] = y_train
    train_path = processed_dir / "train_data.csv"
    train_df.to_csv(train_path, index=False)
    print(f"\n✅ Saved training data: {train_path}")
    print(f"   Shape: {train_df.shape}")
    
    # Save testing data
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['fire_alarm'] = y_test
    test_path = processed_dir / "test_data.csv"
    test_df.to_csv(test_path, index=False)
    print(f"\n✅ Saved testing data: {test_path}")
    print(f"   Shape: {test_df.shape}")
    
    # Save scaler parameters (CRITICAL for firmware!)
    scaler_path = processed_dir / "scaler_parameters.json"
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=4)
    print(f"\n✅ Saved scaler parameters: {scaler_path}")
    
    print("\n" + "="*70)
    print("🔥 CRITICAL FILE SAVED!")
    print("="*70)
    print(f"\n📄 File: {scaler_path}")
    print("\n⚠️  YOU MUST:")
    print("   1. Copy scaler_parameters.json to firmware project")
    print("   2. Hardcode min/max values in fire_detection_app.c")
    print("   3. Use EXACT same normalization in STM32 code")
    print("\n💡 See: 04_Model_Deployment/INSTRUCTIONS.md for integration example")
    print("="*70)
    
    return train_path, test_path, scaler_path

def generate_preprocessing_report(X_train, X_test, y_train, y_test, 
                                  scaler_params, apply_smote_flag):
    """Generate comprehensive preprocessing report"""
    print("\n" + "="*70)
    print("STEP 10: Generating Preprocessing Report")
    print("="*70)
    
    report = []
    report.append("="*70)
    report.append("DATA PREPROCESSING REPORT")
    report.append("Campus Fire Detection System")
    report.append("="*70)
    report.append("")
    
    # Dataset summary
    report.append("1. DATASET SUMMARY")
    report.append("-" * 70)
    report.append(f"   Training samples:   {len(X_train):,}")
    report.append(f"   Testing samples:    {len(X_test):,}")
    report.append(f"   Total samples:      {len(X_train) + len(X_test):,}")
    report.append(f"   Features:           {X_train.shape[1]}")
    report.append("")
    
    # Feature names
    report.append("2. FEATURES")
    report.append("-" * 70)
    for i, feature in enumerate(scaler_params['features'], 1):
        report.append(f"   {i}. {feature}")
    report.append("")
    
    # Class distribution
    report.append("3. CLASS DISTRIBUTION (Training Set)")
    report.append("-" * 70)
    train_no_fire = np.sum(y_train == 0)
    train_fire = np.sum(y_train == 1)
    report.append(f"   No Fire (0): {train_no_fire:,} samples ({train_no_fire/len(y_train)*100:.2f}%)")
    report.append(f"   Fire (1):    {train_fire:,} samples ({train_fire/len(y_train)*100:.2f}%)")
    if apply_smote_flag:
        report.append("   ✅ SMOTE applied to balance classes")
    report.append("")
    
    # Scaler parameters
    report.append("4. NORMALIZATION PARAMETERS (MinMax [0,1])")
    report.append("-" * 70)
    for i, feature in enumerate(scaler_params['features']):
        min_val = scaler_params['min_values'][i]
        max_val = scaler_params['max_values'][i]
        report.append(f"   {feature:12s}:  [{min_val:10.6f}, {max_val:10.6f}]")
    report.append("")
    
    # Critical reminder
    report.append("5. FIRMWARE INTEGRATION")
    report.append("-" * 70)
    report.append("   ⚠️  CRITICAL: Copy scaler parameters to STM32 code!")
    report.append("   File: scaler_parameters.json")
    report.append("   Location: Dataset/processed/")
    report.append("")
    report.append("   C code example:")
    report.append("   static const ScalerParams_t scaler = {")
    for i, feature in enumerate(scaler_params['features']):
        min_val = scaler_params['min_values'][i]
        max_val = scaler_params['max_values'][i]
        report.append(f"       .{feature}_min = {min_val:.6f}f,")
        report.append(f"       .{feature}_max = {max_val:.6f}f,")
    report.append("   };")
    report.append("")
    
    # Next steps
    report.append("6. NEXT STEPS")
    report.append("-" * 70)
    report.append("   ✅ Data preprocessing complete")
    report.append("   → Run: 03_train_basic_model.py")
    report.append("   → Then: 05_export_onnx.py")
    report.append("")
    report.append("="*70)
    
    # Save report
    report_text = "\n".join(report)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "preprocessing_report.txt", "w") as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✅ Saved: results/preprocessing_report.txt")

def main():
    """Main execution function"""
    print("\n" + "🔥"*35)
    print("CAMPUS FIRE DETECTION - DATA PREPROCESSING")
    print("🔥"*35 + "\n")
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return
    
    # Select features
    df_clean = select_features(df)
    
    # Analyze class balance
    needs_smote = analyze_class_balance(df_clean)
    
    # Train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(df_clean)
    
    # Store original for visualization
    X_train_original = X_train.copy()
    
    # Apply SMOTE if needed
    if needs_smote:
        X_train, y_train = apply_smote(X_train, y_train)
    
    # Fit scaler on training data
    scaler, scaler_params = fit_scaler(X_train)
    
    # Transform data
    X_train_scaled, X_test_scaled = transform_data(scaler, X_train, X_test)
    
    # Visualize
    visualize_distributions(X_train_original, X_train_scaled, y_train)
    
    # Save processed data
    train_path, test_path, scaler_path = save_processed_data(
        X_train_scaled, X_test_scaled, y_train, y_test, scaler_params
    )
    
    # Generate report
    generate_preprocessing_report(
        X_train_scaled, X_test_scaled, y_train, y_test, 
        scaler_params, needs_smote
    )
    
    print("\n" + "="*70)
    print("✅ DATA PREPROCESSING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("   1. Dataset/processed/train_data.csv")
    print("   2. Dataset/processed/test_data.csv")
    print("   3. Dataset/processed/scaler_parameters.json  🔥 CRITICAL!")
    print("   4. results/05_normalization_comparison.png")
    print("   5. results/06_class_balance_after_smote.png")
    print("   6. results/preprocessing_report.txt")
    print("\n📋 Next step: Run 03_train_basic_model.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()