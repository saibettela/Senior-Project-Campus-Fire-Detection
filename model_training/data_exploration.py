#!/usr/bin/env python3
"""
01_data_exploration.py
Campus Fire Detection System

Purpose: Explore and visualize the Kaggle smoke detection dataset

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_dataset():
    """Load the smoke detection dataset"""
    print("="*60)
    print("STEP 1: Loading Dataset")
    print("="*60)
    
    # Path relative to project root (parent of model_training)
    _project_root = Path(__file__).resolve().parent.parent
    data_path = _project_root / "Dataset" / "raw" / "smoke_detection_iot.csv"
    
    if not data_path.exists():
        print(f"\n❌ ERROR: Dataset not found at {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Samples: {len(df):,}")
    print(f"   Features: {len(df.columns)}")
    
    return df

def basic_info(df):
    """Display basic dataset information"""
    print("\n" + "="*60)
    print("STEP 2: Basic Dataset Information")
    print("="*60)
    
    print("\n📋 Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    print("\n📊 Data Types:")
    print(df.dtypes)
    
    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ No missing values!")
    else:
        print(missing[missing > 0])
    
    print("\n📈 Statistical Summary:")
    print(df.describe())
    
    return df

def analyze_target(df):
    """Analyze the target variable (Fire Alarm)"""
    print("\n" + "="*60)
    print("STEP 3: Target Variable Analysis")
    print("="*60)
    
    fire_counts = df['Fire Alarm'].value_counts()
    fire_pct = df['Fire Alarm'].value_counts(normalize=True) * 100
    
    print("\n🎯 Fire Alarm Distribution:")
    print(f"   No Fire (0): {fire_counts[0]:,} samples ({fire_pct[0]:.2f}%)")
    print(f"   Fire (1):    {fire_counts[1]:,} samples ({fire_pct[1]:.2f}%)")
    
    # Visualize class distribution
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot
    fire_counts.plot(kind='bar', ax=ax[0], color=['green', 'red'])
    ax[0].set_title('Fire Alarm Distribution (Count)', fontsize=14, fontweight='bold')
    ax[0].set_xlabel('Fire Alarm')
    ax[0].set_ylabel('Count')
    ax[0].set_xticklabels(['No Fire', 'Fire'], rotation=0)
    ax[0].grid(axis='y', alpha=0.3)
    
    # Pie chart
    ax[1].pie(fire_counts, labels=['No Fire', 'Fire'], autopct='%1.1f%%',
              colors=['green', 'red'], startangle=90)
    ax[1].set_title('Fire Alarm Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "01_target_distribution.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: results/01_target_distribution.png")
    plt.close()

def select_project_features(df):
    """Select only features hardware can measure"""
    print("\n" + "="*60)
    print("STEP 4: Feature Selection")
    print("="*60)
    
    # Features hardware actually has
    project_features = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]']
    target = 'Fire Alarm'
    
    print("\n✅ Features YOUR hardware can measure:")
    for i, feature in enumerate(project_features, 1):
        sensor = ""
        if 'Temperature' in feature or 'Humidity' in feature:
            sensor = "→ SHT41"
        elif 'TVOC' in feature:
            sensor = "→ SGP41"
        elif 'eCO2' in feature:
            sensor = "→ SCD41"
        print(f"   {i}. {feature:20s} {sensor}")
    
    print(f"\n🎯 Target: {target}")
    
    # Create filtered dataset
    df_filtered = df[project_features + [target]].copy()
    
    # Rename for cleaner code
    df_filtered.columns = ['temperature', 'humidity', 'tvoc', 'eco2', 'fire_alarm']
    
    print(f"\n📊 Filtered dataset shape: {df_filtered.shape}")
    
    return df_filtered, project_features

def visualize_feature_distributions(df_filtered):
    """Visualize how features differ between fire and no-fire"""
    print("\n" + "="*60)
    print("STEP 5: Feature Distribution Analysis")
    print("="*60)
    
    features = ['temperature', 'humidity', 'tvoc', 'eco2']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sensor Distributions: Fire vs No Fire', fontsize=16, fontweight='bold')
    
    for idx, feature in enumerate(features):
        ax = axes[idx // 2, idx % 2]
        
        # Separate data by class
        fire_data = df_filtered[df_filtered['fire_alarm'] == 1][feature]
        no_fire_data = df_filtered[df_filtered['fire_alarm'] == 0][feature]
        
        # Plot histograms
        ax.hist(no_fire_data, bins=50, alpha=0.6, label='No Fire', 
                color='green', density=True, edgecolor='black')
        ax.hist(fire_data, bins=50, alpha=0.6, label='Fire', 
                color='red', density=True, edgecolor='black')
        
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{feature.replace("_", " ").title()} Distribution', 
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Print statistics
        print(f"\n📊 {feature.upper()}:")
        print(f"   No Fire: mean={no_fire_data.mean():.2f}, std={no_fire_data.std():.2f}")
        print(f"   Fire:    mean={fire_data.mean():.2f}, std={fire_data.std():.2f}")
    
    plt.tight_layout()
    plt.savefig("results/02_feature_distributions.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: results/02_feature_distributions.png")
    plt.close()

def analyze_correlations(df_filtered):
    """Analyze feature correlations"""
    print("\n" + "="*60)
    print("STEP 6: Correlation Analysis")
    print("="*60)
    
    # Compute correlation matrix
    corr_matrix = df_filtered.corr()
    
    # Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/03_correlation_matrix.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: results/03_correlation_matrix.png")
    plt.close()
    
    print("\n📊 Correlation with Fire Alarm:")
    fire_corr = corr_matrix['fire_alarm'].sort_values(ascending=False)
    for feature, corr_value in fire_corr.items():
        if feature != 'fire_alarm':
            indicator = "🔥" if abs(corr_value) > 0.3 else "  "
            print(f"   {indicator} {feature:15s}: {corr_value:+.4f}")

def visualize_time_series(df, df_filtered):
    """Show sensor readings around a fire event"""
    print("\n" + "="*60)
    print("STEP 7: Time Series Visualization")
    print("="*60)
    
    # Find a fire event in the middle of the dataset
    fire_indices = df_filtered[df_filtered['fire_alarm'] == 1].index
    
    if len(fire_indices) == 0:
        print("❌ No fire events found in dataset!")
        return
    
    # Pick a fire event from the middle
    fire_idx = fire_indices[len(fire_indices)//2]
    
    # Create window around fire event
    window_size = 100
    window_start = max(0, fire_idx - window_size)
    window_end = min(len(df_filtered), fire_idx + window_size)
    
    window_df = df_filtered.iloc[window_start:window_end].reset_index(drop=True)
    
    # Plot
    features = ['temperature', 'humidity', 'tvoc', 'eco2']
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(f'Sensor Readings Around Fire Event (Index {fire_idx})', 
                 fontsize=16, fontweight='bold')
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        ax.plot(window_df.index, window_df[feature], linewidth=2, 
                label=feature.replace('_', ' ').title(), color='blue')
        
        # Highlight fire region
        fire_region = window_df[window_df['fire_alarm'] == 1]
        if len(fire_region) > 0:
            ax.axvspan(fire_region.index[0], fire_region.index[-1], 
                      alpha=0.3, color='red', label='Fire Detected')
        
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        if idx == 3:
            ax.set_xlabel('Sample Index', fontsize=11)
    
    plt.tight_layout()
    plt.savefig("results/04_fire_event_timeline.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: results/04_fire_event_timeline.png")
    print(f"   Fire event at index: {fire_idx}")
    print(f"   Window: {window_start} to {window_end}")
    plt.close()

def generate_summary_report(df, df_filtered):
    """Generate a text summary report"""
    print("\n" + "="*60)
    print("STEP 8: Generating Summary Report")
    print("="*60)
    
    report = []
    report.append("="*70)
    report.append("DATASET EXPLORATION SUMMARY REPORT")
    report.append("Campus Fire Detection System")
    report.append("="*70)
    report.append("")
    
    # Dataset info
    report.append("1. DATASET INFORMATION")
    report.append("-" * 70)
    report.append(f"   Total samples: {len(df):,}")
    report.append(f"   Total features: {len(df.columns)}")
    report.append(f"   Dataset shape: {df.shape}")
    report.append(f"   Missing values: {df.isnull().sum().sum()}")
    report.append("")
    
    # Class distribution
    fire_counts = df_filtered['fire_alarm'].value_counts()
    fire_pct = df_filtered['fire_alarm'].value_counts(normalize=True) * 100
    report.append("2. CLASS DISTRIBUTION")
    report.append("-" * 70)
    report.append(f"   No Fire (0): {fire_counts[0]:,} samples ({fire_pct[0]:.2f}%)")
    report.append(f"   Fire (1):    {fire_counts[1]:,} samples ({fire_pct[1]:.2f}%)")
    report.append(f"   Class balance: {fire_counts[1] / fire_counts[0]:.2f}")
    report.append("")
    
    # Selected features
    report.append("3. FEATURES USED IN MODEL")
    report.append("-" * 70)
    features = ['temperature', 'humidity', 'tvoc', 'eco2']
    for i, feat in enumerate(features, 1):
        sensor = ""
        if 'temperature' in feat or 'humidity' in feat:
            sensor = "(SHT41)"
        elif 'tvoc' in feat:
            sensor = "(SGP41)"
        elif 'eco2' in feat:
            sensor = "(SCD41)"
        report.append(f"   {i}. {feat:15s} {sensor}")
    report.append("")
    
    # Feature statistics
    report.append("4. FEATURE STATISTICS")
    report.append("-" * 70)
    for feat in features:
        stats = df_filtered[feat].describe()
        report.append(f"   {feat.upper()}:")
        report.append(f"      Mean:   {stats['mean']:10.2f}")
        report.append(f"      Std:    {stats['std']:10.2f}")
        report.append(f"      Min:    {stats['min']:10.2f}")
        report.append(f"      Max:    {stats['max']:10.2f}")
        report.append("")
    
    # Correlations
    corr_matrix = df_filtered.corr()
    fire_corr = corr_matrix['fire_alarm'].sort_values(ascending=False)
    report.append("5. CORRELATION WITH FIRE ALARM")
    report.append("-" * 70)
    for feature, corr_value in fire_corr.items():
        if feature != 'fire_alarm':
            report.append(f"   {feature:15s}: {corr_value:+.4f}")
    report.append("")
    
    # Next steps
    report.append("6. NEXT STEPS")
    report.append("-" * 70)
    report.append("   ✅ Dataset exploration complete")
    report.append("   → Run: 02_data_preprocessing.py")
    report.append("   → Then: 03_train_basic_model.py")
    report.append("")
    report.append("="*70)
    
    # Save report
    report_text = "\n".join(report)
    with open("results/exploration_summary.txt", "w") as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✅ Saved: results/exploration_summary.txt")

def main():
    """Main execution function"""
    print("\n" + "🔥"*30)
    print("CAMPUS FIRE DETECTION - DATA EXPLORATION")
    print("🔥"*30 + "\n")
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return
    
    # Basic info
    df = basic_info(df)
    
    # Analyze target
    analyze_target(df)
    
    # Select project features
    df_filtered, project_features = select_project_features(df)
    
    # Visualize distributions
    visualize_feature_distributions(df_filtered)
    
    # Correlation analysis
    analyze_correlations(df_filtered)
    
    # Time series
    visualize_time_series(df, df_filtered)
    
    # Summary report
    generate_summary_report(df, df_filtered)
    
    print("\n" + "="*70)
    print("✅ DATA EXPLORATION COMPLETE!")
    print("="*70)
    print("\nGenerated files in results/:")
    print("   1. 01_target_distribution.png")
    print("   2. 02_feature_distributions.png")
    print("   3. 03_correlation_matrix.png")
    print("   4. 04_fire_event_timeline.png")
    print("   5. exploration_summary.txt")
    print("\n📋 Next step: Run 02_data_preprocessing.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
