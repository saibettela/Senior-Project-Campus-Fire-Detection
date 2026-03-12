"""
==============================================================================
05_export_onnx.py - Export XGBoost Model to ONNX Format
Campus Fire Detection System - Cal Poly San Luis Obispo
Team: Joseph DeChaine, Kai Gottschalk, Saigaurav Bettela
Advisor: Dr. Mohammad Ghamari
==============================================================================

PURPOSE:
    Converts the trained XGBoost fire detection model into ONNX format
    for deployment on the STM32L4Q5T6P microcontroller via X-CUBE-AI.

INPUT FILES (from previous steps):
    - Models/fire_detection_xgboost.pkl      (trained model)
    - Models/fire_detection_xgboost.json     (XGBoost native format)
    - Models/model_metadata.json             (training metrics)
    - Dataset/processed/scaler_parameters.json (normalization params)
    - Dataset/processed/test_data.csv        (for validation)

OUTPUT FILES:
    - Models/fire_detection.onnx             ← CRITICAL: Import into STM32CubeMX
    - Models/onnx_export_report.txt          (validation report)
    - results/12_onnx_validation.png         (prediction comparison plot)

NEXT STEP:
    Follow 04_Model_Deployment/INSTRUCTIONS.md to import fire_detection.onnx
    into STM32CubeMX with X-CUBE-AI.

REQUIREMENTS:
    pip install onnx onnxruntime onnxmltools skl2onnx xgboost scikit-learn

==============================================================================
"""

import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 0: Verify Dependencies
# ============================================================

def check_dependencies():
    """Verify all required libraries are installed."""
    print("\n" + "="*70)
    print("STEP 0: Checking Dependencies")
    print("="*70)

    required = {
        'onnx':        'onnx',
        'onnxruntime': 'onnxruntime',
        'onnxmltools': 'onnxmltools',
        'skl2onnx':    'skl2onnx',
        'xgboost':     'xgboost',
        'sklearn':     'scikit-learn',
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}  ← MISSING")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}")
        raise ImportError(f"Please install missing packages before continuing.")

    print("\n✅ All dependencies satisfied.\n")


# ============================================================
# STEP 1: Load Model and Metadata
# ============================================================

def load_model_and_metadata():
    """Load the trained XGBoost model and its metadata."""
    print("="*70)
    print("STEP 1: Loading Trained Model")
    print("="*70)

    project_root = Path(__file__).resolve().parent.parent
    models_dir   = project_root / "Models"

    # --- Load pickle model ---
    pkl_path = models_dir / "fire_detection_xgboost.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Model not found: {pkl_path}\n"
            "Please run 03_train_basic_model.py first."
        )

    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)
    print(f"   ✅ Loaded model: {pkl_path}")

    # --- Load metadata ---
    meta_path = models_dir / "model_metadata.json"
    metadata  = {}
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        acc = metadata.get('performance_metrics', {}).get('test', {}).get('accuracy', 'N/A')
        if acc != 'N/A':
            print(f"   📊 Stored test accuracy: {float(acc)*100:.2f}%")
    else:
        print(f"   ⚠️  model_metadata.json not found — skipping metadata load.")

    print(f"\n   Model type:    {type(model).__name__}")
    print(f"   n_estimators:  {model.n_estimators}")
    print(f"   max_depth:     {model.max_depth}")
    print(f"   n_features:    {model.n_features_in_}")

    return model, metadata, project_root


# ============================================================
# STEP 2: Load Scaler Parameters
# ============================================================

def load_scaler_params(project_root):
    """Load the MinMax scaler parameters used during training."""
    print("\n" + "="*70)
    print("STEP 2: Loading Scaler Parameters")
    print("="*70)

    scaler_path = project_root / "Dataset" / "processed" / "scaler_parameters.json"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler parameters not found: {scaler_path}\n"
            "Please run 02_data_preprocessing.py first."
        )

    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)

    features   = scaler_params['features']
    min_values = scaler_params['min_values']
    max_values = scaler_params['max_values']

    print(f"\n   Features and normalization ranges:")
    print(f"   {'Feature':<14} {'Min':>12}  {'Max':>12}")
    print(f"   {'-'*42}")
    for feat, mn, mx in zip(features, min_values, max_values):
        print(f"   {feat:<14} {mn:>12.4f}  {mx:>12.4f}")

    return scaler_params


# ============================================================
# STEP 3: Load Test Data for Validation
# ============================================================

def load_test_data(project_root):
    """Load the held-out test dataset for ONNX validation."""
    print("\n" + "="*70)
    print("STEP 3: Loading Test Data for Validation")
    print("="*70)

    test_path = project_root / "Dataset" / "processed" / "test_data.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_path}\n"
            "Please run 02_data_preprocessing.py first."
        )

    df = pd.read_csv(test_path)
    feature_cols = ['temperature', 'humidity', 'tvoc', 'eco2']
    X_test = df[feature_cols].values.astype(np.float32)
    y_test = df['fire_alarm'].values

    print(f"\n   ✅ Loaded {len(df):,} test samples")
    print(f"   Fire samples:    {int(y_test.sum()):,} ({y_test.mean()*100:.1f}%)")
    print(f"   No-fire samples: {int((1-y_test).sum()):,} ({(1-y_test).mean()*100:.1f}%)")

    return X_test, y_test


# ============================================================
# STEP 4: Convert to ONNX
# ============================================================

def convert_to_onnx(model, project_root):
    """
    Convert the XGBoost classifier to ONNX format using onnxmltools.
    Output shape is compatible with X-CUBE-AI on STM32.
    """
    print("\n" + "="*70)
    print("STEP 4: Converting XGBoost Model to ONNX")
    print("="*70)

    import onnxmltools
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    # Input: 4 float features (temperature, humidity, tvoc, eco2)
    n_features   = model.n_features_in_
    initial_type = [('float_input', FloatTensorType([None, n_features]))]

    print(f"\n   Converting model...")
    print(f"   Input shape:  [batch, {n_features}]  (float32)")
    print(f"   Output:       binary classification probabilities + label")

    onnx_model = convert_xgboost(
        model,
        initial_types=initial_type,
        target_opset=12          # Opset 12 is well-supported by X-CUBE-AI
    )

    # Save ONNX model
    models_dir  = project_root / "Models"
    models_dir.mkdir(exist_ok=True)
    onnx_path   = models_dir / "fire_detection.onnx"

    import onnx
    onnx.save_model(onnx_model, str(onnx_path))

    file_size_kb = onnx_path.stat().st_size / 1024
    print(f"\n   ✅ ONNX model saved: {onnx_path}")
    print(f"   File size: {file_size_kb:.1f} KB")

    # Also copy to 04_Model_Deployment folder if it exists
    deploy_dir  = project_root / "04_Model_Deployment"
    if deploy_dir.exists():
        import shutil
        deploy_onnx = deploy_dir / "fire_detection.onnx"
        shutil.copy2(str(onnx_path), str(deploy_onnx))
        print(f"   ✅ Copied to deployment folder: {deploy_onnx}")

    return onnx_model, onnx_path


# ============================================================
# STEP 5: Validate ONNX Model
# ============================================================

def validate_onnx(model_sklearn, onnx_path, X_test, y_test):
    """
    Run inference with both the original sklearn model and the ONNX model,
    then compare predictions to confirm export accuracy.
    """
    print("\n" + "="*70)
    print("STEP 5: Validating ONNX Model")
    print("="*70)

    import onnxruntime as rt
    from sklearn.metrics import accuracy_score, confusion_matrix

    # --- Original sklearn predictions ---
    sklearn_preds = model_sklearn.predict(X_test)
    sklearn_probs = model_sklearn.predict_proba(X_test)[:, 1]
    sklearn_acc   = accuracy_score(y_test, sklearn_preds)

    # --- ONNX Runtime predictions ---
    sess       = rt.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    # Run in batches to handle large test sets gracefully
    batch_size  = 1000
    onnx_preds  = []
    onnx_probs  = []

    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i + batch_size]
        pred  = sess.run(None, {input_name: batch})
        # pred[0] = label array, pred[1] = list of dicts {0: p0, 1: p1}
        onnx_preds.extend(pred[0].tolist())
        batch_probs = [d[1] for d in pred[1]]
        onnx_probs.extend(batch_probs)

    onnx_preds = np.array(onnx_preds)
    onnx_probs = np.array(onnx_probs)
    onnx_acc   = accuracy_score(y_test, onnx_preds)

    # --- Agreement between sklearn and ONNX ---
    agreement     = np.mean(sklearn_preds == onnx_preds) * 100
    disagreements = int(np.sum(sklearn_preds != onnx_preds))

    # --- Confusion matrix (ONNX) ---
    cm = confusion_matrix(y_test, onnx_preds)
    tn, fp, fn, tp = cm.ravel()
    false_alarm_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0.0
    detection_rate   = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0

    print(f"\n   Original XGBoost accuracy:  {sklearn_acc*100:.4f}%")
    print(f"   ONNX Runtime accuracy:      {onnx_acc*100:.4f}%")
    print(f"   Prediction agreement:       {agreement:.4f}%")
    if disagreements > 0:
        print(f"   ⚠️  Disagreements:          {disagreements} samples")
    else:
        print(f"   ✅ Perfect agreement — no prediction drift during export!")

    print(f"\n   ONNX Confusion Matrix (Test Set):")
    print(f"   {'':16s}  Pred No-Fire  Pred Fire")
    print(f"   {'Actual No-Fire':16s}  {tn:>10,}    {fp:>8,}")
    print(f"   {'Actual Fire':16s}  {fn:>10,}    {tp:>8,}")

    print(f"\n   False Alarm Rate:  {false_alarm_rate:.2f}%  "
          f"({'✅ PASS' if false_alarm_rate <= 3.0 else '❌ FAIL'} DR-05: ≤3%)")
    print(f"   Detection Rate:    {detection_rate:.2f}%  "
          f"({'✅ PASS' if onnx_acc >= 0.90 else '❌ FAIL'} DR-06: ≥90%)")

    metrics = {
        'sklearn_accuracy': float(sklearn_acc),
        'onnx_accuracy':    float(onnx_acc),
        'agreement_pct':    float(agreement),
        'disagreements':    int(disagreements),
        'false_alarm_rate': float(false_alarm_rate),
        'detection_rate':   float(detection_rate),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp),
                             'fn': int(fn), 'tp': int(tp)},
        'dr05_pass': bool(false_alarm_rate <= 3.0),
        'dr06_pass': bool(onnx_acc >= 0.90),
    }

    return metrics, sklearn_probs, onnx_probs


# ============================================================
# STEP 6: Generate Validation Plot
# ============================================================

def generate_validation_plot(metrics, sklearn_probs, onnx_probs, y_test, project_root):
    """Create a visual comparison of sklearn vs ONNX predictions."""
    print("\n" + "="*70)
    print("STEP 6: Generating Validation Plot")
    print("="*70)

    results_dir = project_root / "03_Model_Training" / "results"
    if not results_dir.exists():
        results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ONNX Export Validation\nCampus Fire Detection System',
                 fontsize=16, fontweight='bold', y=1.01)

    # ---- Plot 1: Probability scatter (sklearn vs ONNX) ----
    ax = axes[0, 0]
    sample_idx = np.random.choice(len(sklearn_probs), min(2000, len(sklearn_probs)), replace=False)
    colors = ['#FF6B6B' if y == 1 else '#4ECDC4' for y in y_test[sample_idx]]
    ax.scatter(sklearn_probs[sample_idx], onnx_probs[sample_idx],
               c=colors, alpha=0.4, s=10, edgecolors='none')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect agreement')
    ax.set_xlabel('XGBoost Probability', fontsize=11)
    ax.set_ylabel('ONNX Runtime Probability', fontsize=11)
    ax.set_title('Probability Comparison\n(sklearn vs ONNX)', fontsize=12, fontweight='bold')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fire_patch    = mpatches.Patch(color='#FF6B6B', label='Fire')
    no_fire_patch = mpatches.Patch(color='#4ECDC4', label='No Fire')
    ax.legend(handles=[fire_patch, no_fire_patch, 
                        plt.Line2D([0],[0], color='black', linestyle='--', label='y = x')],
              fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: Probability difference distribution ----
    ax = axes[0, 1]
    diff = onnx_probs - sklearn_probs
    ax.hist(diff, bins=50, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero difference')
    ax.set_xlabel('ONNX Prob − XGBoost Prob', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Prediction Difference Distribution\n(should be near zero)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    max_diff = np.max(np.abs(diff))
    ax.text(0.05, 0.95, f'Max |diff| = {max_diff:.6f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ---- Plot 3: Confusion matrix ----
    ax = axes[1, 0]
    cm = metrics['confusion_matrix']
    cm_array = np.array([[cm['tn'], cm['fp']],
                          [cm['fn'], cm['tp']]])
    im = ax.imshow(cm_array, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Fire', 'Fire'], fontsize=11)
    ax.set_yticklabels(['No Fire', 'Fire'], fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title('ONNX Confusion Matrix\n(Test Set)', fontsize=12, fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm_array[i, j]:,}',
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color='white' if cm_array[i, j] > cm_array.max() / 2 else 'black')

    # ---- Plot 4: Summary metrics bar chart ----
    ax = axes[1, 1]
    labels = ['ONNX\nAccuracy', 'Detection\nRate', 'False Alarm\nRate']
    values = [metrics['onnx_accuracy'] * 100,
              metrics['detection_rate'],
              metrics['false_alarm_rate']]
    thresholds = [90.0, 90.0, 3.0]
    colors_bar  = ['#2ecc71' if (i < 2 and v >= t) or (i == 2 and v <= t)
                   else '#e74c3c'
                   for i, (v, t) in enumerate(zip(values, thresholds))]

    bars = ax.bar(labels, values, color=colors_bar,
                  edgecolor='black', linewidth=2, width=0.5)
    ax.axhline(90.0, color='green',  linestyle='--', lw=2, label='90% target (DR-06)')
    ax.axhline(3.0,  color='orange', linestyle='--', lw=2, label='3% limit (DR-05)')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('ONNX Model Performance\nvs Design Requirements',
                 fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = results_dir / '12_onnx_validation.png'
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")
    return plot_path


# ============================================================
# STEP 7: Inspect ONNX Model Graph
# ============================================================

def inspect_onnx_model(onnx_path):
    """Print ONNX model graph info for debugging / report purposes."""
    print("\n" + "="*70)
    print("STEP 7: ONNX Model Inspection")
    print("="*70)

    import onnx
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print("   ✅ ONNX model structure is valid (onnx.checker passed)")

    graph = model.graph
    print(f"\n   Inputs:")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value > 0 else '?' 
                 for d in inp.type.tensor_type.shape.dim]
        print(f"     - {inp.name}: float32 {shape}")

    print(f"\n   Outputs:")
    for out in graph.output:
        print(f"     - {out.name}")

    print(f"\n   Nodes (operators): {len(graph.node)}")
    op_counts = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"     {op}: {count}")

    print(f"\n   ONNX opset version: {model.opset_import[0].version}")
    return model


# ============================================================
# STEP 8: Save Export Report
# ============================================================

def save_export_report(onnx_path, metrics, scaler_params, project_root):
    """Write a comprehensive text report of the ONNX export."""
    print("\n" + "="*70)
    print("STEP 8: Saving Export Report")
    print("="*70)

    models_dir = project_root / "Models"
    report_path = models_dir / "onnx_export_report.txt"

    lines = []
    lines.append("=" * 70)
    lines.append("ONNX EXPORT REPORT")
    lines.append("Campus Fire Detection System — XGBoost to ONNX")
    lines.append(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("1. OUTPUT FILE")
    lines.append("-" * 70)
    lines.append(f"   Path:      {onnx_path}")
    lines.append(f"   Size:      {onnx_path.stat().st_size / 1024:.1f} KB")
    lines.append(f"   Format:    ONNX opset 12")
    lines.append(f"   Input:     float_input [batch, 4] (float32)")
    lines.append(f"   Features:  temperature, humidity, tvoc, eco2")
    lines.append("")

    lines.append("2. VALIDATION RESULTS (on held-out test set)")
    lines.append("-" * 70)
    lines.append(f"   Original XGBoost accuracy:  {metrics['sklearn_accuracy']*100:.4f}%")
    lines.append(f"   ONNX Runtime accuracy:      {metrics['onnx_accuracy']*100:.4f}%")
    lines.append(f"   Prediction agreement:       {metrics['agreement_pct']:.4f}%")
    lines.append(f"   Disagreements:              {metrics['disagreements']} samples")
    lines.append("")

    lines.append("3. DESIGN REQUIREMENTS VERIFICATION")
    lines.append("-" * 70)
    lines.append(f"   DR-06 (accuracy ≥ 90%):      {metrics['onnx_accuracy']*100:.2f}%  "
                 f"{'✅ PASS' if metrics['dr06_pass'] else '❌ FAIL'}")
    lines.append(f"   DR-05 (false alarms ≤ 3%):   {metrics['false_alarm_rate']:.2f}%  "
                 f"{'✅ PASS' if metrics['dr05_pass'] else '❌ FAIL'}")
    lines.append("")

    lines.append("4. SCALER PARAMETERS (copy to STM32 firmware)")
    lines.append("-" * 70)
    lines.append("   ⚠️  CRITICAL: These values MUST match your C firmware exactly!")
    lines.append("")
    lines.append("   C struct example:")
    lines.append("   static const ScalerParams_t scaler = {")
    for feat, mn, mx in zip(scaler_params['features'],
                             scaler_params['min_values'],
                             scaler_params['max_values']):
        lines.append(f"       .{feat}_min = {mn:.6f}f,")
        lines.append(f"       .{feat}_max = {mx:.6f}f,")
    lines.append("   };")
    lines.append("")
    lines.append("   Normalization formula (apply before inference):")
    lines.append("   normalized = (raw_value - min) / (max - min)")
    lines.append("")

    lines.append("5. STM32 DEPLOYMENT NEXT STEPS")
    lines.append("-" * 70)
    lines.append("   1. Open STM32CubeMX")
    lines.append("   2. Install X-CUBE-AI expansion pack (if not already done)")
    lines.append("   3. Add AI component → X-CUBE-AI → Application Template")
    lines.append("   4. Import: fire_detection.onnx")
    lines.append("   5. Set quantization: int8 (reduces size, minimal accuracy loss)")
    lines.append("   6. Click 'Analyze' → check memory requirements")
    lines.append("   7. Generate Code → copy AI files to 05_Firmware/")
    lines.append("   8. Hardcode scaler parameters into fire_detection_app.c")
    lines.append("")
    lines.append("   Refer to: 04_Model_Deployment/INSTRUCTIONS.md for full details.")
    lines.append("")
    lines.append("=" * 70)

    with open(report_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"   ✅ Saved report: {report_path}")
    return report_path


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "🔥" * 35)
    print("CAMPUS FIRE DETECTION — EXPORT TO ONNX")
    print("🔥" * 35)

    # Step 0: Dependencies
    check_dependencies()

    # Step 1: Load model
    model, metadata, project_root = load_model_and_metadata()

    # Step 2: Scaler params
    scaler_params = load_scaler_params(project_root)

    # Step 3: Test data
    X_test, y_test = load_test_data(project_root)

    # Step 4: Convert
    onnx_model, onnx_path = convert_to_onnx(model, project_root)

    # Step 5: Validate
    metrics, sklearn_probs, onnx_probs = validate_onnx(model, onnx_path, X_test, y_test)

    # Step 6: Plot
    generate_validation_plot(metrics, sklearn_probs, onnx_probs, y_test, project_root)

    # Step 7: Inspect graph
    inspect_onnx_model(onnx_path)

    # Step 8: Report
    save_export_report(onnx_path, metrics, scaler_params, project_root)

    # Final summary
    print("\n" + "=" * 70)
    print("✅ ONNX EXPORT COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"   1. Models/fire_detection.onnx          ← Import into STM32CubeMX")
    print(f"   2. Models/onnx_export_report.txt")
    print(f"   3. results/12_onnx_validation.png")

    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    print(f"   ONNX Accuracy:      {metrics['onnx_accuracy']*100:.2f}%")
    print(f"   False Alarm Rate:   {metrics['false_alarm_rate']:.2f}%")
    print(f"   Agreement w/ orig:  {metrics['agreement_pct']:.4f}%")
    print(f"\n   DR-06 (≥90% accuracy):    {'✅ PASS' if metrics['dr06_pass'] else '❌ FAIL'}")
    print(f"   DR-05 (≤3% false alarms): {'✅ PASS' if metrics['dr05_pass'] else '❌ FAIL'}")
    print("\n📋 Next step: Follow 04_Model_Deployment/INSTRUCTIONS.md")
    print("   Import fire_detection.onnx into STM32CubeMX → X-CUBE-AI")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()