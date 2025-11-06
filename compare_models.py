"""
Compare CNN with ML Models

Compare the performance of CNN vs Random Forest vs XGBoost.

Author: Anish
Date: November 6, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_results():
    """Load results from all models."""
    results_dir = Path('results/metrics')
    
    results = {}
    
    # Load ML results (optimized thresholds)
    ml_file = results_dir / 'ml_models_optimized.csv'
    if ml_file.exists():
        ml_df = pd.read_csv(ml_file)
        
        # Get best Random Forest (threshold 0.15)
        rf_rows = ml_df[(ml_df['model'] == 'Random Forest') & (ml_df['threshold'] == 0.15)]
        if len(rf_rows) > 0:
            rf_best = rf_rows.iloc[0]
            results['Random Forest'] = {
                'accuracy': rf_best['accuracy'],
                'precision': rf_best['precision'],
                'recall': rf_best['recall'],
                'f1_score': rf_best['f1_score'],
                'model_type': 'ML'
            }
        
        # Get best XGBoost (threshold 0.10)
        xgb_rows = ml_df[(ml_df['model'] == 'XGBoost') & (ml_df['threshold'] == 0.10)]
        if len(xgb_rows) > 0:
            xgb_best = xgb_rows.iloc[0]
            results['XGBoost'] = {
                'accuracy': xgb_best['accuracy'],
                'precision': xgb_best['precision'],
                'recall': xgb_best['recall'],
                'f1_score': xgb_best['f1_score'],
                'model_type': 'ML'
            }
    
    # Load CNN results
    cnn_file = results_dir / 'cnn_results.csv'
    if cnn_file.exists():
        cnn_df = pd.read_csv(cnn_file)
        cnn_row = cnn_df.iloc[0]
        results['CNN (1D)'] = {
            'accuracy': cnn_row['accuracy'],
            'precision': cnn_row['precision'],
            'recall': cnn_row['recall'],
            'f1_score': cnn_row['f1_score'],
            'model_type': 'Deep Learning'
        }
    
    # Load baseline results
    baseline_file = results_dir / 'baseline_results_hai.csv'
    if baseline_file.exists():
        baseline_df = pd.read_csv(baseline_file)
        if_row = baseline_df[baseline_df['method'] == 'isolation_forest'].iloc[0]
        results['Isolation Forest'] = {
            'accuracy': if_row['accuracy'],
            'precision': if_row['precision'],
            'recall': if_row['recall'],
            'f1_score': if_row['f1_score'],
            'model_type': 'Baseline'
        }
    
    return results


def create_comparison_table(results):
    """Create a formatted comparison table."""
    comparison_data = []
    
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Type': metrics['model_type'],
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}"
        })
    
    df = pd.DataFrame(comparison_data)
    return df


def plot_comparison(results, output_path='results/plots/model_comparison.png'):
    """Create visualization comparing all models."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = {'Baseline': '#FF6B6B', 'ML': '#4ECDC4', 'Deep Learning': '#45B7D1'}
    
    for ax, metric in zip(axes, metrics):
        values = [results[model][metric] for model in models]
        model_colors = [colors[results[model]['model_type']] for model in models]
        
        bars = ax.bar(range(len(models)), values, color=model_colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2%}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(metric.capitalize().replace('_', ' '))
        ax.set_title(f'{metric.capitalize().replace("_", " ")} Comparison', fontweight='bold', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['Baseline'], label='Baseline'),
        Patch(facecolor=colors['ML'], label='ML Models'),
        Patch(facecolor=colors['Deep Learning'], label='Deep Learning')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_path}")
    plt.close()


def main():
    """Main comparison function."""
    print(f"\n{'='*70}")
    print("Model Performance Comparison")
    print(f"{'='*70}\n")
    
    # Load all results
    print("Loading results from all models...")
    results = load_all_results()
    
    if not results:
        print("❌ No results found! Run models first.")
        return
    
    print(f"✓ Loaded results for {len(results)} models\n")
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    
    print("Performance Comparison:")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70)
    
    # Save comparison table
    output_file = Path('results/metrics/all_models_comparison.csv')
    comparison_df.to_csv(output_file, index=False)
    print(f"\n✓ Comparison table saved to: {output_file}")
    
    # Create visualization
    plot_comparison(results)
    
    # Find best model
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    
    # Best by each metric
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        best_model = max(results.items(), key=lambda x: x[1][metric])
        print(f"Best {metric.capitalize()}: {best_model[0]} ({best_model[1][metric]:.4f})")
    
    # Overall winner (based on F1-score)
    winner = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\n🏆 Overall Winner (F1-Score): {winner[0]}")
    print(f"   Accuracy: {winner[1]['accuracy']:.4f}")
    print(f"   Precision: {winner[1]['precision']:.4f}")
    print(f"   Recall: {winner[1]['recall']:.4f}")
    print(f"   F1-Score: {winner[1]['f1_score']:.4f}")
    
    print(f"\n{'='*70}")
    print("✅ Comparison complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
