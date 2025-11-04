"""
Compare Standard vs Few-Shot Performance

This script compares the performance of standard and few-shot modes
across different models and languages.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_summary(file_path: str) -> Dict:
    """Load summary.json file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Ensure all expected fields exist with default 0.0
            metrics = {
                'total_samples': data.get('total_samples', 0),
                'compiled_rate': data.get('compiled_rate', 0.0),
                'branch_coverage': data.get('branch_coverage', 0.0),
                'line_coverage': data.get('line_coverage', 0.0),
                'test_pass_rate': data.get('test_pass_rate', 0.0),
                'invocation_rate': data.get('invocation_rate', 0.0)
            }
            return metrics
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return None


def collect_results(base_dir: str) -> List[Dict]:
    """
    Collect all results from evaluation directory.
    
    Args:
        base_dir: Base directory containing evaluation results
        
    Returns:
        List of result dictionaries
    """
    results = []
    base_path = Path(base_dir)
    
    for lang_dir in base_path.iterdir():
        if not lang_dir.is_dir():
            continue
            
        language = lang_dir.name
        
        for mode_dir in lang_dir.iterdir():
            if not mode_dir.is_dir() or mode_dir.name not in ['standard', 'fewshot']:
                continue
                
            mode = mode_dir.name
            
            for model_dir in mode_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                model_name = model_dir.name
                summary_path = model_dir / "summary.json"
                
                if summary_path.exists():
                    summary = load_summary(str(summary_path))
                    
                    if summary:
                        result = {
                            'Language': language,
                            'Mode': mode,
                            'Model': model_name,
                            'Total Samples': summary.get('total_samples', 0),
                            'Compiled Rate (%)': summary.get('compiled_rate', 0.0),
                            'Branch Coverage (%)': summary.get('branch_coverage', 0.0),
                            'Line Coverage (%)': summary.get('line_coverage', 0.0),
                            'Test Pass Rate (%)': summary.get('test_pass_rate', 0.0),
                            'Invocation Rate (%)': summary.get('invocation_rate', 0.0)
                        }
                        results.append(result)
    
    return results


def create_comparison_table(results: List[Dict], metric: str = 'Test Pass Rate (%)') -> pd.DataFrame:
    """
    Create a comparison table with standard and few-shot side by side for a specific metric.
    
    Args:
        results: List of result dictionaries
        metric: The metric to compare (default: 'Test Pass Rate (%)')
        
    Returns:
        Pandas DataFrame with comparison
    """
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No results found!")
        return df
    
    # Pivot to get standard and fewshot side by side
    pivot_df = df.pivot_table(
        index=['Language', 'Model'],
        columns='Mode',
        values=metric,
        aggfunc='first'
    ).reset_index()
    
    # Ensure both columns exist
    if 'standard' not in pivot_df.columns:
        pivot_df['standard'] = 0.0
    if 'fewshot' not in pivot_df.columns:
        pivot_df['fewshot'] = 0.0
    
    # Calculate absolute difference and relative improvement
    pivot_df['Absolute Diff'] = (pivot_df['fewshot'] - pivot_df['standard']).round(4)
    pivot_df['Relative Improvement (%)'] = ((pivot_df['fewshot'] - pivot_df['standard']) / pivot_df['standard'] * 100).round(2)
    pivot_df['Relative Improvement (%)'] = pivot_df['Relative Improvement (%)'].replace([float('inf'), -float('inf')], 0).fillna(0)
    
    # Reorder columns to ensure correct order before renaming
    pivot_df = pivot_df[['Language', 'Model', 'standard', 'fewshot', 'Absolute Diff', 'Relative Improvement (%)']]
    
    # Rename columns for clarity
    pivot_df.columns = ['Language', 'Model', f'Standard {metric}', f'Few-Shot {metric}', 'Absolute Diff', 'Relative Improvement (%)']
    
    # Sort by language and model
    pivot_df = pivot_df.sort_values(['Language', 'Model'])
    
    return pivot_df


def create_summary_by_language(results: List[Dict], metric: str = 'Test Pass Rate (%)') -> pd.DataFrame:
    """
    Create a summary table grouped by language for a specific metric.
    
    Args:
        results: List of result dictionaries
        metric: The metric to summarize
        
    Returns:
        Pandas DataFrame with language-level summary
    """
    df = pd.DataFrame(results)
    
    if df.empty:
        return df
    
    # Group by language and mode
    summary = df.groupby(['Language', 'Mode'])[metric].mean().reset_index()
    
    # Pivot to get standard and fewshot side by side
    pivot_summary = summary.pivot_table(
        index='Language',
        columns='Mode',
        values=metric,
        aggfunc='first'
    ).reset_index()
    
    # Ensure both columns exist
    if 'standard' not in pivot_summary.columns:
        pivot_summary['standard'] = 0.0
    if 'fewshot' not in pivot_summary.columns:
        pivot_summary['fewshot'] = 0.0
    
    # Calculate improvement
    pivot_summary['Absolute Diff'] = (pivot_summary['fewshot'] - pivot_summary['standard']).round(4)
    pivot_summary['Relative Improvement (%)'] = ((pivot_summary['fewshot'] - pivot_summary['standard']) / pivot_summary['standard'] * 100).round(2)
    pivot_summary['Relative Improvement (%)'] = pivot_summary['Relative Improvement (%)'].replace([float('inf'), -float('inf')], 0).fillna(0)
    
    # Reorder columns to ensure correct order before renaming
    pivot_summary = pivot_summary[['Language', 'standard', 'fewshot', 'Absolute Diff', 'Relative Improvement (%)']]
    
    # Rename columns
    pivot_summary.columns = ['Language', f'Avg Standard {metric}', f'Avg Few-Shot {metric}', 'Absolute Diff', 'Relative Improvement (%)']
    
    return pivot_summary


def create_summary_by_model(results: List[Dict], metric: str = 'Test Pass Rate (%)') -> pd.DataFrame:
    """
    Create a summary table grouped by model for a specific metric.
    
    Args:
        results: List of result dictionaries
        metric: The metric to summarize
        
    Returns:
        Pandas DataFrame with model-level summary
    """
    df = pd.DataFrame(results)
    
    if df.empty:
        return df
    
    # Group by model and mode
    summary = df.groupby(['Model', 'Mode'])[metric].mean().reset_index()
    
    # Pivot to get standard and fewshot side by side
    pivot_summary = summary.pivot_table(
        index='Model',
        columns='Mode',
        values=metric,
        aggfunc='first'
    ).reset_index()
    
    # Ensure both columns exist
    if 'standard' not in pivot_summary.columns:
        pivot_summary['standard'] = 0.0
    if 'fewshot' not in pivot_summary.columns:
        pivot_summary['fewshot'] = 0.0
    
    # Calculate improvement
    pivot_summary['Absolute Diff'] = (pivot_summary['fewshot'] - pivot_summary['standard']).round(4)
    pivot_summary['Relative Improvement (%)'] = ((pivot_summary['fewshot'] - pivot_summary['standard']) / pivot_summary['standard'] * 100).round(2)
    pivot_summary['Relative Improvement (%)'] = pivot_summary['Relative Improvement (%)'].replace([float('inf'), -float('inf')], 0).fillna(0)
    
    # Reorder columns to ensure correct order before renaming
    pivot_summary = pivot_summary[['Model', 'standard', 'fewshot', 'Absolute Diff', 'Relative Improvement (%)']]
    
    # Rename columns
    pivot_summary.columns = ['Model', f'Avg Standard {metric}', f'Avg Few-Shot {metric}', 'Absolute Diff', 'Relative Improvement (%)']
    
    # Sort by improvement
    pivot_summary = pivot_summary.sort_values('Relative Improvement (%)', ascending=False)
    
    return pivot_summary


def print_markdown_table(df: pd.DataFrame, title: str):
    """Print DataFrame as a markdown table."""
    print(f"\n## {title}\n")
    print(df.to_markdown(index=False))
    print()


def save_to_csv(df: pd.DataFrame, filename: str):
    """Save DataFrame to CSV file."""
    df.to_csv(filename, index=False)
    print(f"✓ Saved to: {filename}")


def main():
    """Main function."""
    # Configuration
    EVAL_DIR = "evaluation/evaluation_rs"
    OUTPUT_DIR = "evaluation"
    
    # All metrics to compare
    METRICS = [
        'Compiled Rate (%)',
        'Branch Coverage (%)',
        'Line Coverage (%)',
        'Test Pass Rate (%)',
        'Invocation Rate (%)'
    ]
    
    print("=" * 80)
    print("Standard vs Few-Shot Performance Comparison")
    print("=" * 80)
    print()
    
    # Collect results
    print("Collecting results...")
    results = collect_results(EVAL_DIR)
    print(f"✓ Found {len(results)} result files")
    print()
    
    if not results:
        print("⚠ No results found. Please check the evaluation directory.")
        return
    
    print("Creating comparison tables for all metrics...")
    print()
    
    # Create comparison tables for each metric
    for metric in METRICS:
        print(f"\n{'=' * 80}")
        print(f"Metric: {metric}")
        print('=' * 80)
        
        # 1. Detailed comparison (all models and languages)
        detailed_comparison = create_comparison_table(results, metric)
        print_markdown_table(detailed_comparison, f"Detailed Comparison: {metric}")
        
        # Save to CSV
        safe_metric_name = metric.replace(' ', '_').replace('(%)', '').replace('(', '').replace(')', '')
        save_to_csv(detailed_comparison, os.path.join(OUTPUT_DIR, f"comparison_{safe_metric_name}_detailed.csv"))
        
        # 2. Summary by language
        language_summary = create_summary_by_language(results, metric)
        print_markdown_table(language_summary, f"Summary by Language: {metric}")
        save_to_csv(language_summary, os.path.join(OUTPUT_DIR, f"comparison_{safe_metric_name}_by_language.csv"))
        
        # 3. Summary by model
        model_summary = create_summary_by_model(results, metric)
        print_markdown_table(model_summary, f"Summary by Model: {metric}")
        save_to_csv(model_summary, os.path.join(OUTPUT_DIR, f"comparison_{safe_metric_name}_by_model.csv"))
        
        # 4. Overall statistics for this metric
        print(f"\n### Overall Statistics for {metric}\n")
        df = pd.DataFrame(results)
        
        overall_standard = df[df['Mode'] == 'standard'][metric].mean()
        overall_fewshot = df[df['Mode'] == 'fewshot'][metric].mean()
        overall_improvement = ((overall_fewshot - overall_standard) / overall_standard * 100) if overall_standard > 0 else 0
        
        print(f"Overall Average Standard {metric}:  {overall_standard:.4f}")
        print(f"Overall Average Few-Shot {metric}:  {overall_fewshot:.4f}")
        print(f"Overall Average Improvement:        {overall_improvement:.2f}%")
        print()
        
        # Count wins
        wins = (detailed_comparison[f'Few-Shot {metric}'] > detailed_comparison[f'Standard {metric}']).sum()
        total = len(detailed_comparison)
        print(f"Few-Shot wins: {wins}/{total} ({wins/total*100:.1f}%)")
        print()
    
    # Create a consolidated summary across all metrics
    print("\n" + "=" * 80)
    print("Consolidated Summary Across All Metrics")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    summary_data = []
    
    for metric in METRICS:
        overall_standard = df[df['Mode'] == 'standard'][metric].mean()
        overall_fewshot = df[df['Mode'] == 'fewshot'][metric].mean()
        overall_improvement = ((overall_fewshot - overall_standard) / overall_standard * 100) if overall_standard > 0 else 0
        
        summary_data.append({
            'Metric': metric,
            'Avg Standard': round(overall_standard, 4),
            'Avg Few-Shot': round(overall_fewshot, 4),
            'Absolute Diff': round(overall_fewshot - overall_standard, 4),
            'Improvement (%)': round(overall_improvement, 2)
        })
    
    consolidated_df = pd.DataFrame(summary_data)
    print_markdown_table(consolidated_df, "Overall Performance Comparison")
    save_to_csv(consolidated_df, os.path.join(OUTPUT_DIR, "comparison_consolidated_summary.csv"))
    
    print("=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
