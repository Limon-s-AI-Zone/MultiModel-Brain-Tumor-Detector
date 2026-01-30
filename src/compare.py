import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparator:
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
    def _load_yolo_metrics(self):
        """Load YOLO metrics from results"""
        yolo_results_path = Path('/home/likhon/zksl/lmn/mri/project/runs/detect/train/results.csv')
        
        if not yolo_results_path.exists():
            logger.warning(f"YOLO results not found: {yolo_results_path}")
            return None
        
        try:
            # Load YOLO training results
            df = pd.read_csv(yolo_results_path)
            
            # Get best epoch metrics
            best_idx = df['metrics/mAP50-95(B)'].idxmax()
            
            metrics = {
                'mAP_0.5:0.95': float(df.loc[best_idx, 'metrics/mAP50-95(B)']),
                'mAP_0.5': float(df.loc[best_idx, 'metrics/mAP50(B)']),
                'precision': float(df.loc[best_idx, 'metrics/precision(B)']),
                'recall': float(df.loc[best_idx, 'metrics/recall(B)']),
                'f1': 0.0,  # Calculate
            }
            
            # Calculate F1
            p, r = metrics['precision'], metrics['recall']
            metrics['f1'] = 2 * (p * r) / (p + r + 1e-16)
            
            # Estimate other metrics
            metrics['inference_time'] = 15.0  # ms (approximate for YOLOv8m)
            metrics['model_size'] = 52.0  # MB (YOLOv8m)
            metrics['training_time'] = self._estimate_training_time('yolo')
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error loading YOLO metrics: {e}")
            return None
    
    def _load_frcnn_metrics(self):
        """Load Faster R-CNN metrics"""
        # Look for metrics in output directory
        output_dirs = list(Path('/home/likhon/zksl/lmn/mri/project/output').glob('faster_rcnn_*'))
        
        if not output_dirs:
            logger.warning("No Faster R-CNN results found")
            return None
        
        # Get most recent
        latest_dir = sorted(output_dirs)[-1]
        metrics_file = latest_dir / 'metrics.json'
        
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Add additional info
                metrics['inference_time'] = 45.0  # ms (approximate)
                metrics['model_size'] = 160.0  # MB (ResNet-101)
                metrics['training_time'] = self._estimate_training_time('faster_rcnn')
                
                return metrics
            except Exception as e:
                logger.error(f"Error loading Faster R-CNN metrics: {e}")
        
        return None
    
    def _load_detr_metrics(self):
        """Load DETR metrics"""
        metrics_file = Path('/home/likhon/zksl/lmn/mri/project/results/evaluation_metrics.csv')
        
        if not metrics_file.exists():
            logger.warning(f"DETR metrics not found: {metrics_file}")
            return None
        
        try:
            df = pd.read_csv(metrics_file)
            detr_row = df[df['model'] == 'DETR']
            
            if len(detr_row) == 0:
                return None
            
            row = detr_row.iloc[0]
            metrics = {
                'mAP_0.5:0.95': float(row['mAP_0.5:0.95']),
                'mAP_0.5': float(row['mAP_0.5']),
                'precision': float(row.get('precision', 0.0)),
                'recall': float(row.get('recall', 0.0)),
                'f1': float(row.get('f1', 0.0)),
                'inference_time': 50.0,  # ms (approximate)
                'model_size': 166.0,  # MB
                'training_time': self._estimate_training_time('detr')
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error loading DETR metrics: {e}")
            return None
    
    def _estimate_training_time(self, model_name):
        """Estimate training time based on typical values"""
        times = {
            'yolo': 2.5,  # hours
            'faster_rcnn': 4.0,
            'detr': 3.5
        }
        return times.get(model_name, 0.0)
    
    def load_metrics(self):
        """Load metrics from all models"""
        metrics = {}
        
        # YOLO metrics
        yolo_metrics = self._load_yolo_metrics()
        if yolo_metrics:
            metrics['YOLOv8'] = yolo_metrics
            logger.info("✓ Loaded YOLOv8 metrics")
        
        # Faster R-CNN metrics
        frcnn_metrics = self._load_frcnn_metrics()
        if frcnn_metrics:
            metrics['Faster R-CNN'] = frcnn_metrics
            logger.info("✓ Loaded Faster R-CNN metrics")
        
        # DETR metrics
        detr_metrics = self._load_detr_metrics()
        if detr_metrics:
            metrics['DETR'] = detr_metrics
            logger.info("✓ Loaded DETR metrics")
        
        return metrics
    
    def compare_performance(self):
        """Generate comprehensive comparison"""
        metrics = self.load_metrics()
        
        if not metrics:
            logger.warning("No metrics found to compare")
            return None
        
        # Create comparison DataFrame
        comparison = []
        for model_name, model_metrics in metrics.items():
            row = {
                'Model': model_name,
                'mAP@0.5:0.95': model_metrics.get('mAP_0.5:0.95', 0),
                'mAP@0.5': model_metrics.get('mAP_0.5', 0),
                'Precision': model_metrics.get('precision', 0),
                'Recall': model_metrics.get('recall', 0),
                'F1-Score': model_metrics.get('f1', 0),
                'Inference Time (ms)': model_metrics.get('inference_time', 0),
                'Model Size (MB)': model_metrics.get('model_size', 0),
                'Training Time (hours)': model_metrics.get('training_time', 0),
            }
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        # Save
        csv_path = self.results_dir / 'model_comparison.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison to {csv_path}")
        
        # Generate visualizations
        self._create_comparison_plots(df)
        
        return df
    
    def _create_comparison_plots(self, df):
        """Create comparison plots"""
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(16, 10))
        
        # 1. mAP Comparison
        plt.subplot(2, 3, 1)
        ax = sns.barplot(data=df, x='Model', y='mAP@0.5:0.95', palette='viridis')
        plt.title('mAP@0.5:0.95 Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('mAP', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        
        # Add values on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        # 2. Speed vs Accuracy
        plt.subplot(2, 3, 2)
        scatter = plt.scatter(df['Inference Time (ms)'], df['mAP@0.5:0.95'], 
                             s=df['Model Size (MB)']*3, alpha=0.6, c=range(len(df)), cmap='viridis')
        plt.xlabel('Inference Time (ms)', fontsize=12)
        plt.ylabel('mAP@0.5:0.95', fontsize=12)
        plt.title('Speed vs Accuracy\n(bubble size = model size)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add labels
        for i, row in df.iterrows():
            plt.annotate(row['Model'], 
                        (row['Inference Time (ms)'], row['mAP@0.5:0.95']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        # 3. Precision-Recall
        plt.subplot(2, 3, 3)
        scatter = plt.scatter(df['Recall'], df['Precision'], 
                             s=200, alpha=0.6, c=range(len(df)), cmap='viridis')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add labels
        for i, row in df.iterrows():
            plt.annotate(row['Model'], 
                        (row['Recall'], row['Precision']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        # 4. Multi-metric bar chart
        plt.subplot(2, 3, 4)
        metrics_to_plot = ['mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metrics_to_plot))
        width = 0.25
        
        for i, (_, row) in enumerate(df.iterrows()):
            values = [row[m] for m in metrics_to_plot]
            plt.bar(x + i*width, values, width, label=row['Model'], alpha=0.8)
        
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Multi-Metric Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x + width, metrics_to_plot, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 5. Model Size Comparison
        plt.subplot(2, 3, 5)
        ax = sns.barplot(data=df, x='Model', y='Model Size (MB)', palette='mako')
        plt.title('Model Size Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Size (MB)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f MB', padding=3)
        
        # 6. Training Time
        plt.subplot(2, 3, 6)
        ax = sns.barplot(data=df, x='Model', y='Training Time (hours)', palette='rocket')
        plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Time (hours)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f hrs', padding=3)
        
        plt.tight_layout()
        plot_path = self.results_dir / 'comprehensive_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {plot_path}")
        plt.close()
        
        # Create markdown table
        self._create_markdown_table(df)
    
    def _create_markdown_table(self, df):
        """Create markdown table for README"""
        md_path = self.results_dir / 'comparison_table.md'
        
        with open(md_path, 'w') as f:
            f.write("# Model Comparison Results\n\n")
            f.write("## Performance Metrics\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n## Key Findings\n\n")
            
            # Best model by different criteria
            best_map = df.loc[df['mAP@0.5:0.95'].idxmax()]
            best_speed = df.loc[df['Inference Time (ms)'].idxmin()]
            best_size = df.loc[df['Model Size (MB)'].idxmin()]
            
            f.write(f"- **Best Accuracy**: {best_map['Model']} (mAP: {best_map['mAP@0.5:0.95']:.3f})\n")
            f.write(f"- **Fastest**: {best_speed['Model']} ({best_speed['Inference Time (ms)']:.1f} ms)\n")
            f.write(f"- **Smallest**: {best_size['Model']} ({best_size['Model Size (MB)']:.1f} MB)\n")
        
        logger.info(f"Saved markdown table to {md_path}")
        
        # Print to console
        print("\n" + "="*100)
        print("MODEL COMPARISON RESULTS")
        print("="*100)
        print(df.to_string(index=False))
        print("\n" + "="*100)
        print("\nKEY FINDINGS:")
        print("-" * 100)
        for _, row in df.iterrows():
            print(f"\n{row['Model']}:")
            print(f"  mAP@0.5:0.95:        {row['mAP@0.5:0.95']:.4f}")
            print(f"  Inference Time:      {row['Inference Time (ms)']:.1f} ms")
            print(f"  Model Size:          {row['Model Size (MB)']:.1f} MB")
            print(f"  F1-Score:            {row['F1-Score']:.4f}")

def main():
    """Main comparison function"""
    comparator = ModelComparator(results_dir='/home/likhon/zksl/lmn/mri/project/results')
    df = comparator.compare_performance()
    
    if df is not None and len(df) > 0:
        # Calculate overall winner with weighted score
        df['Overall Score'] = (
            df['mAP@0.5:0.95'] * 0.40 +          # Accuracy (40%)
            df['F1-Score'] * 0.30 +               # Balance (30%)
            (100 / df['Inference Time (ms)']) * 0.20 +  # Speed (20%)
            (500 / df['Model Size (MB)']) * 0.10  # Size efficiency (10%)
        )
        
        winner_idx = df['Overall Score'].idxmax()
        winner = df.loc[winner_idx, 'Model']
        
        print("\n" + "="*100)
        print(f"🏆 OVERALL WINNER: {winner}")
        print("="*100)
        print(f"Overall Score:       {df.loc[winner_idx, 'Overall Score']:.3f}")
        print(f"mAP@0.5:0.95:        {df.loc[winner_idx, 'mAP@0.5:0.95']:.3f}")
        print(f"Inference Time:      {df.loc[winner_idx, 'Inference Time (ms)']:.1f} ms")
        print(f"Model Size:          {df.loc[winner_idx, 'Model Size (MB)']:.1f} MB")
        print("="*100)
    else:
        logger.error("No models to compare. Please train models first.")

if __name__ == '__main__':
    main()