# import json
# import numpy as np
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import logging
# import torch
# from PIL import Image
# import os

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def evaluate_coco_predictions(gt_path, pred_path, model_name):
#     """Evaluate predictions using COCO metrics"""
#     try:
#         coco_gt = COCO(gt_path)
#         coco_dt = coco_gt.loadRes(pred_path)
        
#         coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         coco_eval.summarize()
        
#         # Extract metrics
#         metrics = {
#             'model': model_name,
#             'mAP_0.5:0.95': float(coco_eval.stats[0]),
#             'mAP_0.5': float(coco_eval.stats[1]),
#             'mAP_0.75': float(coco_eval.stats[2]),
#             'mAP_small': float(coco_eval.stats[3]),
#             'mAP_medium': float(coco_eval.stats[4]),
#             'mAP_large': float(coco_eval.stats[5]),
#             'AR_max1': float(coco_eval.stats[6]),
#             'AR_max10': float(coco_eval.stats[7]),
#             'AR_max100': float(coco_eval.stats[8]),
#             'AR_small': float(coco_eval.stats[9]),
#             'AR_medium': float(coco_eval.stats[10]),
#             'AR_large': float(coco_eval.stats[11]),
#         }
        
#         # Calculate precision, recall, F1
#         if len(coco_eval.eval['precision']) > 0:
#             precision = np.mean(coco_eval.eval['precision'][coco_eval.eval['precision'] > -1])
#             recall = float(coco_eval.stats[8])  # AR@100
#             metrics['precision'] = float(precision) if not np.isnan(precision) else 0.0
#             metrics['recall'] = recall
#             metrics['f1'] = 2 * (metrics['precision'] * recall) / (metrics['precision'] + recall + 1e-16)
#         else:
#             metrics['precision'] = 0.0
#             metrics['recall'] = 0.0
#             metrics['f1'] = 0.0
        
#         return metrics, coco_eval
    
#     except Exception as e:
#         logger.error(f"Error evaluating {model_name}: {e}")
#         return None, None

# def generate_detr_predictions(model_path, data_dir, annotations_path, output_path):
#     """Generate DETR predictions in COCO format"""
#     from transformers import DetrForObjectDetection, DetrImageProcessor
    
#     logger.info("Loading DETR model...")
#     processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
#     model = DetrForObjectDetection.from_pretrained(
#         "facebook/detr-resnet-50",
#         num_labels=4,
#         ignore_mismatched_sizes=True
#     )
    
#     # Load trained weights
#     state_dict = torch.load(model_path, map_location='cpu')
#     model.load_state_dict(state_dict)
#     model.eval()
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
    
#     # Load annotations to get image list
#     with open(annotations_path, 'r') as f:
#         coco_data = json.load(f)
    
#     predictions = []
    
#     logger.info(f"Generating predictions for {len(coco_data['images'])} images...")
    
#     for img_info in coco_data['images']:
#         img_path = Path(data_dir) / img_info['file_name']
#         if not img_path.exists():
#             logger.warning(f"Image not found: {img_path}")
#             continue
        
#         # Load and process image
#         image = Image.open(img_path).convert('RGB')
#         inputs = processor(images=image, return_tensors="pt")
#         inputs = {k: v.to(device) for k, v in inputs.items()}
        
#         # Inference
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         # Post-process
#         target_sizes = torch.tensor([image.size[::-1]]).to(device)
#         results = processor.post_process_object_detection(
#             outputs, 
#             threshold=0.05, 
#             target_sizes=target_sizes
#         )[0]
        
#         # Convert to COCO format
#         for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
#             x1, y1, x2, y2 = box.cpu().numpy()
#             predictions.append({
#                 'image_id': img_info['id'],
#                 'category_id': int(label.cpu().numpy()) + 1,  # Convert to 1-indexed
#                 'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
#                 'score': float(score.cpu().numpy())
#             })
    
#     # Save predictions
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, 'w') as f:
#         json.dump(predictions, f)
    
#     logger.info(f"Saved {len(predictions)} predictions to {output_path}")
#     return output_path

# def evaluate_yolo(model_path, data_yaml):
#     """Evaluate YOLO model"""
#     try:
#         from ultralytics import YOLO
        
#         logger.info("Evaluating YOLO model...")
#         model = YOLO(model_path)
#         results = model.val(data=data_yaml, split='val')
        
#         metrics = {
#             'model': 'YOLOv8',
#             'mAP_0.5:0.95': float(results.box.map),
#             'mAP_0.5': float(results.box.map50),
#             'mAP_0.75': float(results.box.map75),
#             'precision': float(results.box.mp),
#             'recall': float(results.box.mr),
#             'f1': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-16)
#         }
        
#         return metrics
#     except Exception as e:
#         logger.error(f"Error evaluating YOLO: {e}")
#         return None

# def main():
#     """Main evaluation function"""
#     results_dir = Path('/home/likhon/zksl/lmn/mri/project/results')
#     results_dir.mkdir(parents=True, exist_ok=True)
    
#     # Paths
#     gt_path = '/home/likhon/zksl/lmn/mri/project/data/coco/val/annotations.json'
#     val_images_dir = '/home/likhon/zksl/lmn/mri/project/data/coco/val/images'
    
#     all_metrics = []
    
#     # 1. Evaluate DETR
#     detr_model_path = '/home/likhon/zksl/lmn/mri/project/checkpoints/detr/final_model.pth'
#     if Path(detr_model_path).exists():
#         try:
#             logger.info("="*60)
#             logger.info("EVALUATING DETR")
#             logger.info("="*60)
            
#             # Generate predictions
#             detr_pred_path = results_dir / 'detr_predictions.json'
#             generate_detr_predictions(
#                 detr_model_path,
#                 val_images_dir,
#                 gt_path,
#                 str(detr_pred_path)
#             )
            
#             # Evaluate
#             metrics, _ = evaluate_coco_predictions(gt_path, str(detr_pred_path), 'DETR')
#             if metrics:
#                 all_metrics.append(metrics)
#         except Exception as e:
#             logger.error(f"Error evaluating DETR: {e}")
#     else:
#         logger.warning(f"DETR model not found: {detr_model_path}")
    
#     # 2. Evaluate YOLO
#     yolo_model_path = '/home/likhon/zksl/lmn/mri/project/runs/detect/train/weights/best.pt'
#     yolo_data_yaml = '/home/likhon/zksl/lmn/mri/project/configs/yolo_data.yaml'
#     if Path(yolo_model_path).exists():
#         try:
#             logger.info("="*60)
#             logger.info("EVALUATING YOLO")
#             logger.info("="*60)
            
#             metrics = evaluate_yolo(yolo_model_path, yolo_data_yaml)
#             if metrics:
#                 all_metrics.append(metrics)
#         except Exception as e:
#             logger.error(f"Error evaluating YOLO: {e}")
#     else:
#         logger.warning(f"YOLO model not found: {yolo_model_path}")
    
#     # Create results DataFrame
#     if not all_metrics:
#         logger.error("No models were successfully evaluated!")
#         return None
    
#     df = pd.DataFrame(all_metrics)
    
#     # Save results
#     csv_path = results_dir / 'evaluation_metrics.csv'
#     df.to_csv(csv_path, index=False)
#     logger.info(f"Saved metrics to {csv_path}")
    
#     # Create visualization
#     plt.figure(figsize=(12, 6))
    
#     # Plot 1: mAP comparison
#     plt.subplot(1, 2, 1)
#     ax = sns.barplot(data=df, x='model', y='mAP_0.5:0.95')
#     plt.title('Model Comparison (mAP@0.5:0.95)', fontsize=14, fontweight='bold')
#     plt.ylabel('mAP@0.5:0.95', fontsize=12)
#     plt.xlabel('Model', fontsize=12)
#     plt.ylim(0, 1.0)
    
#     # Add value labels on bars
#     for container in ax.containers:
#         ax.bar_label(container, fmt='%.3f', padding=3)
    
#     # Plot 2: Precision-Recall
#     plt.subplot(1, 2, 2)
#     if 'precision' in df.columns and 'recall' in df.columns:
#         scatter = plt.scatter(df['recall'], df['precision'], s=200, alpha=0.6, c=range(len(df)))
#         plt.xlabel('Recall', fontsize=12)
#         plt.ylabel('Precision', fontsize=12)
#         plt.title('Precision vs Recall', fontsize=14, fontweight='bold')
#         plt.grid(True, alpha=0.3)
        
#         # Add model labels
#         for i, row in df.iterrows():
#             plt.annotate(row['model'], 
#                         (row['recall'], row['precision']),
#                         xytext=(10, 5), textcoords='offset points',
#                         fontsize=10, fontweight='bold')
    
#     plt.tight_layout()
#     plot_path = results_dir / 'model_comparison.png'
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     logger.info(f"Saved plot to {plot_path}")
#     plt.close()
    
#     # Print results table
#     print("\n" + "="*80)
#     print("MODEL EVALUATION RESULTS")
#     print("="*80)
#     print(df.to_string(index=False))
#     print("\n" + "="*80)
    
#     # Print key metrics
#     print("\nKEY METRICS SUMMARY:")
#     print("-"*80)
#     for _, row in df.iterrows():
#         print(f"\n{row['model']}:")
#         print(f"  mAP@0.5:0.95: {row['mAP_0.5:0.95']:.4f}")
#         print(f"  mAP@0.5:     {row['mAP_0.5']:.4f}")
#         if 'precision' in row:
#             print(f"  Precision:   {row['precision']:.4f}")
#         if 'recall' in row:
#             print(f"  Recall:      {row['recall']:.4f}")
#         if 'f1' in row:
#             print(f"  F1-Score:    {row['f1']:.4f}")
    
#     # Determine best model
#     best_model = df.loc[df['mAP_0.5:0.95'].idxmax()]
#     print("\n" + "="*80)
#     print(f"🏆 BEST MODEL: {best_model['model']}")
#     print(f"   mAP@0.5:0.95: {best_model['mAP_0.5:0.95']:.4f}")
#     print("="*80)
    
#     return df

# if __name__ == '__main__':
#     main()


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import torch
from PIL import Image
import os
import time
from typing import Dict, List, Tuple

# COCO evaluation
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Model-specific imports
from transformers import DetrForObjectDetection, DetrImageProcessor
from ultralytics import YOLO

# Detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# Configuration
# ==========================================
class EvaluationConfig:
    # Paths
    GT_PATH = '/home/likhon/zksl/lmn/mri/project/data/coco/val/annotations.json'
    VAL_IMAGES_DIR = '/home/likhon/zksl/lmn/mri/project/data/coco/val/images'
    RESULTS_DIR = Path('/home/likhon/zksl/lmn/mri/project/results')
    
    # Model paths
    DETR_MODEL = '/home/likhon/zksl/lmn/mri/project/checkpoints/detr/final_model.pth'
    YOLO_MODEL = '/home/likhon/zksl/lmn/mri/project/runs/detect/train/weights/best.pt'
    FRCNN_OUTPUT_DIR = '/home/likhon/zksl/lmn/mri/project/output'
    
    # Classes
    CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    
    def __init__(self):
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# DETR Evaluator
# ==========================================
class DETREvaluator:
    def __init__(self, model_path, config: EvaluationConfig):
        self.config = config
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        logger.info("Loading DETR model...")
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=4,
            ignore_mismatched_sizes=True
        )
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
    def generate_predictions(self, output_path: str) -> str:
        """Generate predictions in COCO format"""
        with open(self.config.GT_PATH, 'r') as f:
            coco_data = json.load(f)
        
        predictions = []
        inference_times = []
        
        logger.info(f"Generating DETR predictions for {len(coco_data['images'])} images...")
        
        for idx, img_info in enumerate(coco_data['images']):
            img_path = Path(self.config.VAL_IMAGES_DIR) / img_info['file_name']
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference with timing
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)
            inference_times.append((time.time() - start_time) * 1000)  # Convert to ms
            
            # Post-process
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=0.05,
                target_sizes=target_sizes
            )[0]
            
            # Convert to COCO format
            for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
                x1, y1, x2, y2 = box.cpu().numpy()
                predictions.append({
                    'image_id': img_info['id'],
                    'category_id': int(label.cpu().numpy()) + 1,  # 1-indexed
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'score': float(score.cpu().numpy())
                })
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(coco_data['images'])} images")
        
        # Save predictions
        with open(output_path, 'w') as f:
            json.dump(predictions, f)
        
        avg_inference_time = np.mean(inference_times)
        logger.info(f"✓ DETR predictions saved. Avg inference: {avg_inference_time:.2f} ms")
        
        return output_path, avg_inference_time

# ==========================================
# Faster R-CNN Evaluator
# ==========================================
class FasterRCNNEvaluator:
    def __init__(self, output_dir: str, config: EvaluationConfig):
        self.config = config
        self.output_dir = output_dir
        
        # Find latest model
        output_dirs = list(Path(output_dir).glob('faster_rcnn_*'))
        if not output_dirs:
            raise FileNotFoundError(f"No Faster R-CNN models found in {output_dir}")
        
        self.model_dir = sorted(output_dirs)[-1]
        logger.info(f"Using Faster R-CNN model from: {self.model_dir}")
        
        # Setup config
        self.cfg = self._setup_config()
        self.predictor = DefaultPredictor(self.cfg)
        
    def _setup_config(self):
        """Setup Detectron2 config"""
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        )
        
        # Load trained model
        model_path = self.model_dir / "model_final.pth"
        if not model_path.exists():
            model_path = self.model_dir / "best_model.pth"
        
        cfg.MODEL.WEIGHTS = str(model_path)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        cfg.INPUT.FORMAT = "BGR"
        
        return cfg
    
    def generate_predictions(self, output_path: str) -> Tuple[str, float]:
        """Generate predictions"""
        with open(self.config.GT_PATH, 'r') as f:
            coco_data = json.load(f)
        
        predictions = []
        inference_times = []
        
        logger.info(f"Generating Faster R-CNN predictions...")
        
        for idx, img_info in enumerate(coco_data['images']):
            img_path = Path(self.config.VAL_IMAGES_DIR) / img_info['file_name']
            if not img_path.exists():
                continue
            
            # Load image
            import cv2
            image = cv2.imread(str(img_path))
            
            # Inference
            start_time = time.time()
            outputs = self.predictor(image)
            inference_times.append((time.time() - start_time) * 1000)
            
            # Extract predictions
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            
            # Convert to COCO format
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                predictions.append({
                    'image_id': img_info['id'],
                    'category_id': int(cls) + 1,  # 1-indexed
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'score': float(score)
                })
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(coco_data['images'])} images")
        
        # Save
        with open(output_path, 'w') as f:
            json.dump(predictions, f)
        
        avg_inference_time = np.mean(inference_times)
        logger.info(f"✓ Faster R-CNN predictions saved. Avg inference: {avg_inference_time:.2f} ms")
        
        return output_path, avg_inference_time

# ==========================================
# YOLO Evaluator
# ==========================================
class YOLOEvaluator:
    def __init__(self, model_path: str, config: EvaluationConfig):
        self.config = config
        self.model = YOLO(model_path)
        
    def evaluate(self) -> Dict:
        """Run YOLO validation"""
        logger.info("Evaluating YOLO model...")
        
        # Run validation
        results = self.model.val(
            data='/home/likhon/zksl/lmn/mri/project/configs/yolo_data.yaml',
            split='val',
            save_json=True,
            save_hybrid=True
        )
        
        # Extract metrics
        metrics = {
            'model': 'YOLOv8m',
            'mAP_0.5:0.95': float(results.box.map),
            'mAP_0.5': float(results.box.map50),
            'mAP_0.75': float(results.box.map75),
            'mAP_small': float(results.box.maps[0]) if len(results.box.maps) > 0 else 0.0,
            'mAP_medium': float(results.box.maps[1]) if len(results.box.maps) > 1 else 0.0,
            'mAP_large': float(results.box.maps[2]) if len(results.box.maps) > 2 else 0.0,
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'inference_time': float(results.speed['inference'])
        }
        
        # Calculate F1
        p, r = metrics['precision'], metrics['recall']
        metrics['f1'] = 2 * (p * r) / (p + r + 1e-16)
        
        logger.info(f"✓ YOLO evaluation complete. mAP: {metrics['mAP_0.5:0.95']:.4f}")
        
        return metrics

# ==========================================
# COCO Metrics Calculator
# ==========================================
def evaluate_coco_predictions(gt_path: str, pred_path: str, model_name: str) -> Dict:
    """Evaluate using COCO metrics"""
    try:
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(pred_path)
        
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = {
            'model': model_name,
            'mAP_0.5:0.95': float(coco_eval.stats[0]),
            'mAP_0.5': float(coco_eval.stats[1]),
            'mAP_0.75': float(coco_eval.stats[2]),
            'mAP_small': float(coco_eval.stats[3]),
            'mAP_medium': float(coco_eval.stats[4]),
            'mAP_large': float(coco_eval.stats[5]),
            'AR_max1': float(coco_eval.stats[6]),
            'AR_max10': float(coco_eval.stats[7]),
            'AR_max100': float(coco_eval.stats[8]),
        }
        
        # Calculate precision, recall, F1
        if len(coco_eval.eval['precision']) > 0:
            precision = np.mean(coco_eval.eval['precision'][coco_eval.eval['precision'] > -1])
            recall = float(coco_eval.stats[8])
            metrics['precision'] = float(precision) if not np.isnan(precision) else 0.0
            metrics['recall'] = recall
            metrics['f1'] = 2 * (metrics['precision'] * recall) / (metrics['precision'] + recall + 1e-16)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        return None

# ==========================================
# Visualization
# ==========================================
def create_comprehensive_plots(df: pd.DataFrame, save_dir: Path):
    """Create comprehensive comparison plots"""
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 12))
    
    # 1. mAP Comparison
    plt.subplot(2, 3, 1)
    ax = sns.barplot(data=df, x='model', y='mAP_0.5:0.95', palette='viridis')
    plt.title('mAP@0.5:0.95 Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('mAP@0.5:0.95', fontsize=12)
    plt.xlabel('')
    plt.ylim(0, 1.0)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    # 2. Speed vs Accuracy
    plt.subplot(2, 3, 2)
    scatter = plt.scatter(
        df['inference_time'], 
        df['mAP_0.5:0.95'],
        s=300,
        alpha=0.6,
        c=range(len(df)),
        cmap='viridis'
    )
    plt.xlabel('Inference Time (ms)', fontsize=12)
    plt.ylabel('mAP@0.5:0.95', fontsize=12)
    plt.title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for i, row in df.iterrows():
        plt.annotate(
            row['model'],
            (row['inference_time'], row['mAP_0.5:0.95']),
            xytext=(10, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold'
        )
    
    # 3. Precision-Recall
    plt.subplot(2, 3, 3)
    scatter = plt.scatter(
        df['recall'],
        df['precision'],
        s=300,
        alpha=0.6,
        c=range(len(df)),
        cmap='viridis'
    )
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    
    for i, row in df.iterrows():
        plt.annotate(
            row['model'],
            (row['recall'], row['precision']),
            xytext=(10, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold'
        )
    
    # 4. Multi-metric comparison
    plt.subplot(2, 3, 4)
    metrics_to_plot = ['mAP_0.5:0.95', 'mAP_0.5', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=row['model'], alpha=0.8)
    
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Multi-Metric Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x + width, metrics_to_plot, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 5. Object size performance
    plt.subplot(2, 3, 5)
    size_metrics = ['mAP_small', 'mAP_medium', 'mAP_large']
    x = np.arange(len(size_metrics))
    width = 0.25
    
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in size_metrics]
        plt.bar(x + i*width, values, width, label=row['model'], alpha=0.8)
    
    plt.xlabel('Object Size', fontsize=12)
    plt.ylabel('mAP', fontsize=12)
    plt.title('Performance by Object Size', fontsize=14, fontweight='bold')
    plt.xticks(x + width, ['Small', 'Medium', 'Large'])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 6. F1-Score comparison
    plt.subplot(2, 3, 6)
    ax = sns.barplot(data=df, x='model', y='f1', palette='mako')
    plt.title('F1-Score Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12)
    plt.xlabel('')
    plt.ylim(0, 1.0)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.tight_layout()
    plot_path = save_dir / 'comprehensive_evaluation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plots to {plot_path}")
    plt.close()

# ==========================================
# Main Evaluation
# ==========================================
def main():
    """Run comprehensive evaluation"""
    config = EvaluationConfig()
    all_metrics = []
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE MODEL EVALUATION")
    logger.info("="*80)
    
    # 1. Evaluate DETR
    if Path(config.DETR_MODEL).exists():
        try:
            logger.info("\n" + "="*80)
            logger.info("EVALUATING DETR")
            logger.info("="*80)
            
            evaluator = DETREvaluator(config.DETR_MODEL, config)
            pred_path, inf_time = evaluator.generate_predictions(
                str(config.RESULTS_DIR / 'detr_predictions.json')
            )
            
            metrics = evaluate_coco_predictions(config.GT_PATH, pred_path, 'DETR')
            if metrics:
                metrics['inference_time'] = inf_time
                all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"DETR evaluation failed: {e}")
    
    # 2. Evaluate Faster R-CNN
    if Path(config.FRCNN_OUTPUT_DIR).exists():
        try:
            logger.info("\n" + "="*80)
            logger.info("EVALUATING FASTER R-CNN")
            logger.info("="*80)
            
            evaluator = FasterRCNNEvaluator(config.FRCNN_OUTPUT_DIR, config)
            pred_path, inf_time = evaluator.generate_predictions(
                str(config.RESULTS_DIR / 'frcnn_predictions.json')
            )
            
            metrics = evaluate_coco_predictions(config.GT_PATH, pred_path, 'Faster R-CNN')
            if metrics:
                metrics['inference_time'] = inf_time
                all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Faster R-CNN evaluation failed: {e}")
    
    # 3. Evaluate YOLO
    if Path(config.YOLO_MODEL).exists():
        try:
            logger.info("\n" + "="*80)
            logger.info("EVALUATING YOLO")
            logger.info("="*80)
            
            evaluator = YOLOEvaluator(config.YOLO_MODEL, config)
            metrics = evaluator.evaluate()
            if metrics:
                all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"YOLO evaluation failed: {e}")
    
    # Create results DataFrame
    if not all_metrics:
        logger.error("No models were successfully evaluated!")
        return None
    
    df = pd.DataFrame(all_metrics)
    
    # Save results
    csv_path = config.RESULTS_DIR / 'final_evaluation_metrics.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Saved metrics to {csv_path}")
    
    # Create visualizations
    create_comprehensive_plots(df, config.RESULTS_DIR)
    
    # Print results
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("\n" + "="*80)
    
    # Print per-model summary
    print("\nDETAILED METRICS:")
    print("-"*80)
    for _, row in df.iterrows():
        print(f"\n{row['model']}:")
        print(f"  mAP@0.5:0.95:    {row['mAP_0.5:0.95']:.4f}")
        print(f"  mAP@0.5:         {row['mAP_0.5']:.4f}")
        print(f"  Precision:       {row['precision']:.4f}")
        print(f"  Recall:          {row['recall']:.4f}")
        print(f"  F1-Score:        {row['f1']:.4f}")
        print(f"  Inference Time:  {row['inference_time']:.2f} ms")
    
    # Determine winner
    print("\n" + "="*80)
    print("BEST MODELS BY CATEGORY")
    print("="*80)
    print(f"🏆 Best Accuracy:  {df.loc[df['mAP_0.5:0.95'].idxmax(), 'model']} "
          f"(mAP: {df['mAP_0.5:0.95'].max():.4f})")
    print(f"⚡ Fastest:        {df.loc[df['inference_time'].idxmin(), 'model']} "
          f"({df['inference_time'].min():.2f} ms)")
    print(f"⚖️  Best Balance:   {df.loc[df['f1'].idxmax(), 'model']} "
          f"(F1: {df['f1'].max():.4f})")
    print("="*80)
    
    return df

if __name__ == '__main__':
    main()