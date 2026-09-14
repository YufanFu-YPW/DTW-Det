import json
import time

import torch
import torch.amp

from src.data import CocoEvaluator
from src.data.dataset_eval.coco_eval_visdrone import VisdroneCocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)


@torch.no_grad()
def evaluate_save(coco_evaluator: CocoEvaluator, model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device,
             output_dir):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = postprocessors.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator = VisdroneCocoEvaluator(base_ds, iou_types)

    total_time = 0.0
    total_images = 0
    warmup = 10  # 前10个batch不计时（GPU预热）

    # 新增：收集所有检测结果 - 两种格式
    all_detections_grouped = []  # 按图像分组格式
    all_detections_coco = []  # COCO官方格式（每个检测一条）

    for idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        start_time = time.time()

        outputs = model(samples)

        torch.cuda.synchronize()
        end_time = time.time()

        if idx >= warmup:
            batch_time = end_time - start_time
            total_time += batch_time
            total_images += samples.shape[0]

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        # 收集当前batch的检测结果 - 两种格式
        for target, result in zip(targets, results):
            image_id = target['image_id'].item()

            # 将检测结果转换为numpy/tolist
            boxes = result['boxes'].cpu().numpy()  # [x1,y1,x2,y2]格式
            scores = result['scores'].cpu().numpy()
            labels = result['labels'].cpu().numpy()

            # 格式1：按图像分组（便于查看）
            grouped_detection = {
                'image_id': image_id,
                'boxes': boxes.tolist(),
                'scores': scores.tolist(),
                'labels': labels.tolist()
            }
            # all_detections_grouped.append(grouped_detection)

            # 格式2：COCO官方格式（每个检测一条，用于评估）
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                coco_detection = {
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # [x,y,width,height]
                    'score': float(score)
                }
                # all_detections_coco.append(coco_detection)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # ===== 保存检测结果 =====
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存按图像分组的检测结果（便于可视化）
        if all_detections_grouped:
            grouped_file = output_dir / "detection_results_grouped.json"
            with open(grouped_file, "w") as f:
                json.dump(all_detections_grouped, f, indent=2)
            print(f"Saved grouped detection results to {grouped_file}")

        # 保存COCO格式的检测结果（可直接用于评估）
        if all_detections_coco:
            coco_file = output_dir / "coco_detection_results.json"
            with open(coco_file, "w") as f:
                json.dump(all_detections_coco, f, indent=2)
            print(f"Saved COCO-format results to {coco_file}")

        # 保存评估指标（可选）
        if 'bbox' in iou_types and hasattr(coco_evaluator, 'coco_eval'):
            stats = coco_evaluator.coco_eval['bbox'].stats
            # metrics_dict = {
            #     'AP': float(stats[0]),
            #     'AP_50': float(stats[1]),
            #     'AP_75': float(stats[2]),
            #     'AP_small': float(stats[3]),
            #     'AP_medium': float(stats[4]),
            #     'AP_large': float(stats[5]),
            #     'AR_1': float(stats[6]),
            #     'AR_10': float(stats[7]),
            #     'AR_100': float(stats[8]),
            #     'AR_small': float(stats[9]),
            #     'AR_medium': float(stats[10]),
            #     'AR_large': float(stats[11])
            # }
            # print(stats)
            metrics_dict = {
                'AP': float(stats[0]),
                'AP_25': float(stats[1]),
                'AP_50': float(stats[2]),
                'AP_75': float(stats[3]),
                'AP_verytiny': float(stats[4]),
                'AP_tiny': float(stats[5]),
                'AP_small': float(stats[6]),
                'AP_medium': float(stats[7]),
                'AR_1': float(stats[8]),
                'AR_100': float(stats[9]),
                'AR_1500': float(stats[10]),
                'AR_verytiny': float(stats[11]),
                # 'AR_tiny': float(stats[12]),
                # 'AR_small': float(stats[13]),
                # 'AR_medium': float(stats[14]),
                # 'LRP': float(stats[15]),
                # 'LRP_Loc': float(stats[16]),
                # 'LRP_FP': float(stats[17]),
                # 'LRP_FN': float(stats[18])
            }
            metrics_file = output_dir / "coco_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics_dict, f, indent=2)
            print(f"Saved metrics to {metrics_file}")

        # 保存eval.pth（原始评估结果）
        if 'bbox' in iou_types and hasattr(coco_evaluator, 'coco_eval'):
            if hasattr(coco_evaluator.coco_eval['bbox'], 'eval'):
                torch.save(coco_evaluator.coco_eval['bbox'].eval, output_dir / "eval.pth")
                print(f"Saved eval.pth to {output_dir / 'eval.pth'}")

    # ===== 输出延迟 =====
    if total_images > 0:
        avg_latency = total_time / total_images * 1000
        fps = 1000.0 / avg_latency
        print(f"\n[Latency] Avg latency: {avg_latency:.2f} ms / image")
        print(f"[Latency] FPS: {fps:.2f}")

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator