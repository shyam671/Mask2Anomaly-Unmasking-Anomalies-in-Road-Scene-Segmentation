#Anomaly Train
CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1  --config-file configs/cityscapes/semantic-segmentation/anomaly_train.yaml
#Anomaly Fine-Tune
CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1  --config-file configs/cityscapes/semantic-segmentation/anomaly_ft.yaml


## Anomaly Inference
#RoadAnomaly21
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input '/home/shyam/Mask2Former/unk-eval/RoadAnomaly21/images/*.png' --config-file '/home/shyam/Mask2Anomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' 
#RoadObsticle21
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input '/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp' --config-file '/home/shyam/Mask2Anomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' 
#FS L&F val
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input '/home/shyam/Mask2Former/unk-eval/FS_LostFound_full/images/*.png' --config-file '/home/shyam/Mask2Anomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' 
# FS-Static
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input '/home/shyam/Mask2Former/unk-eval/fs_static/images/*.jpg' --config-file '/home/shyam/Mask2Anomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' 
#RoadAnomaly
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input '/home/shyam/Mask2Former/unk-eval/RoadAnomaly/images/*.jpg' --config-file '/home/shyam/Mask2Anomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' 

#StreetHazard Training
CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 --config-file '/home/shyam/Mask2Anomaly/configs/streethazard/streethazard_training.yaml'
#StreetHazard Fine-Tuning
CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 --config-file '/home/shyam/Mask2Anomaly/configs/streethazard/streethazard_ft.yaml' 
#StreetHazard Inference
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference_streethazard.py --input '/home/shyam/Mask2Anomaly/datasets/streethazard/test/images/*.png' --config-file '/home/shyam/Mask2Anomaly/configs/streethazard/streethazard_inference.yaml' --weights '/home/shyam/Mask2Former/output/streethazard-swin-base-coco-supervised/best_model.pth'
