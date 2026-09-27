import json

nb_path = "/workspace/Advanced-Reading-On-Computer-Vision/Lab04_two_stages/submit/23001934_NguyenTrongThanh_Lab04_v6.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 0: Header
nb['cells'][0]['source'] = """# CAPSTONE PROJECT: AUTOMATED PSEUDO-LABELING AND FASTER R-CNN FINE-TUNING PIPELINE
End-to-end practical project workflow on urban traffic object detection:
1. Collect and partition 2,000 training images (pseudo-labeling set) and 1,000 independent test images.
2. Generate automated bounding box predictions (pseudo-labeling) using pretrained Faster R-CNN ResNet-50 FPN v2.
3. Regroup granular object classes (car, bus, truck, motorcycle, bicycle -> vehicle; person -> pedestrian) and export standardized Pascal VOC XML annotations.
4. Statistically analyze and visualize pseudo-labeled ground truth on the training distribution.
5. Construct custom PyTorch Dataset and fine-tune Faster R-CNN Box Predictor with Early Stopping.
6. Evaluate generalization performance and visualize qualitative detections on 1,000 novel test images.
7. Synthesize key findings and project conclusions."""

# Cell 1: Section 1
nb['cells'][1]['source'] = """## 1. Environment Setup and Configuration"""

# Cell 4: Section 2
nb['cells'][4]['source'] = """## 2. Dataset Collection and Splitting (2,000 Train, 1,000 Test Images)"""

# Cell 6: Section 3
nb['cells'][6]['source'] = """## 3. Automated Pseudo-Labeling and Pascal VOC XML Annotation Export"""

# Cell 8: Section 4
nb['cells'][8]['source'] = """## 4. Statistics and Visualization of the Pseudo-Labeled Dataset (2,000 Images)"""

# Cell 11: Observation 1
nb['cells'][11]['source'] = """**Observation:**
- Pretrained Faster R-CNN achieves comprehensive object coverage on urban road scenes, reliably detecting vehicles and pedestrians under diverse lighting conditions and viewpoints.
- Regrouping granular sub-classes (car, bus, truck, motorcycle, bicycle into 'vehicle') unifies sparse individual classes into a single high-density super-class. This eliminates class imbalance between rare vehicle types (e.g. buses, trucks) and common ones (cars), providing a well-balanced learning objective."""

# Cell 12: Section 5
nb['cells'][12]['source'] = """## 5. Dataset Definition and Faster R-CNN Fine-Tuning with Early Stopping"""

# Cell 17: Observation 2
nb['cells'][17]['source'] = """**Observation:**
- Training was configured for up to 25 epochs equipped with an Early Stopping mechanism (patience = 4, min_delta = 0.003) evaluated on an 80/20 train/validation split.
- Both Training and Validation losses drop steeply across the first few epochs as the Box Predictor head quickly adapts its 1024-d representations to the 2 target classes ('vehicle', 'pedestrian').
- As validation loss stabilized near its minimum (epoch 11), the Early Stopping mechanism detected 4 consecutive epochs without significant reduction and terminated training at epoch 15, successfully preserving the optimal weights while preventing overfitting to pseudo-label noise."""

# Cell 18: Section 6
nb['cells'][18]['source'] = """## 6. Independent Evaluation on 1,000 Novel Test Images"""

# Cell 21: Observation 3
nb['cells'][21]['source'] = """**Observation:**
- The fine-tuned detector demonstrates strong generalization capability on the 1,000 unseen test images, successfully recognizing distant vehicles, dense clusters, and partially occluded pedestrians.
- The streamlined two-class taxonomy substantially reduces inter-class ambiguity (such as between light trucks and SUVs) while maintaining crisp bounding box localization inherited from the pre-trained FPN backbone."""

# Cell 22: Section 7
nb['cells'][22]['source'] = """## 7. Summary and Conclusions
1. **Data Preparation**: Curated 2,000 training images and 1,000 independent test images centered on urban traffic.
2. **Pseudo-Labeling**: Generated automatic bounding box annotations using Faster R-CNN ResNet-50 FPN v2 at score threshold >= 0.5.
3. **Class Regrouping**: Unified granular classes into two super-classes (`vehicle` and `pedestrian`) and saved standardized Pascal VOC XML files.
4. **Fine-Tuning with Early Stopping**: Trained the Box Predictor with AMP FP16 and an 80/20 train/val split. Early stopping safely concluded training at epoch 15, restoring the best model from epoch 11.
5. **Independent Testing**: Evaluated on 1,000 novel test images at 14.2 FPS, detecting 2,023 vehicles and 4,824 pedestrians with a high mean confidence of 0.8811."""

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully translated all markdown cells and observations in v6 notebook to English!")
