
# Faster R-CNN Mini-Labs & PyTorch Fine-Tuning (VOC)

This package contains:
- **mini_labs_frcnn_modules.ipynb** — Four mini-labs (FPN, Anchors/RPN, RoI Align, NMS) with simple visual demos.
- **frcnn_pytorch_finetune_voc.ipynb** — End-to-end PyTorch fine-tuning on VOC-style data (train/val, save, and infer).
- **(Optional) TensorFlow OD API** — Quick commands below if you want to fine-tune Faster R-CNN in TensorFlow.

## Dataset Layout (VOC-style)
```
dataset/
  JPEGImages/
    xxxx.jpg
  Annotations/
    xxxx.xml
  ImageSets/
    Main/
      train.txt
      val.txt
```

## Quick Start
1. Open **mini_labs_frcnn_modules.ipynb** and change `IMG_PATH` to one of your images.
2. Open **frcnn_pytorch_finetune_voc.ipynb**, set `DATA_ROOT` and `CLASSES`, then run all cells.

## TensorFlow OD API (Optional)
Install (Linux/macOS best; on Windows, prebuilt protoc is recommended):
```bash
pip install tensorflow==2.13 tensorflow_io
pip install git+https://github.com/tensorflow/models.git#egg=object-detection&subdirectory=research/object_detection
```

Prepare data (convert VOC/COCO -> TFRecord), create `label_map.pbtxt`, and a `pipeline.config` for Faster R-CNN (+FPN).

Train:
```bash
python models/research/object_detection/model_main_tf2.py   --model_dir=outputs/tf_frcnn_run   --pipeline_config_path=tf_frcnn/pipeline.config   --num_train_steps=20000   --sample_1_of_n_eval_examples=1
```

Export SavedModel for inference:
```bash
python models/research/object_detection/exporter_main_v2.py   --input_type=image_tensor   --pipeline_config_path=tf_frcnn/pipeline.config   --trained_checkpoint_dir=outputs/tf_frcnn_run   --output_directory=outputs/tf_frcnn_exported
```

Happy training!
