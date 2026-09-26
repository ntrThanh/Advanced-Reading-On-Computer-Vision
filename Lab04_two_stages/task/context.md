# TỔNG HỢP CONTEXT & KẾ HOẠCH BÀI THỰC HÀNH LAB 04
**Môn học**: Advanced Reading in Computer Vision (MAT3563)  
**Chủ đề**: Lab 04 - Two-Stages Object Detection Models (Faster R-CNN)  
**Sinh viên**: Nguyễn Trọng Thành (MSSV: 23001934)  

---

## 1. Tổng quan mục tiêu bài thực hành Lab 04
Theo tài liệu hướng dẫn `Lap4_two_stages_object_detection (1).pdf`, mục tiêu chính gồm:
1. Nắm vững và thực hành từng mắt xích kiến trúc của Faster R-CNN (FPN, RPN/Anchors, RoI Pool vs RoI Align, NMS, RoI Heads).
2. Chạy suy luận (Inference) với mô hình đã huấn luyện sẵn (Pretrained Weights trên COCO).
3. Hiểu hiện tượng Out-of-Distribution (OOD) khi dự đoán các đối tượng chưa có trong tập dữ liệu ban đầu.
4. Tinh chỉnh / Huấn luyện lại (Fine-tune) Faster R-CNN trên tập dữ liệu mới (chuẩn Pascal VOC hoặc COCO).
5. Thực hiện các bài kiểm thử, mổ xẻ từng module và nghiên cứu triệt tiêu (Ablation Studies).

---

## 2. Bảng theo dõi tiến độ công việc (Checklist)

| Mục | Nội dung chi tiết | Trạng thái | File thực hiện |
| :--- | :--- | :---: | :--- |
| **Mục 3.1** | FPN (Feature Pyramid Network): Trích xuất feature đa tỷ lệ, heatmap | Đã hoàn thành | `submit/23001934_NguyenTrongThanh_Lab04_v1.ipynb` |
| **Mục 3.2** | Anchors & RPN: Sinh 9 anchors/cell (22,500 boxes), trượt và lọc NMS sơ bộ | Đã hoàn thành | `submit/23001934_NguyenTrongThanh_Lab04_v1.ipynb` |
| **Mục 3.3** | RoI Pool vs RoI Align: Chuẩn hóa vùng đặc trưng (7, 7), so sánh bilinear | Đã hoàn thành | `submit/23001934_NguyenTrongThanh_Lab04_v1.ipynb` |
| **Mục 3.4** | NMS: Thử nghiệm lọc chồng lấn bounding box với các ngưỡng IoU khác nhau | Đã hoàn thành | `submit/23001934_NguyenTrongThanh_Lab04_v1.ipynb` |
| **Mục 4.1** | Chạy suy luận Faster R-CNN v2 Pretrained (PyTorch), trực quan hóa ảnh | Đã hoàn thành | `submit/23001934_NguyenTrongThanh_Lab04_v2.ipynb` |
| **Mục 4.2** | Chạy suy luận nhanh Faster R-CNN bằng TensorFlow Hub (`resnet50_v1`) | Chưa làm | Đề xuất cho các bước sau |
| **Mục 5.1** | Fine-tune Faster R-CNN (Giai đoạn 1: Huấn luyện Box Predictor trên tập Aquarium) | Đã hoàn thành | `submit/23001934_NguyenTrongThanh_Lab04_v3.ipynb` |
| **Mục 5.2** | Fine-tune chuẩn với TensorFlow Object Detection API (TFRecord + pipeline) | Chưa làm | Tùy chọn nâng cao |
| **Mục 6** | Mổ xẻ từng bước trong Faster R-CNN trên 01 ảnh thật (FPN -> RPN -> RoI Align -> Heads) | Chưa làm | Đề xuất thực hiện trong `v4` |
| **Mục 7** | Bài tập nâng cao & Kiểm thử (Ablation FPN, RoI Align vs Pool, IoU sweep) | Chưa làm | Đề xuất thực hiện sau fine-tune |
| **Bài tập thực tế** | Thu thập 2000 ảnh, gán nhãn tự động, gộp nhãn, huấn luyện và test trên 1000 ảnh mới | Chưa làm | Dự án cuối lab |

---

## 3. Chi tiết các file nộp bài đã thực hiện

### 3.1 File `v1`: `submit/23001934_NguyenTrongThanh_Lab04_v1.ipynb`
- **Nội dung**: Hoàn thành toàn bộ **Mục 3 (MINI-LABS)**.
- **Các thành phần**:
  - `3.1 FPN`: Xây dựng module SimpleFPN tối giản, trích xuất đặc trưng C3, C4, C5 thành P3, P4, P5 và hiển thị heatmap đối chiếu.
  - `3.2 Anchors & RPN`: Sinh lưới anchor đa tỷ lệ và aspect ratio, biểu diễn mật độ anchor trượt trên ảnh gốc.
  - `3.3 RoI Pool vs RoI Align`: Trích xuất patch kích thước 7x7 từ feature map, so sánh cơ chế làm tròn tọa độ và nội suy song tuyến.
  - `3.4 NMS`: Lọc bounding box với các ngưỡng IoU khác nhau (0.3, 0.5, 0.7), trực quan hóa trước và sau lọc.
- **Phong cách nhận xét**: Viết ngắn gọn, súc tích, giải thích cơ chế kỹ thuật.

### 3.2 File `v2`: `submit/23001934_NguyenTrongThanh_Lab04_v2.ipynb`
- **Nội dung**: Hoàn thành **Mục 4.1 (PyTorch torchvision Pretrained Inference)** độc lập (chỉ chứa phần làm tiếp theo yêu cầu).
- **Quy chuẩn mã nguồn**:
  - Tuyệt đối không chứa comment (`#`) hoặc chú thích trong code cell.
  - Định dạng chuẩn PEP 8, biến đặt tên trực quan (`image_path`, `filtered_boxes`, `categories`...).
  - Thiết kế `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` tương thích cả GPU và CPU.
  - Tự động quét tìm thư mục ảnh qua nhiều đường dẫn khả dĩ (`possible_dirs`) để chạy được trên local hoặc Google Colab.
- **Kết quả thực nghiệm trên 4 ảnh thực tế**:
  - `image copy 2.png` (Sư tử - Lion): Dự đoán `dog` (97.56%).
  - `image copy 3.png` (Báo - Leopard): Dự đoán `cat` (99.49%).
  - `image copy.png` (Hổ - Tiger): Dự đoán `zebra` (99.93%).
  - `image.png` (Vẹt - Macaw): Dự đoán `bird` (99.87%).
- **Nhận xét**: Viết dạng các đoạn văn ngắn, không bôi đen, giải thích tự nhiên hiện tượng Out-of-Distribution do tập nhãn COCO không có sư tử, báo, hổ.

### 3.3 File `v3`: `submit/23001934_NguyenTrongThanh_Lab04_v3.ipynb`
- **Nội dung**: Hoàn thành **Mục 5.1 (Fine-tune Faster R-CNN Giai đoạn 1 - Box Predictor)**.
- **Các thành phần**:
  - `5.1 Chuẩn bị dữ liệu`: Xây dựng class `AquariumDataset` đọc dữ liệu COCO format từ tập Aquarium Combined (448 ảnh train, 127 ảnh valid), trích xuất bounding box và mapping 7 loài sinh vật biển (`fish`, `jellyfish`, `penguin`, `puffin`, `shark`, `starfish`, `stingray`).
  - `5.2 Cấu hình mô hình`: Nạp Faster R-CNN v2 pretrained, thay thế `roi_heads.box_predictor` bằng `FastRCNNPredictor` (8 lớp tính cả background). Đóng băng toàn bộ mạng và chỉ mở gradient cho `box_predictor` (giảm số tham số cần train xuống còn 41,000 tham số, tương đương 0.1% tổng tham số).
  - `5.3 Huấn luyện siêu tốc & Đánh giá chuẩn YOLO`: Tối ưu với Effective Batch Size = 32 (thông qua Gradient Accumulation 8 bước và micro-batch 4 kết hợp AMP FP16 kích hoạt Tensor Cores trên GPU RTX 3060), tích hợp bảng đánh giá chuẩn YOLO (Precision, Recall, mAP@50, mAP@50:95 cho toàn bộ và từng lớp đối tượng), cơ chế Early Stopping theo dõi đỉnh mAP50, và tự động ghi log vào `outputs/stage1_training_log.json`. Trọng số tốt nhất được lưu tại `outputs/fasterrcnn_aquarium_stage1.pt`.
  - `5.4 Trực quan hóa suy luận`: Kiểm thử mô hình sau khi tinh chỉnh trên các mẫu ảnh từ tập validation, vẽ bounding box và hiển thị điểm tin cậy rõ nét theo từng loài.
- **Quy chuẩn mã nguồn**: Tuyệt đối không chứa comment trong code cell, tương thích đa môi trường (Colab/Server/Local) qua `possible_data_dirs`, nhận xét súc tích phân tích cơ chế đóng băng và khả năng thích ứng miền nhãn mới.

---

## 4. Phân tích hiện tượng nhận diện sai lệch Out-of-Distribution (OOD)
Hiện tượng mô hình nhận diện sai tên loài thú trong thực nghiệm Mục 4.1 là một bài học lý thuyết cốt lõi trong Computer Vision:

1. **Giới hạn không gian nhãn của MS COCO**:
   - Mô hình Faster R-CNN pretrained chỉ được học 80 lớp đối tượng thông dụng.
   - Tập COCO hoàn toàn **không có nhãn `lion` (sư tử), `leopard` (báo) hay `tiger` (hổ)**.
   - Nhóm động vật có vú ăn thịt trong COCO chỉ có đại diện là `cat` và `dog`.
2. **Cơ chế suy diễn cưỡng bức của tầng phân loại (Softmax)**:
   - Báo thuộc họ Mèo (Felidae) chia sẻ nhiều đặc trưng hình thái với mèo nhà $\rightarrow$ gán nhãn `cat` với độ tin cậy 99.49%.
   - Hổ có họa tiết sọc vằn tương phản cao kích hoạt đặc trưng sọc $\rightarrow$ gán nhãn `zebra` (ngựa vằn) 99.93%.
   - Sư tử có bờm và đầu mõm dài được liên tưởng gần nhất với các giống chó lớn $\rightarrow$ gán nhãn `dog` 97.56%.
   - Vẹt trùng với nhãn siêu lớp `bird` có sẵn $\rightarrow$ gán nhãn đúng `bird` 99.87%.
3. **Kết luận**:
   - Về mặt định vị (Localization): Mô hình khoanh Bounding Box cực kỳ chính xác.
   - Về mặt phân loại (Classification): Bị giới hạn bởi không gian nhãn ban đầu. Đây là lý do bắt buộc phải chuyển sang **Mục 5: Fine-tune**.

---

## 5. Đặc tả chi tiết các công việc cần làm tiếp theo

### 5.1 Mục 5.1: Fine-tune Faster R-CNN với PyTorch (Ưu tiên làm `v3`)
- **Tập dữ liệu**: Dùng Pascal VOC dataset (hoặc bộ dữ liệu nhỏ có nhãn XML). Cấu trúc gồm:
  ```
  dataset/
    Annotations/*.xml
    JPEGImages/*.jpg
    ImageSets/Main/train.txt, val.txt
  ```
- **Xây dựng Dataset Class**: Viết `VOCDataset(torch.utils.data.Dataset)` parse file XML lấy toạ độ `xmin, ymin, xmax, ymax` và nhãn đối tượng.
- **Tùy chỉnh Head mô hình**:
  ```python
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  ```
- **Training & Validation Loop**: Sử dụng optimizer SGD (momentum=0.9, weight_decay=0.0005), scheduler StepLR, in train_loss và val_loss theo từng epoch.
- **Lưu và kiểm thử**: Lưu `frcnn_voc.pt` và chạy suy luận vẽ bounding box với nhãn mới đã tinh chỉnh.
- **Tham khảo mẫu**: Đã có file [frcnn_pytorch_finetune_voc.ipynb](file:///media/trong-thanh/Data/University/Fourth%20Year/Computer%20Vision%20%28Advance%29/Advanced-Reading-On-Computer-Vision/Lab04_two_stages/frcnn_pytorch_finetune_voc.ipynb).

### 5.2 Mục 4.2: Suy luận bằng TensorFlow Hub (Tùy chọn)
- Sử dụng `tensorflow_hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1")`.
- Chạy suy luận nhanh trên 1-2 ảnh và trực quan hóa bounding box.

### 5.3 Mục 6: Mổ xẻ từng module Faster R-CNN trên 01 ảnh thật
- Thực hiện trích xuất tuần tự qua 5 bước trên cùng một ảnh:
  1. Trích xuất đặc trưng đa tỷ lệ từ Backbone ResNet50 + FPN ($P_3, P_4, P_5, P_6$).
  2. RPN: Đặt anchors trên từng level, tính điểm objectness và lọc ra top-k proposals bằng NMS.
  3. RoI Align: Cắt các patch $7 \times 7$ tương ứng với vùng proposal.
  4. RoI Heads: Đi qua Fast R-CNN head để phân loại và tinh chỉnh box.
  5. Second-stage NMS: Lọc chồng lấn lần 2 theo từng lớp để xuất kết quả cuối.

### 5.4 Mục 7: Bài tập nâng cao & Kiểm thử (Ablations)
- **Ablation FPN**: Tắt FPN (chỉ dùng tầng $C_5$), đánh giá khả năng phát hiện vật thể nhỏ.
- **RoI Pool vs RoI Align**: Thay `aligned=False` và so sánh độ suy giảm độ chính xác tọa độ.
- **NMS IoU Sweep**: Đo lường sự thay đổi của số lượng box và Precision/Recall khi quét ngưỡng IoU = {0.3, 0.5, 0.7}.
- **Augmentation**: Thêm RandomFlip, Scale, ColorJitter trong torchvision transforms và so sánh loss curve.

### 5.5 Bài tập thực tế lớn (Capstone Project)
- Thu thập khoảng 2,000 ảnh về một lĩnh vực cụ thể.
- Chạy pseudo-labeling bằng Faster R-CNN pretrained.
- Gộp nhãn (ví dụ: các dòng xe $\rightarrow$ vehicles, các loài thú $\rightarrow$ wild_animals...), xuất định dạng Pascal VOC XML.
- Huấn luyện lại mô hình trên tập 2,000 ảnh này.
- Thu thập thêm 1,000 ảnh kiểm thử độc lập để đánh giá độ chính xác thực tế.

---

## 6. Quy chuẩn định dạng mã nguồn & nộp bài
1. **Mã nguồn**:
   - Viết sạch, chuẩn PEP 8, thụt lề 4 space.
   - Không comment (`#`), không docstring trong các ô code cell.
   - Hỗ trợ cả GPU và CPU thông qua cơ chế tự động phát hiện `device`.
   - Tìm kiếm thư mục ảnh/dữ liệu bằng mảng `possible_dirs` để chạy linh hoạt trên mọi máy.
2. **Nhận xét (Markdown)**:
   - Viết ngắn gọn, rõ ràng theo từng đoạn văn.
   - Tránh bôi đen rườm rà, tập trung phân tích bản chất thuật toán và ý nghĩa thực nghiệm.
