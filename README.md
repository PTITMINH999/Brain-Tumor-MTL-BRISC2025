# Brain Tumor Classification and Segmentation

## Giới thiệu dự án

Brain Tumor Classification and Segmentation là hệ thống phân tích ảnh MRI não dựa trên Deep Learning nhằm hỗ trợ phát hiện, phân loại và phân đoạn khối u não một cách tự động. Dự án được xây dựng theo hướng end-to-end, bao gồm toàn bộ quy trình từ tiền xử lý dữ liệu, huấn luyện mô hình, đánh giá hiệu năng, suy luận (inference) và triển khai ứng dụng web.

Hệ thống giải quyết đồng thời hai bài toán quan trọng trong xử lý ảnh y tế:

- **Brain Tumor Classification**: Phân loại ảnh MRI thành các nhóm khối u não.
- **Brain Tumor Segmentation**: Xác định và phân đoạn vùng khối u trên ảnh MRI.

### Dataset

Dự án sử dụng bộ dữ liệu **BRISC2025 (Brain Tumor MRI Dataset for Segmentation and Classification)** gồm:

- 6.000 ảnh MRI T1-weighted.
- 5.000 ảnh huấn luyện và 1.000 ảnh kiểm thử.
- 4 lớp dữ liệu:
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - No Tumor

- Mask phân đoạn được gán nhãn và kiểm định bởi các chuyên gia y tế.
- Bao gồm ba mặt phẳng MRI:
  - Axial
  - Coronal
  - Sagittal

Dữ liệu thô được lưu trong:

```text
data/raw/brisc2025/
```

và được tiền xử lý thành dữ liệu huấn luyện trong:

```text
data/processed/
```

### Kiến trúc mô hình

Dự án triển khai nhiều kiến trúc Deep Learning để phục vụ cả phân loại và phân đoạn:

- U-Net Classifier: Mô hình phân loại dựa trên encoder của U-Net.
- U-Net: Mô hình phân đoạn ảnh y tế với kiến trúc encoder-decoder và skip connections.
- Attention U-Net: Bổ sung cơ chế Attention giúp tập trung vào các vùng quan trọng trên ảnh MRI.
- Multi-Task Learning Model: Chia sẻ encoder cho cả hai nhiệm vụ Classification và Segmentation nhằm tối ưu hóa việc học đặc trưng.

### Chức năng chính

- Tiền xử lý dữ liệu MRI và segmentation mask.
- Huấn luyện mô hình phân loại khối u não.
- Huấn luyện mô hình phân đoạn vùng khối u.
- Huấn luyện mô hình Multi-Task Learning.
- Đánh giá mô hình trên tập kiểm thử.
- Thực hiện suy luận trên ảnh MRI đơn lẻ hoặc theo lô.
- Triển khai giao diện web bằng Flask.
- Hiển thị kết quả trực quan thông qua ảnh overlay giữa MRI và vùng khối u.

### Công nghệ sử dụng

- Python
- PyTorch
- Torchvision
- OpenCV
- Albumentations
- NumPy
- SciPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn
- Flask

---

## Hướng dẫn cài đặt và chạy chương trình

### 1. Clone project

```bash
git clone https://github.com/PTITMINH999/Brain-Tumor-MTL-BRISC2025.git
cd Brain-Tumor-MTL-BRISC2025/SourceCode
```

### 2. Tạo môi trường Python

```bash
conda create -n brain python=3.12
conda activate brain
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Chuẩn bị dữ liệu

Đặt bộ dữ liệu BRISC2025 vào:

```text
data/raw/brisc2025/
```

### 5. Tiền xử lý dữ liệu

```bash
python validation.py
```

Sau khi hoàn thành, dữ liệu đã xử lý sẽ được lưu tại:

```text
data/processed/
```

Đồng thời hệ thống sẽ:

- Kiểm tra tính hợp lệ của dữ liệu.
- Sinh thống kê tiền xử lý.
- Sinh biểu đồ phân bố dữ liệu.
- Lưu báo cáo vào file JSON.

---

### Huấn luyện mô hình

#### Classification Model

```bash
python train.py --task clf
```

#### Segmentation Model

```bash
python train.py --task seg
```

#### Multi-Task Learning Model

```bash
python train.py --task joint
```

#### Attention U-Net

```bash
python train.py --task attn
```

Trong quá trình huấn luyện, hệ thống sẽ tự động:

- Lưu checkpoint tốt nhất vào thư mục `checkpoints/`
- Lưu log huấn luyện vào thư mục `logs/`
- Lưu learning curves vào thư mục `results/`
- Đánh giá mô hình trên tập test sau khi huấn luyện

---

### Chạy Inference bằng CLI

Dự đoán cho một ảnh MRI:

```bash
python inference.py --image path/to/image.png
```

Dự đoán cho một thư mục ảnh:

```bash
python inference.py --folder path/to/images
```

Tùy chỉnh thư mục lưu kết quả:

```bash
python inference.py --image path/to/image.png --output results/demo_results
```

Kết quả được lưu trong:

```text
results/demo_results/
```

---

### Chạy giao diện Web Flask

Khởi động ứng dụng:

```bash
python app.py
```

Sau khi chạy thành công, truy cập:

```text
http://127.0.0.1:5000
```

### Cách sử dụng

1. Tải lên ảnh MRI não.
2. (Tùy chọn) Tải lên Ground Truth Mask.
3. Thực hiện dự đoán.
4. Hệ thống sẽ hiển thị:
   - Kết quả phân loại khối u.
   - Kết quả phân đoạn vùng khối u.
   - Ảnh overlay trực quan.
   - Thông tin tổng hợp kết quả từ các mô hình.

Các ảnh tải lên được lưu tại:

```text
static/uploads/
```

Kết quả dự đoán được lưu tại:

```text
static/results/
```

### Citation

```text
@article{fateh2025brisc,
title={Brisc: Annotated dataset for brain tumor segmentation and classification with swin-hafnet},
author={Fateh, Amirreza and Rezvani, Yasin and Moayedi, Sara and Rezvani, Sadjad and Fateh, Fatemeh and Fateh, Mansoor and Abolghasemi, Vahid},
journal={arXiv preprint arXiv:2506.14318},
year={2025}
}
```
