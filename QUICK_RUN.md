# Hướng dẫn Chạy nhanh (Quick Run)

## 1. Huấn luyện (Training)

Lệnh chạy mặc định dùng file cấu hình `config.yaml`:
```bash
python main.py --train --config config.yaml
```

**Các tham số có thể ghi đè qua dòng lệnh (CLI Overrides):**
- Thay đổi Batch Size: `--batch_size <size>` (Mặc định: `8`)
- Thay đổi Learning Rate: `--learning_rate <lr>` (Mặc định: `0.0001`)
- Thay đổi số lượng Epochs: `--num_epochs <epochs>` (Mặc định: `20`)

Ví dụ chạy ghi đè tham số:
```bash
python main.py --train --config config.yaml --batch_size 8 --num_epochs 20
```

---

## 2. Đánh giá độc lập (Test Evaluation)

Đánh giá mô hình sử dụng trọng số **mới nhất** tự động tìm thấy trong `./results/`:
```bash
python main.py --eval --config config.yaml --weights_path latest
```

Đánh giá mô hình sử dụng một file trọng số cụ thể:
```bash
python main.py --eval --config config.yaml --weights_path ./results/YYYY_MM_DD_HH_MM_SS/models/best_model.pth
```

---

## 3. Cấu trúc kết quả đầu ra (Training Outputs)

Mỗi lần chạy `--train`, một thư mục mới dạng `YYYY_MM_DD_HH_MM_SS` (GMT+7) sẽ được tạo ra trong `./results/`:
* `training_log.txt`: Log chi tiết quá trình train & val chạy thời gian thực.
* `configuration.txt`: Toàn bộ tham số cấu hình đã sử dụng.
* `evaluation_result.txt`: Báo cáo chỉ số lỗi (RMSE, MAE,...) chạy trên tập test sau khi train xong.
* `models/`: Chứa các checkpoint (`checkpoint_X.pth`) và file model tốt nhất (`best_model.pth`).
