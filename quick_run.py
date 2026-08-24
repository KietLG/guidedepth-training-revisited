import os
import shutil
import kagglehub

# 1. Định nghĩa thư mục đích mà bạn muốn lưu
target_dir = "./dataset"

# 2. Tải bộ dataset về thư mục mặc định của kagglehub
print("Đang tải dataset từ Kaggle...")
downloaded_path = kagglehub.dataset_download("awsaf49/nyuv2-official-split-dataset")
print(f"Đã tải xong vào thư mục tạm: {downloaded_path}")

# 3. Tạo thư mục 'dataset' nếu nó chưa tồn tại
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"Đã tạo thư mục mới: {target_dir}")

# 4. Di chuyển dữ liệu về thư mục mong muốn
# Lưu ý: Nếu thư mục 'dataset' đã có dữ liệu cũ, bạn có thể muốn xóa hoặc ghi đè tùy nhu cầu.
for item in os.listdir(downloaded_path):
    source_item = os.path.join(downloaded_path, item)
    target_item = os.path.join(target_dir, item)
    
    # Nếu file/thư mục đã tồn tại ở đích thì xóa trước khi ghi đè
    if os.path.exists(target_item):
        if os.path.isdir(target_item):
            shutil.rmtree(target_item)
        else:
            os.remove(target_item)
            
    # Copy hoặc di chuyển dữ liệu sang
    shutil.move(source_item, target_item)

print(f"--- THÀNH CÔNG ---")
print(f"Dữ liệu của bạn hiện đã được lưu tại: {os.path.abspath(target_dir)}")