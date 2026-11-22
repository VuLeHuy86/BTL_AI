# BTL_AI

Bài tập lớn triển khai các mô hình học máy và học sâu của CS188 Project 5. Repo có hai phần chạy độc lập trên Kaggle. Mỗi phần có một notebook và một thư mục chứa mã nguồn cùng dữ liệu.

## Nội dung project

| Phần | Nội dung |
| --- | --- |
| Q1 | Perceptron cho phân loại nhị phân |
| Q2 | MLP hồi quy để xấp xỉ hàm `sin(x)` |
| Q3 | MLP phân loại ảnh chữ số MNIST |
| Q4 | RNN nhận dạng ngôn ngữ từ chuỗi ký tự của một từ |
| Q5 | CNN phân loại ảnh chữ số trên tập con MNIST |
| Q6 | Scaled Dot-Product Attention với causal mask |
| Mở rộng | LinearSVC cho Q3, CNN 1D ở cấp ký tự cho Q4 và Character-GPT (Q7) sinh văn bản Shakespeare |

Hai mô hình mở rộng LinearSVC và CNN 1D được dùng để đối chiếu với MLP và RNN trong phần chính. Q7 dùng lại cơ chế attention để dự đoán ký tự tiếp theo.

## Môi trường và thư viện

Project chạy bằng **Python 3** trong **Kaggle Notebooks**. Q1–Q6 chạy trên CPU. Trong phần mở rộng, LinearSVC chạy trên CPU, còn CNN 1D và Character-GPT dùng CUDA khi có GPU. Bật GPU trong phần cài đặt notebook để rút ngắn thời gian huấn luyện hai mô hình này.

| Thư viện | Vai trò |
| --- | --- |
| PyTorch (`torch`) | Xây dựng và huấn luyện mô hình, tính loss, xử lý tensor và nạp dữ liệu |
| NumPy (`numpy`) | Đọc dữ liệu `.npz` và tính các chỉ số đánh giá |
| Matplotlib (`matplotlib`) | Phụ thuộc của autograder và phần hiển thị trong mã nguồn Q1–Q6 |
| scikit-learn (`sklearn`) | Huấn luyện và đánh giá LinearSVC trong phần mở rộng |

Các notebook tự kiểm tra thư viện và gọi `pip` khi cần cài thêm. Dữ liệu MNIST và Language ID nằm trong thư mục `data/` của mỗi bộ source. Văn bản Shakespeare cho Q7 nằm ở `optional/source_optional/input.txt`.

## Cấu trúc thư mục

```text
BTL_AI/
├── README.md
├── Q1-Q6/
│   ├── notebook1.ipynb
│   └── source_Q1_Q6/
└── optional/
    ├── notebook2.ipynb
    └── source_optional/
```

## Hướng dẫn chạy trên Kaggle

Tải repo về máy và đăng nhập tài khoản Kaggle. Chọn cặp notebook và thư mục nguồn theo phần muốn chạy.

| Phần | Notebook để import | Thư mục để tạo Dataset | Tên Dataset gợi ý |
| --- | --- | --- | --- |
| Q1–Q6 | [notebook1.ipynb](Q1-Q6/notebook1.ipynb) | [source_Q1_Q6](Q1-Q6/source_Q1_Q6) | `source-q1-q6` |
| Mở rộng | [notebook2.ipynb](optional/notebook2.ipynb) | [source_optional](optional/source_optional) | `source-optional` |

1. **Import notebook.** Tạo một Kaggle Notebook mới, dùng chức năng **Import Notebook** và chọn file `.ipynb` tương ứng trong bảng.
2. **Tạo Dataset.** Nén thư mục nguồn tương ứng thành `source_Q1_Q6.zip` hoặc `source_optional.zip`, rồi tải ZIP lên khi tạo Kaggle Dataset. Giữ nguyên thư mục con `data/` và các file nguồn khi nén. Bạn cũng có thể dùng Dataset chứa các file đã giải nén sẵn, miễn là cấu trúc thư mục được giữ nguyên.
3. **Gắn Dataset.** Mở lại notebook đã import, chọn **Add Input** và thêm Dataset vừa tạo. Sao chép đường dẫn của thư mục Dataset từ mục **Input**.
4. **Điền đường dẫn.** Trong cell đầu tiên, dán đường dẫn Kaggle vào `DATASET_PATH`. Ví dụ với phần Q1–Q6:

   ```python
   DATASET_PATH = "/kaggle/input/datasets/<ten-tai-khoan>/source-q1-q6"
   ```

   Với phần mở rộng:

   ```python
   DATASET_PATH = "/kaggle/input/datasets/<ten-tai-khoan>/source-optional"
   ```

   Hai đường dẫn trên là ví dụ. Dùng đúng đường dẫn Kaggle hiển thị cho Dataset của bạn. Bạn cũng có thể trỏ thẳng tới file ZIP trong Dataset.
5. **Chạy notebook.** Chọn **Run All** để chạy các cell theo thứ tự. Khi chạy phần mở rộng, có thể bật GPU trước bước này.
6. **Chạy phần còn lại.** Lặp lại các bước trên với notebook và Dataset tương ứng ở hàng còn lại.

## Xem kết quả

| Phần | Kết quả hiển thị | Thư mục lưu log |
| --- | --- | --- |
| Q1–Q6 | Log quá trình huấn luyện và kết quả autograder từng câu. Q6 hiển thị kết quả kiểm thử Attention. | `/kaggle/working/project5_q1_q6_reports` |
| Mở rộng | Kết quả LinearSVC, loss và độ chính xác của CNN 1D, loss theo iteration và văn bản sinh ra từ Character-GPT. | `/kaggle/working/project5_extension_reports` |

Xem log ngay dưới cell chạy. Mỗi thư mục kết quả có file `summary.txt` và log riêng cho từng câu hoặc mô hình.
