# Practice 3 — Get started with Hugging Face

Project thực hành Python đáp ứng hai bài trong đề. Ngôn ngữ dữ liệu: **tiếng Anh**.
Không dùng mô hình này để đánh giá chất lượng phân loại tiếng Việt.

## 1. Nội dung

| Bài | File | Các bước |
|---|---|---|
| Exercise 1 | `exercise1_sentiment.py` | Tải mô hình SST-2 có sẵn, tokenize, in token/ID, dự đoán nhãn và xác suất |
| Exercise 2 | `exercise2_finetune.py` | Tải IMDb, chia dữ liệu, tokenize, cấu hình TrainingArguments, Trainer, fine-tune, đánh giá |
| Dự đoán mới | `predict.py` | Nạp mô hình đã lưu, phân loại câu nhập vào |
| Notebook | `Practice3.ipynb` | Chạy và xem từng bước, có thể dùng để trình bày |

Bài 1 dùng `distilbert/distilbert-base-uncased-finetuned-sst-2-english`.
Bài 2 bắt đầu từ `distilbert/distilbert-base-uncased`, huấn luyện đầu phân loại nhị phân.
IMDb có 25.000 mẫu train và 25.000 mẫu test có nhãn; bỏ phần unsupervised.
Nhãn `0 = NEGATIVE`, `1 = POSITIVE`.

## 2. Cài đặt Windows / VS Code

Giải nén project, mở đúng thư mục `HuggingFace_Practice3` bằng **File → Open Folder**.
Dùng **Python 3.11** cho bộ phiên bản được chỉ định ở đây; nếu máy đang dùng Python 3.14,
hãy tạo môi trường riêng bằng Python 3.11. Không cần gỡ Python cũ.

Chạy trong terminal PowerShell của VS Code:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Chọn Ctrl+Shift+P → **Python: Select Interpreter** → `.venv\Scripts\python.exe`.
Các lệnh dưới gọi Python trực tiếp nên không cần kích hoạt môi trường hoặc đổi ExecutionPolicy.
Cần Internet lần đầu để tải thư viện, mô hình và IMDb; nên để trống vài GB đĩa.
Mặc định chạy CPU nếu PyTorch không nhận CUDA. Bản PyTorch cài từ pip không bảo đảm có CUDA.
Nếu muốn GPU, cài bản PyTorch phù hợp theo https://pytorch.org/get-started/locally/.
Kiểm tra thiết bị: `.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`.

## 3. Chạy bài 1

```powershell
.\.venv\Scripts\python.exe exercise1_sentiment.py
.\.venv\Scripts\python.exe exercise1_sentiment.py --text "I love this movie!" "This film was terrible."
```

Terminal in văn bản, tokens, input_ids, nhãn và score. Lưu ở `outputs/exercise1.json`.
Score là xác suất softmax của nhãn được chọn, không phải độ chính xác toàn bộ mô hình.

## 4. Chạy bài 2

Thử toàn bộ luồng với ít dữ liệu trước (vẫn tải mô hình và bộ dữ liệu):

```powershell
.\.venv\Scripts\python.exe exercise2_finetune.py --train-size 64 --val-size 32 --test-size 32 --epochs 1
```

Huấn luyện bài thực hành mặc định: 2.000 train, 500 validation, 500 test, 2 epoch:

```powershell
.\.venv\Scripts\python.exe exercise2_finetune.py
```

Chạy dữ liệu đầy đủ, giữ riêng 5.000 validation:

```powershell
.\.venv\Scripts\python.exe exercise2_finetune.py --train-size 20000 --val-size 5000 --test-size 25000 --epochs 3
```

Chế độ ít dữ liệu chỉ kiểm tra luồng; không dùng kết quả đó kết luận chất lượng.
Train và validation lấy từ các đoạn không giao nhau của tập train đã shuffle.
Chọn checkpoint theo **validation F1**; chỉ dùng test để đánh giá cuối.
Seed mặc định 42; thiết bị và thư viện vẫn có thể gây chênh lệch kết quả.
Chạy lại cùng `--output-dir` sẽ ghi đè báo cáo/mô hình; dùng thư mục khác để so sánh thí nghiệm.

## 5. Kết quả được tạo sau khi chạy

Trong `outputs/finetune/`:

- `model/`: mô hình và tokenizer đã fine-tune, dùng cho predict.py.
- `metrics.json`: accuracy, precision, recall, F1 và test loss.
- `classification_report.txt`: chỉ số từng lớp và macro/weighted average.
- `confusion_matrix.png`: ma trận nhầm lẫn; hàng là nhãn thật, cột là dự đoán.
- `loss_curve.png`: loss huấn luyện và validation theo epoch.
- `training_history.json`: lịch sử loss và chỉ số validation.
- `config.json`: tham số thí nghiệm.

Precision/recall/F1 trong metrics.json tính với lớp POSITIVE là lớp dương.
Accuracy = số dự đoán đúng / tổng mẫu. F1 là trung bình điều hòa precision và recall.
Không có số liệu huấn luyện được điền sẵn trong project.

```powershell
.\.venv\Scripts\python.exe predict.py --text "The acting was excellent!" "I regret watching this movie."
```

## 6. Notebook

Mở `Practice3.ipynb`, chọn kernel `.venv`, chạy từng cell từ trên xuống.
Notebook gọi cùng mã nguồn nên chỉ cần sửa code ở một nơi.
Mỗi cell có giải thích mục đích; cell huấn luyện có thể chạy lâu trên CPU.

## 7. Lỗi thường gặp

- `No module named ...`: cài requirements bằng đúng `.venv\Scripts\python.exe`.
- Không có Python 3.11: cài Python 3.11 rồi chạy lại lệnh tạo môi trường.
- CUDA out of memory: thêm `--batch-size 2 --max-length 64`, hoặc `--cpu`.
- Không tải được mô hình/dataset: kiểm tra Internet tới huggingface.co; chạy lại sau khi mạng ổn định.
- Cảnh báo classifier weights mới khởi tạo ở bài 2: bình thường, đầu phân loại cần được huấn luyện.
- Muốn ép CPU: thêm `--cpu` vào lệnh bài 2.

## 8. Kiểm tra và phạm vi xác minh

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Project được kiểm tra cú pháp Python, cấu trúc notebook, lệnh CLI và kiểm thử cách tính metric.
Môi trường tạo file không cài PyTorch/Transformers và chưa chạy tải mô hình hoặc fine-tune end-to-end.
Bộ phiên bản requirements là cấu hình đề xuất cố định; cần cài và chạy thử trên máy của bạn.

## 9. Gợi ý nội dung báo cáo

1. Mục tiêu hai bài và ý nghĩa sentiment analysis.
2. Mô hình, dữ liệu, cách chia train/validation/test và tokenization.
3. Siêu tham số: learning rate 2e-5, batch 8, max length 128, epoch 2.
4. Chụp đầu ra bài 1; chèn classification_report và hai biểu đồ từ lần chạy thật.
5. Nhận xét lỗi dự đoán, hạn chế dữ liệu nhỏ, cắt văn bản dài và phạm vi tiếng Anh.

## Tài liệu

- https://huggingface.co/docs/transformers/tasks/sequence_classification
- https://huggingface.co/docs/transformers/main_classes/trainer
- https://huggingface.co/datasets/stanfordnlp/imdb
- https://huggingface.co/distilbert/distilbert-base-uncased
- https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
