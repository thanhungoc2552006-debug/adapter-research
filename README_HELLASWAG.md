# Một run HellaSwag trên Modal H100

Cấu hình đã chọn: Llama-2-7B, PoLAR rank 8, seed 0, 5 epoch,
learning rate 0.004, lambda 0.005, alpha 32, dropout 0.05,
cutoff 512, micro-batch 1, effective batch 128, FP16.
Mốc tham chiếu Table 2 của bài báo cho cấu hình này: **82.14% acc_norm**.

## Thêm vào repo adapter-research

Giải nén các file này vào thư mục gốc `adapter-research`:

- `modal_hellaswag.py`
- `hellaswag_runner/`
- `tests/test_hellaswag.py`
- `README_HELLASWAG.md`
- `validation/hellaswag_data_audit.json`

Không ghi đè `modal_app.py`, `polar_runner/` hoặc các CSV ARC.
Đây là entrypoint và package riêng, vẫn dùng mã nguồn tác giả ở commit
`7c9eeabe54cf94e2051632829019c86c4198b3df`. Không thay công thức adapter hay landing.
Các chỉnh sửa adapter trong fork chưa tự được dùng bởi image này.

Giữ `requirements-local.txt` hiện có (`modal==1.6.3`).
Secret `polar-credentials` trong workspace Modal hiện tại cần có `HF_TOKEN`,
`WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`. Giữ tài khoản W&B academic cũ.
Project W&B mặc định vẫn là `polar-table2-arc`; run mới có tên `polar_r8_hellaswag`
và group `hellaswag`, nên có thể lọc riêng trong cùng project.

## Chạy từ PowerShell trong adapter-research

Xem cấu hình mặc định tại máy, không dùng Modal/cloud:

```powershell
python -c "from hellaswag_runner.experiment import make_plan; p=make_plan({}); print(len(p), 'run:', p[0]['adapter'], p[0]['rank'], p[0]['task'])"
```

Kết quả phải ghi `1 run: polar 8 hellaswag`.
Nếu dùng `--stage plan` qua Modal CLI, Modal vẫn có thể build image/khởi tạo App;
stage này không submit train GPU.

Kiểm tra dataset và quyền W&B, tải cache bằng CPU trước:

```powershell
python -m modal run modal_hellaswag.py --stage setup
```

Kiểm tra log có `hellaswag train: 39905 validation: 10042`.
Bước này không submit GPU, nhưng có thể build image/tải model ở lần đầu.

Chạy đúng một lượt train đầy đủ:

```powershell
python -m modal run --detach modal_hellaswag.py --stage train --ranks 8 --adapters polar --tasks hellaswag --gpu H100 --experiment hellaswag-polar-r8-v1
```

Lượt này khởi tạo adapter mới từ base Llama-2-7B; không tiếp tục adapter đã train ARC.
Chờ `Submitted all jobs` trước khi đóng terminal/tắt máy.
Không gửi lại cùng lệnh khi App trước vẫn đang chạy.

Smoke test là tùy chọn, không phải một phần của kết quả reproduction:

```powershell
python -m modal run --detach modal_hellaswag.py --stage smoke
```

Smoke chỉ train 2 optimizer steps và eval 16 mẫu, lưu dưới `_smoke`.
Không so sánh accuracy smoke với Table 2.

## Dữ liệu và cách đánh giá

- Dataset chính thức `Rowan/hellaswag`, revision đã kiểm tra CPU:
  `218ec52e09a7e7462a5400043bb9a69a41d06b76`. Model revision khóa tại lần setup đầu.
- Train: toàn bộ 39,905 mẫu `train`.
- Đánh giá: toàn bộ 10,042 mẫu **validation có nhãn**. Không dùng test không có nhãn.
- Giữ nguyên `process_docs`, prompt và các lựa chọn của harness đi kèm repo tác giả.
- Thay alias cũ `hellaswag` bằng dataset Parquet chính thức `Rowan/hellaswag`.
  Không thực thi dataset loading script từ xa.
- `acc_norm` dùng độ dài ký tự theo code harness được pin. Paper mô tả byte-length;
  giữ hành vi code, ghi cả acc và acc_norm. Đây là một giới hạn khi đối chiếu.
- Batch hiệu dụng của runner là 128. Script gốc micro-batch 6 trên một GPU có thể
  cho effective batch 126 do chia nguyên. Một seed/H100 không bảo đảm trùng số bài báo.

HellaSwag có nhiều mẫu hơn ARC-e khoảng 18 lần, và các đoạn văn thường dài hơn.
Không ước lượng thời gian bằng smoke 2 bước; xem tốc độ thực sau khi train ổn định.
Mỗi call timeout 23 giờ. Nếu bị ngắt, chạy lại cùng lệnh sau khi call cũ đã dừng
để resume checkpoint hoàn chỉnh gần nhất.

## Xem và lưu kết quả

Modal App: `polar-hellaswag-modal`. Chỉ dùng tối đa một GPU trong App này;
không chạy đồng thời App ARC nếu muốn chỉ thuê một GPU tổng cộng.
Container có `scaledown_window=2`, không giữ GPU sau khi hết công việc.

Volume kết quả vẫn là `polar-table2-results`, cache vẫn là `polar-table2-hf-cache`.
Thư mục mới: `/hellaswag-polar-r8-v1/`.
Checkpoint mỗi 10 optimizer steps, giữ hai checkpoint mới nhất; explicit Volume commit.
Dataset/model revision, fingerprint, môi trường và GPU thực tế được ghi cùng run.

W&B: loss/LR/epoch, stable rank và orthogonality của 15 module ở block 5/15/25,
GPU memory. Đo diagnostics mỗi 20 optimizer steps, không thay đổi gradient.
Accuracy chỉ xuất hiện sau khi train đủ 5 epoch và đánh giá xong:
`eval/acc_norm_pct`, `eval/acc_pct`, `eval/paper_acc_norm_pct=82.14`,
`eval/delta_pp`, `eval/split=validation`.

Xuất CSV, không cấp GPU:

```powershell
python -m modal run modal_hellaswag.py --stage report --experiment hellaswag-polar-r8-v1
```

File local: `hellaswag-polar-r8-v1_comparison.csv`.
File remote: `/hellaswag-polar-r8-v1/table2_hellaswag_comparison.csv`.

## Kiểm tra và commit

```powershell
python -m unittest discover -s tests -p test_hellaswag.py -v
git add modal_hellaswag.py hellaswag_runner tests/test_hellaswag.py README_HELLASWAG.md validation/hellaswag_data_audit.json
git diff --cached --stat
git commit -m "Add HellaSwag reproduction on Modal"
git push
```

Không commit API key, `wandb/`, cache hoặc checkpoint.

## Nguồn

- https://github.com/kcc-lion/polar/tree/7c9eeabe54cf94e2051632829019c86c4198b3df
- https://arxiv.org/html/2506.03133v2#S4.T2
- https://huggingface.co/datasets/Rowan/hellaswag

Đã có 9 kiểm tra CPU/offline cho cấu hình, pin YAML, report, W&B và checkpoint.
Đã kiểm tra dữ liệu thật bằng datasets 3.5.0: đủ 39,905 prompt train,
10,042 mẫu validation có nhãn, dùng preprocessing và prompt generator của commit gốc.
Chi tiết trong `validation/hellaswag_data_audit.json`.
Chưa chạy GPU Modal hay dùng credential của bạn trong quá trình tạo gói này.
