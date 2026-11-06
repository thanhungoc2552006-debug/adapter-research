# Adapter Research — ARC pipeline

Một pipeline dùng chung cho LoRA, DoRA, PoLAR, ISR và ISR fixed Frobenius norm trên ARC-Easy / ARC-Challenge. Thêm method không cần copy lại worker, tạo app Modal riêng hay viết lại reporter.

## 1. Cài vào repo đang dùng

Giải nén gói này **ra một thư mục riêng bên ngoài `adapter-research`**. Mở terminal tại thư mục vừa giải nén, nơi có `modal_pipeline.py` và `tools/`.

Xem trước việc chuyển đổi:

```powershell
python tools/install_into_repo.py --root "C:\Users\FPT\Documents\STUDY\LAB\DGM\Implementation\adapter-research"
```

Áp dụng:

```powershell
python tools/install_into_repo.py --root "C:\Users\FPT\Documents\STUDY\LAB\DGM\Implementation\adapter-research" --apply
```

Code cũ được chuyển vào `archives/unified-migration/<thời điểm>/original/`; không cần tự xóa các runner cũ. Công cụ giữ nguyên `.git`, `.gitignore` hiện có, `reports/`, `run_receipts/` và các thư mục dependency. Dữ liệu trên Modal không bị chuyển hoặc xóa. Chi tiết: [MIGRATION.md](docs/MIGRATION.md).

Sau đó mở terminal trong repo:

```powershell
cd "C:\Users\FPT\Documents\STUDY\LAB\DGM\Implementation\adapter-research"
python -m pip install -r requirements-local.txt
python -m adapter_research methods
python -m adapter_research plan --config configs/arc_fixednorm_r8.json
```

Máy local chỉ cần Modal SDK cho các lệnh chạy cloud; không cần cài PyTorch local. Công cụ cài đặt cần Python 3.9 trở lên.

## 2. Trước tiên, cập nhật bảng W&B bằng kết quả đã có

```powershell
python -m modal run modal_pipeline.py --stage report --config configs/arc_report_existing.json
```

Đây là **CPU report**, đọc adapter cuối và evaluation đã lưu; không train lại baseline và không cấp H100. Reporter tìm các experiment trong Volume, bỏ smoke run, chọn ARC-Challenge/rank 8/seed 0 theo config. Nó nhận diện cả code cũ và code mới, bao gồm ISR λ=0.1 và fixed-norm nếu các run tương ứng có đủ file đã lưu.

Với entity/project hiện tại và `comparison.label="main"`, reporter dùng lại ID `cmp1a5e96e4fe188cf04116` của bảng so sánh hiện tại. Terminal in link sau khi publish. Nếu đổi label thì tạo bảng so sánh riêng.

Kết quả local ở `reports/comparison_arc_challenge_r8/`: `summary.csv`, `per_module.csv`, `per_layer_norms.csv`, `report.md`, `report.json`, `wandb_comparison.json`. Bản cloud ở `/results/_reports/arc_challenge/r8/main/`.

## 3. Chạy method theo cùng quy trình

Chỉ chạy hai lệnh sau khi muốn **train một experiment mới**; không cần chạy lại chúng để xem kết quả cũ.

```powershell
python -m modal run --detach modal_pipeline.py --stage smoke --config configs/arc_fixednorm_r8.json
python -m modal run --detach modal_pipeline.py --stage train --config configs/arc_fixednorm_r8.json
```

Smoke: 2 optimizer updates, đánh giá giới hạn 16 mẫu, lưu riêng `_smoke/`. Xem smoke thành công rồi mới chạy train. Train: dùng protocol đầy đủ và sau đó tự chạy CPU report để thêm kết quả vào cùng bảng W&B.

Pipeline tái sử dụng Secret `polar-credentials`, Volume `polar-table2-results`, cache `polar-table2-hf-cache` và các layer image phù hợp. CPU setup kiểm tra quyền truy cập/provenance và chuẩn bị cache trước khi cấp GPU. Thay source có thể cần build lại layer chứa code; không cần tạo lại credentials hoặc các Volume. Chọn đúng Modal profile/workspace `thanh101pedri8` và environment `main` như trước.

| Config | Mục đích |
|---|---|
| `arc_report_existing.json` | Tổng hợp kết quả cũ, CPU |
| `arc_fixednorm_r8.json` | ARC-C/r8, ISR λ=0.1, Frobenius target 1 |
| `arc_fixednorm_control_r8.json` | ARC-C/r8, rescale nhưng λ=0, làm đối chứng |
| `arc_isr_r8.json` | ARC-C/r8, ISR λ=0.01, không rescale |
| `arc_baselines_r8.json` | Train lại LoRA/DoRA/PoLAR trong experiment v2 nếu chủ động muốn |
| `arc_both_tasks.json` | Fixed-norm trên cả ARC-E và ARC-C |

Mặc định dùng **một H100**, các cấu hình chạy lần lượt. Baseline cũ được dùng làm dữ liệu so sánh; không tự train lại khi chạy config của ISR/fixed-norm. Rerun cùng config bỏ qua cấu hình đã hoàn tất trước khi cấp GPU; cấu hình dở dang dùng checkpoint hợp lệ.

## 4. Thêm method tiếp theo

```powershell
python -m adapter_research new-method --name my_method
```

Lệnh tạo:

```text
adapter_research/methods/my_method/__init__.py
adapter_research/methods/my_method/controller.py
configs/arc_my_method_r8.json
```

Sửa loss trong `controller.py`, sửa tham số trong config, rồi dùng cùng các lệnh `plan → smoke → train` với config mới. Registry tự nhận method; reporter tự lấy tên/λ/ρ từ metadata. **Template mới ban đầu vẫn dùng nguyên toán học ISR**; cần implement ý tưởng của bạn trước khi xem nó là một method khác.

Hướng dẫn giao diện controller, checkpoint và cách thêm hyperparameter: [ADDING_METHOD.md](docs/ADDING_METHOD.md).

## Cấu trúc và kết quả đo

```text
modal_pipeline.py          Một entrypoint Modal
configs/                   Chọn task, method và hyperparameter
adapter_research/methods/  Chỉ phần riêng của từng method
adapter_research/pipeline/ Setup, train, checkpoint, evaluation dùng chung
adapter_research/reporting/Đọc run cũ/mới, tính norm, publish W&B
tests/                     Kiểm tra toán học và pipeline
docs/                      Kiến trúc, chuyển đổi, thêm method, validation
tools/                     Công cụ cài đặt
legacy/                    Source cũ để tra cứu và resume experiment cũ
```

Đo **ΔW thực sự**: LoRA/ISR là `γBA`, PoLAR dùng factorization tương ứng. Reporter tính spectral/Frobenius/stable rank cho từng module ở từng layer từ adapter cuối bằng QR/SVD nhỏ, kiểm tra đủ 160 adapter. DoRA chưa có phép dựng đầy đủ ΔW từ base + magnitude trong reporter này nên các norm là N/A; vẫn báo accuracy.

Biểu đồ norm theo `Layer` mô tả adapter **cuối training giữa các layer**, không phải diễn biến theo thời gian. Log training/projection dùng `train/global_step` mới cho biết thay đổi qua các update. Sai số `acc_norm_stderr` là sai số chuẩn theo tập đánh giá, không phải độ lệch chuẩn qua nhiều seed.

Protocol mặc định giữ bản gốc: Llama-2-7B, rank 8, alpha 32, dropout .05, effective batch 128, microbatch 1, epochs 5, cutoff 512. Số optimizer updates do Trainer và kích thước task quyết định; ARC-C trong pipeline cũ khoảng 40, ARC-E khoảng 85. Không diễn giải 40 là giới hạn chung cho mọi config. `acc_norm` của harness đã pin chuẩn hóa theo **độ dài ký tự** của đáp án.

Các checkpoint v1 không được resume trực tiếp bằng trainer v2. Kết quả v1 vẫn được đọc để so sánh; training mới dùng tên experiment v2 và kế thừa model/dataset revisions của baseline. Xem [ARCHITECTURE.md](docs/ARCHITECTURE.md) và [VALIDATION.md](docs/VALIDATION.md).
