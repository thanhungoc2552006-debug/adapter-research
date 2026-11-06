# Validation của gói refactor

Đã kiểm tra bằng CPU trong môi trường torch 2.6.0+cpu, transformers 4.50.3, accelerate 1.6.0, NumPy 2.2.4, safetensors 0.5.3, W&B 0.22.3 và Modal 1.6.3. PEFT 0.14.1.dev0 lấy từ source tác giả đã pin. Đây là môi trường kiểm tra, không phải requirements phải cài trên máy Windows để submit Modal.

## Kiểm tra tự động

Toàn bộ **37 tests đã pass** trên CPU.

- Gradient ISR qua cả tử và mẫu; Gram Frobenius chính xác; PI và phổ QR/SVD; zero/rank-1/warm-up.
- Gradient accumulation 1/2/4/128; auxiliary một lần/update; phần accumulation cuối không đủ batch; AMP scaling/clipping/overflow trên CPU.
- λ=0 khớp baseline khi có dropout; fixed-norm giữ target, không đổi A/AdamW moments; AMP skip không chiếu.
- Đối chiếu old/new fixed-norm controller cho cùng gradient và cập nhật optimizer trong phép thử.
- Tiny Llama với PEFT thật: train/save/reload và resume khớp uninterrupted run.
- Checkpoint thật có ready marker cho baseline và controller, từ chối sai method/source identity.
- Registry/config/scaffold, source hashes, dữ liệu legacy/v2, norm parser, DoRA N/A, auto-discovery và W&B offline.
- Orchestration mock: report không gọi setup/train, bỏ run complete trước GPU spawn, chỉ submit pending run.
- Installer: preview không sửa file, apply giữ Git/results/ignore và backup code cũ; từ chối gói sai checksum trước mutation.

Chạy toàn bộ trong môi trường có dependency kiểm tra:

```bash
PYTHONPATH=.:tests python -m unittest discover -s tests -v
```

Trên PowerShell đặt `$env:PYTHONPATH=".;tests"` rồi chạy `python -m unittest discover -s tests -v`. Các test ML cần đúng dependency đã nêu, không chỉ `requirements-local.txt`.

Đã kiểm tra CLI plan/methods và Modal `run --help` không submit remote jobs. Có kiểm tra cài toàn bộ gói vào bản repo mô phỏng từ ZIP đầu vào.

## Giới hạn thực tế

Chưa chạy full ARC/Llama-2-7B trên H100 với bản refactor này, chưa xác thực cloud credential và chưa publish vào W&B thật. Các test W&B chạy offline; orchestration dùng mock. Smoke trên chính workspace Modal là bước xác minh tiếp theo trước một full train mới. Kết quả accuracy cũ không được thay bằng số đo tổng hợp giả.

Reporter CPU chỉ cần NumPy/safetensors/W&B. Nếu adapter legacy chỉ lưu `.bin` thay vì `.safetensors`, cần môi trường local có torch để đọc nó; image report gọn mặc định không cài torch. DoRA norm chưa được implement đầy đủ. Không có multi-GPU hoặc HellaSwag trong pipeline active.
