# Chuyển từ ba runner sang một pipeline

## Những gì được giữ

- Secret `polar-credentials` và hai Volume cũ; không upload lại checkpoint chỉ vì đổi cấu trúc source.
- Model/dataset revisions từ `table2-arc-resume/source_lock.json` để so sánh cùng dữ liệu và base model.
- Kết quả baseline/ISR/fixed-norm cũ được reporter đọc trực tiếp.
- `.git`, `.gitignore` hiện có, `reports/`, `run_receipts/`, source dependency như `peft/` và `lm-evaluation-harness/` trong repo local.

## Những gì được chuyển

Các entrypoint Modal/reporting cũ, `polar_runner/`, `isr_runner/`, `fixednorm_runner/`, test cũ và các mục HellaSwag được nhận diện sẽ chuyển vào `archives/unified-migration/<timestamp>/original/`. Gói mới có bản `legacy/` của source đã cung cấp; bản trong archives mới là bản gốc chính xác của repo local tại thời điểm cài.

HellaSwag không còn trong config, registry, pipeline hoặc test đang dùng. File cũ nếu tồn tại được lưu trong archive để tránh mất công việc.

`tools/install_into_repo.py` mặc định chỉ preview. Khi `--apply`, nó kiểm tra toàn bộ SHA-256 của gói trước khi sửa target, lưu lịch sử vào `migration.json`, chuyển file cũ rồi cài file mới. Nếu bị ngắt, trạng thái là `interrupted` và các file đã lưu vẫn nằm trong `original/`; không có rollback tự động. Không chạy installer khi đang chỉnh file hoặc train bằng repo đó.

Nếu cần khôi phục, đọc `migration.json` để biết file mới đã cài và đường dẫn cũ đã chuyển; lấy các file tương ứng từ `original/` về đúng vị trí sau khi cất bản mới sang nơi riêng. Không copy đè mù toàn bộ repo, không đụng `.git` hoặc kết quả.

## Checkpoint cũ

V2 đổi source fingerprint, tên file controller state và cơ chế xác minh identity. Nó không nạp checkpoint v1 để tiếp tục train. Muốn resume experiment cũ, dùng source gốc trong archive ở một thư mục làm việc riêng, với đúng layout/entrypoint cũ. `legacy/` cũng giữ source của ZIP đầu vào để tra cứu; kết quả cũ không cần resume để reporter đo lại norm.

Training mới dùng `experiment` mới, ví dụ `arc-fixednorm-v2`. CPU setup từ chối dùng thư mục experiment v1 làm output v2. Khi thay code training chung, chọn tên experiment mới; khi chỉ thêm method mới, các method cũ không đổi source identity.

## Sau cài đặt

1. Mở repo và chạy `python -m adapter_research methods`.
2. Chạy report với `configs/arc_report_existing.json` để kiểm tra LoRA/PoLAR/ISR/fixed-norm đã có.
3. Thêm method bằng `new-method`, sửa code và config, rồi `plan → smoke → train`.
4. Xem `git status` / `git diff` trước khi commit. Archive source có thể commit hoặc giữ local theo lựa chọn của bạn; không commit token, cache, adapter weights hoặc dữ liệu lớn.

Không cần train lại các baseline cũ để sử dụng pipeline mới. Nếu baseline experiment thực tế ở một tên khác, sửa cả `baseline_experiment` và `comparison.baseline_experiment`; để danh sách `comparison.experiments` phù hợp. Auto-discovery vẫn tìm run theo nội dung `run_config.json`, không đoán λ từ tên thư mục.
