# Bản đồ và kế hoạch dọn adapter-research

Khảo sát ngày 06/11/2026 từ file `workspace-tree.txt` do bạn gửi. Có 9.257 dòng trong cây thư mục; chưa có nội dung code hiện tại hay Git status của repo Windows. Các kết luận về đường dẫn được xác nhận từ cây thư mục; cơ chế Modal/hash được đối chiếu với các bản code đã gửi trước đó.

## Kết luận và điều chỉnh đề xuất

Repo hiện tại gồm cả source PoLAR/PEFT/evaluation của tác giả và code nghiên cứu thêm vào. Hai cây `lm-evaluation-harness/` và `peft/` chiếm khoảng 98% số dòng của cây thư mục (đây là tỷ lệ số dòng, không phải dung lượng hoặc số file). Chúng là dependency có chức năng, không phải các bản ZIP trùng cần xóa.

Vì vậy, chưa áp dụng ngay cấu trúc `apps/` và `src/adapter_research/` đã phác thảo ở tin trước. Bước đầu phù hợp là gom file sinh ra và làm rõ bản đồ code; bước sau mới refactor imports/đóng gói Modal dựa trên source thực tế.

## Code nên giữ đúng vị trí trong lượt dọn đầu

| Thành phần hiện có | Vai trò | Hành động |
|---|---|---|
| `modal_app.py` + `polar_runner/` | Pipeline baseline LoRA/DoRA/PoLAR trên ARC | Giữ vị trí và nội dung |
| `modal_lora_isr.py` + `isr_runner/` | Method ISR, chưa ép Frobenius norm | Giữ vị trí và nội dung |
| `modal_lora_isr_fixednorm.py` + `fixednorm_runner/` | ISR + rescale norm sau AdamW | Giữ vị trí và nội dung |
| `arc_r8_report.py` + `modal_arc_r8_report.py` | Đọc adapter cuối, đo SR/Frobenius/spectral | Giữ một bộ hiện hành ở root |
| `wandb_rank8_table.py` + `modal_wandb_rank8.py` | Đăng bảng và chart so sánh W&B | Giữ cùng bộ reporter hiện hành |
| `compare_arc_results.py` | So sánh CSV của pipeline trước | Giữ; chưa kết luận có thể bỏ chỉ từ tên file |
| `peft/` | Source PEFT gồm các tuner của nhiều method | Giữ; nhiều folder là chức năng, không phải bản trùng |
| `lm-evaluation-harness/` | Evaluation và rất nhiều task | Giữ; đóng/collapse trong Explorer khi không đọc |
| `finetune_commonsense.py`, `callbacks/`, `stable_rank/`, `prompter.py`, `lm_eval_prompter.py`, `templates/` | Các thành phần source upstream | Giữ đường dẫn |
| `scripts/`, `notebooks/`, Gemma/GLUE files, `assets/` | Các workflow/tài liệu upstream | Giữ; chưa có bằng chứng là file thừa |
| `README.md`, `README_VI.md`, `README_LORA_ISR_ARC.md`, `VALIDATION_LORA_ISR.md` | Tài liệu gốc và hướng dẫn đã dùng | Giữ trong lượt đầu; kiểm tra liên kết trước khi chuyển vào docs |
| `requirements*.txt`, `LICENSE` | Dependency và giấy phép | Giữ |

Các file cùng tên `worker.py`, `tracking.py`, `checkpointing.py` ở ba runner là những phần tương tự về vai trò. Cây thư mục không chứng minh nội dung giống nhau; không deduplicate chỉ dựa trên tên.

Ở các bản Modal đã gửi, entrypoint tìm runner theo thư mục cạnh file và image sinh script bằng patch source upstream. Toàn bộ file Python trong runner tham gia source hash để kiểm tra resume. Di chuyển/gộp source cần kiểm tra lại imports, `add_local_dir`, script patch, checkpoint và run identity.

## Các kết quả hiện có nên giữ nguyên

```text
reports/
  arcc_r8_with_fixednorm/
  arc_challenge_r8/
  arc_challenge_r8_with_lambda_0p1/
```

Mỗi thư mục hiện có `summary.csv`, `per_module.csv`, `per_layer_norms.csv`, `report.json`, `report.md`, `wandb_comparison.json`.

Chúng là các bộ report/snapshot khác nhau. Cùng tên file không có nghĩa là cùng dữ liệu. Script dọn không sửa hoặc thay thế ba thư mục này, và không đổi đường dẫn report bạn đang dùng để đăng W&B.

`experiments/table2_arc/comparison.csv` cũng được giữ: chưa biết nó có cùng nội dung với CSV nào khác.

## Danh sách di chuyển cụ thể của lượt đầu

| File đang ở root | Đích, giữ nguyên tên file |
|---|---|
| `arc-isr-arcc-l003_comparison.csv` | `reports/run_summaries/` |
| `arc-lora-isr-v1_comparison.csv` | `reports/run_summaries/` |
| `arcc-isr-fixednorm-rho1-lam0p1-v1_comparison.csv` | `reports/run_summaries/` |
| `arc-isr-arcc-l003_train_calls.json` | `run_receipts/` |
| `arc-lora-isr-v1_smoke_calls.json` | `run_receipts/` |
| `arc-lora-isr-v1_train_calls.json` | `run_receipts/` |
| `arcc-isr-fixednorm-rho1-lam0p1-v1_smoke_calls.json` | `run_receipts/` |
| `arcc-isr-fixednorm-rho1-lam0p1-v1_train_calls.json` | `run_receipts/` |
| `hellaswag-polar-r8-v1_train_calls.json` | `run_receipts/` |
| `workspace-tree.txt` | `docs/inventory/` |

Ba CSV là bảng tóm tắt run; sáu JSON là biên nhận submit/call ID của Modal, không phải trọng số checkpoint. Không suy ra giá trị lambda chính xác từ chuỗi `l003`: cần đối chiếu run_config.json khi làm báo cáo khoa học.

Sau lượt đầu:

```text
adapter-research/
  README.md và các tài liệu/lệnh hiện tại
  modal_*.py
  arc_r8_report.py
  wandb_rank8_table.py
  compare_arc_results.py
  polar_runner/
  isr_runner/
  fixednorm_runner/
  peft/
  lm-evaluation-harness/
  callbacks/, stable_rank/, templates/, scripts/, notebooks/, assets/
  experiments/table2_arc/comparison.csv
  reports/
    arcc_r8_with_fixednorm/
    arc_challenge_r8/
    arc_challenge_r8_with_lambda_0p1/
    run_summaries/                  # 3 CSV được gom vào đây
  run_receipts/                     # 6 JSON call receipt
  docs/
    WORKSPACE_AUDIT_VI.md
    inventory/workspace-tree.txt
  tools/organize_workspace.py
  tests/
  validation/
  archives/workspace_cleanup/.../manifest.json
```

Không di chuyển/xóa cache trong script. Có thể thêm `__pycache__/` và `*.py[cod]` vào `.gitignore` nếu chưa có; file .gitignore chưa được cung cấp nên gói này không ghi đè nó. Gitignore không tự untrack file đã commit. Không dùng lệnh xóa hàng loạt trước khi xem Git status.

## Điểm cần kiểm tra: HellaSwag và test fixed-norm

Có `tests/test_hellaswag.py`, `validation/hellaswag_data_audit.json`, `hellaswag-polar-r8-v1_train_calls.json` và cache `modal_hellaswag...pyc`, nhưng không thấy `modal_hellaswag.py` hay `hellaswag_runner/` trong cây thư mục. Đây là dấu hiệu cần kiểm tra bằng Git/source trước khi nói pipeline HellaSwag còn đầy đủ. File .pyc không thay thế source để phát triển/đóng gói lên Modal. Gói dọn không tự phục hồi bằng một phiên bản code cũ khác.

Root `tests/` có test baseline/ISR/HellaSwag, chưa thấy test fixed-norm. Điều này phù hợp với việc trước đó chỉ copy các file cần chạy. Nếu muốn repo lưu đủ kiểm chứng cho method mới, bổ sung các test và tài liệu từ ZIP fixed-norm đúng phiên bản. Không chép đè code training đang dùng chỉ để bổ sung test.

## Cách dùng script trên Windows

Giải nén gói này rồi copy hai thư mục `tools/` và `docs/` vào root `adapter-research`. Đứng ở root repo:

```powershell
# Chỉ in kế hoạch, chưa di chuyển gì
python .\tools\organize_workspace.py

# Thực hiện các mục đã in
python .\tools\organize_workspace.py --apply
```

Script chỉ xử lý 10 tên file trong bảng ở trên, không quét và dọn toàn bộ repo. File không còn ở root được skip. Nếu đích đã tồn tại, dừng trước khi di chuyển bất kỳ file nào trong lượt đó. Nó không gọi Modal, W&B, Git hay GPU; không sửa source hoặc checkpoint ở Modal Volume.

Một recovery manifest có SHA-256 từng file được ghi ở `archives/workspace_cleanup/<timestamp>/manifest.json`. Đường dẫn và lệnh hoàn tác được in ra sau khi chạy.

```powershell
# Thay đường dẫn minh họa bằng manifest mà script in ra
python .\tools\organize_workspace.py --undo "archives/workspace_cleanup/TIMESTAMP/manifest.json"
python .\tools\organize_workspace.py --undo "archives/workspace_cleanup/TIMESTAMP/manifest.json" --apply
```

Undo chỉ khôi phục các file đã được di chuyển và không đổi nội dung. Nếu file đích đã bị sửa hoặc xuất hiện file khác ở đường dẫn gốc, script dừng để tránh ghi đè. Nó giữ các thư mục rỗng và lịch sử recovery; không cố xóa thư mục project.

Các entrypoint cũ vẫn có thể tạo CSV/receipt mới ở root khi bạn chạy thêm thí nghiệm. Lượt dọn này xử lý các file đã tồn tại trong cây thư mục, không thay đổi nơi các entrypoint sẽ ghi trong tương lai. Muốn sửa tận gốc phần output path, cần đọc source hiện tại và thay đổi có kiểm chứng ở lượt refactor sau.

## Lượt refactor tiếp theo, sau khi có source thực tế

1. Giữ một bản Git commit tái hiện được các run hiện tại.
2. Đối chiếu ba runner bằng diff; tách setup/data/evaluation/checkpoint/reporting dùng chung.
3. Đặt ISR và fixed-norm trong các module method riêng; biểu diễn lambda/rho/rank/task bằng config, ghi config vào run identity như hiện tại.
4. Gom reporting và thêm một command thống nhất nhận danh sách experiment. Giữ wrapper cho các command cũ nếu cần.
5. Kiểm tra import/đóng gói Modal, lambda=0 control, rescale, checkpoint resume và kết quả report trước khi chạy GPU mới.

Tên thư mục không tự làm thay đổi nguồn code trên Modal: các image đã gửi clone một revision upstream rồi copy runner lên. Khi muốn sửa implementation PEFT local và train nó trên Modal, cần chủ động đưa source đó vào image thay vì giả định file local tự đồng bộ.

## Kiểm chứng gói dọn

Đã chạy mô phỏng dựa trên đúng 10 tên file từ cây thư mục: preview không tạo file/thư mục; apply đưa file đến đúng chỗ và giữ nguyên hash; lần apply tiếp theo không có việc; undo khôi phục nguyên nội dung; đích trùng và file đã thay đổi làm script dừng; các file source và report snapshot trong fixture giữ nguyên. Đây là kiểm chứng trên bản mô phỏng, không phải đã sửa repo Windows của bạn.
