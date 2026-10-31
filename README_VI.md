# PoLAR Table 2 trên Modal — ARC-e và ARC-c

Gói này chuyển reproduction từ Colab sang **Modal Functions + persistent Volumes**. Không cần GPU trên máy cá nhân, không cần Colab, không cần đưa key vào code.

- Dùng mã gốc `kcc-lion/polar`, commit `7c9eeabe54cf94e2051632829019c86c4198b3df`.
- Base model: **meta-llama/Llama-2-7b-hf**, không dùng bản Chat.
- LoRA, DoRA, PoLAR; rank 4/8/16/32; mỗi task train riêng 5 epoch; tối đa 24 lượt.
- Mặc định **một A100 80 GB**, các lượt xếp hàng và dùng GPU lần lượt. Có thể chọn A100 40 GB bằng CLI.
- W&B 0.22.3 hỗ trợ key dài; `MPLBACKEND=Agg` tránh lỗi matplotlib inline từ Colab.
- Checkpoint, optimizer/scheduler/scaler/RNG và kết quả được giữ trên Modal Volume, không phụ thuộc terminal mở hay máy tính bật.

**Đã kiểm tra:** cú pháp, import/CLI với Modal 1.6.3; patch áp dụng đúng commit gốc; 5 kiểm tra về kế hoạch, checkpoint, bảng kết quả và đổi workspace W&B. **Chưa build image hoặc chạy Llama/A100 trên tài khoản Modal thật**, nên hãy chạy smoke test trước. Image build và các lệnh cần quyền Modal chỉ được thực thi khi bạn chạy.

## 1. Chuẩn bị trên máy tính

Cài Python 3.10 trở lên (khuyến nghị 3.11/3.12). Giải nén gói và mở terminal **trong folder có `modal_app.py`**:

```bash
python -m pip install -r requirements-local.txt
python -m modal setup
```

Lệnh setup mở trình duyệt để đăng nhập Modal. Máy tính chỉ cần cài Modal SDK, không cài PyTorch hay tải Llama về máy. Có thể dùng VS Code terminal / PowerShell.

Xem các tùy chọn mà không chạy GPU:

```bash
python -m modal run modal_app.py --help
```

## 2. Tạo Modal Secret cho đúng tài khoản mới

Trong [Modal Dashboard → Secrets](https://modal.com/secrets), tạo **Custom Secret** tên chính xác:

```text
polar-credentials
```

Thêm các key-value:

| Tên biến | Giá trị bạn điền |
|---|---|
| `HF_TOKEN` | Hugging Face read token, tài khoản đã được cấp quyền Llama 2 |
| `WANDB_API_KEY` | Key W&B của tài khoản hiện tại; giữ nguyên toàn bộ key |
| `WANDB_ENTITY` | Username/team slug có quyền ghi; phần ngay sau `wandb.ai/` trong URL workspace |
| `WANDB_PROJECT` | `polar-table2-arc` hoặc project riêng bạn muốn dùng |

Không điền tên hiển thị thay cho entity. Nếu đổi tài khoản: cập nhật API key **và entity** trong Secret. Không gửi key vào chat hay lưu trong Git. Secret được inject khi Function khởi động; kết thúc các job cũ rồi khởi chạy job mới để nhận Secret mới.

Vào [trang model](https://huggingface.co/meta-llama/Llama-2-7b-hf), bảo đảm tài khoản của HF_TOKEN có quyền truy cập. Có hạn mức/credit cho GPU trên Modal trước khi chạy. Gói dùng tài nguyên tính phí khi chạy; không tự triển khai hoặc khởi động GPU trong bước kiểm tra trên máy.

## 3. Smoke test trước — một cấu hình

```bash
python -m modal run --detach modal_app.py --stage smoke --ranks 4 --adapters polar --tasks arc_easy
```

Quá trình thực hiện:

1. Build image Python 3.10, cài dependencies theo repo, thay riêng W&B 0.19.9 → **0.22.3**.
2. CPU setup kiểm tra quyền HF và quyền ghi workspace W&B bằng run có tag `connection-check`. Cache model/dataset trên Volume; lần đầu cần chờ tải trọng số.
3. GPU train **2 optimizer steps**, eval **16 mẫu**, lưu riêng dưới `_smoke/`.
4. Output có link Modal App/Function logs và link `W&B run:`.

Dùng `--detach` để các job đã submit tiếp tục chạy khi terminal bị ngắt. Ban đầu vẫn cần giữ kết nối cho đến khi setup xong và thông báo **Submitted all jobs** xuất hiện. Build/setup chưa hoàn tất thì chưa có job GPU nào được submit.

Smoke tiêu thụ thời gian vì dùng model thật và batch hiệu dụng 128, không phải một phép tính nhỏ. Kết quả smoke không dùng đối chiếu Table 2. Muốn kiểm tra cả 3 phương pháp trước, chạy thêm:

```bash
python -m modal run --detach modal_app.py --stage smoke --ranks 4 --adapters lora,dora,polar --tasks arc_easy
```

## 4. Chạy reproduction

Khởi đầu 6 lượt rank 8:

```bash
python -m modal run --detach modal_app.py --stage train --ranks 8 --adapters lora,dora,polar --tasks arc_easy,arc_challenge
```

Đủ 24 lượt để reproduce hai cột ARC-e/ARC-c:

```bash
python -m modal run --detach modal_app.py --stage train --ranks 4,8,16,32 --adapters lora,dora,polar --tasks arc_easy,arc_challenge
```

Các lượt được submit trước, xếp hàng, **max_containers=1** nên một App chỉ dùng một GPU train tại một thời điểm. Không chạy đồng thời hai lệnh suite cho cùng `--experiment`; giới hạn một GPU không ngăn được hai App riêng biệt ghi trùng một lượt.

Mỗi lượt là một Function call riêng, timeout 23 giờ; cả suite không bị gom thành một Function call 24 giờ. Khi lỗi cấu hình/quyền truy cập, code không tự retry nhiều lần. Sau khi sửa, chạy lại đúng lệnh để bỏ qua lượt hoàn thành và resume lượt chưa xong. Modal có thể retry sự cố hạ tầng theo cơ chế nền tảng; checkpoint vẫn cần thiết.

### Cấu hình GPU/batch

```bash
python -m modal run --detach modal_app.py --stage train --ranks 8 --adapters polar --tasks arc_easy --gpu A100-40GB --micro-batch 1
```

Mặc định `--micro-batch 1 --effective-batch 128 --dtype fp16`, bật gradient checkpointing, không lượng tử hóa. Để thử tăng tốc trên A100 80 GB có thể dùng `--micro-batch 4`, nhưng sẽ tạo run ID khác vì batch grouping thay đổi. Luôn smoke cùng cấu hình trước. `effective-batch` phải chia hết cho `micro-batch`.

`--eval-batch 1` giảm bộ nhớ lúc eval mà không đổi cấu hình train. `--no-diagnostics` tắt stable rank/orthogonality; `--no-use-wandb` tắt gửi lên W&B. HF_TOKEN vẫn cần thiết. `--save-steps 10` mặc định giữ 2 checkpoint mới nhất.

`--experiment table2-arc-v1` mặc định. Dùng tên mới, ví dụ `--experiment table2-arc-test2`, khi muốn chạy một bộ độc lập hoặc khi thay source code. Nếu không, notebook kiểm tra source/revision để tránh resume nhầm.

## 5. Xem tiến trình

### Modal

Mở App **polar-table2-arc-modal** trên [Modal Dashboard](https://modal.com/apps). Xem danh sách queued/running calls và logs của `setup` hoặc `train_job`. Nếu terminal đang chạy, log cũng stream ra terminal. Các call ID được lưu trong file cục bộ `table2-arc-v1_train_calls.json` khi submit.

### W&B

Project mặc định `polar-table2-arc`. Mỗi tổ hợp `(adapter, rank, task)` có một run. Train và eval nối cùng run ID; smoke được gắn tag `smoke`, lượt thật tag `reproduction`.

| Metric | Ý nghĩa |
|---|---|
| `train/loss` | Loss train theo optimizer step |
| `train/learning_rate`, `train/epoch`, `train/grad_norm` khi Trainer ghi | Learning rate, tiến độ epoch, norm gradient trước clipping |
| `stable_rank/layer5.*`, layer15, layer25 | Stable rank ở từng module LoRA/PoLAR được chọn |
| `diagnostics/stable_rank_mean` | Trung bình ở các module theo dõi, không phải toàn model |
| `diagnostics/orthogonality_A_mean`, `...B_mean` | Mức sai lệch trực chuẩn bình phương |
| `gpu/allocated_GB`, `reserved_GB`, `peak_allocated_GB` | Bộ nhớ PyTorch ở các mốc đo |
| `eval/acc_norm_pct`, `eval/acc_pct` | Accuracy sau evaluation |
| `eval/paper_acc_norm_pct`, `eval/delta_pp` | Số bài báo và chênh lệch điểm phần trăm; không ghi cho smoke |

Stable rank tính bằng QR mỏng của Aᵀ/B, SVD trên lõi r×r; không dựng ΔW lớn. Observer không sửa gradient. Không log stable rank cho DoRA vì BA không phải toàn bộ cập nhật có magnitude/normalization của DoRA. Gram matrix A/B vẫn được đo. Đo mỗi 20 optimizer steps và đầu/cuối train; cập nhật bằng 0 thì stable rank chưa xác định và được bỏ qua.

`LandingProjectionCallback` gốc vẫn hoạt động cho PoLAR. Với gradient accumulation 128, biểu đồ không cập nhật sau từng mẫu. Test accuracy chỉ xuất hiện khi eval xong; không có đường accuracy từng bước nếu không bổ sung validation.

Đổi workspace W&B tạo một namespace logging khác dưới `wandb_targets/`, giữ checkpoint. Metrics cũ vẫn nằm ở workspace cũ. Nếu một lượt đã eval xong trước khi đổi tài khoản, lần chạy lại có thể gửi lại accuracy sang workspace mới; không tái tạo loss lịch sử chưa ghi vào workspace mới.

## 6. Resume và lưu dữ liệu

Volumes được tạo tự động:

```text
polar-table2-results
  /table2-arc-v1/source_lock.json
  /table2-arc-v1/dataset_manifest.json
  /table2-arc-v1/environment.txt
  /table2-arc-v1/modal_changes.diff
  /table2-arc-v1/runs/<run_id>/run_config.json
  /table2-arc-v1/runs/<run_id>/checkpoint-*/
  /table2-arc-v1/runs/<run_id>/adapter_model.safetensors
  /table2-arc-v1/runs/<run_id>/eval_results.json
  /table2-arc-v1/runs/<run_id>/train.log
  /table2-arc-v1/runs/<run_id>/wandb_targets/<namespace>/identity.json
  /table2-arc-v1/_smoke/...

polar-table2-hf-cache
  /hub/...
  /datasets/...
```

Sau `Trainer.on_save`, code ghi dấu checkpoint hoàn tất; process cha gọi Volume.commit(). Chỉ resume checkpoint có đủ adapter, optimizer, scheduler, RNG, trainer state, scaler khi FP16 và dấu hoàn tất. Mất tiến trình trước checkpoint đầu tiên thì lượt đó phải bắt đầu lại; phần sau checkpoint gần nhất có thể phải tính lại. Background commits của Modal bổ sung việc lưu nhưng không thay thế checkpoint.

Train và eval là hai process riêng để trả VRAM trước khi nạp model đánh giá. Nếu train đã xong mà eval lỗi, chạy lại lệnh sẽ chỉ eval. Nếu raw eval JSON đã có mà W&B lỗi, chạy lại sẽ chỉ gửi W&B. Không xóa Volume khi ngắt App.

### Dữ liệu Colab cũ

Bản Modal có storage/run IDs riêng, **không tự truy cập Google Drive hoặc nhập checkpoint Colab**. Nếu còn dữ liệu trên Drive, giữ nguyên để sao lưu; đừng xóa vì chuyển sang Modal. Đưa checkpoint Colab vào pipeline này cần đối chiếu config/revisions và ánh xạ đường dẫn trước, không chỉ copy file vào một run mới. Gói này mặc định chạy reproduction trên Modal từ đầu.

## 7. Tổng hợp và tải kết quả

Khi suite hoàn tất và terminal còn kết nối, CSV được viết về folder hiện tại. Để tạo báo cáo bất kỳ lúc nào (CPU, không cấp GPU):

```bash
python -m modal run modal_app.py --stage report
```

File cục bộ: `table2-arc-v1_comparison.csv`. Báo cáo không lấy smoke và không tính Avg. của 8 task vì chỉ chạy 2 task.

Xem file trên Volume:

```bash
python -m modal volume ls polar-table2-results /table2-arc-v1
```

Tải CSV:

```bash
python -m modal volume get polar-table2-results /table2-arc-v1/table2_arc_comparison.csv ./table2_arc_comparison.csv
```

Tải toàn bộ thư mục thí nghiệm, gồm checkpoint và logs (có thể lớn):

```bash
python -m modal volume get polar-table2-results /table2-arc-v1 ./modal_results
```

Nếu dùng tên experiment khác, thay đường dẫn tương ứng.

## 8. Mức độ tương đồng với bài báo

- Mục tiêu **Table 2**, cấu hình **Table 10 trong v2** (Table 9 trong v1), Appendix G.1.
- LoRA/DoRA learning rate `4e-4`; PoLAR `4e-3`, lambda `5e-3`; alpha 32, dropout .05, 5 epoch, cutoff 512, seed 0; AdamW, cosine, warmup .03; target modules đúng script gốc.
- Giữ FP32 base weights sau `prepare_model_for_kbit_training()` như repo, AMP fp16. Không dùng 4-bit/8-bit.
- Batch mục tiêu 128; script một GPU gốc dùng micro-batch 6 và chia nguyên nên thực tế 126. Muốn mô phỏng số này dùng `--effective-batch 126 --micro-batch 1` (run khác).
- Bật checkpointing, tắt compile, shuffle seed tường minh, chia process train/eval và thêm persistence/logging. Đây là các điều chỉnh so với script gốc, không bảo đảm khớp từng số thập phân trên GH200/H100.
- Dùng metric `acc_norm` của **harness trong repo**. Paper mô tả byte normalization, nhưng `ConfigurableTask.process_results` ở commit này dùng độ dài ký tự; gói giữ hành vi code. Ghi cả `acc` và `acc_norm`.
- Không grid search/chọn rank trên test. Mỗi rank báo cáo độc lập. Không giả định PoLAR phải thắng trên ARC.
- Image cài requirements gốc, thay W&B; model/dataset revision được khóa lần đầu và lưu trong `source_lock.json`. Bản GPU/driver thực tế được lưu trong `attempt_*.json`.

## 9. Khi lỗi

- **403 W&B:** kiểm tra API key và entity cùng tài khoản có quyền ghi. Setup kiểm tra quyền trước khi submit GPU. Không sửa key thành 40 ký tự.
- **401/403 HF:** xin quyền model, dùng token của đúng tài khoản.
- **OOM train:** mặc định micro-batch 1; chọn A100-80GB. Nếu đang thử batch 4, giảm về 1 tạo một cấu hình riêng.
- **OOM eval:** chạy lại với `--eval-batch 1`; train xong sẽ không train lại.
- **FP16 NaN:** giữ logs; có thể thử `--dtype bf16` như một biến thể mới, không trộn kết quả với FP16.
- **Timeout/preemption:** chạy lại đúng lệnh sau khi job cũ đã dừng. Checkpoint hoàn tất gần nhất được dùng.
- **Source changed:** dùng tên `--experiment` mới, không ghi đè source lock cũ.
- **Chưa thấy W&B run:** có thể vẫn đang build image/cache model hoặc chờ GPU; xem Modal logs.
- **CLI không tìm thấy folder/module:** mở terminal tại folder đã giải nén chứa `modal_app.py` và `polar_runner/`.

## 10. Cấu trúc mã

- `modal_app.py`: image, Secret/Volume, CPU setup, GPU worker, submission, report.
- `polar_runner/prepare.py`: patch có kiểm tra source gốc; không tự viết lại adapter.
- `polar_runner/worker.py`: source/cache lock, subprocess train/eval, commit Volume.
- `polar_runner/checkpointing.py`: dấu checkpoint hoàn tất và chọn checkpoint resume.
- `polar_runner/tracking.py`: W&B lifecycle, QR stable rank và Gram diagnostics.
- `polar_runner/evaluate.py`: harness evaluation và log kết quả.
- `polar_runner/experiment.py`: kế hoạch, reference Table 2, thu thập báo cáo.
- `tests/test_pipeline.py`: kiểm tra offline không cần GPU hay tài khoản Modal.

Chạy kiểm tra cục bộ:

```bash
python -m unittest discover -s tests -v
```

## Nguồn đã đối chiếu

- https://github.com/kcc-lion/polar/tree/7c9eeabe54cf94e2051632829019c86c4198b3df
- https://arxiv.org/html/2506.03133v2
- https://modal.com/docs/guide/gpu
- https://modal.com/docs/guide/volumes
- https://modal.com/docs/examples/long-training
- https://modal.com/docs/guide/secrets
- https://modal.com/docs/cli/latest/run
- https://modal.com/docs/cli/latest/volume
