# LoRA-ISR trên ARC-e / ARC-c, dùng pipeline Modal cũ

Đây là code bổ sung cho repo `adapter-research`: huấn luyện **LoRA + inverse-stable-rank regularization**, rồi đánh giá ARC-e / ARC-c giống runner LoRA/PoLAR đã chạy. Tên method trong config: `lora_isr`.

Hai run mặc định: **ISR rank 8 + ARC-e**, **ISR rank 8 + ARC-c**. Chúng chạy tuần tự, tối đa **một H100** cho app này. Method mới bắt đầu từ pretrained Llama-2-7B với adapter mới; không dùng checkpoint LoRA đã train làm initialization. Kết quả baseline cũ được dùng khi so sánh.

## 1. Thêm code vào repo trên Windows

Giải nén toàn bộ nội dung ZIP vào thư mục `adapter-research` đang có. Bạn sẽ có:

```text
adapter-research/
  modal_app.py                (runner ARC cũ, giữ nguyên)
  polar_runner/              (giữ nguyên)
  modal_lora_isr.py           (entrypoint mới)
  isr_runner/                (method và pipeline mới)
  compare_arc_results.py
  README_LORA_ISR_ARC.md
  VALIDATION_LORA_ISR.md
  tests/test_lora_isr.py
```

Mở PowerShell tại repo:

```powershell
cd "$env:USERPROFILE\Downloads\adapter-research"
python -m modal profile current
```

Chọn đúng workspace có Volume `polar-table2-results` và backup ARC đã upload. Dùng lại Secret **`polar-credentials`** với:

| Key | Giá trị |
|---|---|
| `HF_TOKEN` | Token có quyền tải `meta-llama/Llama-2-7b-hf` |
| `WANDB_API_KEY` | API key của tài khoản academic đang dùng |
| `WANDB_ENTITY` | `thanh-nc2416352-hanoi-university-of-science-and-technology` |
| `WANDB_PROJECT` | `polar-table2-arc` |

Không đưa token vào file Python hay Git. Không cần cài PyTorch/CUDA trên Windows để submit Modal; môi trường GPU được build trong container. Nếu máy chưa có Modal: `python -m pip install modal` và `python -m modal token new`.

## 2. Chuẩn bị trên CPU, chưa cấp GPU

Giả sử experiment chứa baseline đã chạy xong là **`table2-arc-resume`** như pipeline trước:

```powershell
python -m modal run modal_lora_isr.py --stage setup --baseline-experiment table2-arc-resume
```

Code sẽ kiểm tra W&B, quyền tải model, dataset và cache; đồng thời dùng lại **model revision / dataset revision** trong `source_lock.json` của baseline. Nếu tên experiment cũ khác, sửa `--baseline-experiment` cho đúng thư mục trực tiếp bên trong Volume. Không trỏ tới thư mục lồng sai như `table2-arc-v1/table2-arc-v1`.

Experiment mới mặc định **`arc-lora-isr-v1`**, không ghi đè kết quả cũ. Nếu đổi source code sau khi đã setup, hãy dùng experiment mới, ví dụ `--experiment arc-lora-isr-v2`, ở tất cả các lệnh tiếp theo. Kiểm tra source hash ngăn resume nhầm code.

Có thể xem cấu hình trên máy mà không submit GPU bằng:

```powershell
python -c "import modal_lora_isr as m; m.main()"
```

## 3. Smoke test trước

```powershell
python -m modal run --detach modal_lora_isr.py --stage smoke --tasks arc_easy --ranks 8 --adapters lora_isr --gpu H100 --save-steps 1 --baseline-experiment table2-arc-resume
```

Smoke chạy **hai optimizer updates** và evaluate 16 mẫu. Update đầu warm-up; update sau thử bật auxiliary. Nó cũng thử lưu `sr_state.pt`, nhưng chưa chứng minh resume trên CUDA. Kiểm tra Modal log không có traceback, và W&B run `smoke_lora_isr_r8_arc_easy` có `train/sr/modules_total = 160`, sau warm-up có `train/sr/modules_active > 0`, `train/sr/aux_backward_calls > 0`. Nếu AMP skip update đầu, hai bước có thể chưa bật auxiliary: kiểm tra `train/sr/optimizer_skipped` và xử lý trước full run.

Kết quả smoke nằm trong `_smoke`, không được gộp vào báo cáo full. Smoke vẫn dùng GPU và có tính usage. Đợi app smoke hoàn tất rồi mới chạy full để tránh hai app dùng hai GPU cùng lúc.

## 4. Chạy thật ARC-e và ARC-c

```powershell
python -m modal run --detach modal_lora_isr.py --stage train --tasks arc_easy,arc_challenge --ranks 8 --adapters lora_isr --gpu H100 --sr-lambda 0.01 --baseline-experiment table2-arc-resume
```

Đây là **hai full runs mới**, không chạy lại LoRA/PoLAR. Chạy lại đúng lệnh sẽ bỏ qua run đã evaluate xong, hoặc resume checkpoint đầy đủ gần nhất. Sau dòng `Submitted all jobs`, cả hai job đã được submit; `--detach` cho phép chúng tiếp tục khi terminal ngắt kết nối. File `arc-lora-isr-v1_train_calls.json` ghi call IDs để theo dõi.

Khi function kết thúc, container sẽ được thu hồi với scaledown 2 giây; không giữ GPU chạy vô hạn. Storage của Volume vẫn tồn tại. Không thể hứa tổng thời gian trước smoke trên H100: task loss vẫn như baseline, còn auxiliary/PI và audit làm thêm công việc. Xem tốc độ thực tế trong Modal log/W&B; không khởi chạy lại lệnh chỉ vì một job đang Pending.

Muốn so sánh tất cả rank đã có baseline, đổi `--ranks 4,8,16,32`: sẽ thành **8 full runs**, không phải một run thay rank giữa chừng.

## 5. Tổng hợp và so sánh với baseline cũ

Sau khi full train/evaluate xong:

```powershell
python -m modal run modal_lora_isr.py --stage report
```

CSV tải về **thư mục PowerShell đang đứng**: `arc-lora-isr-v1_comparison.csv`. Sau đó dùng CSV cũ của bạn, ví dụ `table2-arc-resume_comparison.csv`:

```powershell
python compare_arc_results.py --baseline .\table2-arc-resume_comparison.csv --new .\arc-lora-isr-v1_comparison.csv --output .\arc_isr_vs_baselines.csv
```

Nếu CSV cũ ở thư mục khác, sửa đường dẫn `--baseline`. Script ghép theo task/rank/seed/test split, lấy **acc_norm (%)**, tính `ISR - LoRA` và `ISR - PoLAR` theo điểm phần trăm; không tự chọn kết quả tốt nhất. Nếu thiếu baseline cùng rank/seed, ô tương ứng để trống. CSV legacy không có cột seed được hiểu là seed 0; có thể đổi bằng `--baseline-seed`. Script không thể chứng minh mọi cấu hình đều giống nhau từ CSV; đối chiếu thêm `run_config.json`, `source_lock.json`, `baseline_provenance.json`.

ISR là method mới nên không có accuracy paper để điền. Code để trống `paper_pct` / `delta_pp` cho ISR. Bảng so sánh với chính baseline bạn đã chạy mới là kết quả cần đọc.

## 6. Đúng method đã đề xuất ở những đâu?

| Phần | File / hành vi |
|---|---|
| Forward chuẩn | Dùng adapter backend `lora` trong PEFT của repo PoLAR; `W0 + gamma BA` |
| Optimizer | AdamW của pipeline gốc; chỉ A/B trainable, pretrained bị đóng băng |
| Loss | `CE + lambda_t * sum(s/d)/M`; M cố định là toàn bộ 160 ma trận |
| PI | `regularizer.py`: no-grad, cache vector v, RNG riêng từng module |
| Tử số | `s = ||gamma B(A v)||²`, tính lại với A/B có gradient |
| Mẫu số | `d = gamma² tr((BᵀB)(AAᵀ))`, không detach, không thêm epsilon vào tỷ lệ |
| Ngưỡng | Nếu `sqrt(d) <= 1e-6`, bỏ term đó nhưng vẫn chia tổng cho M |
| Warm-up | `sr_config.py`: lambda=0 thì bỏ hoàn toàn auxiliary/Gram/PI |
| Backward | `trainer.py`: sau CE accumulation, một lần/update, trước AMP unscale và clipping |
| Resume | `sr_state.pt` chứa PI vectors, RNG, schedule/counters; lưu cùng checkpoint Trainer |

Code không dựng full `BA` để regularize; chỉ PI và Gram matrices nhỏ. Không chuẩn hóa hoặc chiếu A/B sau AdamW, không ép Frobenius norm bằng 1. Vẫn giữ dropout LoRA của baseline; penalty áp dụng cho ma trận cập nhật tham số `gamma BA`.

## 7. Cấu hình thử đầu tiên

| Tham số | Mặc định |
|---|---|
| Model | `meta-llama/Llama-2-7b-hf` |
| Tasks | ARC-Easy và ARC-Challenge, train riêng/evaluate riêng |
| Rank | 8 |
| Alpha / dropout | 32 / 0.05 |
| Target modules | q_proj, k_proj, v_proj, up_proj, down_proj ở 32 blocks: 160 ma trận |
| Epochs / LR | 5 / 0.0004 |
| Batch | micro 1, effective 128, accumulation 128 |
| Precision | fp16 AMP; weights base FP32 sau bước prepare như baseline cũ |
| Cutoff / scheduler | 512 / cosine, LR warm-up 3% |
| ISR lambda max | 0.01 |
| ISR warm-up / ramp | 5% / 10% tổng optimizer updates |
| PI | 20 iterations khi tạo/reset vector; 5 iterations mỗi update tiếp theo |
| Exact spectral audit | QR + SVD core r×r, tất cả module mỗi 20 attempted updates và lần active đầu |
| Seed | 0 |
| Checkpoint | Mỗi 10 global steps, giữ 2 checkpoint gần nhất |

Với `u` updates thành công đã hoàn tất, `T = Trainer.state.max_steps`, `W=ceil(.05T)`, `R=max(1,ceil(.10T))`: lambda=0 khi u<W, ngược lại `lambda_max * min(1,(u-W+1)/R)`. AMP overflow không tăng u. Trainer global_step vẫn đếm cả attempted updates theo hành vi framework; vì vậy hai counter được log riêng. ISR warm-up khác warm-up learning rate.

Đây là **cấu hình pilot**, không khẳng định lambda 0.01 tốt nhất. Nếu muốn tune, chạy với `--eval-split validation` và experiment riêng, ví dụ:

```powershell
python -m modal run --detach modal_lora_isr.py --stage train --experiment arc-isr-val-l001 --eval-split validation --sr-lambda 0.01 --baseline-experiment table2-arc-resume --gpu H100
```

Validation run vẫn train trên train split, chỉ đổi split evaluate. Script so sánh không trộn validation với test. Chọn lambda trên validation rồi chốt cấu hình để evaluate test; không chọn best lambda bằng accuracy test. Đổi lambda/PI/warm-up/seed/rank tạo run ID mới. Khi đổi source code cần đổi cả experiment name.

`--sr-lambda 0` là ablation: giữ nguyên LoRA task loss, không chạy auxiliary/PI. Method vẫn mang label ISR để tách riêng thí nghiệm. Không thay đổi các run baseline đã hoàn thành.

## 8. Kết quả, checkpoint, W&B nằm ở đâu?

Modal Volume **`polar-table2-results`**:

```text
arc-lora-isr-v1/
  source_lock.json
  baseline_provenance.json
  dataset_manifest.json
  environment.txt
  modal_changes.diff
  modal_source_sha256.txt
  _smoke/<run_id>/...
  runs/<run_id>/
    run_config.json
    attempt_*.json
    train.log
    eval.log
    sr_modules.json
    sr_metrics.jsonl
    checkpoint-N/
      adapter_model.safetensors
      adapter_config.json
      optimizer.pt
      scheduler.pt
      scaler.pt
      trainer_state.json
      rng_state.pth
      sr_state.pt
      checkpoint_ready.json
    adapter_model.safetensors
    adapter_config.json
    train_complete.json
    eval_results.json
    wandb_targets/...
  table2_arc_comparison.csv
```

Tên adapter file có thể là `.bin` tùy serialization. Checkpoint chỉ được đánh dấu ready sau khi lưu ISR state; parent process theo dõi marker để commit Volume. Nếu bị ngắt giữa lần lưu, runner dùng checkpoint đầy đủ trước đó. Lưu adapter cuối chưa thay thế optimizer/ISR state trong checkpoint.

Base model/tokenizer nằm trong cache Volume `polar-table2-hf-cache`, không upload chúng lên Git hay W&B. Khi chuyển Modal workspace, cần chuyển results Volume và Secret như lần trước; state chỉ ở tài khoản cũ sẽ không tự xuất hiện ở tài khoản mới.

Mở project academic đang dùng:

https://forge.coreweave.com/wandb/thanh-nc2416352-hanoi-university-of-science-and-technology/polar-table2-arc

Hoặc click `W&B run:` trong Modal log. Lọc run tên `lora_isr_r8_arc_easy` / `lora_isr_r8_arc_challenge`, và `is_smoke = false` khi so sánh chính thức.

| Metric | Đọc như thế nào |
|---|---|
| `train/loss` | CE task loss, không cộng auxiliary vào số CE hiển thị |
| `train/sr/lambda` | Lambda thực tế dùng cho update |
| `train/sr/aux_raw` | Tổng inverse stable rank đã skip nhỏ / M |
| `train/sr/aux_weighted` | Lambda × auxiliary raw |
| `train/sr/modules_active` / `skipped_small_norm` | Có bao nhiêu cập nhật đủ lớn để regularize |
| `train/sr/aux_backward_calls` | Số lần backward auxiliary thực tế |
| `train/sr/successful_updates` / `optimizer_skipped` | Schedule và AMP overflow |
| `train/sr/pi_sigma2_ratio_min` | PI numerator / exact sigma_max²; gần 1 là ước lượng tốt |
| `train/sr/exact_stable_rank_mean_all` | Exact stable rank trung bình các module active trong audit toàn bộ 160 modules |
| `diagnostics/stable_rank_mean` | Observer cũ: chỉ 15 modules thuộc blocks 5,15,25; dùng cùng metric này khi so với baseline cũ |
| `eval/acc_norm_pct` | Accuracy chuẩn hóa độ dài, có sau evaluate cuối run |
| `eval/acc_pct` | Accuracy chưa chuẩn hóa |

Audit ISR đo weights trước optimizer update; observer cũ đo sau update ở các bước được chọn. Không coi hai mean là cùng tập ma trận/cùng thời điểm. PI ratio có thể thấp khi singular values gần nhau; tăng `--sr-pi-iters` nếu audit cho thấy ước lượng chưa tốt. Không kết luận method cải thiện accuracy chỉ từ stable rank tăng.

Các metric `compute_seconds` / `backward_wall_seconds` là wall-clock đo từ Python, không phải benchmark GPU kernel chuẩn xác. Accuracy chỉ được evaluate cuối training để giữ protocol cũ. Không có validation accuracy ở mỗi epoch trong pipeline mặc định.

## 9. Commit vào GitHub

Sau khi giải nén và kiểm tra:

```powershell
git add modal_lora_isr.py isr_runner compare_arc_results.py README_LORA_ISR_ARC.md VALIDATION_LORA_ISR.md tests/test_lora_isr.py
git diff --cached --stat
git commit -m "Add inverse stable rank regularization for LoRA on ARC"
git push
```

Chỉ add các file source được liệt kê. Đừng add backup, token, checkpoint, model cache. CSV kết quả có thể commit riêng vào thư mục experiments sau khi chạy xong.

Đã có kiểm thử CPU thật với PyTorch, Trainer, Accelerate và PEFT của upstream; xem `VALIDATION_LORA_ISR.md`. Chưa chạy full Llama-2-7B trên H100 ở môi trường viết code này, vì vậy cần thực hiện smoke trước full run.
