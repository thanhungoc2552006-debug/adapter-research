# Thêm LoRA auxiliary method

## Giao diện hiện có

Registry quét các module/package trong `adapter_research/methods/` và lấy biến `SPEC: MethodSpec`. Không sửa danh sách method trong Modal worker hoặc reporter.

```powershell
python -m adapter_research new-method --name my_method
python -m adapter_research methods
```

`__init__.py` chứa metadata nhẹ: id, label, PEFT backend, learning rate, defaults, validator, controller factory, source modules và factorization cho reporter. Không import torch/controller tại đây vì CPU reporter và local planner cũng import metadata.

Controller được import trễ khi tạo Trainer. Starter kế thừa `SRController`, có các thành phần sau:

| Thành phần | Khi được gọi / nhiệm vụ |
|---|---|
| `factors` | Danh sách A/B được chọn, dùng kiểm tra chỉ adapter được train |
| `configure(max_steps)` | Train bắt đầu; xác định lịch warm-up/ramp, kiểm tra tổng số update khi resume |
| `weighted_loss()` | Một lần tại ranh giới gradient accumulation, trước unscale/clipping; trả scalar có graph đã nhân λ hoặc `None` |
| `after_optimizer(skipped)` | Sau optimizer step; chỉ thay đổi tham số/counter thành công khi AMP không skip |
| `metrics` | Dictionary các scalar log; dùng prefix riêng cho method |
| `aux_backward_calls` | Counter do Trainer cập nhật khi backward auxiliary |
| `state_dict()` / `load_state_dict()` | Counter, vector PI, generator state và tham số cần thiết để resume |

Không tự gọi `backward()`, `optimizer.step()` hoặc scale loss theo số microbatch trong controller. Shared Trainer đảm nhiệm accumulation/AMP. `weighted_loss()` độc lập dữ liệu đầu vào; một loss cần activation của từng batch sẽ cần mở rộng giao diện Trainer, không chỉ thay controller này.

## Sửa toán học

Sửa `methods/my_method/controller.py`. Starter ban đầu gọi `super().weighted_loss()` nên chưa có toán học mới. Giữ warm-up, bảo vệ zero norm và state PI nếu vẫn dùng ISR. Khi thay logic, thêm phép kiểm tra gradient/zero/resume phù hợp.

Nếu cần hyperparameter mới:

1. Thêm `config.py` chứa dataclass/validator cho method mới.
2. Thêm khóa vào `SPEC.defaults` và đổi `SPEC.validate_parameters` thành đường dẫn validator mới.
3. Sửa factory để tạo đúng config/controller.
4. Thêm mọi module ảnh hưởng toán học vào `SPEC.source_modules`.
5. Nếu muốn hiện giá trị trên bảng, thêm cặp `(config_key, display_name)` vào `label_parameters`.

Tên tham số không có trong defaults bị từ chối để phát hiện lỗi gõ như `lamda`. Không bỏ validator chỉ để tránh lỗi cấu hình.

`factorization='BA'` chỉ đúng nếu ΔW vẫn là γBA. Các backend hiện có là `lora`, `dora`, `polar`, với reporter BA/BSA/unavailable. Method thêm loại adapter/trọng số mới cần implement PEFT backend và phép đo ΔW tương ứng; scaffold chỉ dành cho LoRA auxiliary/post-update methods.

## Chạy

Sửa `configs/arc_my_method_r8.json`. Các trường không ghi sẽ kế thừa mặc định, xem `adapter_research/config.py`. Đặt `experiment` mới cho một họ thử nghiệm. Ví dụ chọn một λ khác:

```json
{
  "experiment": "arc-my-method-v2",
  "tasks": ["arc_challenge"],
  "ranks": [8],
  "seeds": [0],
  "methods": [{"name": "my_method", "parameters": {"lambda_max": 0.1}}]
}
```

```powershell
python -m adapter_research plan --config configs/arc_my_method_r8.json
python -m modal run --detach modal_pipeline.py --stage smoke --config configs/arc_my_method_r8.json
python -m modal run --detach modal_pipeline.py --stage train --config configs/arc_my_method_r8.json
```

Chờ smoke thành công rồi chạy train. Không cần chạy stage setup riêng: smoke/train đã gọi CPU setup. Train kết thúc tự tạo/cập nhật cùng bảng so sánh W&B.

Model/dataset revision, protocol, method parameters, source chung và source method được ghi vào run identity. Thêm sibling method hoặc sửa báo cáo không đổi identity của method cũ. Sửa code training chung buộc dùng experiment mới. Sửa source method/λ tạo run ID mới; checkpoint khác identity không được nạp lẫn.

Với validation: đặt `training.eval_split="validation"` và `comparison.after_train=false`. Bảng báo cáo chính chỉ nhận full test evaluation. Không chọn hyperparameter dựa trên test accuracy.

## Fixed-norm hiện tại

`fixednorm/controller.py` kế thừa ISR, chỉ thêm hook sau AdamW. Khi không AMP skip và norm đủ lớn, `projection.py` đo d=‖γBA‖F rồi nhân B với ρ/d; A và AdamW moments giữ nguyên. Mục tiêu áp dụng cho mỗi ΔW, không phải norm của B hoặc toàn model. B khởi tạo zero chưa thể chiếu lên norm dương; phép chiếu bắt đầu sau update tạo ΔW đủ lớn.

Rescale vô hướng tự nó không đổi stable rank. ISR thay hình dạng phổ; chiếu giữ Frobenius norm để nghiên cứu spectral norm dưới ràng buộc đó. Đây là post-update rescale, không phải Riemannian AdamW. Dùng config λ=0/rho=1 để kiểm tra riêng ảnh hưởng của rescale.
