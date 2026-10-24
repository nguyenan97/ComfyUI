# Chạy ComfyUI trên Precision 5550

Profile cân bằng cho Quadro T2000 Max-Q 4 GB, cho phép VieNeu-TTS tiếp tục chạy.

## Khởi động

```powershell
rtk proxy powershell -NoProfile -File D:\Projects\nguyenan97\ComfyUI\run-balanced.ps1
```

Mở http://127.0.0.1:8188. Profile chỉ listen trên loopback. Log và PID nằm trong `user/default/local-run/`.

Workflow mặc định dùng checkpoint đã có `v1-5-pruned-emaonly-fp16.safetensors`, 512×512, batch 1, 24 steps, DPM++ 2M / Karras, CFG 7. Preset cũng được lưu tại `user/default/workflows/balanced-sd15.json`; có thể kéo file này vào canvas nếu trình duyệt đang giữ workflow cũ.

T2000 nằm trong danh sách GPU có FP16 bị lỗi của `comfy/model_management.py`, nên profile dùng compute FP32 dù checkpoint lưu fp16. Low VRAM, reserve 1,5 GB, offload đồng bộ và allocator thông thường được chọn để hạn chế tranh chấp bộ nhớ. VAE dùng FP32 trên GPU với memory management của ComfyUI.

## Tạo ảnh bằng API local

```powershell
rtk proxy D:\Projects\nguyenan97\ComfyUI\.venv\Scripts\python.exe D:\Projects\nguyenan97\ComfyUI\script_examples\local_image.py --steps 24 --prompt "storybook illustration of an orange fox in a flower meadow, soft sunlight, detailed fur"
```

PNG, API graph và kết quả đo lưu trong `output/`. Script cũng xuất preset UI; các lần render tiếp theo nhúng workflow UI vào PNG để kéo lại vào canvas.

## Kiểm thử thực tế ngày 24/10/2026

| Lần chạy | Thời gian qua API | Peak GPU memory.used quan sát |
|---|---:|---:|
| 512×512, 8 steps, CPU VAE | 47,61 giây | 1.705 MiB |
| 512×512, 8 steps, GPU VAE FP32 | 37,61 giây | 1.705 MiB |
| 512×512, 24 steps, GPU VAE FP32 | 94,39 giây | 2.351 MiB |

Hai ảnh thử 8 steps dùng cùng prompt/seed chỉ lệch tối đa 1 mức màu trên 255. Chọn GPU VAE vì bài thử này nhanh hơn khoảng 21% mà giữ kết quả tương đương. Đây là số đo từ các lần chạy cụ thể, không phải chứng minh cấu hình tối ưu trên mọi workflow.

Đã kiểm tra trực quan PNG 8 và 24 steps; ảnh có nội dung, không đen/rỗng. Ảnh 24 steps: `output/balanced_local_00003_.png`. Các process VieNeu không bị dừng; lần 24 steps quan sát thêm một process GPU khác. Peak là tổng memory.used của GPU, sampling khoảng 3 giây, có thể bỏ sót peak ngắn.

Model hiện tại là SD 1.5 nền, nên độ bám prompt và chất lượng minh họa vẫn có giới hạn. Runtime PyTorch 2.10.0+cu128 dùng attention/eager thông thường; các optimized CUDA operations mới của ComfyUI yêu cầu cu130 đang không được dùng. NVIDIA báo effective power limit 20 W tại thời điểm kiểm tra, so với requested/default 40 W. Hiệu năng thực tế còn phụ thuộc công suất và tác vụ dùng chung GPU.

Profile, workflow mặc định và helper được quản lý trên branch `develop`. Venv, model, output và dữ liệu runtime được Git ignore.
