## LLaVA pipeline

Mô-đun này cung cấp 2 đường ống:
- `llava-7b` - để sử dụng với model LLaVA v0 7B (LLaMa 7B đã được tinh chỉnh)
- `llava-13b` - để sử dụng với model LLaVA v0 13B (LLaMa 13B đã được tinh chỉnh)

[LLaVA](https://github.com/haotian-liu/LLaVA) sử dụng CLIP `openai/clip-vit-large-patch14` làm mô hình tầm nhìn và sau đó là một lớp tuyến tính duy nhất. Đối với 13B, trọng lượng máy chiếu nằm trong `liuhaotian/LLaVA-13b-delta-v0`, và đối với 7B họ đang ở `liuhaotian/LLaVA-7b-delta-v0`.

Các kết hợp tham số được hỗ trợ cho cả kiểu máy thị giác và máy chiếu là: CUDA/32bit, CUDA/16bit, CPU/32bit
