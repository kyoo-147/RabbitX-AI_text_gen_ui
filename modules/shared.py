import argparse
import copy
import os
import sys
from collections import OrderedDict
from pathlib import Path

import yaml

from modules.logging_colors import logger

# Model variables
model = None
tokenizer = None
model_name = 'None'
is_seq2seq = False
model_dirty_from_training = False
lora_names = []

# Generation variables
stop_everything = False
generation_lock = None
processing_message = '*Đang nhập...*'

# UI variables
gradio = {}
persistent_interface_state = {}
need_restart = False

# UI defaults
settings = {
    'dark_theme': True,
    'show_controls': True,
    'start_with': '',
    'mode': 'chat',
    'chat_style': 'cai-chat',
    'prompt-default': 'QA',
    'prompt-notebook': 'QA',
    'preset': 'simple-1',
    'max_new_tokens': 512,
    'max_new_tokens_min': 1,
    'max_new_tokens_max': 4096,
    'negative_prompt': '',
    'seed': -1,
    'truncation_length': 2048,
    'truncation_length_min': 0,
    'truncation_length_max': 200000,
    'max_tokens_second': 0,
    'max_updates_second': 0,
    'prompt_lookup_num_tokens': 0,
    'custom_stopping_strings': '',
    'custom_token_bans': '',
    'auto_max_new_tokens': False,
    'ban_eos_token': False,
    'add_bos_token': True,
    'skip_special_tokens': True,
    'stream': True,
    'character': 'Assistant',
    'name1': 'You',
    'custom_system_message': '',
    'instruction_template_str': "{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not ns.found -%}\n    {{- '' + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + '\\n\\n' -}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- '' + message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{-'### Instruction:\\n' + message['content'] + '\\n\\n'-}}\n        {%- else -%}\n            {{-'### Response:\\n' + message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{-'### Response:\\n'-}}\n{%- endif -%}",
    'chat_template_str': "{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{- name1 + ': ' + message['content'] + '\\n'-}}\n        {%- else -%}\n            {{- name2 + ': ' + message['content'] + '\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}",
    'chat-instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',
    'autoload_model': False,
    'gallery-items_per_page': 50,
    'gallery-open': False,
    'default_extensions': ['gallery'],
}

default_settings = copy.deepcopy(settings)

# Parser copied from https://github.com/vladmandic/automatic
parser = argparse.ArgumentParser(description="RabbitX-AI - All LLMs in one UI - Giao diện cho người dùng cuối sử dụng tất cả các LLMs mà họ muốn", conflict_handler='resolve', add_help=True, formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=55, indent_increment=2, width=200))

# Basic settings
group = parser.add_argument_group('Cấu hình cơ bản')
group.add_argument('--multi-user', action='store_true', help='Chế độ nhiều người dùng. Lịch sử trò chuyện không được lưu hoặc tải tự động. Cảnh báo: điều này có thể không an toàn để chia sẻ công khai.')
group.add_argument('--character', type=str, help='Tên nhân vật sẽ tải trong chế độ trò chuyện theo mặc định.')
group.add_argument('--model', type=str, help='Tên của mô hình sẽ tải theo mặc định.')
group.add_argument('--lora', type=str, nargs='+', help='Danh sách LoRA cần tải. Nếu bạn muốn tải nhiều LoRA, hãy viết các tên cách nhau bằng dấu cách.')
group.add_argument('--model-dir', type=str, default='models/', help='Đường dẫn đến thư mục chứa tất cả các model.')
group.add_argument('--lora-dir', type=str, default='loras/', help='Đường dẫn đến thư mục với tất cả các loras.')
group.add_argument('--model-menu', action='store_true', help='Hiển thị menu mô hình trong thiết bị đầu cuối khi giao diện người dùng web được khởi chạy lần đầu tiên.')
group.add_argument('--settings', type=str, help='Tải cài đặt giao diện mặc định từ tệp yaml này. Xem settings-template.yaml để biết ví dụ. Nếu bạn tạo một tệp có tên settings.yaml, tệp này sẽ được tải theo mặc định mà không cần sử dụng cờ --settings.')
group.add_argument('--extensions', type=str, nargs='+', help='Danh sách các tiện ích mở rộng cần tải. Nếu bạn muốn tải nhiều tiện ích mở rộng, hãy viết các tên cách nhau bằng dấu cách.')
group.add_argument('--verbose', action='store_true', help='In lời nhắc đến thiết bị đầu cuối.')
group.add_argument('--chat-buttons', action='store_true', help='Hiển thị các nút trên tab trò chuyện thay vì menu di chuột.')

# Model loader
group = parser.add_argument_group('Trình tải mô hình')
group.add_argument('--loader', type=str, help='Chọn trình tải mô hình theo cách thủ công, nếu không, nó sẽ được tự động phát hiện. Các tùy chọn hợp lệ: Transformers, llama.cpp, llamacpp_HF, ExLlamav2_HF, ExLlamav2, AutoGPTQ, AutoAWQ, GPTQ-for-LLaMa, ctransformers, QuIP#.')

# Transformers/Accelerate
group = parser.add_argument_group('Transformers/Accelerate')
group.add_argument('--cpu', action='store_true', help='Sử dụng CPU để tạo văn bản. Cảnh báo: Quá trình đào tạo trên CPU cực kỳ chậm.')
group.add_argument('--auto-devices', action='store_true', help='Tự động phân chia mô hình theo (các) GPU và CPU có sẵn.')
group.add_argument('--gpu-memory', type=str, nargs='+', help='Bộ nhớ GPU tối đa tính bằng GiB sẽ được phân bổ cho mỗi GPU. Ví dụ: --gpu-memory 10 cho một GPU, --gpu-memory 10 5 cho hai GPU. Bạn cũng có thể đặt các giá trị trong MiB như --gpu-memory 3500MiB.')
group.add_argument('--cpu-memory', type=str, help='Bộ nhớ CPU tối đa tính bằng GiB để phân bổ cho trọng lượng được giảm tải. Giống như trên.')
group.add_argument('--disk', action='store_true', help='Nếu mô hình quá lớn so với (các) GPU và CPU của bạn cộng lại, hãy gửi các lớp còn lại vào đĩa.')
group.add_argument('--disk-cache-dir', type=str, default='cache', help='Thư mục để lưu bộ đệm đĩa vào. Mặc định là "bộ đệm".')
group.add_argument('--load-in-8bit', action='store_true', help='Tải mô hình với độ chính xác 8 bit (sử dụng bitandbyte).')
group.add_argument('--bf16', action='store_true', help='Tải mô hình với độ chính xác bfloat16. Yêu cầu GPU NVIDIA Ampere.')
group.add_argument('--no-cache', action='store_true', help='Đặt use_cache thành Sai trong khi tạo văn bản. Điều này làm giảm mức sử dụng VRAM một chút nhưng lại làm giảm hiệu năng.')
group.add_argument('--trust-remote-code', action='store_true', help='Đặt Trust_remote_code=True trong khi tải mô hình. Cần thiết cho một số mô hình.')
group.add_argument('--force-safetensors', action='store_true', help='Đặt use_safetensors=True trong khi tải mô hình. Điều này ngăn chặn việc thực thi mã tùy ý.')
group.add_argument('--no_use_fast', action='store_true', help='Đặt use_fast=False trong khi tải mã thông báo (theo mặc định là True). Sử dụng cái này nếu bạn gặp bất kỳ vấn đề nào liên quan đến use_fast.')
group.add_argument('--use_flash_attention_2', action='store_true', help='Đặt use_flash_attention_2=True trong khi tải mô hình.')

# bitsandbytes 4-bit
group = parser.add_argument_group('bitsandbytes 4-bit')
group.add_argument('--load-in-4bit', action='store_true', help='Tải mô hình với độ chính xác 4 bit (sử dụng bitandbyte).')
group.add_argument('--use_double_quant', action='store_true', help='use_double_quant cho 4-bit.')
group.add_argument('--compute_dtype', type=str, default='float16', help='tính toán dtype cho 4-bit. Các tùy chọn hợp lệ: bfloat16, float16, float32.')
group.add_argument('--quant_type', type=str, default='nf4', help='quant_type cho 4-bit. Các tùy chọn hợp lệ: nf4, fp4.')

# llama.cpp
group = parser.add_argument_group('llama.cpp')
group.add_argument('--tensorcores', action='store_true', help='Sử dụng llama-cpp-python được biên dịch với sự hỗ trợ của lõi tensor. Điều này làm tăng hiệu suất trên thẻ RTX. chỉ NVIDIA.')
group.add_argument('--n_ctx', type=int, default=2048, help='Kích thước của bối cảnh lời nhắc.')
group.add_argument('--threads', type=int, default=0, help='Số lượng chủ đề sử dụng.')
group.add_argument('--threads-batch', type=int, default=0, help='Số lượng chủ đề được sử dụng để xử lý hàng loạt/nhắc nhở.')
group.add_argument('--no_mul_mat_q', action='store_true', help='Vô hiệu hóa hạt nhân mulmat.')
group.add_argument('--n_batch', type=int, default=512, help='Số lượng mã thông báo nhắc nhở tối đa được gộp lại với nhau khi gọi llama_eval.')
group.add_argument('--no-mmap', action='store_true', help='Ngăn không cho mmap được sử dụng.')
group.add_argument('--mlock', action='store_true', help='Buộc hệ thống giữ mô hình trong RAM.')
group.add_argument('--n-gpu-layers', type=int, default=0, help='Số lớp cần tải xuống GPU.')
group.add_argument('--tensor_split', type=str, default=None, help='Chia mô hình trên nhiều GPU. Danh sách tỷ lệ được phân tách bằng dấu phẩy. Ví dụ: 18,17.')
group.add_argument('--numa', action='store_true', help='Kích hoạt phân bổ nhiệm vụ NUMA cho llama.cpp.')
group.add_argument('--logits_all', action='store_true', help='Cần phải thiết lập để việc đánh giá mức độ bối rối có thể hoạt động. Nếu không, hãy bỏ qua nó, vì nó làm cho quá trình xử lý nhanh chóng chậm hơn.')
group.add_argument('--no_offload_kqv', action='store_true', help='Không giảm tải K, Q, V cho GPU. Điều này tiết kiệm VRAM nhưng làm giảm hiệu suất.')
group.add_argument('--cache-capacity', type=str, help='Dung lượng bộ đệm tối đa (llama-cpp-python). Ví dụ: 2000MiB, 2GiB. Khi được cung cấp không có đơn vị, byte sẽ được coi là.')
group.add_argument('--row_split', action='store_true', help='Chia mô hình theo hàng trên GPU. Điều này có thể cải thiện hiệu suất đa GPU.')

# ExLlamaV2
group = parser.add_argument_group('ExLlamaV2')
group.add_argument('--gpu-split', type=str, help='Danh sách VRAM (tính bằng GB) được phân tách bằng dấu phẩy để sử dụng cho mỗi thiết bị GPU cho các lớp mô hình. Ví dụ: 20,7,7.')
group.add_argument('--autosplit', action='store_true', help='Tự động phân chia các tensor mô hình trên các GPU có sẵn. Điều này khiến --gpu-split bị bỏ qua.')
group.add_argument('--max_seq_len', type=int, default=2048, help='Độ dài chuỗi tối đa.')
group.add_argument('--cfg-cache', action='store_true', help='ExLlamav2_HF: Tạo bộ đệm bổ sung cho lời nhắc tiêu cực CFG. Cần thiết phải sử dụng CFG với trình tải đó.')
group.add_argument('--no_flash_attn', action='store_true', help='Buộc không sử dụng tính năng chú ý flash.')
group.add_argument('--cache_8bit', action='store_true', help='Sử dụng bộ đệm 8 bit để tiết kiệm VRAM.')
group.add_argument('--num_experts_per_token', type=int, default=2, help='Số lượng chuyên gia để sử dụng cho thế hệ. Áp dụng cho các dòng MoE như Mixtral.')

# AutoGPTQ
group = parser.add_argument_group('AutoGPTQ')
group.add_argument('--triton', action='store_true', help='Sử dụng triton.')
group.add_argument('--no_inject_fused_attention', action='store_true', help='Vô hiệu hóa việc sử dụng sự chú ý hợp nhất, điều này sẽ sử dụng ít VRAM hơn nhưng phải trả giá bằng suy luận chậm hơn.')
group.add_argument('--no_inject_fused_mlp', action='store_true', help='Chỉ chế độ Triton: vô hiệu hóa việc sử dụng MLP hợp nhất, điều này sẽ sử dụng ít VRAM hơn với chi phí suy luận chậm hơn.')
group.add_argument('--no_use_cuda_fp16', action='store_true', help='Điều này có thể làm cho mô hình nhanh hơn trên một số hệ thống.')
group.add_argument('--desc_act', action='store_true', help='Đối với các mô hình không có quantize_config.json, tham số này được sử dụng để xác định xem có đặt desc_act hay không trong BaseQuantizeConfig.')
group.add_argument('--disable_exllama', action='store_true', help='Vô hiệu hóa kernel ExLlama, có thể cải thiện tốc độ suy luận trên một số hệ thống.')
group.add_argument('--disable_exllamav2', action='store_true', help='Vô hiệu hóa hạt nhân ExLlamav2.')

# GPTQ-for-LLaMa
group = parser.add_argument_group('GPTQ-for-LLaMa')
group.add_argument('--wbits', type=int, default=0, help='Tải mô hình được lượng tử hóa trước với độ chính xác được chỉ định tính bằng bit. 2, 3, 4 và 8 được hỗ trợ.')
group.add_argument('--model_type', type=str, help='Loại mô hình của mô hình tiền lượng tử hóa. Hiện tại LLaMA, OPT và GPT-J được hỗ trợ.')
group.add_argument('--groupsize', type=int, default=-1, help='Kích thước nhóm.')
group.add_argument('--pre_layer', type=int, nargs='+', help='Số lượng lớp để phân bổ cho GPU. Việc đặt tham số này sẽ cho phép giảm tải CPU cho các kiểu máy 4 bit. Đối với nhiều gpu, hãy viết các số cách nhau bằng dấu cách, ví dụ --pre_layer 30 60.')
group.add_argument('--checkpoint', type=str, help='Đường dẫn đến tệp điểm kiểm tra lượng tử hóa. Nếu không được chỉ định, nó sẽ được tự động phát hiện.')
group.add_argument('--monkey-patch', action='store_true', help='Áp dụng bản vá khỉ để sử dụng LoRA với các mô hình lượng tử hóa.')

# HQQ
group = parser.add_argument_group('HQQ')
group.add_argument('--hqq-backend', type=str, default='PYTORCH_COMPILE', help='Phần phụ trợ cho trình tải HQQ. Tùy chọn hợp lệ: PYTORCH, PYTORCH_COMPILE, ATEN.')

# DeepSpeed
group = parser.add_argument_group('DeepSpeed')
group.add_argument('--deepspeed', action='store_true', help='Cho phép sử dụng DeepSpeed ZeRO-3 để suy luận thông qua tích hợp Transformers.')
group.add_argument('--nvme-offload-dir', type=str, help='DeepSpeed: Thư mục sử dụng để giảm tải ZeRO-3 NVME.')
group.add_argument('--local_rank', type=int, default=0, help='DeepSpeed: Đối số tùy chọn cho thiết lập phân tán.')

# RoPE
group = parser.add_argument_group('RoPE')
group.add_argument('--alpha_value', type=float, default=1, help='Hệ số alpha nhúng theo vị trí để chia tỷ lệ NTK RoPE. Sử dụng cái này hoặc nén_pos_emb, không phải cả hai.')
group.add_argument('--rope_freq_base', type=int, default=0, help='Nếu lớn hơn 0, sẽ được sử dụng thay cho alpha_value. Hai cái đó có liên quan bởi wire_freq_base = 10000 * alpha_value ^ (64 / 63).')
group.add_argument('--compress_pos_emb', type=int, default=1, help="Hệ số nén nhúng vị trí. Nên đặt thành (độ dài ngữ cảnh) / (độ dài ngữ cảnh ban đầu của mô hình). Bằng 1/rope_freq_scale.")

# Gradio
group = parser.add_argument_group('Gradio')
group.add_argument('--listen', action='store_true', help='Làm cho giao diện người dùng web có thể truy cập được từ mạng cục bộ của bạn.')
group.add_argument('--listen-port', type=int, help='Cổng nghe mà máy chủ sẽ sử dụng.')
group.add_argument('--listen-host', type=str, help='Tên máy chủ mà máy chủ sẽ sử dụng.')
group.add_argument('--share', action='store_true', help='Tạo một URL công khai. Điều này hữu ích để chạy giao diện người dùng web trên Google Colab hoặc tương tự.')
group.add_argument('--auto-launch', action='store_true', default=False, help='Mở giao diện người dùng web trong trình duyệt mặc định khi khởi chạy.')
group.add_argument('--gradio-auth', type=str, help='Đặt mật khẩu xác thực Gradio ở định dạng "tên người dùng:mật khẩu". Nhiều thông tin xác thực cũng có thể được cung cấp cùng với "u1:p1,u2:p2,u3:p3".', default=None)
group.add_argument('--gradio-auth-path', type=str, help='Đặt đường dẫn tệp xác thực Gradio. Tệp phải chứa một hoặc nhiều cặp người dùng:mật khẩu có cùng định dạng như trên.', default=None)
group.add_argument('--ssl-keyfile', type=str, help='Đường dẫn đến tệp khóa chứng chỉ SSL.', default=None)
group.add_argument('--ssl-certfile', type=str, help='Đường dẫn đến tệp chứng chỉ chứng chỉ SSL.', default=None)

# API
group = parser.add_argument_group('API')
group.add_argument('--api', action='store_true', help='Kích hoạt tiện ích mở rộng API.')
group.add_argument('--public-api', action='store_true', help='Tạo URL công khai cho API bằng Cloudflare.')
group.add_argument('--public-api-id', type=str, help='ID đường hầm cho Đường hầm Cloudflare có tên. Sử dụng cùng với tùy chọn public-api.', default=None)
group.add_argument('--api-port', type=int, default=5000, help='Cổng nghe cho API.')
group.add_argument('--api-key', type=str, default='', help='Khóa xác thực API.')
group.add_argument('--admin-key', type=str, default='', help='Khóa xác thực API cho các tác vụ của quản trị viên như tải và dỡ mô hình. Nếu không được đặt, sẽ giống như --api-key.')
group.add_argument('--nowebui', action='store_true', help='Không khởi chạy giao diện người dùng Gradio. Hữu ích khi khởi chạy API ở chế độ độc lập.')

# Multimodal
group = parser.add_argument_group('Multimodal')
group.add_argument('--multimodal-pipeline', type=str, default=None, help='Đường ống đa mô hình được sử dụng. Ví dụ: llava-7b, llava-13b.')

# Deprecated parameters
# group = parser.add_argument_group('Deprecated')

args = parser.parse_args()
args_defaults = parser.parse_args([])
provided_arguments = []
for arg in sys.argv[1:]:
    arg = arg.lstrip('-').replace('-', '_')
    if hasattr(args, arg):
        provided_arguments.append(arg)

deprecated_args = []


def do_cmd_flags_warnings():

    # Deprecation warnings
    for k in deprecated_args:
        if getattr(args, k):
            logger.warning(f'Cờ --{k} không được dùng nữa và sẽ sớm bị xóa. Làm ơn hãy xóa cờ đó.')

    # Security warnings
    if args.trust_remote_code:
        logger.warning('trust_remote_code được cho phép. Điều này rất nguy hiểm.')
    if 'COLAB_GPU' not in os.environ and not args.nowebui:
        if args.share:
            logger.warning("Tính năng gradio \"share link\" sử dụng tệp thực thi độc quyền để tạo đường hầm ngược. Sử dụng nó một cách cẩn thận.")
        if any((args.listen, args.share)) and not any((args.gradio_auth, args.gradio_auth_path)):
            logger.warning("\nBạn có khả năng hiển thị giao diện người dùng web trên toàn bộ Internet mà không có bất kỳ mật khẩu truy cập nào.\nBạn có thể tạo một mật khẩu có cờ \"--gradio-auth\" như thế này:\n\n--gradio-auth tên người dùng:password\n \nĐảm bảo thay thế tên người dùng:mật khẩu bằng tên riêng của bạn.")
            if args.multi_user:
                logger.warning('\nChế độ nhiều người dùng mang tính thử nghiệm cao và không nên chia sẻ công khai.')


def fix_loader_name(name):
    if not name:
        return name

    name = name.lower()
    if name in ['llamacpp', 'llama.cpp', 'llama-cpp', 'llama cpp']:
        return 'llama.cpp'
    if name in ['llamacpp_hf', 'llama.cpp_hf', 'llama-cpp-hf', 'llamacpp-hf', 'llama.cpp-hf']:
        return 'llamacpp_HF'
    elif name in ['transformers', 'huggingface', 'hf', 'hugging_face', 'hugging face']:
        return 'Transformers'
    elif name in ['autogptq', 'auto-gptq', 'auto_gptq', 'auto gptq']:
        return 'AutoGPTQ'
    elif name in ['gptq-for-llama', 'gptqforllama', 'gptqllama', 'gptq for llama', 'gptq_for_llama']:
        return 'GPTQ-for-LLaMa'
    elif name in ['exllama', 'ex-llama', 'ex_llama', 'exlama']:
        return 'ExLlama'
    elif name in ['exllamav2', 'exllama-v2', 'ex_llama-v2', 'exlamav2', 'exlama-v2', 'exllama2', 'exllama-2']:
        return 'ExLlamav2'
    elif name in ['exllamav2-hf', 'exllamav2_hf', 'exllama-v2-hf', 'exllama_v2_hf', 'exllama-v2_hf', 'exllama2-hf', 'exllama2_hf', 'exllama-2-hf', 'exllama_2_hf', 'exllama-2_hf']:
        return 'ExLlamav2_HF'
    elif name in ['ctransformers', 'ctranforemrs', 'ctransformer']:
        return 'ctransformers'
    elif name in ['autoawq', 'awq', 'auto-awq']:
        return 'AutoAWQ'
    elif name in ['quip#', 'quip-sharp', 'quipsharp', 'quip_sharp']:
        return 'QuIP#'
    elif name in ['hqq']:
        return 'HQQ'


def add_extension(name, last=False):
    if args.extensions is None:
        args.extensions = [name]
    elif last:
        args.extensions = [x for x in args.extensions if x != name]
        args.extensions.append(name)
    elif name not in args.extensions:
        args.extensions.append(name)


def is_chat():
    return True


def load_user_config():
    '''
    Loads custom model-specific settings
    '''
    if Path(f'{args.model_dir}/config-user.yaml').exists():
        file_content = open(f'{args.model_dir}/config-user.yaml', 'r').read().strip()

        if file_content:
            user_config = yaml.safe_load(file_content)
        else:
            user_config = {}
    else:
        user_config = {}

    return user_config


args.loader = fix_loader_name(args.loader)

# Activate the multimodal extension
if args.multimodal_pipeline is not None:
    add_extension('multimodal')

# Activate the API extension
if args.api or args.public_api:
    add_extension('openai', last=True)

# Load model-specific settings
with Path(f'{args.model_dir}/config.yaml') as p:
    if p.exists():
        model_config = yaml.safe_load(open(p, 'r').read())
    else:
        model_config = {}

# Load custom model-specific settings
user_config = load_user_config()

model_config = OrderedDict(model_config)
user_config = OrderedDict(user_config)
