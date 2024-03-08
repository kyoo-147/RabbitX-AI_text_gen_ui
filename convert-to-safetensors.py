'''

Chuyển đổi mô hình máy biến áp sang định dạng an toàn và phân chia nó.

Điều này giúp tải nhanh hơn (vì có bộ bảo vệ an toàn) và giảm mức sử dụng RAM
trong khi tải (vì sharding).

Dựa trên kịch bản gốc của 81300:

https://gist.github.com/81300/fe5b08bff1cba45296a829b9d6b0f303

'''

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54))
parser.add_argument('MODEL', type=str, default=None, nargs='?', help="Đường dẫn đến mô hình đầu vào.")
parser.add_argument('--output', type=str, default=None, help='Đường dẫn đến thư mục đầu ra (mặc định: models/{model_name}_safetensors).')
parser.add_argument("--max-shard-size", type=str, default="2GB", help="Kích thước tối đa của phân đoạn tính bằng GB hoặc MB (mặc định: %(default)s).")
parser.add_argument('--bf16', action='store_true', help='Tải mô hình với độ chính xác bfloat16. Yêu cầu GPU NVIDIA Ampere.')
args = parser.parse_args()

if __name__ == '__main__':
    path = Path(args.MODEL)
    model_name = path.name

    print(f"Đang tải {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(path)

    out_folder = args.output or Path(f"models/{model_name}_safetensors")
    print(f"Lưu mô hình đã chuyển đổi thành {out_folder} với kích thước phân đoạn tối đa là {args.max_shard_size}...")
    model.save_pretrained(out_folder, max_shard_size=args.max_shard_size, safe_serialization=True)
    tokenizer.save_pretrained(out_folder)
