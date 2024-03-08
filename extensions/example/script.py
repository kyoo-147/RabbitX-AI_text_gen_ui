"""
Một ví dụ về phần mở rộng. Nó không làm gì cả, nhưng bạn có thể thêm các phép biến đổi
trước các câu lệnh return để tùy chỉnh hành vi webui.

Bắt đầu từ history_modifier và kết thúc bằng out_modifier,
các hàm được khai báo theo thứ tự như khi chúng được gọi
thời gian thế hệ.
"""

import gradio as gr
import torch
from transformers import LogitsProcessor

from modules import chat, shared
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
)

params = {
    "display_name": "Example Extension",
    "is_tab": False,
}

class MyLogits(LogitsProcessor):
    """
    Thao tác xác suất cho mã thông báo tiếp theo trước khi nó được lấy mẫu.
    Được sử dụng trong hàm logits_processor_modifier bên dưới.
    """
    def __init__(self):
        pass

    def __call__(self, input_ids, scores):
        # probs = torch.softmax(scores, dim=-1, dtype=torch.float)
        # probs[0] /= probs[0].sum()
        # scores = torch.log(probs / (1 - probs))
        return scores

def history_modifier(history):
    """
    Modifies the chat history.
    Only used in chat mode.
    """
    return history

def state_modifier(state):
    """
    Sửa đổi biến trạng thái, là một từ điển chứa dữ liệu đầu vào
    các giá trị trong giao diện người dùng như thanh trượt và hộp kiểm.
    """
    return state

def chat_input_modifier(text, visible_text, state):
    """
    Sửa đổi chuỗi đầu vào của người dùng trong chế độ trò chuyện (visible_text).
    Bạn cũng có thể sửa đổi cách thể hiện nội bộ của người dùng
    đầu vào (văn bản) để thay đổi cách nó xuất hiện trong lời nhắc.
    """
    return text, visible_text

def input_modifier(string, state, is_chat=False):
    """
    Ở chế độ mặc định/sổ tay, sửa đổi toàn bộ lời nhắc.

    Trong chế độ trò chuyện, nó giống như chat_input_modifier nhưng chỉ được áp dụng
    thành "văn bản", ở đây gọi là "chuỗi" chứ không phải "visible_text".
    """
    return string

def bot_prefix_modifier(string, state):
    """
    Sửa đổi tiền tố cho câu trả lời tiếp theo của bot trong chế độ trò chuyện.
    Theo mặc định, tiền tố sẽ có dạng như "Tên Bot:".
    """
    return string

def tokenizer_modifier(state, prompt, input_ids, input_embeds):
    """
    Sửa đổi id đầu vào và nội dung nhúng.
    Được tiện ích mở rộng đa phương thức sử dụng để nhúng hình ảnh vào lời nhắc.
    Chỉ được sử dụng bởi các bộ tải sử dụng thư viện máy biến áp để lấy mẫu.
    """
    return prompt, input_ids, input_embeds

def logits_processor_modifier(processor_list, input_ids):
    """
    Thêm bộ xử lý nhật ký vào danh sách, cho phép bạn truy cập và sửa đổi
    xác suất mã thông báo tiếp theo.
    Chỉ được sử dụng bởi các bộ tải sử dụng thư viện máy biến áp để lấy mẫu.
    """
    processor_list.append(MyLogits())
    return processor_list

def output_modifier(string, state, is_chat=False):
    """
    Sửa đổi đầu ra LLM trước khi nó được trình bày.

    Trong chế độ trò chuyện, phiên bản đã sửa đổi sẽ đi vào lịch sử['visible'],
    và phiên bản gốc đi vào lịch sử['internal'].
    """
    return string

def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Thay thế chức năng tạo lời nhắc từ lịch sử trò chuyện.
    Chỉ được sử dụng trong chế độ trò chuyện.  
    """
    result = chat.generate_chat_prompt(user_input, state, **kwargs)
    return result

def custom_css():
    """
    Trả về một chuỗi CSS được thêm vào CSS cho webui.
    """
    return ''

def custom_js():
    """
    Trả về một chuỗi javascript được thêm vào javascript
    cho webui.
    """
    return ''

def setup():
    """
    Chỉ được thực thi một lần khi tiện ích mở rộng được nhập.
    """
    pass

def ui():
    """
    Được thực thi khi UI được vẽ. Các phần tử gradient tùy chỉnh và
    trình xử lý sự kiện tương ứng của chúng phải được xác định ở đây.

    Để tìm hiểu về các thành phần gradio, hãy xem tài liệu:
    https://gradio.app/docs/
    """
    pass
