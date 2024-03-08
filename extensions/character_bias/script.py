import os

import gradio as gr

# get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# check if the bias_options.txt file exists, if not, create it
bias_file = os.path.join(current_dir, "bias_options.txt")
if not os.path.isfile(bias_file):
    with open(bias_file, "w") as f:
        f.write("*Tôi rất hạnh phúc*\n*Tôi rất bu*\n*Tôi rất hứng thú*\n*Tôi rất chán*\n*Tôi rất tức giận*")

# read bias options from the text file
with open(bias_file, "r") as f:
    bias_options = [line.strip() for line in f.readlines()]

params = {
    "activate": True,
    "bias string": " *Tôi rất hạnh phúc*",
    "custom string": "",
}


def input_modifier(string):
    """
    Chức năng này được áp dụng cho kiểu nhập văn bản của bạn trước
    chúng được đưa vào mô hình.
    """
    return string


def output_modifier(string):
    """
    Chức năng này được áp dụng cho đầu ra của mô hình.
    """
    return string


def bot_prefix_modifier(string):
    """
    Chức năng này chỉ được áp dụng trong chế độ trò chuyện. Nó sửa đổi
    văn bản tiền tố cho Bot và có thể được sử dụng để định hướng
    hành vi.
    """
    if params['activate']:
        if params['custom string'].strip() != '':
            return f'{string} {params["custom string"].strip()} '
        else:
            return f'{string} {params["bias string"].strip()} '
    else:
        return string


def ui():
    # Gradio elements
    activate = gr.Checkbox(value=params['activate'], label='Kích hoạt tính cách yêu thích')
    dropdown_string = gr.Dropdown(choices=bias_options, value=params["bias string"], label='Tính cách yêu thích', info='Để chỉnh sửa các tùy chọn trong danh sách thả xuống này, hãy chỉnh sửa tệp "bias_options.txt"')
    custom_string = gr.Textbox(value=params['custom string'], placeholder="Nhập chuỗi nhân vật yêu thích", label="Tùy chỉnh nhân vật yêu thích", info='Nếu không trống, sẽ được sử dụng thay cho giá trị trên')

    # Event functions to update the parameters in the backend
    def update_bias_string(x):
        if x:
            params.update({"bias string": x})
        else:
            params.update({"bias string": dropdown_string.get()})
        return x

    def update_custom_string(x):
        params.update({"custom string": x})

    dropdown_string.change(update_bias_string, dropdown_string, None)
    custom_string.change(update_custom_string, custom_string, None)
    activate.change(lambda x: params.update({"activate": x}), activate, None)
