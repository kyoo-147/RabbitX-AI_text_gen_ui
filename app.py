import os
import warnings

from modules import shared

import accelerate  # Việc nhập sớm này khiến GPU Intel vui mừng

import modules.one_click_installer_check
from modules.block_requests import OpenMonkeyPatch, RequestBlocker
from modules.logging_colors import logger

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage không được dùng nữa')
warnings.filterwarnings('ignore', category=UserWarning, message='Sử dụng phương pháp cập nhật không được dùng nữa')
warnings.filterwarnings('ignore', category=UserWarning, message='Trường "model_name" có xung đột')
warnings.filterwarnings('ignore', category=UserWarning, message='Giá trị được truyền vào gr.Dropdown()')
warnings.filterwarnings('ignore', category=UserWarning, message='Trường "model_names" có xung đột')

with RequestBlocker():
    import gradio as gr

import matplotlib

matplotlib.use('Agg')  # Điều này sửa lỗi hiển thị LaTeX trên một số hệ thống

import json
import os
import signal
import sys
import time
from functools import partial
from pathlib import Path
from threading import Lock

import yaml

import modules.extensions as extensions_module
from modules import (
    chat,
    training,
    ui,
    ui_chat,
    ui_default,
    ui_file_saving,
    ui_model_menu,
    ui_notebook,
    ui_parameters,
    ui_session,
    utils
)
from modules.extensions import apply_extensions
from modules.LoRA import add_lora_to_model
from modules.models import load_model
from modules.models_settings import (
    get_fallback_settings,
    get_model_metadata,
    update_model_parameters
)
from modules.shared import do_cmd_flags_warnings
from modules.utils import gradio


def signal_handler(sig, frame):
    logger.info("Đã nhận được Ctrl + C. Chương trình sẽ được thoát trong giấy lát.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def create_interface():

    title = 'RabbitX-AI - All LLMs in one UI - Giao diện cho người dùng cuối sử dụng tất cả các LLMs mà họ muốn'

    # Xác thực mật khẩu
    auth = []
    if shared.args.gradio_auth:
        auth.extend(x.strip() for x in shared.args.gradio_auth.strip('"').replace('\n', '').split(',') if x.strip())
    if shared.args.gradio_auth_path:
        with open(shared.args.gradio_auth_path, 'r', encoding="utf8") as file:
            auth.extend(x.strip() for line in file for x in line.split(',') if x.strip())
    auth = [tuple(cred.split(':')) for cred in auth]

    # Nhập các tiện ích mở rộng và thực thi các hàm setup() của chúng
    if shared.args.extensions is not None and len(shared.args.extensions) > 0:
        extensions_module.load_extensions()

    # Buộc một số sự kiện được kích hoạt khi tải trang
    shared.persistent_interface_state.update({
        'loader': shared.args.loader or 'Transformers',
        'mode': shared.settings['mode'],
        'character_menu': shared.args.character or shared.settings['character'],
        'instruction_template_str': shared.settings['instruction_template_str'],
        'prompt_menu-default': shared.settings['prompt-default'],
        'prompt_menu-notebook': shared.settings['prompt-notebook'],
        'filter_by_loader': shared.args.loader or 'All'
    })

    if Path("cache/pfp_character.png").exists():
        Path("cache/pfp_character.png").unlink()

    # chuỗi css/js
    css = ui.css
    js = ui.js
    css += apply_extensions('css')
    js += apply_extensions('js')

    # Các phần tử trạng thái giao diện
    shared.input_elements = ui.list_interface_input_elements()

    with gr.Blocks(css=css, analytics_enabled=False, title=title, theme=ui.theme) as shared.gradio['interface']:

        # Trạng thái giao diện
        shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})

        # Thông báo âm thanh
        if Path("notification.mp3").exists():
            shared.gradio['audio_notification'] = gr.Audio(interactive=False, value="notification.mp3", elem_id="audio_notification", visible=False)

        # Menu nổi để lưu/xóa tập tin
        ui_file_saving.create_ui()

        # Clipboard tạm thời để lưu tập tin
        shared.gradio['temporary_text'] = gr.Textbox(visible=False)

        # Tab Tạo văn bản
        ui_chat.create_ui()
        ui_default.create_ui()
        ui_notebook.create_ui()

        ui_parameters.create_ui(shared.settings['preset'])  # Tab thông số
        ui_model_menu.create_ui()  # Tab mô hình
        training.create_ui()  # Tab đào tạo
        ui_session.create_ui()  # Tab phiên

        # Sự kiện thế hệ
        ui_chat.create_event_handlers()
        ui_default.create_event_handlers()
        ui_notebook.create_event_handlers()

        # Các sự kiện khác
        ui_file_saving.create_event_handlers()
        ui_parameters.create_event_handlers()
        ui_model_menu.create_event_handlers()

        # Sự kiện ra mắt giao diện
        if shared.settings['dark_theme']:
            shared.gradio['interface'].load(lambda: None, None, None, _js="() => document.getElementsByTagName('body')[0].classList.add('dark')")

        shared.gradio['interface'].load(lambda: None, None, None, _js=f"() => {{{js}}}")
        shared.gradio['interface'].load(None, gradio('show_controls'), None, _js=f'(x) => {{{ui.show_controls_js}; toggle_controls(x)}}')
        shared.gradio['interface'].load(partial(ui.apply_interface_values, {}, use_persistent=True), None, gradio(ui.list_interface_input_elements()), show_progress=False)
        shared.gradio['interface'].load(chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display'))

        extensions_module.create_extensions_tabs()  # Tab tiện ích mở rộng
        extensions_module.create_extensions_block()  # Khối tiện ích mở rộng

    # Khởi chạy giao diện
    shared.gradio['interface'].queue(concurrency_count=64)
    with OpenMonkeyPatch():
        shared.gradio['interface'].launch(
            prevent_thread_lock=True,
            share=True,
            server_name=None if not shared.args.listen else (shared.args.listen_host or '0.0.0.0'),
            server_port=shared.args.listen_port,
            inbrowser=shared.args.auto_launch,
            auth=auth or None,
            ssl_verify=False if (shared.args.ssl_keyfile or shared.args.ssl_certfile) else True,
            ssl_keyfile=shared.args.ssl_keyfile,
            ssl_certfile=shared.args.ssl_certfile
        )


if __name__ == "__main__":

    logger.info("Bắt đầu giao diện người dùng web tạo văn bản")
    do_cmd_flags_warnings()

    # Tải cài đặt tùy chỉnh
    settings_file = None
    if shared.args.settings is not None and Path(shared.args.settings).exists():
        settings_file = Path(shared.args.settings)
    elif Path('settings.yaml').exists():
        settings_file = Path('settings.yaml')
    elif Path('settings.json').exists():
        settings_file = Path('settings.json')

    if settings_file is not None:
        logger.info(f"Đang tải cài đặt từ {settings_file}")
        file_contents = open(settings_file, 'r', encoding='utf-8').read()
        new_settings = json.loads(file_contents) if settings_file.suffix == "json" else yaml.safe_load(file_contents)
        shared.settings.update(new_settings)

    # Cài đặt dự phòng cho mô hình
    shared.model_config['.*'] = get_fallback_settings()
    shared.model_config.move_to_end('.*', last=False)  # Chuyển về đầu

    # Kích hoạt các tiện ích mở rộng được liệt kê trên settings.yaml
    extensions_module.available_extensions = utils.get_available_extensions()
    for extension in shared.settings['default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)

    available_models = utils.get_available_models()

    # Mô hình được xác định thông qua --model
    if shared.args.model is not None:
        shared.model_name = shared.args.model

    # Chọn mô hình từ menu dòng lệnh
    elif shared.args.model_menu:
        if len(available_models) == 0:
            logger.error('Không có mô hình có sẵn! Vui lòng tải xuống ít nhất một mô hình.')
            sys.exit(0)
        else:
            print('Các mô hình sau đây có sẵn:\n')
            for i, model in enumerate(available_models):
                print(f'{i+1}. {model}')

            print(f'\nBạn muốn tải cái nào? 1-{len(available_models)}\n')
            i = int(input()) - 1
            print()

        shared.model_name = available_models[i]

    # Nếu bất kỳ mô hình nào đã được chọn, hãy tải nó
    if shared.model_name != 'None':
        p = Path(shared.model_name)
        if p.exists():
            model_name = p.parts[-1]
            shared.model_name = model_name
        else:
            model_name = shared.model_name

        model_settings = get_model_metadata(model_name)
        update_model_parameters(model_settings, initial=True)  # chiếm đoạt các đối số dòng lệnh

        # Tải mô hình
        shared.model, shared.tokenizer = load_model(model_name)
        if shared.args.lora:
            add_lora_to_model(shared.args.lora)

    shared.generation_lock = Lock()

    if shared.args.nowebui:
        # Khởi động API ở chế độ độc lập
        shared.args.extensions = [x for x in shared.args.extensions if x != 'gallery']
        if shared.args.extensions is not None and len(shared.args.extensions) > 0:
            extensions_module.load_extensions()
    else:
        # Khởi chạy giao diện người dùng web
        create_interface()
        while True:
            time.sleep(0.5)
            if shared.need_restart:
                shared.need_restart = False
                time.sleep(0.5)
                shared.gradio['interface'].close()
                time.sleep(0.5)
                create_interface()
