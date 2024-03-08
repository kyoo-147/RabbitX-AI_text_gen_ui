import gradio as gr

from modules import shared, ui, utils
from modules.github import clone_or_pull_repository
from modules.utils import gradio


def create_ui():
    mu = shared.args.multi_user
    with gr.Tab("Phiên hoạt động", elem_id="session-tab"):
        with gr.Row():
            with gr.Column():
                shared.gradio['reset_interface'] = gr.Button("Áp dụng cờ/tiện ích mở rộng và khởi động lại", interactive=not mu)
                with gr.Row():
                    shared.gradio['toggle_dark_mode'] = gr.Button('Chuyển đổi 💡')
                    shared.gradio['save_settings'] = gr.Button('Lưu giao diện mặc định vào settings.yaml', interactive=not mu)

                with gr.Row():
                    with gr.Column():
                        shared.gradio['extensions_menu'] = gr.CheckboxGroup(choices=utils.get_available_extensions(), value=shared.args.extensions, label="Tiện ích mở rộng có sẵn   ", info='Lưu ý rằng một số tiện ích mở rộng này có thể yêu cầu cài đặt thủ công các yêu cầu Python thông qua lệnh: pip install -r extensions/extension_name/requirements.txt', elem_classes='checkboxgroup-table')

                    with gr.Column():
                        shared.gradio['bool_menu'] = gr.CheckboxGroup(choices=get_boolean_arguments(), value=get_boolean_arguments(active=True), label="Cờ dòng lệnh Boolean", elem_classes='checkboxgroup-table')

            with gr.Column():
                extension_name = gr.Textbox(lines=1, label='Cài đặt hoặc cập nhập tính năng', info='Nhập URL GitHub bên dưới và nhấn Enter. Để biết danh sách các tiện ích mở rộng, hãy xem: https://github.com/oobabooga/text-generation-webui-extensions ⚠️  CẢNH BÁO ⚠️ : tiện ích mở rộng có thể thực thi mã tùy ý. Đảm bảo kiểm tra mã nguồn của họ trước khi kích hoạt chúng.', interactive=not mu)
                extension_status = gr.Markdown()

        shared.gradio['theme_state'] = gr.Textbox(visible=False, value='dark' if shared.settings['dark_theme'] else 'light')
        extension_name.submit(clone_or_pull_repository, extension_name, extension_status, show_progress=False)

        # Reset interface event
        shared.gradio['reset_interface'].click(
            set_interface_arguments, gradio('extensions_menu', 'bool_menu'), None).then(
            lambda: None, None, None, _js='() => {document.body.innerHTML=\'<h1 style="font-family:monospace;padding-top:20%;margin:0;height:100vh;color:lightgray;text-align:center;background:var(--body-background-fill)">Reloading...</h1>\'; setTimeout(function(){location.reload()},2500); return []}')

        shared.gradio['toggle_dark_mode'].click(
            lambda: None, None, None, _js='() => {document.getElementsByTagName("body")[0].classList.toggle("dark")}').then(
            lambda x: 'dark' if x == 'light' else 'light', gradio('theme_state'), gradio('theme_state'))

        shared.gradio['save_settings'].click(
            ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
            ui.save_settings, gradio('interface_state', 'preset_menu', 'extensions_menu', 'show_controls', 'theme_state'), gradio('save_contents')).then(
            lambda: './', None, gradio('save_root')).then(
            lambda: 'settings.yaml', None, gradio('save_filename')).then(
            lambda: gr.update(visible=True), None, gradio('file_saver'))


def set_interface_arguments(extensions, bool_active):
    shared.args.extensions = extensions

    bool_list = get_boolean_arguments()

    for k in bool_list:
        setattr(shared.args, k, False)
    for k in bool_active:
        setattr(shared.args, k, True)
        if k == 'api':
            shared.add_extension('openai', last=True)

    shared.need_restart = True


def get_boolean_arguments(active=False):
    exclude = shared.deprecated_args

    cmd_list = vars(shared.args)
    bool_list = sorted([k for k in cmd_list if type(cmd_list[k]) is bool and k not in exclude + ui.list_model_elements()])
    bool_active = [k for k in bool_list if vars(shared.args)[k]]

    if active:
        return bool_active
    else:
        return bool_list
