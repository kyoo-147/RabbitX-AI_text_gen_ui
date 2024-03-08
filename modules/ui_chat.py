import json
from functools import partial
from pathlib import Path

import gradio as gr
from PIL import Image

from modules import chat, shared, ui, utils
from modules.html_generator import chat_html_wrapper
from modules.text_generation import stop_everything_event
from modules.utils import gradio

inputs = ('Chat input', 'interface_state')
reload_arr = ('history', 'name1', 'name2', 'mode', 'chat_style', 'character_menu')
clear_arr = ('delete_chat-confirm', 'delete_chat', 'delete_chat-cancel')


def create_ui():
    mu = shared.args.multi_user

    shared.gradio['Chat input'] = gr.State()
    shared.gradio['history'] = gr.State({'internal': [], 'visible': []})

    with gr.Tab('Trò chuyện', elem_id='chat-tab', elem_classes=("old-ui" if shared.args.chat_buttons else None)):
        with gr.Row():
            with gr.Column(elem_id='chat-col'):
                shared.gradio['display'] = gr.HTML(value=chat_html_wrapper({'internal': [], 'visible': []}, '', '', 'chat', 'cai-chat', ''))

                with gr.Row(elem_id="chat-input-row"):
                    with gr.Column(scale=1, elem_id='gr-hover-container'):
                        gr.HTML(value='<div class="hover-element" onclick="void(0)"><span style="width: 100px; display: block" id="hover-element-button">&#9776;</span><div class="hover-menu" id="hover-menu"></div>', elem_id='gr-hover')

                    with gr.Column(scale=10, elem_id='chat-input-container'):
                        shared.gradio['textbox'] = gr.Textbox(label='', placeholder='Gửi tin nhắn', elem_id='chat-input', elem_classes=['add_scrollbar'])
                        shared.gradio['show_controls'] = gr.Checkbox(value=shared.settings['show_controls'], label='Hiển thị điều khiển (Ctrl+S)', elem_id='show-controls')
                        shared.gradio['typing-dots'] = gr.HTML(value='<div class="typing"><span></span><span class="dot1"></span><span class="dot2"></span></div>', label='Đang nhập', elem_id='typing-container')

                    with gr.Column(scale=1, elem_id='generate-stop-container'):
                        with gr.Row():
                            shared.gradio['Stop'] = gr.Button('Dừng', elem_id='stop', visible=False)
                            shared.gradio['Generate'] = gr.Button('Khởi tạo', elem_id='Generate', variant='primary')

        # Hover menu buttons
        with gr.Column(elem_id='chat-buttons'):
            with gr.Row():
                shared.gradio['Regenerate'] = gr.Button('Tái khởi tạo (Ctrl + Enter)', elem_id='Regenerate')
                shared.gradio['Continue'] = gr.Button('Tiếp theo (Alt + Enter)', elem_id='Continue')
                shared.gradio['Remove last'] = gr.Button('Xóa câu trả lời cuối cùng (Ctrl + Shift + Backspace)', elem_id='Remove-last')

            with gr.Row():
                shared.gradio['Replace last reply'] = gr.Button('Thay thế câu trả lời cuối cùng (Ctrl + Shift + L)', elem_id='Replace-last')
                shared.gradio['Copy last reply'] = gr.Button('Sao chép câu trả lời cuối cùng (Ctrl + Shift + K)', elem_id='Copy-last')
                shared.gradio['Impersonate'] = gr.Button('Mạo danh (Ctrl + Shift + M)', elem_id='Impersonate')

            with gr.Row():
                shared.gradio['Send dummy message'] = gr.Button('Gửi tin nhắn giả')
                shared.gradio['Send dummy reply'] = gr.Button('Gửi câu trả lời giả')

            with gr.Row():
                shared.gradio['send-chat-to-default'] = gr.Button('Gửi về mặc định')
                shared.gradio['send-chat-to-notebook'] = gr.Button('Gửi tới sổ tay')

        with gr.Row(elem_id='past-chats-row', elem_classes=['pretty_scrollbar']):
            with gr.Column():
                with gr.Row():
                    shared.gradio['unique_id'] = gr.Dropdown(label='Cuộc trò chuyện trước đây', elem_classes=['slim-dropdown'], interactive=not mu)

                with gr.Row():
                    shared.gradio['rename_chat'] = gr.Button('Đổi tên', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_chat'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_chat-confirm'] = gr.Button('Xác nhận', variant='stop', visible=False, elem_classes='refresh-button')
                    shared.gradio['delete_chat-cancel'] = gr.Button('Hủy bỏ', visible=False, elem_classes='refresh-button')
                    shared.gradio['Start new chat'] = gr.Button('Cuộc trò truyện mới', elem_classes='refresh-button')

                with gr.Row(elem_id='rename-row'):
                    shared.gradio['rename_to'] = gr.Textbox(label='Đổi tên thành:', placeholder='Tên mới', visible=False, elem_classes=['no-background'])
                    shared.gradio['rename_to-confirm'] = gr.Button('Xác nhận', visible=False, elem_classes='refresh-button')
                    shared.gradio['rename_to-cancel'] = gr.Button('Hủy bỏ', visible=False, elem_classes='refresh-button')

        with gr.Row(elem_id='chat-controls', elem_classes=['pretty_scrollbar']):
            with gr.Column():
                with gr.Row():
                    shared.gradio['start_with'] = gr.Textbox(label='Bắt đầu trả lời với', placeholder='Sure thing!', value=shared.settings['start_with'], elem_classes=['add_scrollbar'])

                with gr.Row():
                    shared.gradio['mode'] = gr.Radio(choices=['chat', 'chat-instruct', 'instruct'], value='chat', label='Chế độ', info='Xác định cách tạo lời nhắc trò chuyện. Trong chế độ hướng dẫn và hướng dẫn trò chuyện, mẫu hướng dẫn được chọn trong Tham số > Mẫu hướng dẫn phải khớp với mô hình hiện tại.', elem_id='chat-mode')

                with gr.Row():
                    shared.gradio['chat_style'] = gr.Dropdown(choices=utils.get_available_chat_styles(), label='Phong cách trò chuyện', value=shared.settings['chat_style'], visible=shared.settings['mode'] != 'instruct')


def create_chat_settings_ui():
    mu = shared.args.multi_user
    with gr.Tab('Tính cách'):
        with gr.Row():
            with gr.Column(scale=8):
                with gr.Row():
                    shared.gradio['character_menu'] = gr.Dropdown(value=None, choices=utils.get_available_characters(), label='Tính cách', elem_id='character-menu', info='Được sử dụng trong chế độ trò chuyện và hướng dẫn trò chuyện.', elem_classes='slim-dropdown')
                    ui.create_refresh_button(shared.gradio['character_menu'], lambda: None, lambda: {'choices': utils.get_available_characters()}, 'refresh-button', interactive=not mu)
                    shared.gradio['save_character'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_character'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)

                shared.gradio['name1'] = gr.Textbox(value=shared.settings['name1'], lines=1, label='Tên của bạn')
                shared.gradio['name2'] = gr.Textbox(value='', lines=1, label='Tên nhân vật')
                shared.gradio['context'] = gr.Textbox(value='', lines=10, label='Bối cảnh', elem_classes=['add_scrollbar'])
                shared.gradio['greeting'] = gr.Textbox(value='', lines=5, label='Lời chào', elem_classes=['add_scrollbar'])

            with gr.Column(scale=1):
                shared.gradio['character_picture'] = gr.Image(label='Hình ảnh nhân vật', type='pil', interactive=not mu)
                shared.gradio['your_picture'] = gr.Image(label='Ảnh của bạn', type='pil', value=Image.open(Path('cache/pfp_me.png')) if Path('cache/pfp_me.png').exists() else None, interactive=not mu)

    with gr.Tab('Mẫu hướng dẫn'):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    shared.gradio['instruction_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), label='Mẫu hướng dẫn đã lưu', info="Sau khi chọn mẫu xong bấm vào \"Tải\" để tải và áp dụng.", value='None', elem_classes='slim-dropdown')
                    ui.create_refresh_button(shared.gradio['instruction_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button', interactive=not mu)
                    shared.gradio['load_template'] = gr.Button("Tải", elem_classes='refresh-button')
                    shared.gradio['save_template'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_template'] = gr.Button('🗑️ ', elem_classes='refresh-button', interactive=not mu)

            with gr.Column():
                pass

        with gr.Row():
            with gr.Column():
                shared.gradio['custom_system_message'] = gr.Textbox(value=shared.settings['custom_system_message'], lines=2, label='Thông báo hệ thống tùy chỉnh', info='Nếu không trống, sẽ được sử dụng thay cho giá trị mặc định.', elem_classes=['add_scrollbar'])
                shared.gradio['instruction_template_str'] = gr.Textbox(value='', label='Mẫu hướng dẫn', lines=24, info='Thay đổi điều này theo mô hình/LoRA mà bạn đang sử dụng. Được sử dụng trong chế độ hướng dẫn và hướng dẫn trò chuyện.', elem_classes=['add_scrollbar', 'monospace'])
                with gr.Row():
                    shared.gradio['send_instruction_to_default'] = gr.Button('Gửi về mặc định', elem_classes=['small-button'])
                    shared.gradio['send_instruction_to_notebook'] = gr.Button('Gửi tới sổ tay', elem_classes=['small-button'])
                    shared.gradio['send_instruction_to_negative_prompt'] = gr.Button('Gửi tới lời nhắc phủ định', elem_classes=['small-button'])

            with gr.Column():
                shared.gradio['chat_template_str'] = gr.Textbox(value=shared.settings['chat_template_str'], label='Mẫu trò chuyện', lines=22, elem_classes=['add_scrollbar', 'monospace'])
                shared.gradio['chat-instruct_command'] = gr.Textbox(value=shared.settings['chat-instruct_command'], lines=4, label='Lệnh cho chế độ hướng dẫn trò chuyện', info='<|character|> được thay thế bằng tên bot và <|prompt|> được thay thế bằng lời nhắc trò chuyện thông thường.', elem_classes=['add_scrollbar'])

    with gr.Tab('Lịch sử trò chuyện'):
        with gr.Row():
            with gr.Column():
                shared.gradio['save_chat_history'] = gr.Button(value='Lưu lịch sử')

            with gr.Column():
                shared.gradio['load_chat_history'] = gr.File(type='binary', file_types=['.json', '.txt'], label='Lịch sử tải lên JSON')

    with gr.Tab('Tải nhân vật lên'):
        with gr.Tab('YAML hoặc JSON'):
            with gr.Row():
                shared.gradio['upload_json'] = gr.File(type='binary', file_types=['.json', '.yaml'], label='Tệp JSON hoặc YAML', interactive=not mu)
                shared.gradio['upload_img_bot'] = gr.Image(type='pil', label='Ảnh hồ sơ (tùy chọn)', interactive=not mu)

            shared.gradio['Submit character'] = gr.Button(value='Submit', interactive=False)

        with gr.Tab('TavernAI PNG'):
            with gr.Row():
                with gr.Column():
                    shared.gradio['upload_img_tavern'] = gr.Image(type='pil', label='TavernAI PNG File', elem_id='upload_img_tavern', interactive=not mu)
                    shared.gradio['tavern_json'] = gr.State()
                with gr.Column():
                    shared.gradio['tavern_name'] = gr.Textbox(value='', lines=1, label='Tên', interactive=False)
                    shared.gradio['tavern_desc'] = gr.Textbox(value='', lines=4, max_lines=4, label='Sự miêu tả', interactive=False)

            shared.gradio['Submit tavern character'] = gr.Button(value='Nộp', interactive=False)


def create_event_handlers():

    # Obsolete variables, kept for compatibility with old extensions
    shared.input_params = gradio(inputs)
    shared.reload_inputs = gradio(reload_arr)

    shared.gradio['Generate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: (x, ''), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False).then(
        chat.generate_chat_reply_wrapper, gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['textbox'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: (x, ''), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False).then(
        chat.generate_chat_reply_wrapper, gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Regenerate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        partial(chat.generate_chat_reply_wrapper, regenerate=True), gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Continue'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        partial(chat.generate_chat_reply_wrapper, _continue=True), gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Impersonate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: x, gradio('textbox'), gradio('Chat input'), show_progress=False).then(
        chat.impersonate_wrapper, gradio(inputs), gradio('textbox', 'display'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Replace last reply'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.replace_last_reply, gradio('textbox', 'interface_state'), gradio('history')).then(
        lambda: '', None, gradio('textbox'), show_progress=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None)

    shared.gradio['Send dummy message'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.send_dummy_message, gradio('textbox', 'interface_state'), gradio('history')).then(
        lambda: '', None, gradio('textbox'), show_progress=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None)

    shared.gradio['Send dummy reply'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.send_dummy_reply, gradio('textbox', 'interface_state'), gradio('history')).then(
        lambda: '', None, gradio('textbox'), show_progress=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None)

    shared.gradio['Remove last'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.remove_last_message, gradio('history'), gradio('textbox', 'history'), show_progress=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None)

    shared.gradio['Stop'].click(
        stop_everything_event, None, None, queue=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display'))

    if not shared.args.multi_user:
        shared.gradio['unique_id'].select(
            chat.load_history, gradio('unique_id', 'character_menu', 'mode'), gradio('history')).then(
            chat.redraw_html, gradio(reload_arr), gradio('display'))

    shared.gradio['Start new chat'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.start_new_chat, gradio('interface_state'), gradio('history')).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        lambda x: gr.update(choices=(histories := chat.find_all_histories(x)), value=histories[0]), gradio('interface_state'), gradio('unique_id'))

    shared.gradio['delete_chat'].click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, gradio(clear_arr))
    shared.gradio['delete_chat-cancel'].click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, gradio(clear_arr))
    shared.gradio['delete_chat-confirm'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x, y: str(chat.find_all_histories(x).index(y)), gradio('interface_state', 'unique_id'), gradio('temporary_text')).then(
        chat.delete_history, gradio('unique_id', 'character_menu', 'mode'), None).then(
        chat.load_history_after_deletion, gradio('interface_state', 'temporary_text'), gradio('history', 'unique_id')).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, gradio(clear_arr))

    shared.gradio['rename_chat'].click(
        lambda x: x, gradio('unique_id'), gradio('rename_to')).then(
        lambda: [gr.update(visible=True)] * 3, None, gradio('rename_to', 'rename_to-confirm', 'rename_to-cancel'), show_progress=False)

    shared.gradio['rename_to-cancel'].click(
        lambda: [gr.update(visible=False)] * 3, None, gradio('rename_to', 'rename_to-confirm', 'rename_to-cancel'), show_progress=False)

    shared.gradio['rename_to-confirm'].click(
        chat.rename_history, gradio('unique_id', 'rename_to', 'character_menu', 'mode'), None).then(
        lambda: [gr.update(visible=False)] * 3, None, gradio('rename_to', 'rename_to-confirm', 'rename_to-cancel'), show_progress=False).then(
        lambda x, y: gr.update(choices=chat.find_all_histories(x), value=y), gradio('interface_state', 'rename_to'), gradio('unique_id'))

    shared.gradio['rename_to'].submit(
        chat.rename_history, gradio('unique_id', 'rename_to', 'character_menu', 'mode'), None).then(
        lambda: [gr.update(visible=False)] * 3, None, gradio('rename_to', 'rename_to-confirm', 'rename_to-cancel'), show_progress=False).then(
        lambda x, y: gr.update(choices=chat.find_all_histories(x), value=y), gradio('interface_state', 'rename_to'), gradio('unique_id'))

    shared.gradio['load_chat_history'].upload(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.start_new_chat, gradio('interface_state'), gradio('history')).then(
        chat.load_history_json, gradio('load_chat_history', 'history'), gradio('history')).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        lambda x: gr.update(choices=(histories := chat.find_all_histories(x)), value=histories[0]), gradio('interface_state'), gradio('unique_id')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_chat()}}')

    shared.gradio['character_menu'].change(
        chat.load_character, gradio('character_menu', 'name1', 'name2'), gradio('name1', 'name2', 'character_picture', 'greeting', 'context')).success(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.load_latest_history, gradio('interface_state'), gradio('history')).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        lambda x: gr.update(choices=(histories := chat.find_all_histories(x)), value=histories[0]), gradio('interface_state'), gradio('unique_id')).then(
        lambda: None, None, None, _js=f'() => {{{ui.update_big_picture_js}; updateBigPicture()}}')

    shared.gradio['mode'].change(
        lambda x: gr.update(visible=x != 'instruct'), gradio('mode'), gradio('chat_style'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.load_latest_history, gradio('interface_state'), gradio('history')).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        lambda x: gr.update(choices=(histories := chat.find_all_histories(x)), value=histories[0]), gradio('interface_state'), gradio('unique_id'))

    shared.gradio['chat_style'].change(chat.redraw_html, gradio(reload_arr), gradio('display'))
    shared.gradio['Copy last reply'].click(chat.send_last_reply_to_input, gradio('history'), gradio('textbox'), show_progress=False)

    # Save/delete a character
    shared.gradio['save_character'].click(
        lambda x: x, gradio('name2'), gradio('save_character_filename')).then(
        lambda: gr.update(visible=True), None, gradio('character_saver'))

    shared.gradio['delete_character'].click(lambda: gr.update(visible=True), None, gradio('character_deleter'))

    shared.gradio['load_template'].click(
        chat.load_instruction_template, gradio('instruction_template'), gradio('instruction_template_str')).then(
        lambda: "Select template to load...", None, gradio('instruction_template'))

    shared.gradio['save_template'].click(
        lambda: 'My Template.yaml', None, gradio('save_filename')).then(
        lambda: 'instruction-templates/', None, gradio('save_root')).then(
        chat.generate_instruction_template_yaml, gradio('instruction_template_str'), gradio('save_contents')).then(
        lambda: gr.update(visible=True), None, gradio('file_saver'))

    shared.gradio['delete_template'].click(
        lambda x: f'{x}.yaml', gradio('instruction_template'), gradio('delete_filename')).then(
        lambda: 'instruction-templates/', None, gradio('delete_root')).then(
        lambda: gr.update(visible=True), None, gradio('file_deleter'))

    shared.gradio['save_chat_history'].click(
        lambda x: json.dumps(x, indent=4), gradio('history'), gradio('temporary_text')).then(
        None, gradio('temporary_text', 'character_menu', 'mode'), None, _js=f'(hist, char, mode) => {{{ui.save_files_js}; saveHistory(hist, char, mode)}}')

    shared.gradio['Submit character'].click(
        chat.upload_character, gradio('upload_json', 'upload_img_bot'), gradio('character_menu')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_character()}}')

    shared.gradio['Submit tavern character'].click(
        chat.upload_tavern_character, gradio('upload_img_tavern', 'tavern_json'), gradio('character_menu')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_character()}}')

    shared.gradio['upload_json'].upload(lambda: gr.update(interactive=True), None, gradio('Submit character'))
    shared.gradio['upload_json'].clear(lambda: gr.update(interactive=False), None, gradio('Submit character'))
    shared.gradio['upload_img_tavern'].upload(chat.check_tavern_character, gradio('upload_img_tavern'), gradio('tavern_name', 'tavern_desc', 'tavern_json', 'Submit tavern character'), show_progress=False)
    shared.gradio['upload_img_tavern'].clear(lambda: (None, None, None, gr.update(interactive=False)), None, gradio('tavern_name', 'tavern_desc', 'tavern_json', 'Submit tavern character'), show_progress=False)
    shared.gradio['your_picture'].change(
        chat.upload_your_profile_picture, gradio('your_picture'), None).then(
        partial(chat.redraw_html, reset_cache=True), gradio(reload_arr), gradio('display'))

    shared.gradio['send_instruction_to_default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: x.update({'mode': 'instruct', 'history': {'internal': [], 'visible': []}}), gradio('interface_state'), None).then(
        partial(chat.generate_chat_prompt, 'Input'), gradio('interface_state'), gradio('textbox-default')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_default()}}')

    shared.gradio['send_instruction_to_notebook'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: x.update({'mode': 'instruct', 'history': {'internal': [], 'visible': []}}), gradio('interface_state'), None).then(
        partial(chat.generate_chat_prompt, 'Input'), gradio('interface_state'), gradio('textbox-notebook')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_notebook()}}')

    shared.gradio['send_instruction_to_negative_prompt'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: x.update({'mode': 'instruct', 'history': {'internal': [], 'visible': []}}), gradio('interface_state'), None).then(
        partial(chat.generate_chat_prompt, 'Input'), gradio('interface_state'), gradio('negative_prompt')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_generation_parameters()}}')

    shared.gradio['send-chat-to-default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        partial(chat.generate_chat_prompt, '', _continue=True), gradio('interface_state'), gradio('textbox-default')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_default()}}')

    shared.gradio['send-chat-to-notebook'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        partial(chat.generate_chat_prompt, '', _continue=True), gradio('interface_state'), gradio('textbox-notebook')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_notebook()}}')

    shared.gradio['show_controls'].change(None, gradio('show_controls'), None, _js=f'(x) => {{{ui.show_controls_js}; toggle_controls(x)}}')
