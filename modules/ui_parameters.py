from pathlib import Path

import gradio as gr

from modules import loaders, presets, shared, ui, ui_chat, utils
from modules.utils import gradio


def create_ui(default_preset):
    mu = shared.args.multi_user
    generate_params = presets.load_preset(default_preset)
    with gr.Tab("Tham số", elem_id="parameters"):
        with gr.Tab("Thế hệ"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        shared.gradio['preset_menu'] = gr.Dropdown(choices=utils.get_available_presets(), value=default_preset, label='Cài sẵn', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['preset_menu'], lambda: None, lambda: {'choices': utils.get_available_presets()}, 'refresh-button', interactive=not mu)
                        shared.gradio['save_preset'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['delete_preset'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['random_preset'] = gr.Button('🎲', elem_classes='refresh-button')

                with gr.Column():
                    shared.gradio['filter_by_loader'] = gr.Dropdown(label="Lọc theo trình tải", choices=["All"] + list(loaders.loaders_and_params.keys()), value="All", elem_classes='slim-dropdown')

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                            shared.gradio['temperature'] = gr.Slider(0.01, 5, value=generate_params['temperature'], step=0.01, label='Nhiệt độ')
                            shared.gradio['top_p'] = gr.Slider(0.0, 1.0, value=generate_params['top_p'], step=0.01, label='top_p')
                            shared.gradio['min_p'] = gr.Slider(0.0, 1.0, value=generate_params['min_p'], step=0.01, label='min_p')
                            shared.gradio['top_k'] = gr.Slider(0, 200, value=generate_params['top_k'], step=1, label='top_k')
                            shared.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=generate_params['repetition_penalty'], step=0.01, label='repetition_penalty')
                            shared.gradio['presence_penalty'] = gr.Slider(0, 2, value=generate_params['presence_penalty'], step=0.05, label='presence_penalty')
                            shared.gradio['frequency_penalty'] = gr.Slider(0, 2, value=generate_params['frequency_penalty'], step=0.05, label='frequency_penalty')
                            shared.gradio['repetition_penalty_range'] = gr.Slider(0, 4096, step=64, value=generate_params['repetition_penalty_range'], label='repetition_penalty_range')
                            shared.gradio['typical_p'] = gr.Slider(0.0, 1.0, value=generate_params['typical_p'], step=0.01, label='typical_p')
                            shared.gradio['tfs'] = gr.Slider(0.0, 1.0, value=generate_params['tfs'], step=0.01, label='tfs')
                            shared.gradio['top_a'] = gr.Slider(0.0, 1.0, value=generate_params['top_a'], step=0.01, label='top_a')
                            shared.gradio['epsilon_cutoff'] = gr.Slider(0, 9, value=generate_params['epsilon_cutoff'], step=0.01, label='epsilon_cutoff')
                            shared.gradio['eta_cutoff'] = gr.Slider(0, 20, value=generate_params['eta_cutoff'], step=0.01, label='eta_cutoff')

                        with gr.Column():
                            shared.gradio['guidance_scale'] = gr.Slider(-0.5, 2.5, step=0.05, value=generate_params['guidance_scale'], label='guidance_scale', info='Đối với CFG. 1,5 là một giá trị tốt.')
                            shared.gradio['negative_prompt'] = gr.Textbox(value=shared.settings['negative_prompt'], label='Lời nhắc phủ định', lines=3, elem_classes=['add_scrollbar'])
                            shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='penalty_alpha', info='Để tìm kiếm tương phản. do_sample phải được bỏ chọn.')
                            shared.gradio['mirostat_mode'] = gr.Slider(0, 2, step=1, value=generate_params['mirostat_mode'], label='mirostat_mode', info='mode=1 chỉ dành cho llama.cpp.')
                            shared.gradio['mirostat_tau'] = gr.Slider(0, 10, step=0.01, value=generate_params['mirostat_tau'], label='mirostat_tau')
                            shared.gradio['mirostat_eta'] = gr.Slider(0, 1, step=0.01, value=generate_params['mirostat_eta'], label='mirostat_eta')
                            shared.gradio['smoothing_factor'] = gr.Slider(0.0, 10.0, value=generate_params['smoothing_factor'], step=0.01, label='smoothing_factor', info='Kích hoạt Lấy mẫu bậc hai.')
                            shared.gradio['dynamic_temperature'] = gr.Checkbox(value=generate_params['dynamic_temperature'], label='dynamic_temperature')
                            shared.gradio['dynatemp_low'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_low'], step=0.01, label='dynatemp_low', visible=generate_params['dynamic_temperature'])
                            shared.gradio['dynatemp_high'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_high'], step=0.01, label='dynatemp_high', visible=generate_params['dynamic_temperature'])
                            shared.gradio['dynatemp_exponent'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_exponent'], step=0.01, label='dynatemp_exponent', visible=generate_params['dynamic_temperature'])
                            shared.gradio['temperature_last'] = gr.Checkbox(value=generate_params['temperature_last'], label='temperature_last', info='Di chuyển nhiệt độ/nhiệt độ động/lấy mẫu bậc hai đến cuối ngăn xếp bộ lấy mẫu, bỏ qua vị trí của chúng trong "Ưu tiên bộ lấy mẫu".')
                            shared.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='do_sample')
                            shared.gradio['seed'] = gr.Number(value=shared.settings['seed'], label='Hạt giống (-1 cho ngẫu nhiên)')
                            with gr.Accordion('Other parameters', open=False):
                                shared.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=generate_params['encoder_repetition_penalty'], step=0.01, label='encoder_repetition_penalty')
                                shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='no_repeat_ngram_size')
                                shared.gradio['min_length'] = gr.Slider(0, 2000, step=1, value=generate_params['min_length'], label='min_length')
                                shared.gradio['num_beams'] = gr.Slider(1, 20, step=1, value=generate_params['num_beams'], label='num_beams', info='Đối với Beam Search, cùng với length_penalty và Early_stopping.')
                                shared.gradio['length_penalty'] = gr.Slider(-5, 5, value=generate_params['length_penalty'], label='length_penalty')
                                shared.gradio['early_stopping'] = gr.Checkbox(value=generate_params['early_stopping'], label='early_stopping')

                    gr.Markdown("[Learn more](https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab)")

                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['truncation_length'] = gr.Slider(value=get_truncation_length(), minimum=shared.settings['truncation_length_min'], maximum=shared.settings['truncation_length_max'], step=256, label='Cắt bớt lời nhắc đến độ dài này', info='Các mã thông báo ngoài cùng bên trái sẽ bị xóa nếu lời nhắc vượt quá độ dài này. Hầu hết các mô hình yêu cầu tối đa là 2048.')
                            shared.gradio['max_tokens_second'] = gr.Slider(value=shared.settings['max_tokens_second'], minimum=0, maximum=20, step=1, label='Mã thông báo tối đa/giây', info='Để làm cho văn bản có thể đọc được trong thời gian thực.')
                            shared.gradio['max_updates_second'] = gr.Slider(value=shared.settings['max_updates_second'], minimum=0, maximum=24, step=1, label='Cập nhật giao diện người dùng tối đa/giây', info='Đặt cài đặt này nếu bạn gặp hiện tượng lag trong giao diện người dùng trong khi phát trực tuyến.')
                            shared.gradio['prompt_lookup_num_tokens'] = gr.Slider(value=shared.settings['prompt_lookup_num_tokens'], minimum=0, maximum=10, step=1, label='prompt_lookup_num_tokens', info='Kích hoạt giải mã tra cứu nhanh chóng.')

                            shared.gradio['custom_stopping_strings'] = gr.Textbox(lines=1, value=shared.settings["custom_stopping_strings"] or None, label='Chuỗi dừng tùy chỉnh', info='Ngoài các mặc định. Được viết giữa "" và được phân tách bằng dấu phẩy.', placeholder='"\\n", "\\nYou:"')
                            shared.gradio['custom_token_bans'] = gr.Textbox(value=shared.settings['custom_token_bans'] or None, label='Lệnh cấm mã thông báo tùy chỉnh', info='ID mã thông báo cụ thể bị cấm tạo, được phân tách bằng dấu phẩy. ID có thể được tìm thấy trong tab Mặc định hoặc Notebook.')

                        with gr.Column():
                            shared.gradio['auto_max_new_tokens'] = gr.Checkbox(value=shared.settings['auto_max_new_tokens'], label='auto_max_new_tokens', info='Mở rộng max_new_tokens theo độ dài ngữ cảnh có sẵn.')
                            shared.gradio['ban_eos_token'] = gr.Checkbox(value=shared.settings['ban_eos_token'], label='Ban the eos_token', info='Buộc mô hình không bao giờ kết thúc thế hệ sớm.')
                            shared.gradio['add_bos_token'] = gr.Checkbox(value=shared.settings['add_bos_token'], label='Thêm bos_token vào đầu lời nhắc', info='Tắt tính năng này có thể khiến câu trả lời trở nên sáng tạo hơn.')
                            shared.gradio['skip_special_tokens'] = gr.Checkbox(value=shared.settings['skip_special_tokens'], label='Bỏ qua các token đặc biệt', info='Một số kiểu máy cụ thể cần bỏ cài đặt này.')
                            shared.gradio['stream'] = gr.Checkbox(value=shared.settings['stream'], label='Kích hoạt truyền phát văn bản')

                            with gr.Blocks():
                                shared.gradio['sampler_priority'] = gr.Textbox(value=generate_params['sampler_priority'], lines=12, label='Ưu tiên lấy mẫu', info='Tên tham số được phân tách bằng dòng mới hoặc dấu phẩy.')

                            with gr.Row() as shared.gradio['grammar_file_row']:
                                shared.gradio['grammar_file'] = gr.Dropdown(value='None', choices=utils.get_available_grammars(), label='Tải ngữ pháp từ tập tin (.gbnf)', elem_classes='slim-dropdown')
                                ui.create_refresh_button(shared.gradio['grammar_file'], lambda: None, lambda: {'choices': utils.get_available_grammars()}, 'refresh-button', interactive=not mu)
                                shared.gradio['save_grammar'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                                shared.gradio['delete_grammar'] = gr.Button('🗑️ ', elem_classes='refresh-button', interactive=not mu)

                    shared.gradio['grammar_string'] = gr.Textbox(value='', label='Ngữ pháp', lines=16, elem_classes=['add_scrollbar', 'monospace'])

        ui_chat.create_chat_settings_ui()


def create_event_handlers():
    shared.gradio['filter_by_loader'].change(loaders.blacklist_samplers, gradio('filter_by_loader', 'dynamic_temperature'), gradio(loaders.list_all_samplers()), show_progress=False)
    shared.gradio['preset_menu'].change(presets.load_preset_for_ui, gradio('preset_menu', 'interface_state'), gradio('interface_state') + gradio(presets.presets_params()))
    shared.gradio['random_preset'].click(presets.random_preset, gradio('interface_state'), gradio('interface_state') + gradio(presets.presets_params()))
    shared.gradio['grammar_file'].change(load_grammar, gradio('grammar_file'), gradio('grammar_string'))
    shared.gradio['dynamic_temperature'].change(lambda x: [gr.update(visible=x)] * 3, gradio('dynamic_temperature'), gradio('dynatemp_low', 'dynatemp_high', 'dynatemp_exponent'))


def get_truncation_length():
    if 'max_seq_len' in shared.provided_arguments or shared.args.max_seq_len != shared.args_defaults.max_seq_len:
        return shared.args.max_seq_len
    elif 'n_ctx' in shared.provided_arguments or shared.args.n_ctx != shared.args_defaults.n_ctx:
        return shared.args.n_ctx
    else:
        return shared.settings['truncation_length']


def load_grammar(name):
    p = Path(f'grammars/{name}')
    if p.exists():
        return open(p, 'r', encoding='utf-8').read()
    else:
        return ''
