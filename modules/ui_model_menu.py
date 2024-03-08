import importlib
import math
import re
import traceback
from functools import partial
from pathlib import Path

import gradio as gr
import psutil
import torch
from transformers import is_torch_xpu_available

from modules import loaders, shared, ui, utils
from modules.logging_colors import logger
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import (
    apply_model_settings_to_state,
    get_model_metadata,
    save_instruction_template,
    save_model_settings,
    update_model_parameters
)
from modules.utils import gradio


def create_ui():
    mu = shared.args.multi_user

    # Finding the default values for the GPU and CPU memories
    total_mem = []
    if is_torch_xpu_available():
        for i in range(torch.xpu.device_count()):
            total_mem.append(math.floor(torch.xpu.get_device_properties(i).total_memory / (1024 * 1024)))
    else:
        for i in range(torch.cuda.device_count()):
            total_mem.append(math.floor(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)))

    default_gpu_mem = []
    if shared.args.gpu_memory is not None and len(shared.args.gpu_memory) > 0:
        for i in shared.args.gpu_memory:
            if 'mib' in i.lower():
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)))
            else:
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)) * 1000)

    while len(default_gpu_mem) < len(total_mem):
        default_gpu_mem.append(0)

    total_cpu_mem = math.floor(psutil.virtual_memory().total / (1024 * 1024))
    if shared.args.cpu_memory is not None:
        default_cpu_mem = re.sub('[a-zA-Z ]', '', shared.args.cpu_memory)
    else:
        default_cpu_mem = 0

    with gr.Tab("Mô hình", elem_id="model-tab"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            shared.gradio['model_menu'] = gr.Dropdown(choices=utils.get_available_models(), value=lambda: shared.model_name, label='Mô hình', elem_classes='slim-dropdown', interactive=not mu)
                            ui.create_refresh_button(shared.gradio['model_menu'], lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button', interactive=not mu)
                            shared.gradio['load_model'] = gr.Button("Tải", visible=not shared.settings['autoload_model'], elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['unload_model'] = gr.Button("Dỡ bỏ", elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['reload_model'] = gr.Button("Tải lại", elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['save_model_settings'] = gr.Button("Lưu các thiết lập", elem_classes='refresh-button', interactive=not mu)

                    with gr.Column():
                        with gr.Row():
                            shared.gradio['lora_menu'] = gr.Dropdown(multiselect=True, choices=utils.get_available_loras(), value=shared.lora_names, label='LoRA(s)', elem_classes='slim-dropdown', interactive=not mu)
                            ui.create_refresh_button(shared.gradio['lora_menu'], lambda: None, lambda: {'choices': utils.get_available_loras(), 'value': shared.lora_names}, 'refresh-button', interactive=not mu)
                            shared.gradio['lora_menu_apply'] = gr.Button(value='Apply LoRAs', elem_classes='refresh-button', interactive=not mu)

        with gr.Row():
            with gr.Column():
                shared.gradio['loader'] = gr.Dropdown(label="Trình tải mô hình", choices=loaders.loaders_and_params.keys(), value=None)
                with gr.Box():
                    with gr.Row():
                        with gr.Column():
                            with gr.Blocks():
                                for i in range(len(total_mem)):
                                    shared.gradio[f'gpu_memory_{i}'] = gr.Slider(label=f"gpu-memory trong MiB cho thiết bị :{i}", maximum=total_mem[i], value=default_gpu_mem[i])

                                shared.gradio['cpu_memory'] = gr.Slider(label="cpu-memory trong MiB", maximum=total_cpu_mem, value=default_cpu_mem)

                            with gr.Blocks():
                                shared.gradio['transformers_info'] = gr.Markdown('Tham sô load-in-4bit:')
                                shared.gradio['compute_dtype'] = gr.Dropdown(label="compute_dtype", choices=["bfloat16", "float16", "float32"], value=shared.args.compute_dtype)
                                shared.gradio['quant_type'] = gr.Dropdown(label="quant_type", choices=["nf4", "fp4"], value=shared.args.quant_type)

                            shared.gradio['hqq_backend'] = gr.Dropdown(label="hqq_backend", choices=["PYTORCH", "PYTORCH_COMPILE", "ATEN"], value=shared.args.hqq_backend)
                            shared.gradio['n_gpu_layers'] = gr.Slider(label="n-gpu-layers", minimum=0, maximum=256, value=shared.args.n_gpu_layers)
                            shared.gradio['n_ctx'] = gr.Slider(minimum=0, maximum=shared.settings['truncation_length_max'], step=256, label="n_ctx", value=shared.args.n_ctx, info='Độ dài bối cảnh. Hãy thử giảm mức này nếu bạn hết bộ nhớ trong khi tải mô hình.')
                            shared.gradio['tensor_split'] = gr.Textbox(label='tensor_split', info='Danh sách tỷ lệ để phân chia mô hình trên nhiều GPU. Ví dụ: 18,17')
                            shared.gradio['n_batch'] = gr.Slider(label="n_batch", minimum=1, maximum=2048, step=1, value=shared.args.n_batch)
                            shared.gradio['threads'] = gr.Slider(label="threads", minimum=0, step=1, maximum=32, value=shared.args.threads)
                            shared.gradio['threads_batch'] = gr.Slider(label="threads_batch", minimum=0, step=1, maximum=32, value=shared.args.threads_batch)
                            shared.gradio['wbits'] = gr.Dropdown(label="wbits", choices=["None", 1, 2, 3, 4, 8], value=shared.args.wbits if shared.args.wbits > 0 else "None")
                            shared.gradio['groupsize'] = gr.Dropdown(label="groupsize", choices=["None", 32, 64, 128, 1024], value=shared.args.groupsize if shared.args.groupsize > 0 else "None")
                            shared.gradio['model_type'] = gr.Dropdown(label="model_type", choices=["None"], value=shared.args.model_type or "None")
                            shared.gradio['pre_layer'] = gr.Slider(label="pre_layer", minimum=0, maximum=100, value=shared.args.pre_layer[0] if shared.args.pre_layer is not None else 0)
                            shared.gradio['gpu_split'] = gr.Textbox(label='gpu-split', info='Danh sách VRAM (tính bằng GB) được phân tách bằng dấu phẩy để sử dụng cho mỗi GPU. Ví dụ: 20,7,7')
                            shared.gradio['max_seq_len'] = gr.Slider(label='max_seq_len', minimum=0, maximum=shared.settings['truncation_length_max'], step=256, info='Độ dài bối cảnh. Hãy thử giảm mức này nếu bạn hết bộ nhớ trong khi tải mô hình.', value=shared.args.max_seq_len)
                            with gr.Blocks():
                                shared.gradio['alpha_value'] = gr.Slider(label='alpha_value', minimum=1, maximum=8, step=0.05, info='Hệ số alpha nhúng theo vị trí để chia tỷ lệ NTK RoPE. Giá trị được đề xuất (NTKv1): 1,75 cho bối cảnh 1,5x, 2,5 cho bối cảnh 2x. Sử dụng cái này hoặc nén_pos_emb, không phải cả hai.', value=shared.args.alpha_value)
                                shared.gradio['rope_freq_base'] = gr.Slider(label='rope_freq_base', minimum=0, maximum=1000000, step=1000, info='Nếu lớn hơn 0, sẽ được sử dụng thay cho alpha_value. Hai cái đó có liên quan bởi wire_freq_base = 10000 * alpha_value ^ (64/63)', value=shared.args.rope_freq_base)
                                shared.gradio['compress_pos_emb'] = gr.Slider(label='compress_pos_emb', minimum=1, maximum=8, step=1, info='Hệ số nén nhúng vị trí. Nên đặt thành (độ dài ngữ cảnh) / (độ dài ngữ cảnh ban đầu của mô hình). Bằng 1/rope_freq_scale.', value=shared.args.compress_pos_emb)

                            shared.gradio['autogptq_info'] = gr.Markdown('ExLlamav2_HF được khuyến nghị trên AutoGPTQ cho các mô hình có nguồn gốc từ Llama.')
                            shared.gradio['quipsharp_info'] = gr.Markdown('QuIP# hiện phải được cài đặt thủ công.')

                        with gr.Column():
                            shared.gradio['load_in_8bit'] = gr.Checkbox(label="load-in-8bit", value=shared.args.load_in_8bit)
                            shared.gradio['load_in_4bit'] = gr.Checkbox(label="load-in-4bit", value=shared.args.load_in_4bit)
                            shared.gradio['use_double_quant'] = gr.Checkbox(label="use_double_quant", value=shared.args.use_double_quant)
                            shared.gradio['use_flash_attention_2'] = gr.Checkbox(label="use_flash_attention_2", value=shared.args.use_flash_attention_2, info='Đặt use_flash_attention_2=True trong khi tải mô hình.')
                            shared.gradio['auto_devices'] = gr.Checkbox(label="auto-devices", value=shared.args.auto_devices)
                            shared.gradio['tensorcores'] = gr.Checkbox(label="tensorcores", value=shared.args.tensorcores, info='Chỉ dành cho NVIDIA: sử dụng llama-cpp-python được biên dịch với sự hỗ trợ lõi tensor. Điều này làm tăng hiệu suất trên thẻ RTX.')
                            shared.gradio['cpu'] = gr.Checkbox(label="cpu", value=shared.args.cpu, info='llama.cpp: Sử dụng llama-cpp-python được biên dịch mà không cần tăng tốc GPU. Transformers: sử dụng PyTorch ở chế độ CPU.')
                            shared.gradio['row_split'] = gr.Checkbox(label="row_split", value=shared.args.row_split, info='Chia mô hình theo hàng trên GPU. Điều này có thể cải thiện hiệu suất đa gpu.')
                            shared.gradio['no_offload_kqv'] = gr.Checkbox(label="no_offload_kqv", value=shared.args.no_offload_kqv, info='Không giảm tải K, Q, V cho GPU. Điều này tiết kiệm VRAM nhưng làm giảm hiệu suất.')
                            shared.gradio['no_mul_mat_q'] = gr.Checkbox(label="no_mul_mat_q", value=shared.args.no_mul_mat_q, info='Vô hiệu hóa hạt nhân mulmat.')
                            shared.gradio['triton'] = gr.Checkbox(label="triton", value=shared.args.triton)
                            shared.gradio['no_inject_fused_attention'] = gr.Checkbox(label="no_inject_fused_attention", value=shared.args.no_inject_fused_attention, info='Vô hiệu hóa sự chú ý hợp nhất. Sự chú ý hợp nhất cải thiện hiệu suất suy luận nhưng sử dụng nhiều VRAM hơn. Hợp nhất các lớp cho AutoAWQ. Vô hiệu hóa nếu sắp hết VRAM.')
                            shared.gradio['no_inject_fused_mlp'] = gr.Checkbox(label="no_inject_fused_mlp", value=shared.args.no_inject_fused_mlp, info='Chỉ ảnh hưởng đến Triton. Vô hiệu hóa MLP hợp nhất. MLP hợp nhất cải thiện hiệu suất nhưng sử dụng nhiều VRAM hơn. Vô hiệu hóa nếu sắp hết VRAM.')
                            shared.gradio['no_use_cuda_fp16'] = gr.Checkbox(label="no_use_cuda_fp16", value=shared.args.no_use_cuda_fp16, info='Điều này có thể làm cho mô hình nhanh hơn trên một số hệ thống.')
                            shared.gradio['desc_act'] = gr.Checkbox(label="desc_act", value=shared.args.desc_act, info='\'desc_act\', \'wbits\', and \'groupsize\' được sử dụng cho các mẫu cũ không có quantize_config.json.')
                            shared.gradio['no_mmap'] = gr.Checkbox(label="no-mmap", value=shared.args.no_mmap)
                            shared.gradio['mlock'] = gr.Checkbox(label="mlock", value=shared.args.mlock)
                            shared.gradio['numa'] = gr.Checkbox(label="numa", value=shared.args.numa, info='Hỗ trợ NUMA có thể trợ giúp trên một số hệ thống có quyền truy cập bộ nhớ không đồng nhất.')
                            shared.gradio['disk'] = gr.Checkbox(label="disk", value=shared.args.disk)
                            shared.gradio['bf16'] = gr.Checkbox(label="bf16", value=shared.args.bf16)
                            shared.gradio['cache_8bit'] = gr.Checkbox(label="cache_8bit", value=shared.args.cache_8bit, info='Sử dụng bộ đệm 8 bit để tiết kiệm VRAM.')
                            shared.gradio['autosplit'] = gr.Checkbox(label="autosplit", value=shared.args.autosplit, info='Tự động phân chia các tensor mô hình cho các GPU có sẵn.')
                            shared.gradio['no_flash_attn'] = gr.Checkbox(label="no_flash_attn", value=shared.args.no_flash_attn, info='Buộc không sử dụng tính năng chú ý flash.')
                            shared.gradio['cfg_cache'] = gr.Checkbox(label="cfg-cache", value=shared.args.cfg_cache, info='Cần thiết phải sử dụng CFG với trình tải này.')
                            shared.gradio['num_experts_per_token'] = gr.Number(label="Số lượng chuyên gia trên mỗi mã thông báo", value=shared.args.num_experts_per_token, info='Chỉ áp dụng cho các dòng MoE như Mixtral.')
                            with gr.Blocks():
                                shared.gradio['trust_remote_code'] = gr.Checkbox(label="trust-remote-code", value=shared.args.trust_remote_code, info='Đặt Trust_remote_code=True trong khi tải mã thông báo/mô hình. Để bật tùy chọn này, hãy khởi động giao diện người dùng web bằng --trust-remote-code flag.', interactive=shared.args.trust_remote_code)
                                shared.gradio['no_use_fast'] = gr.Checkbox(label="no_use_fast", value=shared.args.no_use_fast, info='Đặt use_fast=False trong khi tải mã thông báo.')
                                shared.gradio['logits_all'] = gr.Checkbox(label="logits_all", value=shared.args.logits_all, info='Cần phải thiết lập tính năng đánh giá độ phức tạp để hoạt động với trình tải này. Nếu không, hãy bỏ qua nó, vì nó làm cho quá trình xử lý nhanh chóng chậm hơn.')

                            shared.gradio['disable_exllama'] = gr.Checkbox(label="disable_exllama", value=shared.args.disable_exllama, info='Vô hiệu hóa kernel ExLlama cho các mô hình GPTQ.')
                            shared.gradio['disable_exllamav2'] = gr.Checkbox(label="disable_exllamav2", value=shared.args.disable_exllamav2, info='Vô hiệu hóa kernel ExLlamav2 cho các mô hình GPTQ.')
                            shared.gradio['gptq_for_llama_info'] = gr.Markdown('Trình tải kế thừa để tương thích với các GPU cũ hơn. ExLlamav2_HF hoặc AutoGPTQ được ưu tiên cho các mẫu GPTQ khi được hỗ trợ.')
                            shared.gradio['exllamav2_info'] = gr.Markdown("ExLlamav2_HF được khuyến nghị thay vì ExLlamav2 để tích hợp tốt hơn với các tiện ích mở rộng và hoạt động lấy mẫu nhất quán hơn trên các trình tải.")
                            shared.gradio['llamacpp_HF_info'] = gr.Markdown("llamacpp_HF tải llama.cpp dưới dạng mô hình Transformers. Để sử dụng nó, bạn cần đặt GGUF của mình vào thư mục con của models/ với các tệp mã thông báo cần thiết.\n\nBạn có thể sử dụng trình đơn \"llamacpp_HF Creator\" để tự động thực hiện việc đó.")

            with gr.Column():
                with gr.Row():
                    shared.gradio['autoload_model'] = gr.Checkbox(value=shared.settings['autoload_model'], label='Tự động tải mô hình', info='Có tải mô hình ngay khi nó được chọn trong danh sách thả xuống Mô hình hay không.', interactive=not mu)

                with gr.Tab("Tải xuống"):
                    shared.gradio['custom_model_menu'] = gr.Textbox(label="Tải xuống mô hình hoặc LoRA", info="Nhập đường dẫn tên người dùng/người mẫu HF, ví dụ: facebook/galactica-125m. Để chỉ định một nhánh, hãy thêm nó vào cuối sau ký tự \":\" như thế này: facebook/galactica-125m:main. Để tải xuống một tệp, hãy nhập tên của nó vào hộp thứ hai.", interactive=not mu)
                    shared.gradio['download_specific_file'] = gr.Textbox(placeholder="Tên tệp (đối với mô hình GGUF)", show_label=False, max_lines=1, interactive=not mu)
                    with gr.Row():
                        shared.gradio['download_model_button'] = gr.Button("Tải xuống", variant='primary', interactive=not mu)
                        shared.gradio['get_file_list'] = gr.Button("Nhận danh sách tập tin", interactive=not mu)

                with gr.Tab("Người sáng tạo llamacpp_HF"):
                    with gr.Row():
                        shared.gradio['gguf_menu'] = gr.Dropdown(choices=utils.get_available_ggufs(), value=lambda: shared.model_name, label='Chọn GGUF của bạn', elem_classes='slim-dropdown', interactive=not mu)
                        ui.create_refresh_button(shared.gradio['gguf_menu'], lambda: None, lambda: {'choices': utils.get_available_ggufs()}, 'refresh-button', interactive=not mu)

                    shared.gradio['unquantized_url'] = gr.Textbox(label="Nhập URL cho mô hình gốc (không được lượng tử hóa)", info="Ví dụ: https://huggingface.co/lmsys/vicuna-13b-v1.5", max_lines=1)
                    shared.gradio['create_llamacpp_hf_button'] = gr.Button("Xác nhận", variant="primary", interactive=not mu)
                    gr.Markdown("Thao tác này sẽ di chuyển tệp gguf của bạn vào thư mục con của `models` cùng với các tệp mã thông báo cần thiết.")

                with gr.Tab("Tùy chỉnh mẫu hướng dẫn"):
                    with gr.Row():
                        shared.gradio['customized_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), value='None', label='Chọn mẫu hướng dẫn mong muốn', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['customized_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button', interactive=not mu)

                    shared.gradio['customized_template_submit'] = gr.Button("Xác nhận", variant="primary", interactive=not mu)
                    gr.Markdown("Điều này cho phép bạn đặt mẫu tùy chỉnh cho mô hình hiện được chọn trong menu \"Trình tải mô hình\". Bất cứ khi nào mô hình được tải, mẫu này sẽ được sử dụng thay cho mẫu được chỉ định trong medatada của mô hình, điều này đôi khi có sai sót.")

                with gr.Row():
                    shared.gradio['model_status'] = gr.Markdown('Không có mô hình nào được tải' if shared.model_name == 'None' else 'Ready')


def create_event_handlers():
    shared.gradio['loader'].change(
        loaders.make_loader_params_visible, gradio('loader'), gradio(loaders.get_all_params())).then(
        lambda value: gr.update(choices=loaders.get_model_types(value)), gradio('loader'), gradio('model_type'))

    # In this event handler, the interface state is read and updated
    # with the model defaults (if any), and then the model is loaded
    # unless "autoload_model" is unchecked
    shared.gradio['model_menu'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        apply_model_settings_to_state, gradio('model_menu', 'interface_state'), gradio('interface_state')).then(
        ui.apply_interface_values, gradio('interface_state'), gradio(ui.list_interface_input_elements()), show_progress=False).then(
        update_model_parameters, gradio('interface_state'), None).then(
        load_model_wrapper, gradio('model_menu', 'loader', 'autoload_model'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['load_model'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        partial(load_model_wrapper, autoload=True), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['reload_model'].click(
        unload_model, None, None).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        partial(load_model_wrapper, autoload=True), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['unload_model'].click(
        unload_model, None, None).then(
        lambda: "Model unloaded", None, gradio('model_status'))

    shared.gradio['save_model_settings'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        save_model_settings, gradio('model_menu', 'interface_state'), gradio('model_status'), show_progress=False)

    shared.gradio['lora_menu_apply'].click(load_lora_wrapper, gradio('lora_menu'), gradio('model_status'), show_progress=False)
    shared.gradio['download_model_button'].click(download_model_wrapper, gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)
    shared.gradio['get_file_list'].click(partial(download_model_wrapper, return_links=True), gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)
    shared.gradio['autoload_model'].change(lambda x: gr.update(visible=not x), gradio('autoload_model'), gradio('load_model'))
    shared.gradio['create_llamacpp_hf_button'].click(create_llamacpp_hf, gradio('gguf_menu', 'unquantized_url'), gradio('model_status'), show_progress=True)
    shared.gradio['customized_template_submit'].click(save_instruction_template, gradio('model_menu', 'customized_template'), gradio('model_status'), show_progress=True)


def load_model_wrapper(selected_model, loader, autoload=False):
    if not autoload:
        yield f"Các cài đặt cho `{selected_model}` đã được cập nhật.\n\nBấm vào \"Tải\" để tải nó."
        return

    if selected_model == 'None':
        yield "Không có mô hình nào được chọn"
    else:
        try:
            yield f"Đang tải `{selected_model}`..."
            unload_model()
            if selected_model != '':
                shared.model, shared.tokenizer = load_model(selected_model, loader)

            if shared.model is not None:
                output = f"Đã tải thành công `{selected_model}`."

                settings = get_model_metadata(selected_model)
                if 'instruction_template' in settings:
                    output += '\n\nNó có vẻ là một mô hình làm theo hướng dẫn với mẫu "{}". Trong tab trò chuyện, nên sử dụng chế độ hướng dẫn hoặc trò chuyện-hướng dẫn.'.format(settings['instruction_template'])

                yield output
            else:
                yield f"Không tải được `{selected_model}`."
        except:
            exc = traceback.format_exc()
            logger.error('Không tải được mô hình.')
            print(exc)
            yield exc.replace('\n', '\n\n')


def load_lora_wrapper(selected_loras):
    yield ("Áp dụng các LoRA sau đây để {}:\n\n{}".format(shared.model_name, '\n'.join(selected_loras)))
    add_lora_to_model(selected_loras)
    yield ("Đã áp dụng thành công LoRAs")


def download_model_wrapper(repo_id, specific_file, progress=gr.Progress(), return_links=False, check=False):
    try:
        downloader = importlib.import_module("download-model").ModelDownloader()

        progress(0.0)
        model, branch = downloader.sanitize_model_and_branch_names(repo_id, None)

        yield ("Lấy link tải từ HF")
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=False, specific_file=specific_file)
        if return_links:
            output = "```\n"
            for link in links:
                output += f"{Path(link).name}" + "\n"

            output += "```"
            yield output
            return

        yield ("Lấy thư mục đầu ra")
        output_folder = downloader.get_output_folder(model, branch, is_lora, is_llamacpp=is_llamacpp)
        if check:
            progress(0.5)

            yield ("Kiểm tra các tập tin đã tải xuống trước đó")
            downloader.check_model_files(model, branch, links, sha256, output_folder)
            progress(1.0)
        else:
            yield (f"Dữ liệu đang tải{'s' if len(links) > 1 else ''} đến `{output_folder}/`")
            downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=is_llamacpp)

            yield (f"Đã lưu thành công mô hình vào `{output_folder}/`.")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def create_llamacpp_hf(gguf_name, unquantized_url, progress=gr.Progress()):
    try:
        downloader = importlib.import_module("download-model").ModelDownloader()

        progress(0.0)
        model, branch = downloader.sanitize_model_and_branch_names(unquantized_url, None)

        yield ("Nhận liên kết tệp mã thông báo từ HF")
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=True)
        output_folder = Path(shared.args.model_dir) / (re.sub(r'(?i)\.gguf$', '', gguf_name) + "-HF")

        yield (f"Đang tải mã thông báo xuống `{output_folder}`")
        downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=False)

        # Move the GGUF
        (Path(shared.args.model_dir) / gguf_name).rename(output_folder / gguf_name)

        yield (f"Đã lưu mô hình vào `{output_folder}/`.\n\nBây giờ bạn có thể tải nó bằng cách sử dụng llamacpp_HF.")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def update_truncation_length(current_length, state):
    if 'loader' in state:
        if state['loader'].lower().startswith('exllama'):
            return state['max_seq_len']
        elif state['loader'] in ['llama.cpp', 'llamacpp_HF', 'ctransformers']:
            return state['n_ctx']

    return current_length
