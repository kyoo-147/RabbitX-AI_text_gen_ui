import os

os.environ["WANDB_MODE"] = "offline"
# os.environ["WANDB_DISABLED"] = "true"

import json
import math
import random
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
import transformers
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict
)
from peft.utils.other import \
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as model_to_lora_modules
from transformers import is_torch_xpu_available
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
)

from modules import shared, ui, utils
from modules.evaluate import (
    calculate_perplexity,
    generate_markdown_table,
    save_past_evaluations
)
from modules.logging_colors import logger
from modules.models import reload_model
from modules.utils import natural_keys

MODEL_CLASSES = {v[1]: v[0] for v in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.items()}
PARAMETERS = ["lora_name", "always_override", "q_proj_en", "v_proj_en", "k_proj_en", "o_proj_en", "gate_proj_en", "down_proj_en", "up_proj_en", "save_steps", "micro_batch_size", "batch_size", "epochs", "learning_rate", "lr_scheduler_type", "lora_rank", "lora_alpha", "lora_dropout", "cutoff_len", "dataset", "eval_dataset", "format", "eval_steps", "raw_text_file", "overlap_len", "newline_favor_len", "higher_rank_limit", "warmup_steps", "optimizer", "hard_cut_string", "train_only_after", "stop_at_loss", "add_eos_token", "min_chars", "report_to"]
WANT_INTERRUPT = False

train_log = {}
train_template = {}


def create_ui():
    mu = shared.args.multi_user
    with gr.Tab("Đào tạo", elem_id="training-tab"):
        with gr.Tab('Đào tạo LoRA', elem_id='lora-train-tab'):
            tmp = gr.State('')
            with gr.Row():
                with gr.Column():
                    gr.Markdown("[Hướng dẫn](https://github.com/oobabooga/text-generation-webui/wiki/05-%E2%80%90-Training-Tab)")

                    with gr.Row():
                        copy_from = gr.Dropdown(label='Sao chép tham số từ', value='None', choices=utils.get_available_loras(), elem_classes=['slim-dropdown'], interactive=not mu)
                        ui.create_refresh_button(copy_from, lambda: None, lambda: {'choices': utils.get_available_loras()}, 'refresh-button', interactive=not mu)

                    with gr.Row():
                        with gr.Column(scale=5):
                            lora_name = gr.Textbox(label='Tên', info='Tên tệp LoRA mới của bạn')
                        with gr.Column():
                            always_override = gr.Checkbox(label='Ghi đè các tập tin hiện có', value=False, info='Nếu tên giống nhau, việc chọn sẽ thay thế tệp hiện có và việc bỏ chọn sẽ tải và tiếp tục từ tệp đó (thứ hạng phải giống nhau).', elem_classes=['no-background'])

                    with gr.Accordion(label='Mô-đun mục tiêu', open=False):
                        gr.Markdown("Nhắm mục tiêu nhiều mô-đun hơn sẽ gần tinh chỉnh hoàn toàn hơn nhưng phải trả giá bằng việc tăng yêu cầu VRAM và kích thước bộ điều hợp.\nLƯU Ý: Chỉ hoạt động với model_id='llama', các loại khác sẽ giữ lại hành vi đào tạo mặc định và không sử dụng các cài đặt này.")
                        with gr.Row():
                            with gr.Column():
                                q_proj_en = gr.Checkbox(label='Cho phép q_proj', value=True)
                            with gr.Column():
                                v_proj_en = gr.Checkbox(label='Cho phép v_proj', value=True)
                            with gr.Column():
                                k_proj_en = gr.Checkbox(label='Cho phép k_proj', value=False)
                            with gr.Column():
                                o_proj_en = gr.Checkbox(label='Cho phép o_proj', value=False)
                            with gr.Column():
                                gate_proj_en = gr.Checkbox(label='Cho phép gate_proj', value=False)
                            with gr.Column():
                                down_proj_en = gr.Checkbox(label='Cho phép down_proj', value=False)
                            with gr.Column():
                                up_proj_en = gr.Checkbox(label='Cho phép up_proj', value=False)

                    with gr.Row():
                        with gr.Column():
                            lora_rank = gr.Slider(label='LoRA Rank', value=32, minimum=0, maximum=1024, step=4, info='Còn được gọi là số thứ nguyên. Giá trị cao hơn = tệp lớn hơn, kiểm soát nội dung nhiều hơn. Giá trị nhỏ hơn = tệp nhỏ hơn, ít quyền kiểm soát hơn. Sử dụng 4 hoặc 8 cho kiểu, 128 hoặc 256 để dạy, 1024+ để biết chi tiết về dữ liệu lớn. Cần nhiều VRAM hơn để đạt thứ hạng cao hơn.')
                            lora_alpha = gr.Slider(label='LoRA Alpha', value=64, minimum=0, maximum=2048, step=4, info='Điều này chia cho thứ hạng sẽ trở thành tỷ lệ của LoRA. Cao hơn có nghĩa là mạnh hơn. Giá trị tiêu chuẩn tốt gấp đôi Thứ hạng của bạn.')
                            batch_size = gr.Slider(label='Batch Size', value=128, minimum=0, maximum=1024, step=4, info='Kích thước lô toàn cầu. Hai kích cỡ lô cùng nhau xác định sự tích lũy độ dốc (gradientAccum = batch / microBatch). Giá trị tích lũy gradient cao hơn dẫn đến chất lượng đào tạo tốt hơn.')
                            micro_batch_size = gr.Slider(label='Micro Batch Size', value=4, minimum=1, maximum=128, step=1, info='Kích thước lô trên mỗi thiết bị (LƯU Ý: nhiều thiết bị chưa được triển khai). Tăng điều này sẽ tăng mức sử dụng VRAM.')
                            cutoff_len = gr.Slider(label='Độ dài Cutoff', minimum=0, maximum=4096, value=256, step=32, info='Độ dài giới hạn để nhập văn bản. Về cơ bản, mỗi dòng văn bản sẽ dài bao nhiêu. Giá trị cao hơn đòi hỏi nhiều VRAM hơn.')

                        with gr.Column():
                            save_steps = gr.Number(label='Lưu sau môi bước n', value=0, info='Nếu trên 0, điểm kiểm tra của LoRA sẽ được lưu mỗi khi vượt qua nhiều bước này.')

                            epochs = gr.Number(label='Epochs', value=3, info='Số lần mỗi mục trong tập dữ liệu sẽ được đưa vào đào tạo. Vì vậy, 1 có nghĩa là cho mỗi mục vào một lần, 5 có nghĩa là cho ăn vào năm lần, v.v..')
                            learning_rate = gr.Textbox(label='Tỷ lệ học tập', value='3e-4', info='Trong ký hiệu khoa học. 3e-4 là điểm khởi đầu tốt. 1e-2 cực cao, 1e-6 cực thấp.')
                            with gr.Row():
                                lr_scheduler_type = gr.Dropdown(label='LR Scheduler', value='linear', choices=['linear', 'constant', 'constant_with_warmup', 'cosine', 'cosine_with_restarts', 'polynomial', 'inverse_sqrt'], info='Bộ lập lịch tốc độ học tập - xác định tốc độ học tập thay đổi như thế nào theo thời gian. "Không đổi" có nghĩa là không bao giờ thay đổi, "tuyến tính" có nghĩa là đi theo đường thẳng từ tốc độ học xuống 0, cosine đi theo đường cong, v.v..', elem_classes=['slim-dropdown'])

                    with gr.Accordion(label='Tùy chọn nâng cao', open=False):
                        with gr.Row():
                            with gr.Column():
                                lora_dropout = gr.Slider(label='LoRA Dropout', minimum=0.0, maximum=1.0, step=0.025, value=0.05, info='Xác suất phần trăm cho việc loại bỏ các lớp LoRA. Điều này có thể giúp giảm bớt việc trang bị quá mức. Hầu hết người dùng nên để mặc định.')
                                stop_at_loss = gr.Slider(label='Dừng lại khi mất mát', minimum=0.0, maximum=3.0, step=0.1, value=0.00, info='Quá trình sẽ tự động dừng khi đạt được giá trị tổn thất mong muốn. (con số hợp lý là 1,5-1,8)')
                                with gr.Row():
                                    optimizer = gr.Dropdown(label='Tối ưu hóa', value='adamw_torch', choices=['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision', 'sgd', 'adagrad'], info='Các tùy chọn triển khai trình tối ưu hóa khác nhau dành cho người dùng nâng cao. Tác dụng của các lựa chọn khác nhau vẫn chưa được ghi chép đầy đủ.', elem_classes=['slim-dropdown'])

                            with gr.Column():
                                warmup_steps = gr.Number(label='Warmup Steps', value=100, info='Đối với nhiều bước này khi bắt đầu, tốc độ học sẽ thấp hơn bình thường. Điều này giúp giảng viên chuẩn bị mô hình và tính toán trước số liệu thống kê để nâng cao chất lượng đào tạo sau khi bắt đầu.')
                                train_only_after = gr.Textbox(label='Train Only After', value='', info='Chỉ xem xét văn bản *sau* chuỗi này trong bất kỳ đoạn cụ thể nào để đào tạo. Đối với tập dữ liệu Alpaca, hãy sử dụng "### Phản hồi:" để chỉ huấn luyện phản hồi và bỏ qua dữ liệu đầu vào.')

                                add_eos_token = gr.Checkbox(label='Thêm EOS token', value=False, info="Thêm mã thông báo EOS cho từng mục tập dữ liệu. Trong trường hợp văn bản thô, EOS sẽ được thêm vào ở phần Hard Cut")

                                higher_rank_limit = gr.Checkbox(label='CHo phép higher ranks', value=False, info='Nếu được chọn, hãy thay đổi thanh trượt Xếp hạng/Alpha ở trên để tăng cao hơn nhiều. Điều này sẽ không hoạt động nếu không có GPU cấp trung tâm dữ liệu.')
                                report_to = gr.Radio(label="Lưu nhật ký chi tiết với", value="None", choices=["None", "wandb", "tensorboard"], interactive=True)

                with gr.Column():
                    with gr.Tab(label='Tập dữ liệu được định dạng'):
                        with gr.Row():
                            format = gr.Dropdown(choices=utils.get_datasets('training/formats', 'json'), value='None', label='Định dạng dữ liệu', info='Tệp định dạng được sử dụng để quyết định cách định dạng dữ liệu đầu vào.', elem_classes=['slim-dropdown'], interactive=not mu)
                            ui.create_refresh_button(format, lambda: None, lambda: {'choices': utils.get_datasets('training/formats', 'json')}, 'refresh-button', interactive=not mu)

                        with gr.Row():
                            dataset = gr.Dropdown(choices=utils.get_datasets('training/datasets', 'json'), value='None', label='Tệp dữ liệu', info='Tệp dữ liệu được sử dụng để đào tạo.', elem_classes=['slim-dropdown'], interactive=not mu)
                            ui.create_refresh_button(dataset, lambda: None, lambda: {'choices': utils.get_datasets('training/datasets', 'json')}, 'refresh-button', interactive=not mu)

                        with gr.Row():
                            eval_dataset = gr.Dropdown(choices=utils.get_datasets('training/datasets', 'json'), value='None', label='Bộ dữ liệu đánh giá', info='Tệp dữ liệu (tùy chọn) được sử dụng để đánh giá mô hình sau khi đào tạo.', elem_classes=['slim-dropdown'], interactive=not mu)
                            ui.create_refresh_button(eval_dataset, lambda: None, lambda: {'choices': utils.get_datasets('training/datasets', 'json')}, 'refresh-button', interactive=not mu)

                        eval_steps = gr.Number(label='Đánh giá mỗi n bước', value=100, info='Nếu một tập dữ liệu đánh giá được đưa ra, hãy kiểm tra nó mỗi khi vượt qua nhiều bước này.')

                    with gr.Tab(label="Tệp văn bản thô"):
                        with gr.Row():
                            raw_text_file = gr.Dropdown(choices=utils.get_datasets('training/datasets', 'txt'), value='None', label='Tệp văn bản', info='Tệp văn bản thô để sử dụng cho đào tạo.', elem_classes=['slim-dropdown'], interactive=not mu)
                            ui.create_refresh_button(raw_text_file, lambda: None, lambda: {'choices': utils.get_datasets('training/datasets', 'txt')}, 'refresh-button', interactive=not mu)

                        with gr.Row():
                            with gr.Column():
                                overlap_len = gr.Slider(label='Chiều dài chồng chéo', minimum=0, maximum=512, value=128, step=16, info='Có bao nhiêu mã thông báo từ đoạn văn bản trước được đưa vào đoạn văn bản tiếp theo. (Bản thân các khối sẽ có kích thước được xác định bởi Độ dài cắt). Đặt mức chồng lấp chính xác bằng một nửa chiều dài đường cắt có thể là lý tưởng.')
                                newline_favor_len = gr.Slider(label='Ưu tiên độ dài cắt dòng mới', minimum=0, maximum=512, value=128, step=16, info='Độ dài (tính bằng ký tự, không phải mã thông báo) của khoảng cách tối đa để dịch chuyển phần cắt chồng chéo để đảm bảo các đoạn được cắt ở dòng mới. Nếu quá thấp, có thể xảy ra vết cắt ở giữa đường.')

                            with gr.Column():
                                hard_cut_string = gr.Textbox(label='Dây cắt cứng', value='\\n\\n\\n', info='Chuỗi biểu thị sự cắt cứng giữa các phần văn bản. Giúp ngăn chặn sự chồng chéo không mong muốn.')
                                min_chars = gr.Number(label='Bỏ qua các khối nhỏ', value=0, info='Bỏ qua các khối Hard Cut có ký tự nhỏ hơn hoặc bằng số này')

                    with gr.Row():
                        start_button = gr.Button("Bắt đầu đào tạo LoRA", variant='primary', interactive=not mu)
                        stop_button = gr.Button("Ngắt", interactive=not mu)

                    output = gr.Markdown(value="Sẵn sàng")

        with gr.Tab('Đánh giá sự bối rối', elem_id='evaluate-tab'):
            with gr.Row():
                with gr.Column():
                    models = gr.Dropdown(utils.get_available_models(), label='Mô hình', multiselect=True, interactive=not mu)
                    evaluate_text_file = gr.Dropdown(choices=['wikitext', 'ptb', 'ptb_new'] + utils.get_datasets('training/datasets', 'txt')[1:], value='wikitext', label='Dữ liệu đầu vào', info='Tệp văn bản thô mà mô hình sẽ được đánh giá trên đó. Các tùy chọn đầu tiên được tải xuống tự động: wikitext, ptb và ptb_new. Các tùy chọn tiếp theo là các tệp văn bản cục bộ của bạn đang được đào tạo/tập dữ liệu.', interactive=not mu)
                    with gr.Row():
                        with gr.Column():
                            stride_length = gr.Slider(label='Stride', minimum=0, maximum=32768, value=512, step=256, info='Được sử dụng để thực hiện việc đánh giá nhanh hơn với chi phí chính xác. 1 = chậm nhất nhưng chính xác nhất. 512 là giá trị chung.')

                        with gr.Column():
                            max_length = gr.Slider(label='max_length', minimum=0, maximum=shared.settings['truncation_length_max'], value=0, step=256, info='Bối cảnh cho mỗi đánh giá. Nếu được đặt thành 0, độ dài ngữ cảnh tối đa cho mô hình sẽ được sử dụng.')

                    with gr.Row():
                        start_current_evaluation = gr.Button("Đánh giá mô hình tải", interactive=not mu)
                        start_evaluation = gr.Button("Đánh giá các mô hình được lựa chọn", interactive=not mu)
                        stop_evaluation = gr.Button("Ngắt", interactive=not mu)

                with gr.Column():
                    evaluation_log = gr.Markdown(value='')

            evaluation_table = gr.Dataframe(value=generate_markdown_table(), interactive=True)
            with gr.Row():
                save_comments = gr.Button('Lưu bình luận', elem_classes="small-button", interactive=not mu)
                refresh_table = gr.Button('Làm mới bảng', elem_classes="small-button", interactive=not mu)

    # Training events
    all_params = [lora_name, always_override, q_proj_en, v_proj_en, k_proj_en, o_proj_en, gate_proj_en, down_proj_en, up_proj_en, save_steps, micro_batch_size, batch_size, epochs, learning_rate, lr_scheduler_type, lora_rank, lora_alpha, lora_dropout, cutoff_len, dataset, eval_dataset, format, eval_steps, raw_text_file, overlap_len, newline_favor_len, higher_rank_limit, warmup_steps, optimizer, hard_cut_string, train_only_after, stop_at_loss, add_eos_token, min_chars, report_to]

    copy_from.change(do_copy_params, [copy_from] + all_params, all_params)
    start_button.click(do_train, all_params, output)
    stop_button.click(do_interrupt, None, None, queue=False)
    higher_rank_limit.change(change_rank_limit, [higher_rank_limit], [lora_rank, lora_alpha])

    # Evaluation events. For some reason, the interrupt event
    # doesn't work with the .then() syntax, so I write them one
    # by one in this ugly but functional way.
    ev = start_evaluation.click(calculate_perplexity, [models, evaluate_text_file, stride_length, max_length], evaluation_log, show_progress=False)
    ev.then(generate_markdown_table, None, evaluation_table, show_progress=False)

    ev_cur = start_current_evaluation.click(
        lambda: ['current model'], None, tmp).then(
        calculate_perplexity, [tmp, evaluate_text_file, stride_length, max_length], evaluation_log, show_progress=False)

    ev_cur.then(generate_markdown_table, None, evaluation_table, show_progress=False)

    stop_evaluation.click(None, None, None, cancels=[ev, ev_cur], queue=False)
    refresh_table.click(generate_markdown_table, None, evaluation_table, show_progress=True)
    save_comments.click(
        save_past_evaluations, evaluation_table, None).then(
        lambda: "Comments saved.", None, evaluation_log, show_progress=False)


def do_interrupt():
    global WANT_INTERRUPT
    WANT_INTERRUPT = True


def do_copy_params(lora_name: str, *args):
    f_name = f"{shared.args.lora_dir}/{clean_path(None, lora_name)}/training_parameters.json"
    if Path(f_name).is_file():
        with open(f_name, 'r', encoding='utf-8') as format_file:
            params: dict[str, str] = json.load(format_file)
    else:
        params = {}

    result = list()
    for i in range(0, len(PARAMETERS)):
        key = PARAMETERS[i]
        if key in params:
            result.append(params[key])
        else:
            result.append(args[i])

    return result


def change_rank_limit(use_higher_ranks: bool):
    mult = 2 if use_higher_ranks else 1
    return {"maximum": 1024 * mult, "__type__": "update"}, {"maximum": 2048 * mult, "__type__": "update"}


def clean_path(base_path: str, path: str):
    """Loại bỏ các ký hiệu bất thường và buộc xây dựng một đường dẫn tương ứng với thư mục dự định."""
    path = path.replace('\\', '/').replace('..', '_')
    if base_path is None:
        return path

    return f'{Path(base_path).absolute()}/{path}'


def backup_adapter(input_folder):
    # Get the creation date of the file adapter_model.bin
    try:
        adapter_file = Path(f"{input_folder}/adapter_model.bin")
        if adapter_file.is_file():

            logger.info("Sao lưu bộ điều hợp LoRA hiện có")
            creation_date = datetime.fromtimestamp(adapter_file.stat().st_ctime)
            creation_date_str = creation_date.strftime("Backup-%Y-%m-%d")

            # Create the new subfolder
            subfolder_path = Path(f"{input_folder}/{creation_date_str}")
            subfolder_path.mkdir(parents=True, exist_ok=True)

            # Check if the file already exists in the subfolder
            backup_adapter_file = Path(f"{input_folder}/{creation_date_str}/adapter_model.bin")
            if backup_adapter_file.is_file():
                print(" - Bản sao lưu đã tồn tại. Bỏ qua quá trình sao lưu.")
                return

            # Copy existing files to the new subfolder
            existing_files = Path(input_folder).iterdir()
            for file in existing_files:
                if file.is_file():
                    shutil.copy2(file, subfolder_path)
    except Exception as e:
        print("Đã xảy ra lỗi trong backup_adapter:", str(e))


def calc_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def do_train(lora_name: str, always_override: bool, q_proj_en: bool, v_proj_en: bool, k_proj_en: bool, o_proj_en: bool, gate_proj_en: bool, down_proj_en: bool, up_proj_en: bool, save_steps: int, micro_batch_size: int, batch_size: int, epochs: int, learning_rate: str, lr_scheduler_type: str, lora_rank: int, lora_alpha: int, lora_dropout: float, cutoff_len: int, dataset: str, eval_dataset: str, format: str, eval_steps: int, raw_text_file: str, overlap_len: int, newline_favor_len: int, higher_rank_limit: bool, warmup_steps: int, optimizer: str, hard_cut_string: str, train_only_after: str, stop_at_loss: float, add_eos_token: bool, min_chars: int, report_to: str):

    if shared.args.monkey_patch:
        from alpaca_lora_4bit.monkeypatch.peft_tuners_lora_monkey_patch import (
            replace_peft_model_with_int4_lora_model
        )
        replace_peft_model_with_int4_lora_model()

    global WANT_INTERRUPT
    WANT_INTERRUPT = False

    # == Input validation / processing ==
    yield "Chuẩn bị đầu vào..."
    lora_file_path = clean_path(None, lora_name)
    if lora_file_path.strip() == '':
        yield "Đầu vào tên tệp LoRA bị thiếu hoặc không hợp lệ."
        return

    lora_file_path = f"{Path(shared.args.lora_dir)}/{lora_file_path}"
    actual_lr = float(learning_rate)
    model_type = type(shared.model).__name__

    if model_type in MODEL_CLASSES:
        model_id = MODEL_CLASSES[model_type]
    else:
        model_id = "llama"
        if model_type == "PeftModelForCausalLM":
            if len(shared.lora_names) > 0:
                yield "Bạn đang cố gắng huấn luyện một LoRA trong khi bạn đã tải một LoRA khác. Điều này sẽ có tác dụng nhưng có thể gây ra những tác dụng không mong muốn. *(Vẫn sẽ tiếp tục sau 5 giây, nhấn `Interrupt` để dừng.)*"
                logger.warning("Đào tạo LoRA trên LoRA khác. Có thể có tác dụng không mong muốn.")
            else:
                yield "ID mẫu không khớp do tải LoRA. Hãy xem xét tải lại mô hình cơ sở. *(Vẫn sẽ tiếp tục sau 5 giây, nhấn `Interrup` để dừng.)*"
                logger.warning("ID mẫu không khớp do tải LoRA. Hãy xem xét tải lại mô hình cơ sở.")
        else:
            yield "ID mẫu không khớp do tải LoRA. Hãy xem xét tải lại mô hình cơ sở..)*"
            logger.warning(f"Chương trình đào tạo LoRA hiện chỉ được xác thực cho các mẫu LLaMA, OPT, GPT-J và GPT-NeoX. (Đã tìm thấy loại mô hình: {model_type})")

        time.sleep(5)

    if shared.args.loader == 'GPTQ-for-LLaMa' and not shared.args.monkey_patch:
        yield "Đào tạo LoRA với GPTQ-for-LLaMa yêu cầu tải bằng `--monkey-patch`"
        return

    if cutoff_len <= 0 or micro_batch_size <= 0 or batch_size <= 0 or actual_lr <= 0 or lora_rank <= 0 or lora_alpha <= 0:
        yield "Không thể nhập số 0."
        return

    gradient_accumulation_steps = batch_size // micro_batch_size
    shared.tokenizer.pad_token_id = 0
    shared.tokenizer.padding_side = "left"

    # Populate target_modules list with chosen X_proj modules. Llama-based models only atm, non-llama will revert to default behavior.
    def list_target_modules(model_id):
        if model_id != "llama" and model_id != "mistral":
            return model_to_lora_modules[model_id]

        available_modules = {
            "gate": gate_proj_en,
            "down": down_proj_en,
            "up": up_proj_en,
            "q": q_proj_en,
            "v": v_proj_en,
            "k": k_proj_en,
            "o": o_proj_en,
        }
        target_mods = [f"{name}_proj" for name, enabled in available_modules.items() if enabled]
        return target_mods

    def encode(text, add_bos_token):
        result = shared.tokenizer.encode(text, truncation=True, max_length=cutoff_len)
        # Check if the first two tokens are BOS
        if len(result) >= 2 and result[:2] == [shared.tokenizer.bos_token_id, shared.tokenizer.bos_token_id]:
            result = result[1:]

        if not add_bos_token and result[0] == shared.tokenizer.bos_token_id:
            result = result[1:]
        return result

    def tokenize(prompt, append_eos_token=False):

        if train_only_after == '' or train_only_after not in prompt:
            input_ids = encode(prompt, True)

            if append_eos_token and input_ids[-1] != shared.tokenizer.eos_token_id and len(input_ids) < cutoff_len:
                input_ids.append(shared.tokenizer.eos_token_id)

            input_ids = [shared.tokenizer.pad_token_id] * (cutoff_len - len(input_ids)) + input_ids
            labels = [1] * len(input_ids)

        else:
            ind = prompt.index(train_only_after) + len(train_only_after)
            before_tokens = encode(prompt[:ind], True)
            after_tokens = encode(prompt[ind:], False)

            if append_eos_token and after_tokens[-1] != shared.tokenizer.eos_token_id:
                after_tokens.append(shared.tokenizer.eos_token_id)

            full_length = len(after_tokens) + len(before_tokens)
            if full_length > cutoff_len:
                after_tokens = after_tokens[:cutoff_len - len(before_tokens)]
            else:
                before_tokens = [shared.tokenizer.pad_token_id] * (cutoff_len - full_length) + before_tokens

            input_ids = before_tokens + after_tokens
            labels = [-100] * len(before_tokens) + [1] * len(after_tokens)

        input_ids = torch.tensor(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(shared.tokenizer.pad_token_id),
        }

    train_template.clear()

    # == Prep the dataset, format, etc ==
    if raw_text_file not in ['None', '']:
        train_template["template_type"] = "raw_text"
        logger.info("Tải tệp dữ liệu văn bản thô")
        fullpath = clean_path('training/datasets', f'{raw_text_file}')
        fullpath = Path(fullpath)
        if fullpath.is_dir():
            logger.info('Thư mục lộ trình đào tạo {}'.format(raw_text_file))
            raw_text = ""
            file_paths = sorted(fullpath.glob('*.txt'), key=lambda path: natural_keys(path.name))
            for file_path in file_paths:
                if file_path.is_file():
                    with file_path.open('r', encoding='utf-8') as file:
                        raw_text += file.read().replace('\r', '')

                    logger.info(f"Đã tải tệp đào tạo: {file_path.name}")
        else:
            with open(clean_path('training/datasets', f'{raw_text_file}.txt'), 'r', encoding='utf-8') as file:
                raw_text = file.read().replace('\r', '')

        cut_string = hard_cut_string.replace('\\n', '\n')
        eos_added = 0
        out_tokens = []
        for text_part in raw_text.split(cut_string):
            if len(text_part.strip()) <= min_chars:
                continue

            tokens = shared.tokenizer.encode(text_part)
            if add_eos_token:
                tokens.append(shared.tokenizer.eos_token_id)
                eos_added += 1

            step = cutoff_len - overlap_len
            if step <= 0:
                yield f"Error: overlap_len ({overlap_len}) không thể lớn hơn hoặc bằng cutoff_len ({cutoff_len})"
                return

            out_tokens.extend(split_chunks(tokens, cutoff_len, step))

        if eos_added > 0:
            print(f"EOS được thêm vào {eos_added} khối văn bản")

        del raw_text  # Note: could be a gig for a large dataset, so delete redundant data as we go to be safe on RAM
        text_chunks = [shared.tokenizer.decode(x) for x in out_tokens]
        del out_tokens
        if newline_favor_len > 0:
            text_chunks = [cut_chunk_for_newline(x, newline_favor_len) for x in text_chunks]

        train_data = Dataset.from_list([tokenize(x) for x in text_chunks])
        del text_chunks
        eval_data = None
    else:
        if dataset in ['None', '']:
            yield "Thiếu lựa chọn đầu vào tập dữ liệu, không thể tiếp tục."
            return

        if format in ['None', '']:
            yield "Thiếu lựa chọn định dạng đầu vào, không thể tiếp tục."
            return

        train_template["template_type"] = "dataset"

        with open(clean_path('training/formats', f'{format}.json'), 'r', encoding='utf-8-sig') as formatFile:
            format_data: dict[str, str] = json.load(formatFile)

        # == store training prompt ==
        for _, value in format_data.items():
            prompt_key = f"template_{len(train_template)}"
            train_template[prompt_key] = value

        def generate_prompt(data_point: dict[str, str]):
            for options, data in format_data.items():
                if set(options.split(',')) == set(x[0] for x in data_point.items() if (type(x[1]) is str and len(x[1].strip()) > 0)):
                    for key, val in data_point.items():
                        if type(val) is str:
                            data = data.replace(f'%{key}%', val)
                    return data
            raise RuntimeError(f'Data-point "{data_point}" không có bộ khóa nào khớp trong định dạng "{list(format_data.keys())}"')

        def generate_and_tokenize_prompt(data_point):
            prompt = generate_prompt(data_point)
            return tokenize(prompt, add_eos_token)

        logger.info("Đang tải tập dữ liệu JSON")
        data = load_dataset("json", data_files=clean_path('training/datasets', f'{dataset}.json'))
        train_data = data['train'].map(generate_and_tokenize_prompt, new_fingerprint='%030x' % random.randrange(16**30))

        if eval_dataset == 'None':
            eval_data = None
        else:
            eval_data = load_dataset("json", data_files=clean_path('training/datasets', f'{eval_dataset}.json'))
            eval_data = eval_data['train'].map(generate_and_tokenize_prompt, new_fingerprint='%030x' % random.randrange(16**30))

    # == We MUST reload model if it went through any previous training, even failed one ==
    if shared.model_dirty_from_training:
        selected_model = shared.model_name
        if selected_model:
            print("\033[1;31;1m(Mô hình đã được sửa đổi trong quá trình đào tạo trước đó, cần phải tải lại...)\033[0;37;0m")
            try:
                yield f"Reloading {selected_model}..."
                reload_model()
                if shared.model is not None:
                    print("Đã tải lại mô hình OK, tiếp tục đào tạo.")
                else:
                    return f"Không tải được {selected_model}."
            except:
                exc = traceback.format_exc()
                logger.error('Lỗi khi tải lại mô hình.')
                print(exc)
                return exc.replace('\n', '\n\n')

    # == Start prepping the model itself ==
    if not hasattr(shared.model, 'lm_head') or hasattr(shared.model.lm_head, 'weight'):
        logger.info("Mô hình đã sẵn sàng")
        if 'quantization_config' in shared.model.config.to_dict():
            prepare_model_for_kbit_training(shared.model)

    # base model is now frozen and should not be reused for any other LoRA training than this one
    shared.model_dirty_from_training = True

    logger.info("Đang chuẩn bị cho việc đào tạo")
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=list_target_modules(model_id),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # == Backup the existing adapter ==
    if not always_override:
        backup_adapter(lora_file_path)

    # == get model trainable params
    model_trainable_params, model_all_params = calc_trainable_parameters(shared.model)

    try:
        logger.info("Taọ mô hình LoRA")
        lora_model = get_peft_model(shared.model, config)
        if not always_override and Path(f"{lora_file_path}/adapter_model.bin").is_file():
            logger.info("Loading existing LoRA data")
            state_dict_peft = torch.load(f"{lora_file_path}/adapter_model.bin", weights_only=True)
            set_peft_model_state_dict(lora_model, state_dict_peft)
    except:
        yield traceback.format_exc().replace('\n', '\n\n')
        return

    if shared.args.monkey_patch:
        from alpaca_lora_4bit.autograd_4bit import Autograd4bitQuantLinear
        from alpaca_lora_4bit.models import Linear4bitLt
        for _, m in lora_model.named_modules():
            if isinstance(m, Autograd4bitQuantLinear) or isinstance(m, Linear4bitLt):
                if m.is_v1_model:
                    m.zeros = m.zeros.half()
                m.scales = m.scales.half()

    class Tracked():
        def __init__(self):
            self.current_steps = 0
            self.max_steps = 0
            self.did_save = False

    tracked = Tracked()
    actual_save_steps = math.ceil(save_steps / gradient_accumulation_steps)

    class Callbacks(transformers.TrainerCallback):
        def on_step_begin(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            tracked.current_steps = state.global_step * gradient_accumulation_steps
            tracked.max_steps = state.max_steps * gradient_accumulation_steps
            if WANT_INTERRUPT:
                control.should_epoch_stop = True
                control.should_training_stop = True
            elif state.global_step > 0 and actual_save_steps > 0 and state.global_step % actual_save_steps == 0:
                lora_model.save_pretrained(f"{lora_file_path}/checkpoint-{tracked.current_steps}/")
                # Save log
                with open(f"{lora_file_path}/checkpoint-{tracked.current_steps}/training_log.json", 'w', encoding='utf-8') as file:
                    json.dump(train_log, file, indent=2)
                # == Save training prompt ==
                with open(f"{lora_file_path}/checkpoint-{tracked.current_steps}/training_prompt.json", 'w', encoding='utf-8') as file:
                    json.dump(train_template, file, indent=2)

        def on_substep_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            tracked.current_steps += 1
            if WANT_INTERRUPT:
                control.should_epoch_stop = True
                control.should_training_stop = True

        def on_log(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, logs, **kwargs):
            train_log.update(logs)
            train_log.update({"current_steps": tracked.current_steps})
            if WANT_INTERRUPT:
                print("\033[1;31;1mĐã bị ngắt bởi người dùng\033[0;37;0m")

            print(f"\033[1;30;40mStep: {tracked.current_steps} \033[0;37;0m", end='')
            if 'loss' in logs:
                loss = float(logs['loss'])
                if loss <= stop_at_loss:
                    control.should_epoch_stop = True
                    control.should_training_stop = True
                    print(f"\033[1;31;1mdùng mất mất đạt {stop_at_loss}.\033[0;37;0m")

    # Fix training for mixed precision models
    for param in shared.model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    trainer = transformers.Trainer(
        model=lora_model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=transformers.TrainingArguments(
            report_to=report_to if report_to != "None" else None,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=math.ceil(warmup_steps / gradient_accumulation_steps),
            num_train_epochs=epochs,
            learning_rate=actual_lr,
            fp16=False if shared.args.cpu or shared.args.bf16 else True,
            bf16=shared.args.bf16,
            optim=optimizer,
            logging_steps=2 if stop_at_loss > 0 else 5,
            evaluation_strategy="steps" if eval_data is not None else "no",
            eval_steps=math.ceil(eval_steps / gradient_accumulation_steps) if eval_data is not None else None,
            save_strategy="steps" if eval_data is not None else "no",
            output_dir=lora_file_path,
            lr_scheduler_type=lr_scheduler_type,
            load_best_model_at_end=eval_data is not None,
            # TODO: Enable multi-device support
            ddp_find_unused_parameters=None,
            no_cuda=shared.args.cpu,
            use_ipex=True if is_torch_xpu_available() and not shared.args.cpu else False
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(shared.tokenizer, mlm=False),
        callbacks=list([Callbacks()])
    )

    lora_model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        lora_model = torch.compile(lora_model)

    # == Save parameters for reuse ==
    with open(f"{lora_file_path}/training_parameters.json", 'w', encoding='utf-8') as file:
        vars = locals()
        json.dump({x: vars[x] for x in PARAMETERS}, file, indent=2)

    # == Save training prompt ==
    with open(f"{lora_file_path}/training_prompt.json", 'w', encoding='utf-8') as file:
        json.dump(train_template, file, indent=2)

    # == Main run and monitor loop ==
    logger.info("Bắt đầu đào tạo")
    yield "Đang bắt đầu..."

    lora_trainable_param, lora_all_param = calc_trainable_parameters(lora_model)

    projections_string = ", ".join([projection.replace("_proj", "") for projection in list_target_modules(model_id)])

    print(f"Đào tạo '{model_id}' mô hình sử dụng ({projections_string}) phép chiếu")

    if lora_all_param > 0:
        print(f"Thông số có thể đào tạo: {lora_trainable_param:,d} ({100 * lora_trainable_param / lora_all_param:.4f} %), Tất cả thông số: {lora_all_param:,d} (Mô hình: {model_all_params:,d})")

    train_log.update({"base_model_name": shared.model_name})
    train_log.update({"base_model_class": shared.model.__class__.__name__})
    train_log.update({"base_loaded_in_4bit": getattr(lora_model, "is_loaded_in_4bit", False)})
    train_log.update({"base_loaded_in_8bit": getattr(lora_model, "is_loaded_in_8bit", False)})
    train_log.update({"projections": projections_string})

    if stop_at_loss > 0:
        print(f"Giám sát tổn thất \033[1;31;1m(Tự động dừng lại tại: {stop_at_loss})\033[0;37;0m")

    if WANT_INTERRUPT:
        yield "Đã ngắt trước khi bắt đầu."
        return

    def log_train_dataset(trainer):
        decoded_entries = []
        # Try to decode the entries and write the log file
        try:
            # Iterate over the first 10 elements in the dataset (or fewer if there are less than 10)
            for i in range(min(10, len(trainer.train_dataset))):
                decoded_text = shared.tokenizer.decode(trainer.train_dataset[i]['input_ids'])
                decoded_entries.append({"value": decoded_text})

            # Write the log file
            Path('logs').mkdir(exist_ok=True)
            with open(Path('logs/train_dataset_sample.json'), 'w') as json_file:
                json.dump(decoded_entries, json_file, indent=4)

            logger.info("Tệp log 'train_dataset_sample.json' được tạo trong thư mục 'log'.")
        except Exception as e:
            logger.error(f"Không thể tạo tệp nhật ký do lỗi: {e}")

    def threaded_run():
        log_train_dataset(trainer)
        trainer.train()
        # Note: save in the thread in case the gradio thread breaks (eg browser closed)
        lora_model.save_pretrained(lora_file_path)
        logger.info("Quá trình đào tạo LoRA đã hoàn tất và được lưu.")
        # Save log
        with open(f"{lora_file_path}/training_log.json", 'w', encoding='utf-8') as file:
            json.dump(train_log, file, indent=2)

    thread = threading.Thread(target=threaded_run)
    thread.start()
    last_step = 0
    start_time = time.perf_counter()

    while thread.is_alive():
        time.sleep(0.5)
        if WANT_INTERRUPT:
            yield "Đang gián đoạn, vui lòng đợi... *(Chạy sẽ dừng sau khi bước đào tạo hiện tại hoàn tất.)*"

        elif tracked.current_steps != last_step:
            last_step = tracked.current_steps
            time_elapsed = time.perf_counter() - start_time
            if time_elapsed <= 0:
                timer_info = ""
                total_time_estimate = 999
            else:
                its = tracked.current_steps / time_elapsed
                if its > 1:
                    timer_info = f"`{its:.2f}` it/s"
                else:
                    timer_info = f"`{1.0/its:.2f}` s/it"

                total_time_estimate = (1.0 / its) * (tracked.max_steps)

            yield f"Đang chạy... **{tracked.current_steps}** / **{tracked.max_steps}** ... {timer_info}, {format_time(time_elapsed)} / {format_time(total_time_estimate)} ... {format_time(total_time_estimate - time_elapsed)} còn lại"

    # Saving in the train thread might fail if an error occurs, so save here if so.
    if not tracked.did_save:
        logger.info("Quá trình đào tạo hoàn tất, đang lưu")
        lora_model.save_pretrained(lora_file_path)

    if WANT_INTERRUPT:
        logger.info("Quá trình đào tạo gián đoạn.")
        yield f"Đã ngắt. LoRA chưa hoàn chỉnh đã được lưu vào `{lora_file_path}`."
    else:
        logger.info("Hoàn thành đào tạo!")
        yield f"Xong! LoRA đã lưu vào `{lora_file_path}`.\n\nTrước khi thử nghiệm LoRA mới của bạn, trước tiên hãy đảm bảo tải lại mô hình vì nó hiện đã bị bẩn sau quá trình đào tạo."


def split_chunks(arr, size, step):
    for i in range(0, len(arr), step):
        yield arr[i:i + size]


def cut_chunk_for_newline(chunk: str, max_length: int):
    if '\n' not in chunk:
        return chunk

    first_newline = chunk.index('\n')
    if first_newline < max_length:
        chunk = chunk[first_newline + 1:]

    if '\n' not in chunk:
        return chunk

    last_newline = chunk.rindex('\n')
    if len(chunk) - last_newline < max_length:
        chunk = chunk[:last_newline]

    return chunk


def format_time(seconds: float):
    if seconds < 120:
        return f"`{seconds:.0f}` giây"

    minutes = seconds / 60
    if minutes < 120:
        return f"`{minutes:.0f}` phút"

    hours = minutes / 60
    return f"`{hours:.0f}` giờ"
