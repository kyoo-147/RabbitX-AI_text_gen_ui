from pathlib import Path

import torch
from peft import PeftModel
from transformers import is_torch_xpu_available

import modules.shared as shared
from modules.logging_colors import logger
from modules.models import reload_model


def add_lora_to_model(lora_names):
    if 'GPTQForCausalLM' in shared.model.__class__.__name__ or shared.args.loader == 'AutoGPTQ':
        add_lora_autogptq(lora_names)
    elif shared.model.__class__.__name__ in ['Exllamav2Model', 'Exllamav2HF'] or shared.args.loader in ['ExLlamav2', 'ExLlamav2_HF']:
        add_lora_exllamav2(lora_names)
    else:
        add_lora_transformers(lora_names)


def get_lora_path(lora_name):
    p = Path(lora_name)
    if p.exists():
        lora_name = p.parts[-1]

    return Path(f"{shared.args.lora_dir}/{lora_name}")


def add_lora_exllamav2(lora_names):

    from exllamav2 import ExLlamaV2Lora

    if isinstance(shared.model.loras, list):
        for lora in shared.model.loras:
            lora.unload()

    if len(lora_names) > 0:
        logger.info("Áp dụng các LoRA sau đây để {}: {}".format(shared.model_name, ', '.join(lora_names)))
        shared.model.loras = []
        for lora_name in lora_names:
            lora_path = get_lora_path(lora_name)
            if shared.model.__class__.__name__ == 'Exllamav2Model':
                lora = ExLlamaV2Lora.from_directory(shared.model.model, str(lora_path))
            else:
                lora = ExLlamaV2Lora.from_directory(shared.model.ex_model, str(lora_path))

            shared.model.loras.append(lora)

        shared.lora_names = lora_names
    else:
        shared.lora_names = []
        shared.model.loras = None


def add_lora_autogptq(lora_names):
    '''
    Chuyển thể từ https://github.com/Ph0rk0z/text-generation-webui-testing
    '''

    try:
        from auto_gptq import get_gptq_peft_model
        from auto_gptq.utils.peft_utils import GPTQLoraConfig
    except:
        logger.error("Phiên bản AutoGPTQ này không hỗ trợ LoRA. Bạn cần cài đặt từ nguồn hoặc đợi bản phát hành mới.")
        return

    if len(lora_names) == 0:
        reload_model()

        shared.lora_names = []
        return
    else:
        if len(lora_names) > 1:
            logger.warning('AutoGPTQ hiện chỉ có thể hoạt động với 1 LoRA. Chỉ cái đầu tiên trong danh sách sẽ được tải.')
        if not shared.args.no_inject_fused_attention:
            logger.warning('Chú ý hợp nhất + AutoGPTQ có thể làm hỏng quá trình tải Lora. Vô hiệu hóa nó.')

        peft_config = GPTQLoraConfig(
            inference_mode=True,
        )

        lora_path = get_lora_path(lora_names[0])
        logger.info("Áp dụng các LoRA sau đây để {}: {}".format(shared.model_name, ', '.join([lora_names[0]])))
        shared.model = get_gptq_peft_model(shared.model, peft_config, lora_path)
        shared.lora_names = [lora_names[0]]
        return


def add_lora_transformers(lora_names):
    prior_set = set(shared.lora_names)
    added_set = set(lora_names) - prior_set
    removed_set = prior_set - set(lora_names)

    # If no LoRA needs to be added or removed, exit
    if len(added_set) == 0 and len(removed_set) == 0:
        return

    # Add a LoRA when another LoRA is already present
    if len(removed_set) == 0 and len(prior_set) > 0 and "__merged" not in shared.model.peft_config.keys():
        logger.info(f"Thêm (các) LoRA có tên {added_set} đến mô hình")
        for lora in added_set:
            shared.model.load_adapter(get_lora_path(lora), lora)

        if len(lora_names) > 1:
            merge_loras()

        shared.lora_names = lora_names
        return

    # If any LoRA needs to be removed, start over
    if len(removed_set) > 0:
        shared.model = shared.model.unload()

    if len(lora_names) > 0:
        params = {}
        if not shared.args.cpu:
            if shared.args.load_in_4bit or shared.args.load_in_8bit:
                params['peft_type'] = shared.model.dtype
            else:
                params['dtype'] = shared.model.dtype
                if hasattr(shared.model, "hf_device_map"):
                    params['device_map'] = {"base_model.model." + k: v for k, v in shared.model.hf_device_map.items()}

        logger.info("Áp dụng các LoRA sau đây để {}: {}".format(shared.model_name, ', '.join(lora_names)))
        shared.model = PeftModel.from_pretrained(shared.model, get_lora_path(lora_names[0]), adapter_name=lora_names[0], **params)
        for lora in lora_names[1:]:
            shared.model.load_adapter(get_lora_path(lora), lora)

        if len(lora_names) > 1:
            merge_loras()

        if not shared.args.load_in_8bit and not shared.args.cpu:
            shared.model.half()
            if not hasattr(shared.model, "hf_device_map"):
                if torch.backends.mps.is_available():
                    device = torch.device('mps')
                    shared.model = shared.model.to(device)
                elif is_torch_xpu_available():
                    device = torch.device("xpu:0")
                    shared.model = shared.model.to(device)
                else:
                    shared.model = shared.model.cuda()

    shared.lora_names = lora_names


def merge_loras():
    if len(list({shared.model.peft_config[adapter].r for adapter in shared.model.peft_config.keys()})) > 1:
        logger.warning("Không thể hợp nhất các LoRA đã tải vì chúng có thứ hạng khác nhau. Chỉ cái đầu tiên sẽ hoạt động.")
        return

    shared.model.add_weighted_adapter(shared.lora_names, [1] * len(shared.lora_names), "__merged")
    shared.model.set_adapter("__merged")
