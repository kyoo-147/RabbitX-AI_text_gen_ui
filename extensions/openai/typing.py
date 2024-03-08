import json
import time
from typing import Dict, List

from pydantic import BaseModel, Field


class GenerationOptions(BaseModel):
    preset: str | None = Field(default=None, description="Tên của tệp trong text-gen-webui/presets (không có phần mở rộng .yaml). Các tham số lấy mẫu bị ghi đè bởi tùy chọn này là các khóa trong hàm default_preset() trong module/presets.py.")
    min_p: float = 0
    dynamic_temperature: bool = False
    dynatemp_low: float = 1
    dynatemp_high: float = 1
    dynatemp_exponent: float = 1
    smoothing_factor: float = 0
    top_k: int = 0
    repetition_penalty: float = 1
    repetition_penalty_range: int = 1024
    typical_p: float = 1
    tfs: float = 1
    top_a: float = 0
    epsilon_cutoff: float = 0
    eta_cutoff: float = 0
    guidance_scale: float = 1
    negative_prompt: str = ''
    penalty_alpha: float = 0
    mirostat_mode: int = 0
    mirostat_tau: float = 5
    mirostat_eta: float = 0.1
    temperature_last: bool = False
    do_sample: bool = True
    seed: int = -1
    encoder_repetition_penalty: float = 1
    no_repeat_ngram_size: int = 0
    min_length: int = 0
    num_beams: int = 1
    length_penalty: float = 1
    early_stopping: bool = False
    truncation_length: int = 0
    max_tokens_second: int = 0
    prompt_lookup_num_tokens: int = 0
    custom_token_bans: str = ""
    sampler_priority: List[str] | str | None = Field(default=None, description="Danh sách các bộ lấy mẫu trong đó các mục đầu tiên sẽ xuất hiện đầu tiên trong ngăn xếp. Ví dụ: [\"top_k\", \"temperature\", \"top_p\"].")
    auto_max_new_tokens: bool = False
    ban_eos_token: bool = False
    add_bos_token: bool = True
    skip_special_tokens: bool = True
    grammar_string: str = ""


class CompletionRequestParams(BaseModel):
    model: str | None = Field(default=None, description="Tham số không được sử dụng. Để thay đổi mô hình, hãy sử dụng điểm cuối /v1/internal/model/load.")
    prompt: str | List[str]
    best_of: int | None = Field(default=1, description="Tham số không được sử dụng.")
    echo: bool | None = False
    frequency_penalty: float | None = 0
    logit_bias: dict | None = None
    logprobs: int | None = None
    max_tokens: int | None = 16
    n: int | None = Field(default=1, description="Tham số không được sử dụng.")
    presence_penalty: float | None = 0
    stop: str | List[str] | None = None
    stream: bool | None = False
    suffix: str | None = None
    temperature: float | None = 1
    top_p: float | None = 1
    user: str | None = Field(default=None, description="Tham số không được sử dụng.")


class CompletionRequest(GenerationOptions, CompletionRequestParams):
    pass


class CompletionResponse(BaseModel):
    id: str
    choices: List[dict]
    created: int = int(time.time())
    model: str
    object: str = "text_completion"
    usage: dict


class ChatCompletionRequestParams(BaseModel):
    messages: List[dict]
    model: str | None = Field(default=None, description="Tham số không được sử dụng. Để thay đổi mô hình, hãy sử dụng /v1/internal/model/load endpoint.")
    frequency_penalty: float | None = 0
    function_call: str | dict | None = Field(default=None, description="Tham số không được sử dụng.")
    functions: List[dict] | None = Field(default=None, description="Tham số không được sử dụng.")
    logit_bias: dict | None = None
    max_tokens: int | None = None
    n: int | None = Field(default=1, description="Tham số không được sử dụng.")
    presence_penalty: float | None = 0
    stop: str | List[str] | None = None
    stream: bool | None = False
    temperature: float | None = 1
    top_p: float | None = 1
    user: str | None = Field(default=None, description="Tham số không được sử dụng.")

    mode: str = Field(default='instruct', description="Tùy chọn hợp lệ: instruct, chat, chat-instruct.")

    instruction_template: str | None = Field(default=None, description="Mẫu hướng dẫn được xác định trong phần tạo văn bản-webui/hướng dẫn-mẫu. Nếu không được đặt, mẫu chính xác sẽ được tự động lấy từ siêu dữ liệu mô hình.")
    instruction_template_str: str | None = Field(default=None, description="A Jinja2 instruction template. If set, will take precedence over everything else.")

    character: str | None = Field(default=None, description="A character defined under text-generation-webui/characters. If not set, the default \"Assistant\" character will be used.")
    user_name: str | None = Field(default=None, description="Tên của bạn (người dùng). Theo mặc định, nó là \"You\".", alias="name1")
    bot_name: str | None = Field(default=None, description="Ghi đè giá trị được đặt theo trường ký tự.", alias="name2")
    context: str | None = Field(default=None, description="Ghi đè giá trị được đặt theo trường ký tự.")
    greeting: str | None = Field(default=None, description="Ghi đè giá trị được đặt theo trường ký tự.")
    chat_template_str: str | None = Field(default=None, description="Mẫu Jinja2 để trò chuyện.")

    chat_instruct_command: str | None = None

    continue_: bool = Field(default=False, description="Làm cho tin nhắn bot cuối cùng trong lịch sử được tiếp tục thay vì bắt đầu một tin nhắn mới.")


class ChatCompletionRequest(GenerationOptions, ChatCompletionRequestParams):
    pass


class ChatCompletionResponse(BaseModel):
    id: str
    choices: List[dict]
    created: int = int(time.time())
    model: str
    object: str = "chat.completion"
    usage: dict


class EmbeddingsRequest(BaseModel):
    input: str | List[str] | List[int] | List[List[int]]
    model: str | None = Field(default=None, description="Tham số không được sử dụng. Để thay đổi mô hình, hãy đặt OPENEDAI_EMBEDDING_MODEL và OPENEDAI_EMBEDDING_DEVICE biến môi trường trước khi khởi động máy chủ.")
    encoding_format: str = Field(default="float", description="Có thể là float hoặc base64.")
    user: str | None = Field(default=None, description="Tham số không được sử dụng.")


class EmbeddingsResponse(BaseModel):
    index: int
    embedding: List[float]
    object: str = "embedding"


class EncodeRequest(BaseModel):
    text: str


class EncodeResponse(BaseModel):
    tokens: List[int]
    length: int


class DecodeRequest(BaseModel):
    tokens: List[int]


class DecodeResponse(BaseModel):
    text: str


class TokenCountResponse(BaseModel):
    length: int


class LogitsRequestParams(BaseModel):
    prompt: str
    use_samplers: bool = False
    top_logits: int | None = 50
    frequency_penalty: float | None = 0
    max_tokens: int | None = 16
    presence_penalty: float | None = 0
    temperature: float | None = 1
    top_p: float | None = 1


class LogitsRequest(GenerationOptions, LogitsRequestParams):
    pass


class LogitsResponse(BaseModel):
    logits: Dict[str, float]


class ModelInfoResponse(BaseModel):
    model_name: str
    lora_names: List[str]


class ModelListResponse(BaseModel):
    model_names: List[str]


class LoadModelRequest(BaseModel):
    model_name: str
    args: dict | None = None
    settings: dict | None = None


class LoraListResponse(BaseModel):
    lora_names: List[str]


class LoadLorasRequest(BaseModel):
    lora_names: List[str]


def to_json(obj):
    return json.dumps(obj.__dict__, indent=4)


def to_dict(obj):
    return obj.__dict__
