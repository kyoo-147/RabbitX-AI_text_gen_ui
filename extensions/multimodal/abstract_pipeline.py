from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from PIL import Image
from transformers import is_torch_xpu_available


class AbstractMultimodalPipeline(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        'tên của đường ống, phải giống như trong --multimodal-pipeline'
        pass

    @staticmethod
    @abstractmethod
    def image_start() -> Optional[str]:
        'trả về chuỗi bắt đầu hình ảnh, biểu diễn chuỗi của mã thông báo bắt đầu hình ảnh hoặc Không có nếu không áp dụng'
        pass

    @staticmethod
    @abstractmethod
    def image_end() -> Optional[str]:
        'trả về chuỗi kết thúc hình ảnh, biểu diễn chuỗi của mã thông báo kết thúc hình ảnh hoặc Không có nếu không áp dụng'
        pass

    @staticmethod
    @abstractmethod
    def placeholder_token_id() -> int:
        'trả về id mã thông báo giữ chỗ'
        pass

    @staticmethod
    @abstractmethod
    def num_image_embeds() -> int:
        'trả về số lượng nhúng được sử dụng bởi một hình ảnh (ví dụ: 256 cho LLaVA)'
        pass

    @abstractmethod
    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        'chuyển tiếp hình ảnh thông qua đường dẫn tầm nhìn và trả lại phần nhúng của chúng'
        pass

    @staticmethod
    @abstractmethod
    def embed_tokens(input_ids: torch.Tensor) -> torch.Tensor:
        'mã thông báo nhúng, chức năng chính xác thay đổi tùy theo LLM, đối với LLaMA thì đó là `shared.model.model.embed_tokens`'
        pass

    @staticmethod
    @abstractmethod
    def placeholder_embeddings() -> torch.Tensor:
        'nhận phần nhúng giữ chỗ nếu có nhiều hình ảnh và `add_all_images_to_prompt` là Sai'
        pass

    def _get_device(self, setting_name: str, params: dict):
        if params[setting_name] is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "xpu:0" if is_torch_xpu_available() else "cpu")
        return torch.device(params[setting_name])

    def _get_dtype(self, setting_name: str, params: dict):
        return torch.float32 if int(params[setting_name]) == 32 else torch.float16
