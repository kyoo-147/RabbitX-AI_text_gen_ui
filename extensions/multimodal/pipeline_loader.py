import traceback
from importlib import import_module
from pathlib import Path
from typing import Tuple

from extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline
from modules import shared
from modules.logging_colors import logger


def _get_available_pipeline_modules():
    pipeline_path = Path(__file__).parent / 'pipelines'
    modules = [p for p in pipeline_path.iterdir() if p.is_dir()]
    return [m.name for m in modules if (m / 'pipelines.py').exists()]


def load_pipeline(params: dict) -> Tuple[AbstractMultimodalPipeline, str]:
    pipeline_modules = {}
    available_pipeline_modules = _get_available_pipeline_modules()
    for name in available_pipeline_modules:
        try:
            pipeline_modules[name] = import_module(f'extensions.multimodal.pipelines.{name}.pipelines')
        except:
            logger.warning(f'Không thể nhận đường ống đa phương thức từ {name}')
            logger.warning(traceback.format_exc())

    if shared.args.multimodal_pipeline is not None:
        for k in pipeline_modules:
            if hasattr(pipeline_modules[k], 'get_pipeline'):
                pipeline = getattr(pipeline_modules[k], 'get_pipeline')(shared.args.multimodal_pipeline, params)
                if pipeline is not None:
                    return (pipeline, k)
    else:
        model_name = shared.args.model.lower()
        for k in pipeline_modules:
            if hasattr(pipeline_modules[k], 'get_pipeline_from_model_name'):
                pipeline = getattr(pipeline_modules[k], 'get_pipeline_from_model_name')(model_name, params)
                if pipeline is not None:
                    return (pipeline, k)

    available = []
    for k in pipeline_modules:
        if hasattr(pipeline_modules[k], 'available_pipelines'):
            pipelines = getattr(pipeline_modules[k], 'available_pipelines')
            available += pipelines

    if shared.args.multimodal_pipeline is not None:
        log = f'Multimodal - ERROR: Không thể tải đa phương thức pipeline "{shared.args.multimodal_pipeline}", đường ống có sẵn là: {available}.'
    else:
        log = f'Multimodal - ERROR: Không thể xác định đường dẫn đa phương thức cho mô hình {shared.args.model}, vui lòng chọn một cách thủ công bằng cách sử dụng --multimodal-pipeline [PIPELINE]. Các đường ống có sẵn là: {available}.'
    logger.critical(f'{log} Vui lòng chỉ định đường dẫn chính xác hoặc tắt tiện ích mở rộng')
    raise RuntimeError(f'{log} Vui lòng chỉ định đường dẫn chính xác hoặc tắt tiện ích mở rộng')
