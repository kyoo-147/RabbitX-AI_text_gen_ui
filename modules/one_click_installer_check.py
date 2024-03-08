from pathlib import Path

from modules.logging_colors import logger

if Path('../webui.py').exists():
    logger.warning('\nCó vẻ như bạn đang chạy phiên bản cũ của '
                   'one-click-installers.\n'
                   'Có vẻ như bạn đang chạy phiên bản lỗi thời của:\n'
                   'https://github.com/oobabooga/text-generation-webui/wiki/Migrating-an-old-one%E2%80%90click-install')
