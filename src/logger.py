import os

import logging
from logging import Logger, FileHandler, Formatter


if not os.path.exists('logs'):
    os.makedirs('logs')

logger: Logger = logging.getLogger(__name__)

file_handler: FileHandler = FileHandler('logs/pipeline.log')
file_handler.setFormatter(Formatter('[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s'))

logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)
