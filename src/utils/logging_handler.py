import os

import logging
from logging import Logger, FileHandler, Formatter


class Logger:
    def __init__(self, file_name: str) -> None:
        self.logger: Logger = logging.getLogger(__name__)
        self.file_handler: FileHandler = FileHandler(f'logs/{file_name}')
    
    def create_logger(self) -> Logger:
        '''
        Returns:
            self.logger: Logger
        '''
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        self.file_handler.setFormatter(Formatter('[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.file_handler)
        self.logger.setLevel(logging.DEBUG)

        return self.logger
