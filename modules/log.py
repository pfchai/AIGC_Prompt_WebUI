# -*- coding: utf-8 -*-

import logging


def get_logger(name: str = 'webui'):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    stream_handler = logging.handlers.TimedRotatingFileHandler('log/webui.log', when='D', backupCount=14, encoding='utf-8')
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger