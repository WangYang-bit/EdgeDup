import logging
import colorlog


def setup_logging(log_file_path):
    formatter = logging.Formatter('%(asctime)s :%(levelname)s:[%(threadName)s] %(message)s')


    logger = logging.getLogger("logger")

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_color_handler = colorlog.StreamHandler()
    console_color_handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s %(levelname)s:[%(threadName)s] %(message)s",
        log_colors={
            'DEBUG': 'blue',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    console_color_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_color_handler)

    return logger


