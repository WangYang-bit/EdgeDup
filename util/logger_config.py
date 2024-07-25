import logging
import colorlog


def setup_logging(log_file_path):
    # 创建一个格式化器，定义日志的格式
    formatter = logging.Formatter('%(asctime)s :%(levelname)s:[%(threadName)s] %(message)s')


    # 创建 logger
    logger = logging.getLogger("logger")

    # 设置全局日志级别
    logger.setLevel(logging.DEBUG)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 创建颜色控制台处理器
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

    # 将处理器添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_color_handler)

    return logger


