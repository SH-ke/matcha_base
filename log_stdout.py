# coding=utf-8
from tqdm.auto import tqdm
import time
import logging
import sys
import datetime
__is_setup_done = False


def setup_logging(log_prefix):
    global __is_setup_done

    if __is_setup_done:
        pass
    else:
        __log_file_name = "./loggings/{}-{}_log_file.txt".format(log_prefix,
                                                      datetime.datetime.utcnow().isoformat().replace(":", "-"))

        __log_format = '%(asctime)s - %(name)-30s - %(levelname)s - %(message)s'
        __console_date_format = '%Y-%m-%d %H:%M:%S'
        __file_date_format = '%Y-%m-%d %H-%M-%S'

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        console_formatter = logging.Formatter(__log_format, __console_date_format)

        file_formatter = logging.Formatter(__log_format, __file_date_format)
        file_handler = logging.FileHandler(__log_file_name, mode='a', delay=True)
        # file_handler = TqdmLoggingHandler2(__log_file_name, mode='a', delay=True)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

        tqdm_handler = TqdmLoggingHandler()
        tqdm_handler.setLevel(logging.DEBUG)
        tqdm_handler.setFormatter(console_formatter)
        root.addHandler(tqdm_handler)

        __is_setup_done = True

class TqdmLoggingHandler(logging.StreamHandler):

    def __init__(self, level=logging.NOTSET):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)
        # from https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit/38739634#38739634
        self.flush()


def example_long_procedure():
    setup_logging('setup')
    __logger = logging.getLogger('logger')
    __logger.setLevel(logging.DEBUG)
    for i in tqdm(range(10), unit_scale=True, dynamic_ncols=True, file=sys.stdout):
        time.sleep(.1)
        __logger.info('foo {}'.format(i))


if __name__ == "__main__":
    example_long_procedure()

