import inspect
import logging
import os
import sys


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        super().__init__()
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return record.levelno != self.passlevel
        else:
            assert isinstance(self.passlevel, object)
            return record.levelno == self.passlevel


_FORMAT = '%(asctime)-15s - %(levelname)s - [%(funcfile)s -> %(func)s] %(message)s'
_formatter = logging.Formatter(_FORMAT)
logging.basicConfig(level='INFO')
auto_logger = logging.getLogger('MatAL')
auto_logger.propagate = False
auto_logger.handlers = []
_ch1 = logging.StreamHandler(sys.stdout)
_ch1.setLevel(logging.INFO)
_ch1.setFormatter(_formatter)
_f1 = SingleLevelFilter(logging.INFO, False)
_ch1.addFilter(_f1)

_ch2 = logging.StreamHandler(sys.stderr)
_ch2.setLevel(logging.WARNING)
_ch2.setFormatter(_formatter)

auto_logger.addHandler(_ch1)
auto_logger.addHandler(_ch2)


def auto_log(message, level='info'):
    assert level in ['debug', 'info', 'warning', 'error', 'critical']
    func = inspect.currentframe().f_back.f_code
    getattr(auto_logger, level)(message, extra={
        'func': func.co_name,
        'funcfile': os.path.basename(func.co_filename)
    })
