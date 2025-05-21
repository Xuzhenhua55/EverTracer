import sys
import time

class LoggerUtils:
    COLORS = {
        "HEADER": "\033[95m",
        "BLUE": "\033[94m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "ENDC": "\033[0m",
        "BOLD": "\033[1m",
        "GRAY": "\033[90m"
    }

    _log_file_path = None
    _log_file_params = None

    @staticmethod
    def set_log_file_path(log_file_path, params_str=None):
        LoggerUtils._log_file_path = log_file_path
        LoggerUtils._log_file_params = params_str
        # 写入参数信息到日志文件
        if log_file_path and params_str:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"[PARAMS] {params_str}\n")

    @staticmethod
    def _write_log(msg_type, msg):
        if LoggerUtils._log_file_path:
            with open(LoggerUtils._log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"[{msg_type} {LoggerUtils._timestamp()}] {msg}\n")

    @staticmethod
    def _timestamp():
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def info(msg):
        print(f"{LoggerUtils.COLORS['BLUE']}[INFO {LoggerUtils._timestamp()}] {msg}{LoggerUtils.COLORS['ENDC']}")
        LoggerUtils._write_log('INFO', msg)

    @staticmethod
    def success(msg):
        print(f"{LoggerUtils.COLORS['GREEN']}[SUCCESS {LoggerUtils._timestamp()}] {msg}{LoggerUtils.COLORS['ENDC']}")
        LoggerUtils._write_log('SUCCESS', msg)

    @staticmethod
    def warning(msg):
        print(f"{LoggerUtils.COLORS['YELLOW']}[WARNING {LoggerUtils._timestamp()}] {msg}{LoggerUtils.COLORS['ENDC']}")
        LoggerUtils._write_log('WARNING', msg)

    @staticmethod
    def error(msg, exit_program=False):
        print(f"{LoggerUtils.COLORS['RED']}[ERROR {LoggerUtils._timestamp()}] {msg}{LoggerUtils.COLORS['ENDC']}")
        LoggerUtils._write_log('ERROR', msg)
        if exit_program:
            sys.exit(1)

    @staticmethod
    def step(msg):
        print(f"{LoggerUtils.COLORS['HEADER']}[STEP] {msg}{LoggerUtils.COLORS['ENDC']}")

    @staticmethod
    def title(msg):
        print(f"\n{LoggerUtils.COLORS['BOLD']}========== {msg} =========={LoggerUtils.COLORS['ENDC']}\n")

    @staticmethod
    def line():
        print(f"{LoggerUtils.COLORS['GRAY']}" + "-" * 60 + f"{LoggerUtils.COLORS['ENDC']}")