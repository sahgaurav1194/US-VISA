"""
Custom exception class for the project.

Wraps any raised exception with the file name and line number where it
occurred, which makes debugging pipeline failures much faster.
"""

import sys


def error_message_detail(error, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    return f"Error occurred in script [{file_name}] at line [{line_no}] : {str(error)}"


class USvisaException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
