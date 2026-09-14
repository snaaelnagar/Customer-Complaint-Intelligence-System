"""
exception.py

Custom exception class for the
Machine Learning / NLP Customer Complaint Intelligence System.

Purpose:
- Capture detailed error information
- Include file name and line number
- Make debugging easier
"""

import sys
from typing import Any


def get_error_details(error: Exception, error_detail: Any) -> str:
    """
    Extract detailed exception information.

    Parameters
    ----------
    error : Exception
        Original exception object

    error_detail : Any
        Typically sys module

    Returns
    -------
    str
        Formatted error message with
        file name and line number
    """

    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in file [{file_name}] "
        f"at line [{line_number}] "
        f"with message [{str(error)}]"
    )


class CustomException(Exception):
    """
    Custom exception class.

    Adds:
    - File name
    - Line number
    - Original exception message
    """

    def __init__(self, error: Exception, error_detail: Any):
        super().__init__(str(error))

        self.error_message = get_error_details(
            error=error,
            error_detail=error_detail
        )

    def __str__(self) -> str:
        return self.error_message


if __name__ == "__main__":

    try:

        print("Testing CustomException")

        value = 10 / 0

    except Exception as e:

        raise CustomException(
            error=e,
            error_detail=sys
        )