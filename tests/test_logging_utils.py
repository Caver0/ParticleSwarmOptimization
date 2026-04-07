from __future__ import annotations

import uuid

from pso_lab.io.logging_utils import setup_logger


def test_setup_logger_creates_log_file(tmp_path):
    logger_name = f"pso_test_{uuid.uuid4().hex}"
    logger = setup_logger(name=logger_name, log_dir=tmp_path)
    handlers = list(logger.handlers)

    try:
        logger.info("This message should be stored in the log file.")

        for handler in handlers:
            handler.flush()

        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1

        log_content = log_files[0].read_text(encoding="utf-8")
        assert "Log file created at" in log_content
        assert "This message should be stored in the log file." in log_content
        assert logger_name in log_files[0].name
    finally:
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
