import logging
import sys

def setup_logger(name: str = "ECG_Pipeline") -> logging.Logger:
    """Sets up and returns a configured logger."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler (clean log)
        from pathlib import Path
        log_path = Path(__file__).resolve().parent.parent / "pipeline.log"
        fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

logger = setup_logger()
