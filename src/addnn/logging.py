import logging
from typing import Optional

def init(filename: Optional[str] = None) -> None:
    logging.basicConfig(filename=filename, format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.DEBUG)
