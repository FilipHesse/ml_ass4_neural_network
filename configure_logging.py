#!/usr/bin/env python3
import logging
import os
from pathlib import Path

def configure_logging():

    Path(os.path.join(os.path.dirname(__file__), "log")).mkdir(parents=True, exist_ok=True)
    LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(filename = os.path.join(os.path.dirname(__file__), "log", "main.log"),
                        level = logging.DEBUG,
                        format=LOG_FORMAT,
                        filemode='w')