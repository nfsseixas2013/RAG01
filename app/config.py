import logging
import sys
from logging.handlers import RotatingFileHandler
from os import environ
from pathlib import Path
from dynaconf import Dynaconf

settings = Dynaconf(settings_files=["./settings.toml","./.secrets.toml"])