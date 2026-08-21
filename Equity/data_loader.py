import rqdatac as rq
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

load_dotenv("/Users/joycelu/cross-asset-quant-lab/.env.")
RICEQUANT_API_KEY = os.getenv("RICEQUANT_API_KEY")
rq.init("license", RICEQUANT_API_KEY)

