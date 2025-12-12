import pandas as pd
from pathlib import Path
from datetime import datetime, UTC
from typing import Optional, Dict
import fast_f1_utils as ffu
import track_info
import battle_detector
import clickhouse_connector

# Constants
PROJECT_ROOT = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"

if __name__ == '__main__':
    battles = battle_detector.detect_races_battles(2025, "MONZA", "R")
    connector = clickhouse_connector.ClickHouseConnector(
        host='n8deexcqaz.germanywestcentral.azure.clickhouse.cloud',
        database='default',
        username='default',
        password='gQRh0ZP_5eBwn'
    )
    connector.insert_battles(battles)
