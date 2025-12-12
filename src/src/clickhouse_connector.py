import clickhouse_connect
import os
from typing import Optional, List, Union
from models import BattleRecord


class ClickHouseConnector:
    """Helper class for ClickHouse connections with typed BattleRecord support."""

    def __init__(self, host: Optional[str] = None, username: Optional[str] = None,
                 password: Optional[str] = None, database: str = "default"):
        self.host = host or os.getenv('CLICKHOUSE_HOST')
        self.username = username or os.getenv('CLICKHOUSE_USER', 'default')
        self.password = password or os.getenv('CLICKHOUSE_PASSWORD')
        self.database = database

        if not self.host or not self.password:
            raise ValueError("ClickHouse credentials not configured")

        self.client = clickhouse_connect.get_client(
            host=self.host,
            port=8443,
            username=self.username,
            password=self.password,
            database=self.database,
            verify=False
        )

    def insert_battles(self, battles: Union[List[BattleRecord], List[dict]]):
        """
        Insert battles into ClickHouse.

        Args:
            battles: List of BattleRecord objects or dictionaries
        """
        if not battles:
            print("No battles to insert")
            return

        # Convert BattleRecord objects to lists
        data = []
        for battle in battles:
            if isinstance(battle, BattleRecord):
                data.append(battle.to_list())
            elif isinstance(battle, dict):
                # Backward compatibility with dict format
                data.append([
                    battle['attacker'],
                    battle['defender'],
                    battle['overtake'],
                    battle['time_stamp'],
                    battle['attacker_speed'],
                    battle['defender_speed'],
                    battle['speed_difference'],
                    battle['lap_number'],
                    battle['safety_car'],
                    battle['yellow_flag'],
                    battle['attacker_tyre_compound'],
                    battle['defender_tyre_compound'],
                    battle['attacker_tyre_age'],
                    battle['defender_tyre_age'],
                    battle['tyre_age_difference'],
                    battle['track'],
                    battle['sector'],
                    battle['sector_type'],
                    battle['is_in_drs_zone'],
                    battle['drs_zone_length'],
                    battle['track_type'],
                    battle['attacker_qualification_rank'],
                    battle['defender_qualification_rank']
                ])
            else:
                raise TypeError(f"Expected BattleRecord or dict, got {type(battle)}")

        self.client.insert(
            'battles',
            data,
            column_names=BattleRecord.column_names()
        )

        print(f"✓ Inserted {len(data)} battles")

    def insert_single_battle(self, battle: BattleRecord):
        """Insert a single battle record."""
        self.insert_battles([battle])

    def close(self):
        """Close connection."""
        self.client.close()