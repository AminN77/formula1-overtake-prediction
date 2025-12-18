import fastf1
from pathlib import Path
import battle_detector
import clickhouse_connector

# Constants
PROJECT_ROOT = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"

# YEARS = range(2022, 2025)
YEARS = [2022]

def ingest_all_sessions():
    connector = clickhouse_connector.ClickHouseConnector(
        host='n8deexcqaz.germanywestcentral.azure.clickhouse.cloud',
        database='default',
        username='default',
        password='gjcvr.oSu~r1n'
    )

    for year in YEARS:
        print(f"\n📅 Processing season {year}")
        schedule = fastf1.get_event_schedule(year)

        for _, event in schedule.iterrows():
            event_name = event["EventName"]

            print(f"  🏁 {event_name}")

            try:
                battles = battle_detector.detect_races_battles(
                    year=year,
                    gp=event_name,  # safer than name
                    identifier="R"
                )

                if battles is None or len(battles) == 0:
                    print("      ⚠ No battles found")
                    continue

                connector.insert_battles(battles)

            except Exception as e:
                # Some sessions simply don't exist (e.g. FP3 on sprint weekends)
                print(f"      ❌ Skipped {event_name}: {e}")


if __name__ == "__main__":
    ingest_all_sessions()
