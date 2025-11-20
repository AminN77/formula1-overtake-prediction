import unittest
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.overtake_detector import build_position_map
import pandas as pd



class TestBuildPositionMap(unittest.TestCase):

    def test_simple_case(self):
        """Test with simple 2-lap, 2-driver scenario"""
        # Create mock data
        data = {
            'LapNumber': [1, 1, 2, 2],
            'Driver': ['HAM', 'VER', 'HAM', 'VER'],
            'Position': [1, 2, 1, 2]
        }
        laps = pd.DataFrame(data)

        result = build_position_map(laps)

        # Expected structure
        expected = {
            1: {'HAM': 1, 'VER': 2},
            2: {'HAM': 1, 'VER': 2}
        }

        self.assertEqual(result, expected)