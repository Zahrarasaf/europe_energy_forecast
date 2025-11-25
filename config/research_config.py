from dataclasses import dataclass
from typing import List

@dataclass
class ResearchConfig:
    DATA_PATH: str = "data/europe_energy.csv"
    COUNTRIES: List[str] = ['DE', 'FR', 'IT', 'ES', 'UK', 'NL', 'BE', 'PL']
    TARGET_COUNTRY: str = 'DE'
    TEST_SIZE: float = 0.2
    SEQUENCE_LENGTH: int = 30
    LAGS: List[int] = [1, 7, 30]
    ROLLING_WINDOWS: List[int] = [7, 30]

config = ResearchConfig()
