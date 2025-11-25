from dataclasses import dataclass
from typing import List

@dataclass
class ResearchConfig:
    # Data parameters
    DATA_PATH: str = "data/europe_energy.csv"
    COUNTRIES: List[str] = ['DE', 'FR', 'IT', 'ES', 'UK', 'NL', 'BE', 'PL']
    TARGET_COUNTRY: str = 'DE'
    
    # Time parameters
    START_DATE: str = "2006-01-01"
    END_DATE: str = "2017-12-31"
    TEST_SIZE: float = 0.2
    FORECAST_HORIZON: int = 30
    
    # Model parameters
    SEQUENCE_LENGTH: int = 30
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    
    # Advanced features
    LAGS: List[int] = [1, 7, 30, 365]
    ROLLING_WINDOWS: List[int] = [7, 30, 90]

config = ResearchConfig()
