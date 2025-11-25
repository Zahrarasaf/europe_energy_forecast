from dataclasses import dataclass
from typing import List

@dataclass
class ResearchConfig:
    # Data parameters
    DATA_PATH: str = "data/europe_energy.csv"
    COUNTRIES: List[str] = None
    TARGET_COUNTRY: str = 'DE'
    
    # Time parameters
    TEST_SIZE: float = 0.2
    FORECAST_HORIZON: int = 30
    
    # Model parameters
    SEQUENCE_LENGTH: int = 30
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    
    # Advanced features
    LAGS: List[int] = [1, 7, 30]
    ROLLING_WINDOWS: List[int] = [7, 30]

    def __post_init__(self):
        if self.COUNTRIES is None:
            self.COUNTRIES = ['DE', 'FR', 'IT', 'ES', 'UK', 'NL', 'BE', 'PL']

config = ResearchConfig()
