from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ResearchConfig:
    # Data parameters
    START_DATE: str = "2015-01-01"
    END_DATE: str = "2024-01-01"
    TARGET_VARIABLE: str = "total_load_actual"
    COUNTRIES: List[str] = None
    
    # Model parameters
    FORECAST_HORIZON: int = 30
    CV_FOLDS: int = 5
    TEST_SIZE: float = 0.2
    
    # Advanced modeling
    TRANSFORMER_PARAMS: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.COUNTRIES is None:
            self.COUNTRIES = ["Germany", "France", "Italy", "Spain"]
        
        if self.TRANSFORMER_PARAMS is None:
            self.TRANSFORMER_PARAMS = {
                "d_model": 64,
                "nhead": 4,
                "num_layers": 3,
                "dropout": 0.1
            }

config = ResearchConfig()
