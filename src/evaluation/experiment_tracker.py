import mlflow
import wandb
import pandas as pd
from typing import Dict, Any
import json

class ResearchExperimentTracker:
    def __init__(self, config):
        self.config = config
        self.setup_tracking()
    
    def setup_tracking(self):
        """Setup MLflow and Weights & Biases"""
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("energy_forecasting_phd")
        
        wandb.init(
            project=self.config.WANDB_PROJECT,
            config=vars(self.config)
        )
    
    def log_experiment(self, 
                      model_name: str,
                      metrics: Dict[str, float],
                      parameters: Dict[str, Any],
                      artifacts: Dict[str, str] = None):
        """Log complete experiment details"""
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(parameters)
            wandb.config.update(parameters)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            wandb.log(metrics)
            
            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path)
            
            # Log model
            if 'model_path' in artifacts:
                mlflow.pytorch.log_model(artifacts['model_path'], "model")
    
    def log_comparative_analysis(self, results_df: pd.DataFrame):
        """Log comparative analysis of all models"""
        wandb.log({"model_comparison": wandb.Table(dataframe=results_df)})
        
        # Create performance visualization
        fig = self._create_performance_plot(results_df)
        wandb.log({"performance_comparison": fig})
    
    def _create_performance_plot(self, results_df):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        results_df.set_index('model')['test_rmse'].plot(kind='bar', ax=ax)
        ax.set_ylabel('RMSE')
        ax.set_title('Model Performance Comparison')
        return fig
