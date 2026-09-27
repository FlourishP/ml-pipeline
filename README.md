# ML Pipeline

End-to-end machine learning pipeline with experiment tracking, model registry, and automated deployment.

## Features
- **Data Ingestion**: Automated data validation and preprocessing
- **Experiment Tracking**: MLflow integration for metrics and artifacts
- **Model Registry**: Versioned models with stage transitions
- **Automated Training**: Pipeline orchestration with retry logic
- **Prediction API**: REST endpoint for model inference
- **Monitoring**: Data drift detection and model performance alerts

## Tech Stack
- **Language**: Python 3.11
- **ML Framework**: scikit-learn, XGBoost
- **Tracking**: MLflow
- **Orchestration**: Prefect
- **Storage**: PostgreSQL, MinIO
- **Infrastructure**: Docker, Docker Compose

## Quick Start
```bash
git clone https://github.com/FlourishP/ml-pipeline
cd ml-pipeline
docker-compose up -d
prefect deployment apply
```

## License
MIT — see [LICENSE](LICENSE)
