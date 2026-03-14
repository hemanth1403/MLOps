# Lab 2: ML Monitoring - Quick Reference

## Quick Start Commands

```bash
# Start all services
docker-compose up -d

# Train models
python src/train.py --model-type random_forest
python src/train.py --model-type xgboost
python src/train.py --model-type logistic_regression

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.038, 0.05, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]}'

# Run simulations
python src/simulate.py --scenario all

# Stop services
docker-compose down
```

## Access Points

| Service     | URL                           | Credentials |
| ----------- | ----------------------------- | ----------- |
| MLflow UI   | http://localhost:5000         | -           |
| Grafana     | http://localhost:3000         | admin/admin |
| Prometheus  | http://localhost:9090         | -           |
| API Docs    | http://localhost:8000/docs    | -           |
| API Health  | http://localhost:8000/health  | -           |
| API Metrics | http://localhost:8000/metrics | -           |

## File Structure

```
Lab2/
├── README.md                       # Main documentation
├── LAB_GUIDE.md                    # Detailed walkthrough
├── docker-compose.yml              # Service orchestration
├── Dockerfile                      # API container
├── requirements.txt                # Python dependencies
├── start.sh                        # Quick start script
├── src/
│   ├── train.py                    # Training with MLflow
│   ├── serve.py                    # FastAPI serving
│   ├── drift_detector.py           # Drift detection
│   ├── simulate.py                 # Simulation scenarios
│   └── utils.py                    # Helper functions
├── config/
    ├── prometheus.yml              # Prometheus config
    ├── alerting_rules.yml          # Alert rules
    ├── alertmanager.yml            # Alert routing
    └── grafana_datasources.yml     # Grafana datasource

```

## Key Features

### MLflow Integration

- Automatic experiment tracking
- Hyperparameter logging
- Metric visualization
- Model registry
- Artifact storage
- Confusion matrix & ROC curves

### Prometheus Monitoring

- Request metrics (count, rate)
- Latency metrics (p50, p95, p99)
- Model metrics (accuracy, confidence)
- System metrics (active requests, errors)
- Feature distributions
- Custom metrics

### Data Drift Detection

- Kolmogorov-Smirnov test
- Population Stability Index (PSI)
- Wasserstein distance
- Multi-feature drift detection
- Automated alerts

### Production Features

- Health checks
- Readiness probes
- Automated alerting
- Docker containerization
- CI/CD integration
- Load testing

## Monitoring Scenarios

| Scenario                | Command                                         | Observes           |
| ----------------------- | ----------------------------------------------- | ------------------ |
| Normal Operation        | `python src/simulate.py --scenario normal`      | Baseline metrics   |
| Performance Degradation | `python src/simulate.py --scenario degradation` | Confidence drop    |
| Data Drift              | `python src/simulate.py --scenario drift`       | Distribution shift |
| High Load               | `python src/simulate.py --scenario load`        | Latency spike      |
| Errors                  | `python src/simulate.py --scenario errors`      | Error rate         |
| All Scenarios           | `python src/simulate.py --scenario all`         | Complete demo      |

## Key Metrics to Monitor

### Model Performance

- `test_accuracy` - Overall accuracy
- `test_f1_score` - F1 score
- `test_precision` - Precision
- `test_recall` - Recall
- `roc_auc` - ROC AUC score

### System Performance

- `model_requests_total` - Total requests
- `model_prediction_latency_seconds` - Latency
- `active_requests` - Concurrent requests
- `model_errors_total` - Error count
- `prediction_confidence` - Confidence distribution

### Data Quality

- PSI score per feature
- KS test p-values
- Feature distributions
- Drift detection flags

## Useful Prometheus Queries

```promql
# Request rate (per second)
rate(model_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m]))

# Error rate
rate(model_errors_total[5m]) / rate(model_requests_total[5m])

# Average confidence
avg(prediction_confidence)

# Requests by status
sum by (status) (rate(model_requests_total[5m]))
```

## Common Alerts

| Alert           | Condition        | Action                            |
| --------------- | ---------------- | --------------------------------- |
| Low Accuracy    | accuracy < 0.7   | Review model, consider retraining |
| High Latency    | p95 > 100ms      | Check system resources, optimize  |
| High Error Rate | errors > 1%      | Check logs, investigate failures  |
| Data Drift      | PSI > 0.2        | Analyze drift, retrain if needed  |
| No Requests     | rate = 0 for 15m | Check service health              |

## Troubleshooting

```bash
# View service logs
docker-compose logs [service-name]

# Check service health
docker-compose ps

# Restart a service
docker-compose restart [service-name]

# View API logs
docker-compose logs -f ml_api

# Check Prometheus targets
# Visit: http://localhost:9090/targets

# Reset everything
docker-compose down -v
docker-compose up -d
```

## Documentation Links

- **Main README**: Comprehensive overview and architecture
- **LAB_GUIDE**: Step-by-step walkthrough with examples
- **This file**: Quick reference for common tasks

## 💡 Pro Tips

1. **Keep Grafana open** when running simulations to see real-time metrics
2. **Compare multiple models** in MLflow before deploying
3. **Monitor confidence scores** - drops indicate potential issues
4. **Set appropriate thresholds** for alerts based on your use case
5. **Use drift detection** proactively, not reactively
6. **Document your experiments** in MLflow with good run names

## Success Criteria

We successfully completed the lab when you can:

- Train multiple models and track them in MLflow
- View model metrics and artifacts in MLflow UI
- Make predictions via the API
- See metrics in Prometheus
- Create dashboards in Grafana
- Detect data drift programmatically
- Run all simulation scenarios
- Understand alert conditions
- Pass CI/CD tests in GitHub Actions

---
