# Grafana, Prometheus, Tempo, Loki — provision-as-code

All observability is provisioned from YAML/JSON checked into git. Edits in
the Grafana UI are not persisted across restarts — change the files here and
redeploy.

## Layout

```
infra/
├── docker/docker-compose.yml            # services: grafana, prometheus, loki, tempo
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/datasources.yml  # Prometheus, Tempo, Loki wiring
│   │   └── dashboards/dashboards.yml    # file provider
│   └── dashboards/
│       ├── agent-latency.json           # p50/p95 per agent, error rate, RPS
│       ├── broker-health.json           # primary/backup up, submit p95, failover spikes
│       └── risk-halts.json              # drawdown, regime, halts, Sharpe 30d, sector exposure
├── prometheus/
│   ├── prometheus.yml                   # scrape API/worker /metrics + otel-collector + self
│   └── rules/ai-investing.yml           # alerts: drawdown, halts, broker, agent latency
└── tempo/tempo.yaml                     # OTLP gRPC ingest on 4317 → local storage
```

## URLs (dev)

- Grafana:    http://localhost:3001  (admin / `GRAFANA_PASSWORD`, default `admin`)
- Prometheus: http://localhost:9090
- Tempo:      http://localhost:3200
- Loki:       http://localhost:3100

## Adding a dashboard

1. Build it in the UI, export JSON ("Share → Export → Save to file"), drop it
   in `infra/grafana/dashboards/`.
2. Pick a stable `uid` so links survive renames.
3. Commit.

## Adding an alert rule

Edit `infra/prometheus/rules/ai-investing.yml`. Prometheus picks rule files up
on reload (`SIGHUP` or container restart). Default labels: `severity` and
`team: ai-investing`.

## Metrics the dashboards expect

The app must export these from `/metrics`. Names are namespaced and chosen so
relabeling is rarely needed:

| Metric                                  | Type      | Labels                  |
|-----------------------------------------|-----------|-------------------------|
| `agent_request_duration_seconds`        | histogram | `agent`                 |
| `agent_request_total`                   | counter   | `agent`, `status`       |
| `broker_up`                             | gauge     | `role`, `broker`        |
| `broker_order_submit_duration_seconds`  | histogram | `broker`                |
| `broker_order_total`                    | counter   | `broker`, `status`      |
| `broker_failover_total`                 | counter   | `from`, `to`            |
| `portfolio_drawdown_ratio`              | gauge     | —                       |
| `portfolio_sharpe_30d`                  | gauge     | —                       |
| `portfolio_sharpe_drop_ratio`           | gauge     | —                       |
| `portfolio_open_positions`              | gauge     | —                       |
| `portfolio_sector_exposure_ratio`       | gauge     | `sector`                |
| `regime_state`                          | gauge     | (0=normal,1=vol,2=stress,3=crisis) |
| `risk_halt_total`                       | counter   | `reason`                |

Wiring these into the existing FastAPI/Temporal services is tracked as a
follow-up; the dashboards work the moment the metrics start flowing.
