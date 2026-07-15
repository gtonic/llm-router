# 📊 LLM Router - Visualisierung & Observability Guide

## Übersicht

Der LLM Router unterstützt drei Ebenen der Visualisierung:

1. **Traces** - Anfrage-Flüsse durch das System (Jaeger)
2. **Metriken** - Latenz, Kosten, Throughput (Prometheus + Grafana)
3. **Logs** - Strukturierte Logs mit Request-IDs

---

## 1. Jaeger - Distributed Tracing

### Quick Start (Docker Compose)

```bash
# Start mit Jaeger
docker compose up -d

# Open Jaeger UI
open http://localhost:16686
```

### Konfiguration

```yaml
# docker-compose.yml
jaeger:
  image: jaegertracing/all-in-one:latest
  ports:
    - "16686:16686"  # Web UI
    - "14268:14268"  # HTTP ingest (legacy)
    - "14250:14250"  # gRPC ingest
    - "4317:4317"    # OTLP gRPC
    - "4318:4318"    # OTLP HTTP
  environment:
    - COLLECTOR_OTLP_ENABLED=true
    - MEMORY_MAX_TRACES=50000
```

### Traces ansehen

1. **Jaeger UI** → http://localhost:16686
2. Service: `llm-router` auswählen
3. "Find Traces" klicken
4. Trace anklicken für detaillierte Span-Ansicht

### Was wird getracet?

- ✅ Router-Decision (Policy, Complexity, Round-Robin)
- ✅ Model-Call (Llama.cpp / OpenAI)
- ✅ Guardrails (PII, Abuse, Safety)
- ✅ Rate-Limiting
- ✅ Kosten & Token-Verbrauch

---

## 2. Prometheus + Grafana - Metriken (Empfohlen)

### Warum Grafana?

- **Echtzeit-Dashboards** für Latenz, Kosten, Fehler
- **Alerting** bei hohen Latenzen oder PII-Detection
- **Kosten-Tracking** pro Model, User, Request
- **Historische Analyse** (Tage/Wochen zurück)

### Docker Compose Extension

```yaml
# docker-compose.monitoring.yml
x-prometheus-volume: &prometheus-volume
  volumes:
    - prometheus-data:/prometheus

services:
  prometheus:
    image: prom/prometheus:latest
    <<: *prometheus-volume
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - llm-router-net

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./configs/grafana/dashboards:/etc/grafana/provisioning/dashboards
    networks:
      - llm-router-net
    depends_on:
      - prometheus

volumes:
  prometheus-data:
  grafana-data:
```

### Prometheus Config

```yaml
# configs/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'llm-router'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['app:8000']
```

### Grafana Dashboards

Nach dem Start:
1. Open http://localhost:3000 (admin/admin)
2. Import Dashboard JSON (siehe unten)
3. Prometheus als Data Source hinzufügen

---

## 3. Console Logging - Development

Für schnelle Debugging-Sessions ohne externe Dependencies:

```bash
# .env
ROUTER_LOG_LEVEL=DEBUG
OTEL_ENABLED=false  # Disables OTLP, uses console exporter
```

Logs mit Request-IDs:
```
[138c3252] PII detected in request - routing to local model
[138c3252] Router decision: policy -> llama-local
[138c3252] Model call: 245ms, 150 tokens, $0.00
```

---

## Empfohlenes Setup für verschiedene Szenarien

### 🔬 Development (Lokal)

```bash
docker compose up -d app jaeger
# Jaeger UI: http://localhost:16686
```

### 📊 Production Monitoring

```bash
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
# Jaeger UI:    http://localhost:16686
# Grafana:      http://localhost:3000
# Prometheus:   http://localhost:9090
```

### ☁️ Cloud (AWS/GCP/Azure)

- **Traces**: Jaeger Operator oder AWS X-Ray
- **Metriken**: CloudWatch Metrics oder Prometheus Operator
- **Logs**: CloudWatch Logs oder Loki

---

## Key Metrics zu beobachten

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `llm_router_request_duration_seconds` | Request Latenz p95 < 5s |
| `llm_router_request_cost_total` | Kosten pro Request |
| `llm_router_pii_detected_total` | PII-Detection Rate |
| `llm_router_route_decision` | Routing-Entscheidungen |
| `llm_router_model_call_errors_total` | Fehler pro Model |
| `llm_router_rate_limit_rejected_total` | Rate-Limit Hits |

---

## Troubleshooting

### Keine Traces in Jaeger

1. Prüfe Container-Logs: `docker logs llm-router | grep OTEL`
2. Teste Console-Exporter: Setze `OTEL_ENABLED=false`
3. Prüfe Netzwerk: `docker exec llm-router ping jaeger`
4. Prüfe Ports: `docker exec jaeger netstat -tlnp | grep 431`

### Grafana kann keine Daten sehen

1. Prometheus erreichbar? `curl http://localhost:9090/-/healthy`
2. Data Source geprüft? Settings → Data Sources → Prometheus
3. Dashboard JSON imported?

---

## Next Steps

1. **Heute**: Jaeger starten und erste Traces ansehen
2. **Diese Woche**: Grafana Dashboard für Latenz/Kosten
3. **Nächste Woche**: Alerting Setup (Slack/Email Notifications)
