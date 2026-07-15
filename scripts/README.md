# Scripts für Apple Container

Dieses Verzeichnis enthält Shell-Skripte für den Aufbau und Betrieb von LLM Router Containern mit [apple/container](https://github.com/apple/container).

## Skripte

### `container-build.sh`

Baut das Container-Image.

```bash
# Build with default tag (latest)
./scripts/container-build.sh

# Build with specific tag
./scripts/container-build.sh v1.0.0
```

### `container-run.sh`

Startet den Container mit konfigurierbaren Optionen.

```bash
# Basic run
./scripts/container-run.sh

# With custom port
./scripts/container-run.sh --port 9000

# With environment file
./scripts/container-run.sh --env-file .env

# With profile volumes
./scripts/container-run.sh --profile ./profiles

# Detached mode
./scripts/container-run.sh --detach
```

### `container-compose.sh`

Ersatz für docker-compose mit apple/container. Startet alle Services (LLM Router + optional Jaeger).

```bash
# Start all services
./scripts/container-compose.sh up

# Stop all services
./scripts/container-compose.sh down

# Show logs
./scripts/container-compose.sh logs

# Show status
./scripts/container-compose.sh status
```

Mit Jaeger für Tracing:

```bash
ENABLE_JAEGER=true ./scripts/container-compose.sh up
```

## Abhängigkeiten

1. **Apple Container System** muss laufen:
   ```bash
   sudo container system start
   ```

2. **VM-Maschine** muss erstellt sein:
   ```bash
   container machine create --arch arm64 llm-mac
   container machine set-default llm-mac
   ```

3. **Image** muss gebaut sein:
   ```bash
   make build
   ```

## Umgebungsvariablen

| Variable | Standard | Beschreibung |
|----------|----------|--------------|
| `LLM_ROUTER_IMAGE` | `llm-router` | Image-Name |
| `LLM_ROUTER_TAG` | `latest` | Image-Tag |
| `LLM_ROUTER_PORT` | `8000` | Host-Port |
| `ENABLE_JAEGER` | `false` | Jaeger aktivieren |

## Fehlerbehebung

### `XPC connection error: Connection invalid`

```bash
sudo container system restart
```

### `failed to list container machines`

```bash
container machine create --arch arm64 llm-mac
```

### Image not found

```bash
make build
```
