# LLM Router — Apple Container Guide

Diese Anleitung beschreibt die Containerisierung mit [apple/container](https://github.com/apple/container) (v1.0.0).

## Voraussetzungen

- macOS mit Apple Silicon (M1/M2/M3/M4)
- macOS 26+ (erforderlich für apple/container)
- Installed: `container` CLI (`/usr/local/bin/container` v1.0.0)

## Schnelleinstieg

```bash
# 1. System-Service starten (benötigt sudo)
sudo container system start

# 2. VM-Maschine erstellen
container machine create --arch arm64 llm-mac

# 3. Maschine als Standard setzen
container machine set-default llm-mac

# 4. Image bauen
make build

# 5. Container starten
make run

# 6. Logs prüfen
make logs
```

## Befehle im Überblick

### System verwalten

```bash
# System starten (sudo erforderlich)
sudo container system start

# System stoppen
container system stop

# Status prüfen
container system status

# System neu starten
sudo container system restart
```

### VM-Maschine verwalten

```bash
# Maschine erstellen
container machine create --arch arm64 llm-mac

# Maschine löschen
container machine delete llm-mac

# Standard-Maschine setzen
container machine set-default llm-mac

# Verfügbare Maschinen auflisten
container machine list

# Maschine stoppen
container machine stop llm-mac

# Maschine starten
container machine start llm-mac
```

### Images bauen und verwalten

```bash
# Image bauen
container build -t llm-router:latest .

# Mit spezifischem Tag
container build -t llm-router:v1.0.0 .

# Alle Images auflisten
container image list

# Image löschen
container image delete llm-router:latest

# Image exportieren
container image save -o llm-router.tar llm-router:latest

# Image importieren
container image load -i llm-router.tar
```

### Container starten und verwalten

```bash
# Container im Hintergrund starten
container run -d --name llm-router -p 8000:8000 llm-router:latest

# Mit Environment Variablen
container run -d --name llm-router \
  -p 8000:8000 \
  -e ROUTER_DEFAULT_STRATEGY=policy \
  -e ROUTER_OTLP_ENABLED=false \
  llm-router:latest

# Mit Volume-Mounts (Profile, Policies, Logs)
container run -d --name llm-router \
  -p 8000:8000 \
  -v $(pwd)/profiles:/app/profiles:ro \
  -v $(pwd)/agent-policies:/app/agent-policies:ro \
  -v $(pwd)/logs:/app/logs:rw \
  llm-router:latest

# Container stoppen
container stop llm-router

# Container starten
container start llm-router

# Container Logs
container logs llm-router
container logs -f llm-router  # Follow

# Container löschen
container delete llm-router

# In Container Shell öffnen
container exec -it llm-router /bin/sh

# Container Statistiken
container stats llm-router

# Alle Container auflisten
container list
```

## Makefile Befehle

Das Makefile stellt bequeme Targets bereit:

```bash
# Alle Befehle anzeigen
make help

# System starten
make container-start

# Maschine erstellen
make machine-create

# Image bauen
make build

# Container starten
make run

# Container Logs anzeigen
make logs

# Container stoppen
make stop

# Alles aufräumen
make clean

# Status prüfen
make check-system
```

## Umgebungsvariablen

Erstelle eine `.env` Datei basierend auf `.env.example`:

```bash
cp .env.example .env
```

Umgebungsvariablen können auch direkt an `container run` übergeben werden:

```bash
container run -d --name llm-router \
  -e ROUTER_DEFAULT_STRATEGY=policy \
  -e ROUTER_DEFAULT_MODEL=llama-3.1-8b \
  -e ROUTER_OTLP_ENABLED=false \
  -p 8000:8000 \
  llm-router:latest
```

## Docker Compose Alternative

Da apple/container kein compose unterstützt, kann ein einfaches Shell-Skript verwendet werden:

```bash
# scripts/container-compose.sh
#!/bin/bash
# Startet alle Services mit apple/container

# 1. LLM Router starten
container run -d --name llm-router \
  -p 8000:8000 \
  -v $(pwd)/profiles:/app/profiles:ro \
  -v $(pwd)/agent-policies:/app/agent-policies:ro \
  -v $(pwd)/logs:/app/logs:rw \
  llm-router:latest

# 2. Optional: Jaeger für Trace-Visualisierung starten
container run -d --name jaeger \
  -p 16686:16686 \
  -e COLLECTOR_OTLP_ENABLED=true \
  jaegertracing/all-in-one:latest
```

## Troubleshooting

### Problem: `XPC connection error: Connection invalid`

Lösung: Container-System neu starten:
```bash
sudo container system restart
```

### Problem: `failed to list container machines`

Lösung: Maschine erstellen:
```bash
container machine create --arch arm64 llm-mac
```

### Problem: Image build schlägt fehl

Lösung: Stelle sicher, dass das System läuft:
```bash
container system status
```

### Problem: Container startet nicht

Lösung: Logs prüfen:
```bash
container logs llm-router
```

## Architektur

```
┌─────────────────────────────────────────────────────────────┐
│  macOS (Apple Silicon)                                      │
├─────────────────────────────────────────────────────────────┤
│  container system (System-Service)                          │
│  ├── containerd (Container-Laufzeit)                        │
│  └── VM-Maschine (llm-mac)                                 │
│      ┌───────────────────────────────────────────────────┐  │
│      │  Linux-Kernel (ARM64)                             │  │
│      │  ┌─────────────────────────────────────────────┐  │  │
│      │  │  LLM Router Container                       │  │  │
│      │  │  ├── Python 3.12                            │  │  │
│      │  │  ├── FastAPI (Port 8000)                    │  │  │
│      │  │  └── uvicorn                               │  │  │
│      │  └─────────────────────────────────────────────┘  │  │
│      └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Vergleich: Docker vs. apple/container

| Feature | Docker Desktop | apple/container |
|---------|---------------|-----------------|
| VM-Betrieb | Linux (HyperKit) | macOS-native (VirtIO) |
| ARM64-Unterstützung | Rosetta-Emulation | Native ARM64 |
| System-Service | Hintergrunddienst | `container system` |
| Maschine | Docker VM | `container machine` |
| Build | `docker build` | `container build` |
| Run | `docker run` | `container run` |
| Volume-Mounts | `-v` | `--volume` |
| Port-Mapping | `-p` | `-p` |
| Compose | Ja | Nein (Shell-Skript) |