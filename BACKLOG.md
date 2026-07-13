# 📋 LLM Router & Gateway — Backlog

> Master task list für den policybasierten LLM-Gateway-Router auf LangChain-Basis.
> Jeder Task ist unabhängig umsetzbar. Mit `[x]` abhaken wenn erledigt.

---

## 📊 Summary

| Phase | Task | Beschreibung | Status | Dependencies |
|---|---|---|---|---|
| **P0** | 1 | Projekt-Setup & Struktur | ⬜ TODO | — |
| **P0** | 2 | config.py — Settings & Pydantic-Modelle | ⬜ TODO | 1 |
| **P1** | 3 | models.py — Request/Response Models | ⬜ TODO | 1 |
| **P1** | 4 | pool/base.py — ModelBackend ABC | ⬜ TODO | 1 |
| **P1** | 5 | pool/local.py — LlamaCPPBackend | ⬜ TODO | 4 |
| **P1** | 6 | pool/remote.py — RemoteBackend | ⬜ TODO | 4 |
| **P1** | 7 | pool/pool.py — ModelPool | ⬜ TODO | 5, 6 |
| **P2** | 8 | tracing/ — OpenTelemetry Setup | ⬜ TODO | 1 |
| **P2** | 9 | routing/base.py — PolicyBase ABC | ⬜ TODO | 1 |
| **P2** | 10 | routing/complexity.py — Komplexitäts-Detektor | ⬜ TODO | 9 |
| **P2** | 11 | routing/policy.py — PolicyMatcher (YAML) | ⬜ TODO | 9 |
| **P2** | 12 | routing/hybrid.py — Hybrid-Routing | ⬜ TODO | 10, 11 |
| **P2** | 13 | routing/round_robin.py — Lastverteilung | ⬜ TODO | 9 |
| **P3** | 14 | guardrails/rate_limiter.py — Rate Limiting | ⬜ TODO | 1 |
| **P3** | 15 | guardrails/pii_filter.py — PII-Erkennung | ⬜ TODO | 1 |
| **P3** | 16 | guardrails/abuse_filter.py — Abuse/Moderation | ⬜ TODO | 1 |
| **P3** | 17 | guardrails/content_safety.py — Content Safety | ⬜ TODO | 1 |
| **P4** | 18 | router.py — Hauptklasse: RouterPolicyEngine | ⬜ TODO | 7, 10-13, 15-17 |
| **P4** | 19 | server/app.py — FastAPI App-Factory | ⬜ TODO | 18 |
| **P4** | 20 | server/routes.py — Endpunkte | ⬜ TODO | 18, 19 |
| **P5** | 21 | Logging — JSONL Audit-Logging | ⬜ TODO | 18 |
| **P5** | 22 | Policy-Konfiguration — default.yaml | ⬜ TODO | 11 |
| **P5** | 23 | Model-Profile — local.yaml, remote.yaml | ⬜ TODO | 5, 6 |
| **P5** | 24 | .env.example & Dokumentation | ⬜ TODO | 1-23 |
| **P6** | 25 | Tests — Unit Tests für alle Module | ⬜ TODO | alle |
| **P6** | 26 | Tests — Integrationstests (E2E) | ⬜ TODO | 25 |
| **P6** | 27 | README.md aktualisieren | ⬜ TODO | 25, 26 |

---

## Detaillierte Tasks (1-10)

---

### Task 1: Projekt-Setup & Struktur

**Beschreibung:** Grundstruktur des llm-router Projekts anlegen.

**Ordnerstruktur:**
```
llm-router/
├── pyproject.toml
├── README.md / AGENTS.md / BACKLOG.md / .env.example
├── src/
│   └── llm_router/
│       ├── __init__.py  config.py  models.py  router.py  logging_setup.py
│       ├── routing/     base.py  complexity.py  policy.py  hybrid.py  round_robin.py
│       ├── pool/        base.py  local.py  remote.py  pool.py
│       ├── guardrails/  __init__.py  rate_limiter.py  pii_filter.py  abuse_filter.py  content_safety.py
│       ├── tracing/     __init__.py  otel_setup.py  span_attributes.py
│       └── server/      __init__.py  app.py  routes.py
├── agent-policies/default.yaml
├── profiles/local.yaml  remote.yaml
└── tests/
```

**pyproject.toml Dependencies:**
```toml
[project]
name = "llm-router"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "langchain>=1.3.9",
    "langchain-openai>=1.3.5",
    "langchain-community>=0.3.31",
    "fastapi>=0.139.0",
    "uvicorn[standard]>=0.34.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
    "opentelemetry-api>=1.28",
    "opentelemetry-sdk>=1.28",
    "opentelemetry-exporter-otlp>=1.28",
    "aiofiles>=25.1.0",
    "pydantic-yaml>=1.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=9.1.1",
    "pytest-asyncio>=0.25.0",
    "httpx>=0.28.0",
    "respx>=0.22.0",
]

[project.scripts]
llm-router = "llm_router.main:main"
llm-router-server = "llm_router.server.app:main"
```

**Acceptance Criteria:**
- [ ] `pyproject.toml` mit allen Dependencies
- [ ] Ordnerstruktur erstellt
- [ ] `pip install -e .` funktioniert
- [ ] `python -c "import llm_router"` funktioniert

---

### Task 2: config.py — Settings & Pydantic-Modelle

**Inhalt:** Zentrale Konfiguration mit Enums, Dataclasses und Pydantic Settings.

**Neue Enums:**
```python
class RoutingStrategy(str, Enum):
    POLICY = "policy"
    COMPLEXITY = "complexity"
    HYBRID = "hybrid"
    ROUND_ROBIN = "round_robin"
    LATENCY = "latency"
    COST = "cost"

class PrivacyLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CLASSIFIED = "classified"
```

**Dataclasses:**
```python
@dataclass
class ModelBackendConfig:
    id: str
    name: str
    type: str          # "local" | "remote" | "edge"
    base_url: str
    api_key: str
    model_name: str
    enabled: bool = True
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 60
    retry_count: int = 3
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0

@dataclass
class PolicyRule:
    id: str
    name: str
    description: str = ""
    conditions: dict = field(default_factory=dict)
    target_model: str = ""
    priority: int = 0
    enabled: bool = True

@dataclass
class GatewaySettings(BaseSettings):
    default_strategy: RoutingStrategy = RoutingStrategy.POLICY
    default_model: str = "llama-local"
    models: list[ModelBackendConfig] = field(default_factory=list)
    models_dir: str = "profiles"
    policies_dir: str = "agent-policies"
    default_policy: str = "default"
    guardrails_input: bool = True
    guardrails_output: bool = True
    pii_redact: bool = True
    pii_max_tokens: int = 4096
    abuse_block_threshold: float = 0.8
    rate_limit_rpm: int = 60
    rate_limit_ppm: int = 60000
    otlp_enabled: bool = True
    otlp_endpoint: str = "http://localhost:4318/v1/traces"
    otlp_protocol: str = "http/protobuf"
    log_dir: str = "logs"
    log_level: str = "INFO"

    model_config = {"env_prefix": "ROUTER_", "env_file": ".env", "extra": "ignore"}
```

**Acceptance Criteria:**
- [ ] `GatewaySettings` lädt aus `.env` (Prefix `ROUTER_`)
- [ ] `ModelBackendConfig` kann aus YAML geladen werden
- [ ] `PolicyRule` kann aus YAML geladen werden
- [ ] Default-Werte sinnvoll gesetzt
- [ ] `pydantic-settings` mit `extra="ignore"`

---

### Task 3: models.py — Request/Response Models

**Modelle zu implementieren (15+):**
- `MessageRole`, `FinishReason` — Enums
- `FunctionCall`, `ToolCall`, `ChatMessage` — Nachrichten-Strukturen
- `ChatCompletionRequest`, `ChatCompletionResponse` — OpenAI-kompatibel
- `ChatCompletionChoice`, `UsageInfo`, `ChunkChoice`, `ChatCompletionChunk` — Streaming
- `ModelInfo`, `ModelListResponse` — Modell-Liste
- `GuardrailCheckRequest`, `GuardrailCheckResponse` — Guardrail-Prüfung
- `RoutingDecision` — Routing-Ergebnis mit Metadata

**Acceptance Criteria:**
- [ ] Alle 15+ Modelle implementiert
- [ ] JSON-Serialisierung (`model_dump(mode="json")`)
- [ ] OpenAI-API-kompatible Feldnamen
- [ ] SSE-Streaming mit `to_sse_event()`
- [ ] Router-spezifische Modelle enthalten
---

### Task 4: pool/base.py — ModelBackend ABC

**Abstrakte Basisklasse für alle Modell-Backends.**

```python
@dataclass
class GenerateResult:
    content: str
    model: str
    usage: UsageInfo
    finish_reason: str
    tool_calls: list[dict] | None = None
    latency_ms: float = 0.0

@dataclass
class UsageInfo:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float = 0.0

@dataclass
class HealthStatus:
    healthy: bool
    latency_ms: float
    error: str | None = None

class ModelBackend(ABC):
    def __init__(self, config: ModelBackendConfig):
        self.config = config

    @abstractmethod
    async def generate(self, messages: list[dict], temperature: float = 0.3,
                       max_tokens: int = 4096, tools: list[dict] | None = None, **kwargs) -> GenerateResult: ...

    @abstractmethod
    async def generate_stream(self, messages: list[dict], temperature: float = 0.3, **kwargs) -> AsyncIterator[GenerateResult]: ...

    @abstractmethod
    async def health_check(self) -> HealthStatus: ...

    async def generate_with_retry(self, messages: list[dict], max_retries: int = 3, **kwargs) -> GenerateResult:
        """Wrapper mit exponentiellem Backoff."""
        ...
```

**Acceptance Criteria:**
- [ ] ABC mit 3 abstract methods
- [ ] `GenerateResult`, `UsageInfo`, `HealthStatus` dataclasses
- [ ] `generate_with_retry` mit exponentiellem Backoff
- [ ] Typ-Hints und Docstrings auf Deutsch

---

### Task 5: pool/local.py — LlamaCPPBackend

**Beschreibung:** Lokales Backend für Llama.cpp / Ollama.

**Implementierung:**
- Nutzt `langchain_openai.ChatOpenAI` als Client
- Connectet zu `base_url` (z.B. `http://localhost:8888/v1`)
- Lädt `api_key` aus Config (kann leer sein)
- Setzt `temperature`, `max_tokens`, `timeout` aus Config
- Health-Check via `httpx` mit Timeout

**Acceptance Criteria:**
- [ ] `generate()` sendet Request an lokalen Endpunkt
- [ ] `generate_stream()` unterstützt SSE
- [ ] `health_check()` prüft `/v1/models` oder `/health`
- [ ] Fehlerbehandlung (ConnectionError → unhealthy)

---

### Task 6: pool/remote.py — RemoteBackend

**Beschreibung:** Remote-Backend für OpenAI-kompatible APIs.

**Implementierung:**
- Gleiche Basis wie `LlamaCPPBackend` (beide nutzen `ChatOpenAI`)
- Unterschied: zwingend `api_key` erforderlich
- Unterstützt mehrere Provider durch `base_url`
- Kosten-Tracking (aus Config: `cost_per_1m_input`, `cost_per_1m_output`)

**Acceptance Criteria:**
- [ ] `generate()` sendet Request an Remote-Endpunkt
- [ ] Kosten wird in `UsageInfo.cost` berechnet
- [ ] Retry-Logik mit exponentiellem Backoff
- [ ] API-Key wird sicher behandelt (nicht geloggt)

---

### Task 7: pool/pool.py — ModelPool

**Beschreibung:** Verwaltung aller Model-Backends mit Health-Checking.

```python
class ModelPool:
    def __init__(self, configs: list[ModelBackendConfig]):
        self._backends: dict[str, ModelBackend] = {}
        self._configs = configs
        self._initialize()

    def _initialize(self):
        for cfg in self._configs:
            if cfg.type == "local":
                self._backends[cfg.id] = LlamaCPPBackend(cfg)
            elif cfg.type == "remote":
                self._backends[cfg.id] = RemoteBackend(cfg)

    def get(self, model_id: str) -> ModelBackend: ...
    def list_models(self) -> list[str]: ...
    async def health_check_all(self) -> dict[str, HealthStatus]: ...
    def get_healthy_models(self) -> list[str]: ...
```

**Acceptance Criteria:**
- [ ] Backends werden lazy initialisiert
- [ ] `get(model_id)` wirft KeyError wenn nicht gefunden
- [ ] `health_check_all()` prüft alle parallel
- [ ] `get_healthy_models()` filtert unhealthy heraus
- [ ] Thread-safe (mit `asyncio.Lock`)

---

### Task 8: tracing/ — OpenTelemetry Setup

**Beschreibung:** OTEL-Initialisierung und Span-Attribute-Builder.

**Files:**
```python
# tracing/otel_setup.py
def setup_otel(service_name: str = "llm-router", otlp_enabled: bool = True, ...):
    """Initialisiert TracerProvider mit OTLP-Exportern."""
    ...

def get_tracer(name: str = "llm-router"):
    """Gibt konfigurierten Tracer zurück."""
    ...

# tracing/span_attributes.py
class SpanAttributes:
    GATEWAY_REQUEST_ID = "gateway.request_id"
    GATEWAY_USER_ID = "gateway.user_id"
    ROUTING_STRATEGY = "routing.strategy"
    ROUTING_POLICY = "routing.policy_matched"
    MODEL_SELECTED = "model.selected"
    MODEL_PROVIDER = "model.provider"
    GUARDRAIL_PII_DETECTED = "guardrail.pii_detected"
    GUARDRAIL_ABUSE_SCORE = "guardrail.abuse_score"
    LATENCY_MS = "latency.ms"
    ...
```

**Trace-Hierarchie:**
```
span_gateway (POST /v1/chat/completions)
├── span_input_guardrails
│   ├── span_rate_limiter
│   ├── span_pii_filter
│   └── span_abuse_filter
├── span_router_decision
│   ├── span_complexity
│   ├── span_policy_match
│   └── span_hybrid_plan
├── span_model_call (local oder remote)
│   └── span_llm_http_call
├── span_output_guardrails
│   ├── span_pii_output
│   └── span_content_safe
└── span_audit
```

**Acceptance Criteria:**
- [ ] `setup_otel()` initialisiert TracerProvider
- [ ] OTLP-Exporter für HTTP/gRPC
- [ ] `SpanAttributes` mit allen Attributen
- [ ] Beispiel-Span für jeden Trace-Typ

---

### Task 9: routing/base.py — PolicyBase ABC

**Abstrakte Basisklasse für alle Routing-Strategien.**

```python
@dataclass
class RoutingResult:
    model_id: str
    strategy: str
    policy_matched: str | None = None
    metadata: dict = field(default_factory=dict)

class PolicyBase(ABC):
    @abstractmethod
    async def route(self, messages: list[dict], user_id: str | None = None,
                    api_key: str | None = None, available_models: list[ModelBackend] | None = None) -> RoutingResult: ...
```

---

### Task 10: routing/complexity.py — Komplexitäts-Detektor

**Analyse-Merkmale:**
- Prompt-Länge (Token-Count via `tiktoken` oder `langchain`)
- Keywords (Code, Analyse, Erklärung, Zusammenfassung)
- Anzahl impliziter Schritte
- Vorhandene Tools/Strukturen

**Mapping:**
```python
COMPLEXITY_TO_MODEL = {
    "low": "smallest_local",
    "medium": "medium_local",
    "high": "large_local",
    "critical": "remote_top",
}
```

**Acceptance Criteria:**
- [ ] `ComplexityDetector` mit `analyze(messages) -> str`
- [ ] Token-Count via `tiktoken` oder `langchain`
- [ ] Keyword-Erkennung für task_type
- [ ] Mapping von Komplexität → Modell-ID
---

### Task 11: routing/policy.py — PolicyMatcher (YAML-Regeln)

**Beschreibung:** Matcht eingehende Anfragen gegen YAML-konfigurierte Policy-Regeln.

**Policy-Datei-Beispiel (`agent-policies/default.yaml`):**
```yaml
rules:
  - id: "no_remote_pii"
    name: "Keine externen Calls für PII"
    conditions:
      contains_pii: true
    target_model: "llama-local"
    priority: 100

  - id: "code_generation"
    name: "Code → lokales großes Modell"
    conditions:
      task_type: "code"
    target_model: "qwen-72b"
    priority: 50

  - id: "default_remote"
    name: "Standard Remote"
    conditions: {}
    target_model: "gpt4o"
    priority: 0
```

**Klasse:**
```python
class PolicyMatcher:
    def __init__(self, policies_dir: str):
        self.rules: list[PolicyRule] = []
        self._load_policies(policies_dir)

    def _load_policies(self, policies_dir: str):
        """Lädt alle YAML-Dateien im policies_dir."""
        ...

    def match(self, messages: list[dict], **kwargs) -> RoutingResult:
        """Prüft alle Regeln nach Priorität. Erste passende Regel gewinnt."""
        ...
```

**Acceptance Criteria:**
- [ ] YAML-Parser für Policy-Dateien
- [ ] `match()` prüft Regeln nach Priorität
- [ ] Fallback auf `default_model` wenn keine Regel passt
- [ ] Logische Operatoren: `all:` (AND), `any:` (OR)

---

### Task 12: routing/hybrid.py — Hybrid-Routing

**Beschreibung:** Splittet komplexe Anfragen in Teilaufgaben.

```python
@dataclass
class HybridStep:
    model: str
    task: str
    description: str = ""
    depends_on: str | None = None

@dataclass
class HybridPlan:
    is_hybrid: bool
    steps: list[HybridStep]

class HybridRouter(PolicyBase):
    async def route(self, messages: list[dict], **kwargs) -> RoutingResult:
        """Erstellt Hybridplan, gibt Modell für ersten Schritt zurück."""
        ...
```

**Beispiel:**
```
Schritt 1: lokales Modell → Entities extrahieren
Schritt 2: Remote-Modell  → Sentiment-Analyse (abhängig von Schritt 1)
Schritt 3: lokales Modell → Formatierung + PII-Prüfung (abhängig von Schritt 2)
```

---

### Task 13: routing/round_robin.py — Lastverteilung

```python
class RoundRobinPolicy(PolicyBase):
    def __init__(self, model_ids: list[str]):
        self.model_ids = model_ids
        self._counter = 0

    async def route(self, **kwargs) -> RoutingResult:
        model = self.model_ids[self._counter % len(self.model_ids)]
        self._counter += 1
        return RoutingResult(model_id=model, strategy="round_robin")
```

---

### Task 14: guardrails/rate_limiter.py — Rate Limiting

```python
@dataclass
class RateLimitResult:
    allowed: bool
    remaining_requests: int
    reset_at: datetime | None = None
    error: str = ""

class RateLimiter:
    def __init__(self, rpm: int = 60, tpm: int = 60000):
        self.requests: dict[str, list[float]] = {}
        self.rpm = rpm
        self.tpm = tpm

    async def check(self, client_id: str, tokens: int = 0) -> RateLimitResult:
        """Prüft ob Request erlaubt ist (Token-Bucket)."""
        ...
```

**Acceptance Criteria:**
- [ ] Requests-per-Minute Limit pro Client
- [ ] Tokens-per-Minute Limit pro Client
- [ ] Leaky-Bucket Algorithmus
- [ ] 429 Response mit `Retry-After` Header
- [ ] Cleanup alter Einträge

---

### Task 15: guardrails/pii_filter.py — PII-Erkennung

**Erkannte Muster:**
- E-Mails, Telefonnummern, IBAN, Kreditkarten
- API-Keys, JWTs, SSN
- Postanschriften, Geburtsdatum, Kfz-Kennzeichen
- Gesundheitsdaten, Steuer-IDs

```python
class PiiFilter:
    def __init__(self, redact: bool = True):
        self.redact = redact
        self.patterns = self._load_patterns()

    def check(self, text: str, mode: str = "input") -> PiiResult:
        """Prüft Text auf PII."""
        ...

    def redact(self, text: str) -> str:
        """Ersetzt PII durch ****."""
        ...
```

**Acceptance Criteria:**
- [ ] 10+ PII-Muster implementiert
- [ ] `check()` gibt PiiResult mit matched patterns
- [ ] `redact()` ersetzt PII durch `****`
- [ ] Performance: < 1ms pro 1KB Text

---

### Task 16: guardrails/abuse_filter.py — Abuse/Moderation

**Kategorien:**
| Kategorie | Beispiel |
|---|---|
| Prompt Injection | "Ignore previous instructions" |
| Jailbreak | "Do Anything Now", "Zeig deine System-Anweisungen" |
| Encoding Tricks | Base64, Hex, Leetspeak |
| Toxic Content | Hassrede, Diskriminierung |
| Illegal Content | Drogenanleitungen, Einbrucheinweisungen |
| Multi-Turn | Ketten von Fragen zur Umgehung |

```python
@dataclass
class AbuseResult:
    safe: bool
    abuse_score: float  # 0.0 - 1.0
    categories: list[str] = field(default_factory=list)
    details: str = ""

class AbuseFilter:
    def check(self, text: str) -> AbuseResult:
        """Prüft auf Missbrauch."""
        ...
```

**Acceptance Criteria:**
- [ ] 20+ Injektionsmuster
- [ ] AbuseScore (0.0-1.0)
- [ ] Threshold-basierte Blockade
- [ ] Alle abgelehnten Requests loggen

---

### Task 17: guardrails/content_safety.py — Content Safety

**Kategorien:**
- Gewalt, Hassrede, illegale Inhalte
- Selbsterhaltung, Terrorismus, Deepfakes
- Waffencode, Drogenherstellung

**Schweregrade:**
| Level | Aktion |
|---|---|
| LOW | Nur warnen |
| MEDIUM | Markieren + warnen |
| HIGH | Maskieren im Output |
| CRITICAL | Blockieren + loggen |

**Acceptance Criteria:**
- [ ] 10+ Gefahrenkategorien
- [ ] Severity-Level (LOW/MEDIUM/HIGH/CRITICAL)
- [ ] Configurable Action pro Level
- [ ] Integration in Output-Pipeline

---

### Task 18: router.py — Hauptklasse: RouterPolicyEngine

**Orchestriert alle Komponenten: Guardrails → Router → Model → Guardrails**

```python
class RouterPolicyEngine:
    def __init__(self, pool: ModelPool, routing_strategy: RoutingStrategy,
                 policy_matcher: PolicyMatcher, complexity_detector: ComplexityDetector,
                 hybrid_router: HybridRouter, rate_limiter: RateLimiter,
                 pii_filter: PiiFilter, abuse_filter: AbuseFilter,
                 content_safety: ContentSafety, tracer: Tracer):
        ...

    async def generate(self, messages: list[dict], user_id: str | None = None,
                       api_key: str | None = None, model: str | None = None, **kwargs) -> GenerateResult:
        """Hauptschleife: Input Guardrails → Router → Model → Output Guardrails → Audit"""
        # 1. Input Guardrails (rate limit, pii, abuse)
        # 2. Router Decision (complexity, policy, hybrid)
        # 3. Model Call (local oder remote)
        # 4. Output Guardrails (pii leak, content safety)
        # 5. Audit + Trace
        ...

    async def generate_stream(self, messages: list[dict], **kwargs) -> AsyncIterator[GenerateResult]:
        """Streaming-Version."""
        ...
```

**Acceptance Criteria:**
- [ ] Integriert alle Guardrails (Input + Output)
- [ ] Integriert alle Routing-Strategien
- [ ] OTEL-Tracing über gesamte Pipeline
- [ ] Error-Handling auf jeder Ebene
- [ ] Logging aller Schritte

---

### Task 19: server/app.py — FastAPI App-Factory

**FastAPI-Anwendung mit Lifespan, CORS, Exception-Handling.**

```python
def create_app() -> FastAPI:
    app = FastAPI(title="LLM Router & Gateway", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
    app.include_router(router)
    return app
```

**Acceptance Criteria:**
- [ ] FastAPI-App mit Lifespan
- [ ] CORS-Middleware
- [ ] Request-Logging
- [ ] Exception-Handler (422, 500)
- [ ] CLI-Entry-Point: `llm-router-server`

---

### Task 20: server/routes.py — Endpunkte

| Endpoint | Method | Beschreibung |
|---|---|---|
| `/v1/chat/completions` | POST | Haupt-Endpunkt (stream/nicht-stream) |
| `/v1/models` | GET | Liste verfügbarer Modelle |
| `/health` | GET | Health-Check |
| `/v1/guardrails/check` | POST | Manuelles Guardrail-Prüf-Endpunkt |
| `/v1/routing/rules` | GET | Sichtbare Policy-Regeln |
| `/v1/admin/reload` | POST | Policies/Profile neu laden |

**Acceptance Criteria:**
- [ ] Alle 6 Endpunkte implementiert
- [ ] `/v1/chat/completions` mit Stream/Non-Stream
- [ ] SSE-Streaming mit `StreamingResponse`
- [ ] OpenAI-kompatible Request/Response-Formate
---

### Task 21: Logging — JSONL Audit-Logging

**Schreiben von Audit-Einträgen nach JSONL pro Request.**

**Audit-Eintrag Struktur:**
```json
{
  "timestamp": "2026-07-13T...",
  "request_id": "abc123",
  "user_id": "user-456",
  "model_selected": "llama-local",
  "routing_strategy": "policy",
  "policy_matched": "no_remote_pii",
  "guardrail_pii_detected": 2,
  "guardrail_abuse_score": 0.0,
  "prompt_tokens": 150,
  "completion_tokens": 200,
  "total_tokens": 350,
  "cost": 0.0,
  "latency_ms": 45.2,
  "error": null,
  "status": "success"
}
```

**Acceptance Criteria:**
- [ ] JSONL-Logger schreibt pro Request einen Eintrag
- [ ] Timestamps sind ISO-8601
- [ ] Files pro Tag (`audit_YYYYMMDD.jsonl`)
- [ ] Alle oben genannten Felder enthalten
- [ ] Rotation: Alte Files nach 30 Tagen löschen

---

### Task 22: Policy-Konfiguration — default.yaml

**Beispiel-Policy-Datei mit sinnvollen Regeln.**

**Inhalt von `agent-policies/default.yaml`:**
```yaml
rules:
  - id: "no_remote_pii"
    name: "Keine externen Calls für PII"
    conditions:
      contains_pii: true
    target_model: "llama-local"
    priority: 100

  - id: "code_generation"
    name: "Code → lokales großes Modell"
    conditions:
      task_type: "code"
    target_model: "qwen-72b"
    priority: 50

  - id: "complex_analysis"
    name: "Komplexe Analyse → GPT-4o"
    conditions:
      complexity: "high"
    target_model: "gpt4o"
    priority: 30

  - id: "default_remote"
    name: "Standard Remote"
    conditions: {}
    target_model: "gpt4o"
    priority: 0
```

**Acceptance Criteria:**
- [ ] default.yaml mit sinnvollen Regeln
- [ ] YAML wird korrekt von PolicyMatcher geladen
- [ ] Regeln sind nach Priorität sortiert

---

### Task 23: Model-Profile — local.yaml, remote.yaml

**Beispiel-Model-Konfigurationen.**

**`profiles/local.yaml`:**
```yaml
- id: "llama-local"
  name: "Llama CPP Local"
  type: "local"
  base_url: "http://localhost:8888/v1"
  api_key: "sk-unsloth-14c1730a00492812a57221693069c210"
  model_name: "Qwen3.6-35B-A3B-local"
  enabled: true
  temperature: 0.3
  max_tokens: 4096
```

**`profiles/remote.yaml`:**
```yaml
- id: "gpt4o"
  name: "OpenAI GPT-4o"
  type: "remote"
  base_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  model_name: "gpt-4o"
  enabled: true
  timeout: 60
  cost_per_1m_input: 2.50
  cost_per_1m_output: 10.00
```

**Acceptance Criteria:**
- [ ] local.yaml für lokales Modell
- [ ] remote.yaml für Remote-Modell
- [ ] Environment-Variable-Expansion (`${VAR}`)
- [ ] Wird von ModelPool geladen

---

### Task 24: .env.example & Dokumentation

**Acceptance Criteria:**
- [ ] `.env.example` mit allen Config-Variablen
- [ ] Kurzanleitung zum Starten
- [ ] Beispiele für Policy-Konfiguration
- [ ] Beispiele für Guardrail-Checks

---

### Task 25: Tests — Unit Tests für alle Module

**Test-Abdeckung:**
- `tests/test_pool/test_base.py` — ModelBackend ABC Tests
- `tests/test_pool/test_local.py` — LlamaCPPBackend Tests (mock)
- `tests/test_pool/test_remote.py` — RemoteBackend Tests (mock)
- `tests/test_pool/test_pool.py` — ModelPool Tests
- `tests/test_routing/test_complexity.py` — Komplexitäts-Detektor Tests
- `tests/test_routing/test_policy.py` — PolicyMatcher Tests
- `tests/test_routing/test_hybrid.py` — HybridRouter Tests
- `tests/test_routing/test_round_robin.py` — RoundRobin Tests
- `tests/test_guardrails/test_rate_limiter.py` — Rate Limiter Tests
- `tests/test_guardrails/test_pii_filter.py` — PII Filter Tests
- `tests/test_guardrails/test_abuse_filter.py` — Abuse Filter Tests
- `tests/test_guardrails/test_content_safety.py` — Content Safety Tests
- `tests/test_tracing/test_otel.py` — OTEL Tests
- `tests/test_router/test_router.py` — RouterPolicyEngine Tests
- `tests/test_server/test_models.py` — Request/Response Tests

**Acceptance Criteria:**
- [ ] Alle Unit-Tests bestanden
- [ ] Mock-Server für externe API-Calls
- [ ] Coverage > 80%

---

### Task 26: Tests — Integrationstests (E2E)

**Acceptance Criteria:**
- [ ] End-to-End Test: Request → Router → Mock-LLM → Response
- [ ] End-to-End Test: PII im Input wird blockiert/geredactiert
- [ ] End-to-End Test: Policy-Regel wird korrekt gematcht
- [ ] End-to-End Test: Streaming funktioniert
- [ ] End-to-End Test: Rate Limiting funktioniert

---

### Task 27: README.md aktualisieren

**README.md mit:**
- Projektbeschreibung
- Quickstart (Installation, Config, Start)
- Architektur-Übersicht
- API-Dokumentation (alle Endpunkte)
- Policy-Beispiele
- Guardrail-Beispiele
- OTEL-Integration
- Troubleshooting
- License

**Acceptance Criteria:**
- [ ] Vollständige Dokumentation
- [ ] Alle Beispiele laufen
- [ ] Architektur-Diagramm enthalten
