# EvalShield-AI

> **Production-Grade End-to-End Test Suite & MLOps Guardrail Pipeline for AI Systems**

[![CI/CD Pipeline](https://github.com/vladzpop11/evalshield-ai/actions/workflows/ci-cd-pipeline.yml/badge.svg)](https://github.com/vladzpop11/evalshield-ai/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![ONNX Runtime INT8](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-005CED.svg)](https://onnxruntime.ai)
[![Pandera](https://img.shields.io/badge/Pandera-Contract_Validation-FF6F00.svg)](https://pandera.readthedocs.io)
[![Playwright](https://img.shields.io/badge/Playwright-E2E_%26_A11y-45ba4b.svg)](https://playwright.dev)
[![WCAG 2.2 AA](https://img.shields.io/badge/Accessibility-WCAG_2.2_AA-6C5CE7.svg)](https://www.w3.org/WAI/standards-guidelines/wcag/)
[![NIST AI RMF 1.0](https://img.shields.io/badge/Compliance-NIST_AI_RMF_1.0-22c55e.svg)](https://www.nist.gov/itl/ai-risk-management-framework)
[![EU AI Act](https://img.shields.io/badge/Compliance-EU_AI_Act_Annex_IV-2563eb.svg)](https://artificialintelligenceact.eu/)

---

## 1. Architectural Overview

EvalShield-AI is a modular, enterprise-grade quality engineering platform and safety guardrail engine for AI/LLM applications. It provides full-stack quality governance across every layer of the modern AI lifecycle:

```mermaid
flowchart TD
    subgraph Ingestion & NLP Feature Extraction
        RAW[Raw Prompt / Response Text] -->|Sub-ms Regex & Entropy| EXTRACT[TextFeatureExtractor]
        METRIC[Pre-computed Float Scores] -->|Caller Override| GATE{Dual-Mode Router}
        EXTRACT --> GATE
    end

    subgraph Serving & Execution Engine
        GATE --> CACHE{LRU Semantic Cache\nSHA-256 Fingerprint}
        CACHE -->|Cache Hit 0.0ms| RESP[Authoritative Verdict]
        CACHE -->|Cache Miss| ROUTER[Champion / Challenger Router]
        ROUTER -->|Synchronous| CHAMP[Champion: INT8 ONNX]
        ROUTER -.->|Async Shadow Traffic| CHAL[Challenger: FP32 ONNX / Sklearn]
        CHAL -.->|Discrepancy Check| DISCREP[discrepancy_log.jsonl]
        CHAMP --> RESP
    end

    subgraph Governance & Observability
        RESP --> PROM[Prometheus /metrics]
        RESP --> NIST[NIST & EU AI Act Report]
        RESP --> SHADOW[/model/shadow-stats]
        RESP --> DRIFT[Two-Sample KS Drift Monitor]
    end
```

---

## 2. Directory Structure

```text
evalshield-ai/
├── .github/
│   └── workflows/
│       └── ci-cd-pipeline.yml         # Multi-stage GitHub Actions CI/CD automation
├── docker/
│   ├── Dockerfile.api                 # Hardened Python 3.10/3.11 slim container (non-root)
│   ├── Dockerfile.frontend            # Lightweight Nginx Alpine reverse proxy
│   └── docker-compose.yml             # Full-stack container orchestration
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                    # Production FastAPI v1.2 with shadow router, cache & Prometheus
│   │   └── schemas.py                 # Pydantic v2 schemas: dual-mode input & compliance models
│   └── ml/
│       ├── __init__.py
│       ├── artifacts/
│       │   ├── evalshield_model_int8.onnx # Quantized INT8 ONNX production model
│       │   ├── evalshield_model.onnx      # Full-precision FP32 ONNX fallback
│       │   ├── evalshield_model.joblib    # Scikit-learn Joblib pipeline fallback
│       │   ├── baseline_distribution.parquet # 4,000 synthetic benchmark records
│       │   ├── metadata.json              # Hyperparameters, metrics & schema
│       │   └── discrepancy_log.jsonl      # Async shadow routing discrepancy log
│       ├── champion_challenger.py     # Asynchronous shadow evaluation router
│       ├── feature_extractor.py       # Zero-dependency in-memory NLP feature extractor
│       ├── inference.py               # 3-tier inference engine (INT8 ONNX -> FP32 -> Joblib)
│       └── model_generator.py         # Training, INT8 dynamic quantization & parquet generation
├── tests/
│   ├── ml_quality/
│   │   ├── __init__.py
│   │   ├── test_adversarial_fuzzing.py # 17 fuzzing tests (homoglyph, base64, DAN, jailbreaks)
│   │   ├── test_champion_challenger.py # 6 shadow routing & telemetry tests
│   │   ├── test_data_contracts.py     # 11 Pandera schema boundary & strictness tests
│   │   ├── test_model_behavior.py     # 6 CheckList behavioral tests (MFT, INV, DIR)
│   │   └── test_metrics_drift.py      # 4 Kolmogorov-Smirnov & PSI drift detection tests
│   ├── api/
│   │   ├── __init__.py
│   │   └── test_api_endpoints.py      # 14 async integration tests (text mode, metrics, audit)
│   ├── e2e_ui/
│   │   ├── specs/
│   │   │   ├── dashboard.spec.ts      # End-to-end interactive workflow tests
│   │   │   ├── visual.spec.ts         # Visual & responsive viewport tests
│   │   │   └── a11y.spec.ts           # WCAG 2.2 AA automated accessibility audits
│   │   ├── playwright.config.ts       # Playwright configuration
│   │   └── package.json               # Node.js dependencies (@axe-core/playwright, etc.)
│   └── performance/
│       └── load-test.js               # k6 load testing script with stages & SLA thresholds
├── index.html                         # WCAG 2.2 AA compliant interactive shield dashboard
├── requirements.txt                   # Locked Python dependencies
├── pytest.ini                         # Pytest runner configuration
└── README.md                          # Architecture, verification & deployment manual
```

---

## 3. Platform Capabilities & Innovations

### A. Dual-Mode Ingestion & NLP Feature Extraction
- **Text Mode (Recommended):** Ingest raw prompt and response texts. In-memory `TextFeatureExtractor` auto-derives all 5 risk dimensions:
  1. *Token Shannon Entropy:* Detects degenerate repetition or chaotic payload gibberish.
  2. *Prompt Injection Probability:* 11 regex patterns, Markdown delimiter abuse, Base64 payload decoding, and non-ASCII density.
  3. *Toxicity Score:* Lexical indicator density with non-linear severity scaling.
  4. *Semantic Drift:* Out-of-domain distance against approved enterprise vocabulary.
  5. *Hallucination Risk:* Lexical prompt-response grounding and speculative qualifier detection.
- **Metric Mode:** Programmatic ingestion of pre-computed float scores with optional partial overrides.

### B. INT8 Dynamic ONNX Quantization & LRU Caching
- **INT8 Quantization:** `evalshield_model_int8.onnx` generated via ONNX Runtime dynamic quantization, achieving ~40% latency reduction.
- **LRU Semantic Cache:** SHA-256 feature fingerprint cache (512 slots) serving repeated inputs at `0.0 ms` latency.
- **3-Tier Failover:** INT8 ONNX $\to$ FP32 ONNX $\to$ Scikit-Learn Joblib pipeline.

### C. Champion / Challenger Shadow Router
- Zero added latency for production callers: Champion responds synchronously while Challenger evaluates mirrored traffic asynchronously via `ThreadPoolExecutor`.
- Tracks real-time decision agreement rates, latency deltas, and logs divergence to `discrepancy_log.jsonl`.
- Live stats endpoint: `GET /model/shadow-stats`.

### D. Sensitive Data & PII Redaction Engine (`PIIDetector`)
- **Automated Discovery & Anonymization:** Identifies sensitive entities and credentials in real-time:
  - *Credit Card Numbers:* Validated via **ISO/IEC 7812 Luhn checksum algorithm** (eliminates false positives).
  - *Social Security Numbers:* US SSN area-group-serial structure.
  - *API Keys & Cloud Secrets:* OpenAI (`sk-...`), AWS Access Keys (`AKIA...`), GitHub Tokens (`ghp_...`), JWTs.
  - *Contact Details:* Emails, international phone numbers, and external IP addresses.
- **Redaction Support:** Replaces sensitive values with non-reversible masked placeholders (`[REDACTED_EMAIL]`, `[REDACTED_CREDIT_CARD]`, etc.).

### E. Risk Explainability & Attribution
- **Normalized Factor Attribution:** Generates explainability weights (`risk_contributions`) per inference that sum to 100%, enabling transparent attribution compliant with EU AI Act Article 13.

### F. Real-Time Token Streaming Guardrail (`/evaluate/stream`)
- **Server-Sent Events (SSE):** Streaming interface for LLM pipelines yielding chunk-by-chunk validation with instant `ABORT_STREAM` signals when safety constraints are breached mid-generation.

### G. Observability & Regulatory Compliance
- **Prometheus Scraper:** `GET /metrics` exposes real-time counters, mode breakdowns, cache hits, PII detections, shadow stats, and p50/p95/p99 quantiles alongside PromQL `_sum` and `_count` series.
- **Audit Export:** `GET /model/compliance-report` outputs structured technical documentation aligned with **NIST AI RMF 1.0** and **EU AI Act Regulation (EU) 2024/1689 Annex IV**.

---

## 4. Local Quickstart

### Prerequisites
- Python 3.10+
- Node.js 20+
- (Optional) Docker & Docker Compose
- (Optional) k6 (for performance load testing)

### Step 1: Clone and Set Up Python Environment

```bash
git clone https://github.com/vladzpop11/evalshield-ai.git
cd evalshield-ai

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Train & Compile Model Artifacts

```bash
python -m src.ml.model_generator
```
*Generates INT8 ONNX, FP32 ONNX, Joblib pipeline, baseline distribution parquet, and metadata JSON under `src/ml/artifacts/`.*

### Step 3: Run the FastAPI Service

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Prometheus Metrics: [http://localhost:8000/metrics](http://localhost:8000/metrics)
- NIST & EU AI Act Report: [http://localhost:8000/model/compliance-report](http://localhost:8000/model/compliance-report)
- Shadow Router Telemetry: [http://localhost:8000/model/shadow-stats](http://localhost:8000/model/shadow-stats)

### Step 4: Open the Web UI Dashboard

Serve the static dashboard:
```bash
python3 -m http.server 8080
```
Visit [http://localhost:8080](http://localhost:8080) to interact with the guardrail playground.

---

## 5. Running the Test Suites

### Full Suite Run (70 Automated Tests)
```bash
pytest tests/ml_quality tests/api -v
```

### A. ML Quality, Fuzzing, PII & Shadow Router Tests
```bash
pytest tests/ml_quality -v
```
Runs:
- `test_adversarial_fuzzing.py`: 17 jailbreak, homoglyph, Base64, and boundary tests.
- `test_pii_detector.py`: 8 ISO/IEC 7812 Luhn card, SSN, API key, and redaction tests.
- `test_champion_challenger.py`: 6 async shadow dispatch, discrepancy, and telemetry tests.
- `test_data_contracts.py`: 11 Pandera schema boundary & null rejection tests.
- `test_model_behavior.py`: 6 CheckList behavioral tests (MFT, INV, DIR).
- `test_metrics_drift.py`: 4 Kolmogorov-Smirnov 2-sample & PSI drift detection tests.

### B. API Integration & Contract Tests
```bash
pytest tests/api -v
```
Runs:
- 18 async tests verifying `/health`, `/metrics`, `/evaluate/stream`, `/model/compliance-report`, `/model/shadow-stats`, PII redaction, explainability weights, batch evaluation, and drift monitoring.

### C. Frontend E2E, Visual & Accessibility Tests
```bash
cd tests/e2e_ui
npm test
npm run test:a11y    # WCAG 2.2 AA audit (0 violations)
npm run test:visual  # Responsive viewport validation
```

### D. Performance & Load Testing with k6
```bash
k6 run tests/performance/load-test.js
```
Validates:
- `http_req_duration`: p(95) < 120ms, p(99) < 250ms
- `http_req_failed`: error rate < 1%

---

## 6. Docker Orchestration

Run the complete multi-service stack:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

Verify service health:
```bash
docker compose -f docker/docker-compose.yml ps
```

Teardown:
```bash
docker compose -f docker/docker-compose.yml down -v
```

---

## 7. Quality Gates & SLA Thresholds

| Metric | Target SLA | Test Enforcement | Status |
| :--- | :--- | :--- | :--- |
| **Model Accuracy** | $\ge 85.0\%$ | `test_metrics_drift.py` | **94.88%** PASS |
| **Model Macro F1** | $\ge 80.0\%$ | `test_metrics_drift.py` | **88.92%** PASS |
| **Single Inference Latency** | $< 30\text{ ms}$ | `test_model_behavior.py` | **1.5 ms** PASS |
| **Adversarial Fuzzing** | $100\%$ attack detection | `test_adversarial_fuzzing.py` | **17/17** PASS |
| **Champion / Challenger Shadow** | Zero user latency impact | `test_champion_challenger.py` | **6/6** PASS |
| **API p95 Response Time** | $< 120\text{ ms}$ (40 VUs) | `load-test.js` (k6) | **PASS** |
| **API Error Rate** | $< 1.0\%$ | `load-test.js` (k6) | **PASS** |
| **Accessibility Compliance** | $0$ WCAG 2.2 AA Violations | `a11y.spec.ts` (Axe-Core) | **0 Violations** PASS |
| **Data Contract Integrity** | $100\%$ schema conformity | `test_data_contracts.py` (Pandera) | **PASS** |
