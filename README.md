# LLM Router System

Production-grade distributed LLM processing pipeline with intelligent routing, caching, and observability.

## 📋 Overview

This system routes LLM requests across multiple providers (OpenAI, Anthropic, Groq) based on:
- Request complexity
- Cost optimization
- Provider health
- Latency requirements

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone <repo-url>
cd llm-router-system

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Start Redis
docker-compose up -d redis

# 6. Run tests
pytest tests/ -v

# 7. Start API server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# 8. Start workers (in another terminal)
rq worker --url redis://localhost:6379
```

## 📊 Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## 🧪 Testing

```bash
# Unit tests
pytest tests/ -v --cov=src

# Load testing
locust -f load_tests/locustfile.py --host http://localhost:8000
```

## 📖 Documentation

- [Architecture](ARCHITECTURE.md)
- [Routing Logic](docs/ROUTING_LOGIC.md)
- [Cost Optimization](docs/COST_OPTIMIZATION.md)
- [Observability](docs/OBSERVABILITY.md)
- [Scaling Strategy](docs/SCALING_STRATEGY.md)

## 🔧 Configuration

See `.env.example` for all available configuration options.

## 📝 License

MIT
