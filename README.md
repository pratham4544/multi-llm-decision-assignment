# Streamlit App - LLM Router System

## Prerequisites

- Python 3.10+
- Redis running on localhost:6379
- Groq API key in `.env` file

## Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Redis (if not running)
redis-server

# 3. Start the FastAPI backend
python -m src.main

# 4. In a new terminal, start Streamlit
streamlit run streamlit_app.py
```

The app opens at **http://localhost:8501**.

## Model Routing

| Query Type | Model | Example |
|---|---|---|
| Simple | llama-3.1-8b-instant | "What is the capital of France?" |
| Coding | mixtral-8x7b-32768 | "Write a Python function to sort a list" |
| Complex | llama-3.3-70b-versatile | "Analyze the economic implications of AI" |
| High latency/rate | gemma2-9b-it | Auto-fallback when system is under load |

## Test Panel

Click the test buttons at the bottom of the app to verify:

- **Test Router** - Backend is alive and routing
- **Test Redis** - Redis connectivity
- **Test Latency** - End-to-end response time
- **Test Model Routing** - Correct model selected for simple, coding, and complex queries
