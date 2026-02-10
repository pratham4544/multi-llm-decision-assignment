"""
Streamlit UI for the LLM Router System.
Provides a chat interface with real-time routing insights and logs.

Run with: streamlit run streamlit_app.py
(Make sure the FastAPI backend is running on port 8000 first)
"""

import time
import requests
import streamlit as st

API_BASE = "http://localhost:8000"


def api_request(method: str, path: str, **kwargs):
    """Make an API request to the FastAPI backend."""
    try:
        resp = getattr(requests, method)(f"{API_BASE}{path}", timeout=30, **kwargs)
        return resp
    except requests.ConnectionError:
        return None


def check_backend_health():
    """Check if the FastAPI backend is reachable."""
    resp = api_request("get", "/ping")
    return resp is not None and resp.status_code == 200


def test_router_working():
    """Test that the router is alive and can process a request."""
    resp = api_request("post", "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "temperature": 0.0,
        "max_tokens": 50,
    }, headers={"X-Client-ID": "test-panel", "X-Client-Tier": "premium"})
    if resp and resp.status_code == 200:
        data = resp.json()
        return True, {
            "model": data.get("model"),
            "provider": data.get("provider"),
            "latency_ms": round(data.get("latency_ms", 0), 1),
            "status": "PASS",
        }
    return False, {"status": "FAIL", "error": resp.text if resp else "Connection failed"}


def test_redis_working():
    """Test Redis connectivity via the health endpoint."""
    resp = api_request("get", "/health")
    if resp and resp.status_code == 200:
        data = resp.json()
        redis_status = data.get("redis", {}).get("status", "unknown")
        return redis_status == "healthy", {
            "redis_status": redis_status,
            "status": "PASS" if redis_status == "healthy" else "FAIL",
        }
    return False, {"status": "FAIL", "error": "Health endpoint unreachable"}


def test_latency():
    """Measure actual end-to-end latency."""
    start = time.time()
    resp = api_request("post", "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Say hello."}],
        "temperature": 0.0,
        "max_tokens": 10,
    }, headers={"X-Client-ID": "test-panel", "X-Client-Tier": "premium"})
    elapsed = (time.time() - start) * 1000

    if resp and resp.status_code == 200:
        data = resp.json()
        return True, {
            "e2e_latency_ms": round(elapsed, 1),
            "provider_latency_ms": round(data.get("latency_ms", 0), 1),
            "model": data.get("model"),
            "status": "PASS",
        }
    return False, {"status": "FAIL", "e2e_latency_ms": round(elapsed, 1)}


def test_model_routing():
    """Verify correct model selection for different query types."""
    # Clear cache first so we get fresh routing decisions (not stale cached responses)
    api_request("post", "/cache/clear")

    # Use unique suffixes to avoid any residual cache hits
    import uuid
    tag = uuid.uuid4().hex[:6]

    test_cases = [
        {
            "name": "Simple -> llama-3.1-8b-instant",
            "message": f"What is the capital of France? [test-{tag}-simple]",
            "expected_model": "llama-3.1-8b-instant",
        },
        {
            "name": "Coding -> mixtral-8x7b-32768",
            "message": f"Write a Python function to sort a list using merge sort algorithm [test-{tag}-code]",
            "expected_model": "mixtral-8x7b-32768",
        },
        {
            "name": "Complex -> llama-3.3-70b-versatile",
            "message": f"Analyze and compare the economic implications of renewable energy adoption across developing nations, synthesize the key findings step by step [test-{tag}-complex]",
            "expected_model": "llama-3.3-70b-versatile",
        },
    ]
    results = []
    all_pass = True
    for tc in test_cases:
        resp = api_request("post", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": tc["message"]}],
            "temperature": 0.0,
            "max_tokens": 50,
        }, headers={"X-Client-ID": "test-panel", "X-Client-Tier": "premium"})

        if resp and resp.status_code == 200:
            data = resp.json()
            actual_model = data.get("model", "unknown")
            cached = data.get("cached", False)
            passed = actual_model == tc["expected_model"]
            if not passed:
                all_pass = False
            results.append({
                "test": tc["name"],
                "expected": tc["expected_model"],
                "actual": actual_model,
                "cached": cached,
                "status": "PASS" if passed else "FAIL",
            })
        else:
            all_pass = False
            results.append({
                "test": tc["name"],
                "expected": tc["expected_model"],
                "actual": "ERROR",
                "status": "FAIL",
            })
    return all_pass, results


# --- Page config ---
st.set_page_config(
    page_title="LLM Router System",
    page_icon="🔀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "request_logs" not in st.session_state:
    st.session_state.request_logs = []

# --- Sidebar ---
with st.sidebar:
    st.title("LLM Router System")
    st.caption("Intelligent multi-provider LLM routing")

    # Backend status
    backend_up = check_backend_health()
    if backend_up:
        st.success("Backend: Connected")
    else:
        st.error("Backend: Offline")
        st.info("Start the backend with:\n```\npython -m src.main\n```")

    st.divider()

    # Chat settings
    st.subheader("Chat Settings")
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant.",
        height=80,
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.number_input("Max Tokens", min_value=1, max_value=32768, value=1024)

    st.divider()

    # Health / Metrics
    st.subheader("System Info")

    if st.button("Refresh System Info", use_container_width=True):
        st.session_state["refresh_info"] = True

    if backend_up:
        # Health
        health_resp = api_request("get", "/health")
        if health_resp and health_resp.status_code == 200:
            health = health_resp.json()
            status = health.get("status", "unknown")
            color = {"healthy": "green", "degraded": "orange", "unhealthy": "red"}.get(status, "gray")
            st.markdown(f"**Health:** :{color}[{status.upper()}]")

            # Provider health
            providers = health.get("providers", {})
            if providers:
                for pname, pinfo in providers.items():
                    pstatus = pinfo.get("status", "unknown") if isinstance(pinfo, dict) else str(pinfo)
                    st.markdown(f"- `{pname}`: {pstatus}")

            # Redis
            redis_info = health.get("redis", {})
            redis_st = redis_info.get("status", "unknown")
            st.markdown(f"**Redis:** {redis_st}")
        else:
            st.warning("Could not fetch health info")

        # Providers
        prov_resp = api_request("get", "/providers")
        if prov_resp and prov_resp.status_code == 200:
            st.divider()
            st.subheader("Available Models")
            prov_data = prov_resp.json()
            for pname, pinfo in prov_data.items():
                models = pinfo.get("models", [])
                for m in models:
                    st.markdown(
                        f"- **{m['id']}** ({m['complexity']}) "
                        f"— ${m['input_cost_per_1k']:.5f}/1K in"
                    )

    st.divider()

    # Cache controls
    st.subheader("Cache")
    if st.button("Clear Cache", use_container_width=True):
        resp = api_request("post", "/cache/clear")
        if resp and resp.status_code == 200:
            st.success("Cache cleared!")
        else:
            st.error("Failed to clear cache")

# --- Main area ---
st.header("Chat")

# Display chat history
for entry in st.session_state.messages:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry["role"] == "assistant" and "meta" in entry:
            meta = entry["meta"]
            cols = st.columns(5)
            cols[0].metric("Provider", meta.get("provider", "—"))
            cols[1].metric("Model", meta.get("model", "—"))
            cols[2].metric("Latency", f"{meta.get('latency_ms', 0):.0f}ms")
            cols[3].metric("Cost", f"${meta.get('cost', 0):.6f}")
            cols[4].metric("Tokens", str(meta.get("total_tokens", 0)))

            if meta.get("cached"):
                st.info(f"Served from cache ({meta.get('cache_type', 'exact')})")

# Chat input
if prompt := st.chat_input("Send a message...", disabled=not backend_up):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build request payload
    api_messages = []
    if system_prompt.strip():
        api_messages.append({"role": "system", "content": system_prompt.strip()})
    for msg in st.session_state.messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # Send to backend
    with st.chat_message("assistant"):
        with st.spinner("Routing request..."):
            start = time.time()
            resp = api_request(
                "post",
                "/v1/chat/completions",
                json={
                    "messages": api_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                headers={
                    "X-Client-ID": "streamlit-ui",
                    "X-Client-Tier": "premium",
                },
            )
            elapsed = time.time() - start

        if resp and resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            meta = {
                "provider": data.get("provider", "—"),
                "model": data.get("model", "—"),
                "latency_ms": data.get("latency_ms", elapsed * 1000),
                "cost": data.get("cost", 0),
                "total_tokens": data.get("usage", {}).get("total_tokens", 0),
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                "cached": data.get("cached", False),
                "cache_type": data.get("cache_type"),
                "trace_id": data.get("trace_id"),
            }

            st.markdown(content)

            # Show routing details
            cols = st.columns(5)
            cols[0].metric("Provider", meta["provider"])
            cols[1].metric("Model", meta["model"])
            cols[2].metric("Latency", f"{meta['latency_ms']:.0f}ms")
            cols[3].metric("Cost", f"${meta['cost']:.6f}")
            cols[4].metric("Tokens", str(meta["total_tokens"]))

            if meta["cached"]:
                st.info(f"Served from cache ({meta.get('cache_type', 'exact')})")

            # Save to session
            st.session_state.messages.append({
                "role": "assistant",
                "content": content,
                "meta": meta,
            })

            # Save to request log
            st.session_state.request_logs.append({
                "prompt": prompt[:80],
                **meta,
                "e2e_time_ms": round(elapsed * 1000, 1),
            })

        elif resp is not None:
            error_detail = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            error_msg = error_detail.get("detail", {}).get("error", {}).get("message", resp.text)
            st.error(f"Error ({resp.status_code}): {error_msg}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {error_msg}",
            })
        else:
            st.error("Could not connect to backend. Is it running?")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Error: Backend unavailable",
            })

# --- Request Log ---
if st.session_state.request_logs:
    st.divider()
    st.subheader("Request Log")

    # Summary metrics
    logs = st.session_state.request_logs
    total_cost = sum(l.get("cost", 0) for l in logs)
    avg_latency = sum(l.get("latency_ms", 0) for l in logs) / len(logs)
    cache_hits = sum(1 for l in logs if l.get("cached"))

    summary_cols = st.columns(4)
    summary_cols[0].metric("Total Requests", len(logs))
    summary_cols[1].metric("Avg Latency", f"{avg_latency:.0f}ms")
    summary_cols[2].metric("Total Cost", f"${total_cost:.6f}")
    summary_cols[3].metric("Cache Hits", str(cache_hits))

    # Log table
    import pandas as pd

    df = pd.DataFrame(logs)
    display_cols = ["prompt", "provider", "model", "latency_ms", "cost", "total_tokens", "cached", "e2e_time_ms"]
    existing = [c for c in display_cols if c in df.columns]
    st.dataframe(df[existing], use_container_width=True, hide_index=True)

    if st.button("Clear Log"):
        st.session_state.request_logs = []
        st.rerun()

# --- Router Test Panel ---
st.divider()
st.header("Router Test Panel")
st.caption("Run diagnostic tests to verify routing, Redis, latency, and model selection")

test_col1, test_col2, test_col3, test_col4 = st.columns(4)

with test_col1:
    if st.button("Test Router", use_container_width=True, type="primary"):
        with st.spinner("Testing router..."):
            passed, result = test_router_working()
        if passed:
            st.success(f"PASS - Model: {result['model']}")
        else:
            st.error(f"FAIL - {result.get('error', 'Unknown error')}")
        st.json(result)

with test_col2:
    if st.button("Test Redis", use_container_width=True, type="primary"):
        with st.spinner("Testing Redis..."):
            passed, result = test_redis_working()
        if passed:
            st.success("PASS - Redis connected")
        else:
            st.error("FAIL - Redis issue")
        st.json(result)

with test_col3:
    if st.button("Test Latency", use_container_width=True, type="primary"):
        with st.spinner("Measuring latency..."):
            passed, result = test_latency()
        if passed:
            st.success(f"PASS - {result['e2e_latency_ms']}ms E2E")
        else:
            st.error(f"FAIL - {result.get('e2e_latency_ms', '?')}ms")
        st.json(result)

with test_col4:
    if st.button("Test Model Routing", use_container_width=True, type="primary"):
        with st.spinner("Running routing tests (3 queries)..."):
            passed, results = test_model_routing()
        if passed:
            st.success("ALL PASS - Models routed correctly")
        else:
            st.error("SOME FAILED - See details below")
        for r in results:
            if r["status"] == "PASS":
                st.success(f"{r['test']}: {r['actual']}")
            else:
                st.error(f"{r['test']}: expected {r['expected']}, got {r['actual']}")
