"""Generator script for notebooks/decider_colab_server.ipynb.

Produces a production-hardened Google Colab notebook to run Strands Decider 2B:
- Dynamic device detection: CUDA GPU when assigned, automatic fallback to CPU mode.
- Avoids duplicate processes on ports 8099 and 8000.
- Health checks Strands Decider BEFORE opening the tunnel.
- Dynamic token loading from .decider_token without hardcoding into source.
- Verifies Bearer token auth policy locally before exposing to tunnel.
- Robust pycloudflared Urls object attribute handling (.tunnel / regex).
- Never prints secrets or tokens directly into notebook cell outputs.
"""

from __future__ import annotations

import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Strands Decider 2B — Authenticated Remote Inference Server\n",
                "\n",
                "**DevFlow AI — AWS Agent Experience Hackathon (`#strands-decider-2b`, `#tool-selection`)**\n",
                "\n",
                "This notebook runs the official **Strands Decider 2B** (`StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`) model server with:\n",
                "1. **Automatic Device Detection**: CUDA GPU if assigned; graceful fallback to CPU mode (`--device cpu`) if Colab assigns a CPU instance.\n",
                "2. **Process Lifecycle Management**: Automatically terminates any stale processes before starting new instances.\n",
                "3. **Verified Health Polling**: Actively verifies that `strands-decider` has loaded weights and responds to `/health` before launching tunnels.\n",
                "4. **Bearer Token Authentication Proxy**: Protects port 8099 behind a FastAPI proxy on port 8000. Unauthorized traffic receives `401 Unauthorized`.\n",
                "5. **Zero-Setup Secure Tunnel**: Exposes port 8000 via Cloudflare Tunnel (`pycloudflared`)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 1: Hardware & Device Verification\n",
                "Verifies whether an NVIDIA GPU is available. If none is assigned, configures CPU-only execution."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import psutil\n",
                "import os\n",
                "\n",
                "ram_gb = psutil.virtual_memory().total / (1024 ** 3)\n",
                "print(f\"Total System RAM: {ram_gb:.2f} GB\")\n",
                "\n",
                "if torch.cuda.is_available():\n",
                "    gpu_name = torch.cuda.get_device_name(0)\n",
                "    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)\n",
                "    DEVICE = \"cuda\"\n",
                "    print(f\"[+] GPU Available: {gpu_name} ({gpu_mem:.2f} GB VRAM)\")\n",
                "    print(\"    Mode: High-Performance GPU Inference (~25-115ms latency)\")\n",
                "else:\n",
                "    DEVICE = \"cpu\"\n",
                "    print(\"[-] GPU Not Available (Colab CPU Runtime)\")\n",
                "    print(f\"    Mode: CPU-Only Inference ({ram_gb:.2f} GB RAM is sufficient for 2B weights)\")\n",
                "    print(\"    Latency will be higher (~400-1200ms), but fully functional.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Install Official Packages & Dependencies\n",
                "Installs `strands-decider`, `fastapi`, `uvicorn`, `httpx`, and `pycloudflared`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q strands-decider fastapi uvicorn httpx pycloudflared\n",
                "print(\"[+] Packages installed successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Configure Authentication Token\n",
                "Sets an authentication token in memory to protect your remote model endpoint. The token is stored in memory and `.decider_token`, but never printed into output cells to prevent credential leakage in notebooks, demos, or version control."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import secrets\n",
                "\n",
                "# Prompt for token or generate random 32-char hex token\n",
                "user_key = input(\"Enter secret API key (press Enter to auto-generate random token): \").strip()\n",
                "API_KEY = user_key if user_key else secrets.token_hex(16)\n",
                "\n",
                "# Save token to private config file (.decider_token)\n",
                "with open(\".decider_token\", \"w\") as f:\n",
                "    f.write(API_KEY)\n",
                "\n",
                "import os\n",
                "os.chmod(\".decider_token\", 0o600)\n",
                "print(\"[+] Secret API key configured securely in memory and '.decider_token'.\")\n",
                "print(\"    (Token is masked and NOT displayed in output cells to prevent accidental leakage).\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Write Secure Reverse Proxy Script\n",
                "This FastAPI proxy listens on port 8000 and requires `Authorization: Bearer <API_KEY>` for all incoming requests. It dynamically loads the secret from `.decider_token`. Unauthorized requests receive `401 Unauthorized`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "proxy_code = '''\n",
                "from fastapi import FastAPI, Header, HTTPException, Request, Response\n",
                "import httpx\n",
                "import os\n",
                "import uvicorn\n",
                "\n",
                "app = FastAPI(title=\"Strands Decider Auth Proxy\")\n",
                "DECIDER_INTERNAL_URL = \"http://127.0.0.1:8099\"\n",
                "\n",
                "def get_secret_key() -> str:\n",
                "    try:\n",
                "        with open(\".decider_token\", \"r\") as f:\n",
                "            return f.read().strip()\n",
                "    except Exception:\n",
                "        return os.environ.get(\"API_KEY\", \"\")\n",
                "\n",
                "def verify_token(authorization: str | None = Header(None)):\n",
                "    secret_key = get_secret_key()\n",
                "    if not authorization:\n",
                "        raise HTTPException(status_code=401, detail=\"Missing Authorization header\")\n",
                "    parts = authorization.split()\n",
                "    if len(parts) != 2 or parts[0].lower() != \"bearer\" or parts[1] != secret_key:\n",
                "        raise HTTPException(status_code=401, detail=\"Invalid Bearer token\")\n",
                "\n",
                "@app.get(\"/health\")\n",
                "async def proxy_health(authorization: str | None = Header(None)):\n",
                "    verify_token(authorization)\n",
                "    async with httpx.AsyncClient() as client:\n",
                "        try:\n",
                "            res = await client.get(f\"{DECIDER_INTERNAL_URL}/health\", timeout=5.0)\n",
                "            return res.json()\n",
                "        except Exception as e:\n",
                "            return {\"online\": False, \"error\": str(e)}\n",
                "\n",
                "@app.post(\"/v1/systemone\")\n",
                "async def proxy_systemone(request: Request, authorization: str | None = Header(None)):\n",
                "    verify_token(authorization)\n",
                "    body = await request.body()\n",
                "    async with httpx.AsyncClient() as client:\n",
                "        try:\n",
                "            res = await client.post(\n",
                "                f\"{DECIDER_INTERNAL_URL}/v1/systemone\",\n",
                "                content=body,\n",
                "                headers={\"Content-Type\": \"application/json\"},\n",
                "                timeout=30.0,\n",
                "            )\n",
                "            return Response(content=res.content, status_code=res.status_code, media_type=\"application/json\")\n",
                "        except Exception as e:\n",
                "            raise HTTPException(status_code=502, detail=f\"Internal Decider error: {e}\")\n",
                "\n",
                "if __name__ == \"__main__\":\n",
                "    uvicorn.run(app, host=\"127.0.0.1\", port=8000)\n",
                "'''\n",
                "\n",
                "with open(\"auth_proxy.py\", \"w\") as f:\n",
                "    f.write(proxy_code.strip())\n",
                "print(\"[+] Created auth_proxy.py successfully.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 5: Launch Model Server, Health Polling & Secure Tunnel\n",
                "Cleans up stale processes, starts `strands-decider serve` on port 8099, actively polls for health readiness, starts the auth proxy on port 8000, and exposes port 8000 via Cloudflare Tunnel."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import re\n",
                "import subprocess\n",
                "import time\n",
                "import requests\n",
                "from pycloudflared import try_cloudflare\n",
                "\n",
                "# 1. Clean up any stale processes from previous runs\n",
                "print(\"[*] Cleaning up any previous server processes...\")\n",
                "subprocess.run(\"fuser -k 8099/tcp 8000/tcp 2>/dev/null || true\", shell=True)\n",
                "subprocess.run(\"pkill -f 'strands-decider' 2>/dev/null || true\", shell=True)\n",
                "subprocess.run(\"pkill -f 'auth_proxy.py' 2>/dev/null || true\", shell=True)\n",
                "subprocess.run(\"pkill -f 'cloudflared' 2>/dev/null || true\", shell=True)\n",
                "time.sleep(2)\n",
                "\n",
                "# 2. Launch strands-decider on internal port 8099\n",
                "model_checkpoint = \"StrandsAgents/strands-decider-2B-qwen3.5-v1-2610\"\n",
                "print(f\"[*] Launching strands-decider on 127.0.0.1:8099 (device: {DEVICE})...\")\n",
                "decider_proc = subprocess.Popen([\n",
                "    \"strands-decider\", \"serve\", model_checkpoint,\n",
                "    \"--host\", \"127.0.0.1\", \"--port\", \"8099\",\n",
                "    \"--device\", DEVICE\n",
                "])\n",
                "\n",
                "# 3. Active health polling — wait until weights are loaded and server returns 200\n",
                "print(\"[*] Polling model server on /health (up to 60s for weight loading)...\")\n",
                "model_ready = False\n",
                "for attempt in range(30):\n",
                "    try:\n",
                "        h = requests.get(\"http://127.0.0.1:8099/health\", timeout=1.5)\n",
                "        if h.status_code == 200:\n",
                "            print(f\"[+] Model server is HEALTHY and READY! (loaded in ~{(attempt + 1) * 2}s)\")\n",
                "            print(f\"    Model details: {h.json()}\")\n",
                "            model_ready = True\n",
                "            break\n",
                "    except Exception:\n",
                "        pass\n",
                "    if decider_proc.poll() is not None:\n",
                "        raise RuntimeError(f\"[-] strands-decider exited prematurely with code {decider_proc.returncode}\")\n",
                "    time.sleep(2)\n",
                "\n",
                "if not model_ready:\n",
                "    raise RuntimeError(\"[-] Model server failed to become healthy on port 8099 within 60s.\")\n",
                "\n",
                "# 4. Launch FastAPI authenticated proxy on port 8000\n",
                "print(\"[*] Starting Authenticated Reverse Proxy on port 8000...\")\n",
                "proxy_proc = subprocess.Popen([\"python\", \"auth_proxy.py\"])\n",
                "time.sleep(2)\n",
                "\n",
                "# Verify proxy local policy\n",
                "r_unauth = requests.get(\"http://127.0.0.1:8000/health\", timeout=2.0)\n",
                "r_auth = requests.get(\"http://127.0.0.1:8000/health\", headers={\"Authorization\": f\"Bearer {API_KEY}\"}, timeout=2.0)\n",
                "if r_unauth.status_code != 401 or r_auth.status_code != 200:\n",
                "    raise RuntimeError(f\"[-] Proxy verification failed: unauth={r_unauth.status_code}, auth={r_auth.status_code}\")\n",
                "print(\"[+] Auth proxy verified: 401 on unauthorized, 200 with Bearer token.\")\n",
                "\n",
                "# 5. Establish Cloudflare Tunnel\n",
                "print(\"[*] Opening Cloudflare Tunnel to authenticated port 8000...\")\n",
                "tunnel_obj = try_cloudflare(port=8000)\n",
                "\n",
                "# Handle Urls object (.tunnel), alternate versions (.url), or string representation\n",
                "raw_tunnel = getattr(tunnel_obj, \"tunnel\", None) or getattr(tunnel_obj, \"url\", None) or str(tunnel_obj)\n",
                "match = re.search(r\"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com\", str(raw_tunnel))\n",
                "if match:\n",
                "    tunnel_url = match.group(0)\n",
                "else:\n",
                "    raise RuntimeError(f\"[-] Could not extract Cloudflare tunnel URL from: {raw_tunnel}\")\n",
                "\n",
                "print(\"\\n\" + \"=\" * 75)\n",
                "print(\"STRANDS DECIDER 2B SERVER IS LIVE AND SECURED!\")\n",
                "print(\"=\" * 75)\n",
                "print(f\"Public Tunnel URL : {tunnel_url}\")\n",
                "print(\"Authentication    : HTTP Bearer Token (Configured in Step 3)\")\n",
                "print(\"=\" * 75)\n",
                "print(\"\\nRun these exact commands in your Windows PowerShell terminal:\")\n",
                "print(f'$env:STRANDS_DECIDER_URL = \"{tunnel_url}\"')\n",
                "print('$env:STRANDS_DECIDER_API_KEY = \"<your-token-from-step-3>\"')\n",
                "print('E:\\\\devflow_venv\\\\Scripts\\\\python.exe verify_live_e2e.py')\n",
                "print(\"=\" * 75)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6: Test Authenticated Inference (In-Notebook Verification)\n",
                "Sends a genuine `#tool-selection` decision query through the authenticated tunnel URL."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import requests\n",
                "\n",
                "headers = {\n",
                "    \"Authorization\": f\"Bearer {API_KEY}\",\n",
                "    \"Content-Type\": \"application/json\",\n",
                "}\n",
                "\n",
                "# 1. Test /health\n",
                "health_res = requests.get(f\"{tunnel_url}/health\", headers=headers, timeout=10)\n",
                "print(\"Health Check Response:\", health_res.json())\n",
                "\n",
                "# 2. Test /v1/systemone with a real tool-selection query\n",
                "payload = {\n",
                "    \"state\": \"Why is my pytest asserting assert 1 == 2?\\n\\nContext:\\nFAILED test_app.py - AssertionError: 1 != 2\",\n",
                "    \"model\": \"StrandsAgents/strands-decider-2B-qwen3.5-v1-2610\",\n",
                "    \"questions\": {\n",
                "        \"tool_choice\": {\n",
                "            \"type\": \"choice\",\n",
                "            \"instructions\": \"Select the most appropriate developer tool to process the user's request.\",\n",
                "            \"criteria\": {\n",
                "                \"explain_code\": \"Explain code structure, syntax, logic, algorithms, or programming concepts.\",\n",
                "                \"debug_error\": \"Analyze exceptions, stack traces, crash logs, and runtime error messages.\",\n",
                "                \"analyze_test\": \"Interpret failing test output, assertions, test runner logs, and expected vs actual values.\",\n",
                "                \"search_docs\": \"Retrieve relevant documentation, API specifications, and architectural guides from the local documentation collection.\"\n",
                "            }\n",
                "        }\n",
                "    }\n",
                "}\n",
                "\n",
                "decision_res = requests.post(f\"{tunnel_url}/v1/systemone\", json=payload, headers=headers, timeout=20)\n",
                "print(\"\\nDecision Response from Live Model:\")\n",
                "print(decision_res.json())"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

os.makedirs("notebooks", exist_ok=True)
notebook_path = os.path.join("notebooks", "decider_colab_server.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated {notebook_path} successfully.")
