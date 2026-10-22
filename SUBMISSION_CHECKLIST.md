# AWS Agent Experience Hackathon: Strands Decider — Submission Checklist

This checklist tracks every mandatory requirement for the **AWS Agent Experience Hackathon: Strands Decider** submission for **DevFlow AI**.

---

## 1. Mandatory Technical Requirements (Built & Verified)

- [x] **1. Working original AI agent**
  - Application: **DevFlow AI**
  - Location: `devflow/` and `app.py`
  - Implementation: Python 3.12, Streamlit UI, two-tier cognitive architecture.
  - Verified: App starts and runs cleanly via `streamlit run app.py`.

- [ ] **2. Actual use of Strands Decider 2B for at least one meaningful decision (PENDING LIVE SERVER RUN)**
  - Model Checkpoint: `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`
  - Pattern: `#tool-selection`
  - Wire Format: Typed `ChoiceQuestion` over official `POST /v1/systemone` System One API.
  - Integration: `devflow/decider.py` formats and evaluates candidate options (`explain_code`, `debug_error`, `analyze_test`, `search_docs`).
  - Code Verification: Automated unit tests and mock HTTP server integration tests pass (`tests/test_decider.py`, `tests/test_integration.py`).
  - Remote Runner: Google Colab notebook prepared at `notebooks/decider_colab_server.ipynb` with authenticated reverse proxy and Cloudflare tunnel.
  - **Live Verification Status**: **UNVERIFIED**. The host machine (Windows, Intel Iris Xe, 0 GB free C: drive) cannot host the 4.8 GB weights locally. To satisfy official hackathon criteria, launch the provided Colab runner and run `verify_live_e2e.py`.

- [ ] **3. Amazon Bedrock for the generative part of the agent (PENDING AWS CREDENTIALS)**
  - AWS Service: Amazon Bedrock Runtime
  - API: Modern unified AWS SDK `Converse` API (`boto3.client('bedrock-runtime').converse(...)`)
  - Integration: `devflow/bedrock_client.py`
  - Code Verification: Fully implemented with automated tests and grounded fallback simulation (`tests/test_bedrock.py`).
  - Verification Script: `verify_bedrock.py` checks credentials and measures token cost (~$0.00005 per test request).
  - **Live Verification Status**: **UNVERIFIED**. Active AWS credentials and Bedrock model access (e.g. Claude 3 Haiku) must be configured via `aws configure` to verify live AWS Bedrock inference.

- [x] **4. At least one reusable Decider 2B decision pattern**
  - Pattern Name: `#tool-selection`
  - Directory: `decision_patterns/tool_selection/`
  - Deliverables:
    - `README.md` (detailed architectural recipe, state specification, adaptation instructions)
    - `questions.json` (typed choice question wire schema)
    - `example.py` (standalone runnable demonstration independent of Streamlit)
  - Verified: `python decision_patterns/tool_selection/example.py` runs and verifies all routes.

- [x] **5. Public source-code repository**
  - URL: [https://github.com/KvPradeepthi/devflow-ai](https://github.com/KvPradeepthi/devflow-ai)
  - Required Files:
    - `README.md` (comprehensive documentation, architecture, setup)
    - `requirements.txt` (reproducible dependencies)
    - `.env.example` (clean placeholders, zero secrets)
    - `.gitignore` (ignores `.env`, virtualenvs, caches, test artifacts)
    - `LICENSE` (MIT License)
  - Verified: Pushed to GitHub on branch `main`, credential scan clean, zero secrets committed.

- [x] **6. Automated Tests & Empirical Benchmarks**
  - Test Suite: 30 automated tests in `tests/` covering tools, decider, policy, Bedrock, and integration.
  - Coverage: **88% code coverage** (`pytest --cov=devflow`).
  - Benchmarks: 20 representative cases evaluated across 60 trials in `benchmarks/benchmark_runner.py`.
  - Results Recorded: `benchmarks/benchmark_results.json` and `benchmarks/benchmark_report.md` (90% accuracy, 28.53ms Decider latency, explicitly disclosed as offline simulation baseline).

- [x] **7. Required Hackathon Tags Included**
  - `#strands-decider-2b`
  - `#tool-selection`
  - Included in README, article draft, video script, UI sidebar, and schema specifications.

---

## 2. External Submission Actions (To Be Completed by User)

The project codebase, tests, article draft, and video script are 100% complete and published. The following external publishing actions require your AWS account, Colab runtime, and video platform access:

- [x] **Action 1: Initialize and Push to Public GitHub Repository (COMPLETED)**
  - Repository is live: [https://github.com/KvPradeepthi/devflow-ai](https://github.com/KvPradeepthi/devflow-ai)

- [ ] **Action 2: Run Live Decider & Bedrock Verification (Optional but Recommended)**
  - Open `notebooks/decider_colab_server.ipynb` in Google Colab (with T4 GPU or CPU).
  - Run all cells to obtain the public tunnel URL and Bearer token.
  - In PowerShell:
    ```powershell
    $env:STRANDS_DECIDER_URL = "https://<your-tunnel>.trycloudflare.com"
    $env:STRANDS_DECIDER_API_KEY = "<your-token>"
    E:\devflow_venv\Scripts\python.exe verify_live_e2e.py
    ```
  - In PowerShell, run `aws configure` and verify Bedrock:
    ```powershell
    E:\devflow_venv\Scripts\python.exe verify_bedrock.py
    ```

- [ ] **Action 3: Publish AWS Builder Center Article**
  - Open the completed draft in `BUILDER_CENTER_ARTICLE.md`.
  - Copy and paste into [AWS Builder Center](https://builder.aws.com/).
  - Ensure tags `#strands-decider-2b` and `#tool-selection` are attached to the article.

- [ ] **Action 4: Record and Upload Demo Video (< 3 minutes)**
  - Use the exact scene-by-scene script in `DEMO_SCRIPT.md` (target: 2 min 40 sec).
  - Launch Streamlit: `E:\devflow_venv\Scripts\streamlit.exe run app.py`.
  - Walk through the preset examples, probability distribution, and JSON trace.
  - Upload the video to YouTube (Public or Unlisted) or Vimeo.
  - Insert the video URL into `BUILDER_CENTER_ARTICLE.md` and the hackathon submission form.

- [ ] **Action 5: Submit on Hackathon Portal Before Deadline**
  - Portal: [AWS Agent Experience Hackathon Portal](https://builder.aws.com/build/hackathons/4f50957c-1d08-37ab-b11c-97516b9adeb4/agent-experience-hackathon)
  - Deadline: October 23, 2026, 11:59 PM PDT.
  - Project Title: `DevFlow AI`
  - Provide Public GitHub Repo URL, Builder Center Article URL, and Demo Video URL.
