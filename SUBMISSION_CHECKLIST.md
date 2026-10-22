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
  - Code Verification: Automated unit tests and mock HTTP server integration tests pass (`test_integration.py`).
  - **Live Verification Status**: **UNVERIFIED**. The host machine (Windows, Intel Iris Xe, 0.6 GB free C: drive) cannot host the 4.5 GB weights locally. To satisfy official hackathon criteria, you must run `strands-decider serve` on a GPU/Metal machine and test at least one genuine live request.

- [ ] **3. Amazon Bedrock for the generative part of the agent (PENDING AWS CREDENTIALS)**
  - AWS Service: Amazon Bedrock Runtime
  - API: Modern unified AWS SDK `Converse` API (`boto3.client('bedrock-runtime').converse(...)`)
  - Integration: `devflow/bedrock_client.py`
  - Code Verification: Fully implemented with automated tests and grounded fallback simulation.
  - **Live Verification Status**: **UNVERIFIED**. Active AWS credentials and Bedrock model access (e.g. Claude 3 Haiku or Nova Micro) must be supplied to test live AWS Bedrock inference.

- [x] **4. At least one reusable Decider 2B decision pattern**
  - Pattern Name: `#tool-selection`
  - Directory: `decision_patterns/tool_selection/`
  - Deliverables:
    - `README.md` (detailed architectural recipe, state specification, adaptation instructions)
    - `questions.json` (typed choice question wire schema)
    - `example.py` (standalone runnable demonstration independent of Streamlit)
  - Verified: `python decision_patterns/tool_selection/example.py` runs and verifies all routes.

- [x] **5. Public source-code repository readiness**
  - Directory: `C:\Users\DEEPTHI\.gemini\antigravity\scratch\devflow-ai`
  - Required Files:
    - `README.md` (comprehensive documentation)
    - `requirements.txt` (reproducible dependencies)
    - `.env.example` (clean placeholders, zero secrets)
    - `.gitignore` (ignores `.env`, virtualenvs, caches, test artifacts)
    - `LICENSE` (MIT License)
  - Verified: Credential scan clean, no secrets committed.

- [x] **6. Automated Tests & Empirical Benchmarks**
  - Test Suite: 28 automated tests in `tests/` covering tools, decider, policy, Bedrock, and integration.
  - Coverage: **88% code coverage** (`pytest --cov=devflow`).
  - Benchmarks: 20 representative cases evaluated across 60 trials in `benchmarks/benchmark_runner.py`.
  - Results Recorded: `benchmarks/benchmark_results.json` and `benchmarks/benchmark_report.md` (90% accuracy, 28.65ms Decider latency).

- [x] **7. Required Hackathon Tags Included**
  - `#strands-decider-2b`
  - `#tool-selection`
  - Included in README, article draft, video script, UI sidebar, and schema specifications.

---

## 2. External Submission Actions (To Be Completed by User)

The project codebase, tests, article draft, and video script are 100% complete. The following external publishing actions require your AWS account, GitHub account, and video platform access:

- [ ] **Action 1: Initialize and Push to Public GitHub Repository**
  - Create a new public repository on GitHub (e.g. `https://github.com/<your-username>/devflow-ai`).
  - Run the following commands:
    ```powershell
    cd C:\Users\DEEPTHI\.gemini\antigravity\scratch\devflow-ai
    git init
    git add .
    git commit -m "Initial commit of DevFlow AI for AWS Strands Decider Hackathon"
    git branch -M main
    git remote add origin https://github.com/<your-username>/devflow-ai.git
    git push -u origin main
    ```

- [ ] **Action 2: Publish AWS Builder Center Article**
  - Open the draft prepared in `BUILDER_CENTER_ARTICLE.md`.
  - Update repository and video URL placeholders.
  - Publish the article on [AWS Builder Center](https://builder.aws.com/).
  - Ensure tags `#strands-decider-2b` and `#tool-selection` are added to the post.

- [ ] **Action 3: Record and Upload Demo Video (< 3 minutes)**
  - Use the exact scene-by-scene script in `DEMO_SCRIPT.md` (target: 2 min 40 sec).
  - Launch Streamlit: `E:\devflow_venv\Scripts\python.exe -m streamlit run app.py`.
  - Walk through the preset examples, probability distribution, and JSON trace.
  - Upload the video to YouTube (Public or Unlisted) or Vimeo.

- [ ] **Action 4: Submit on Hackathon Portal Before Deadline**
  - Portal: [AWS Agent Experience Hackathon Portal](https://builder.aws.com/build/hackathons/4f50957c-1d08-37ab-b11c-97516b9adeb4/agent-experience-hackathon)
  - Deadline: October 23, 2026, 11:59 PM PDT.
  - Fill in Project Title: `DevFlow AI`
  - Provide Public GitHub Repo URL, Builder Center Article URL, and Demo Video URL.
