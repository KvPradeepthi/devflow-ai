"""Standalone AWS Bedrock Converse API Verification Script.

Checks AWS credentials, tests the Bedrock Converse API with Claude 3 Haiku,
and verifies model access without exposing sensitive keys or tokens.

Cost Estimate:
  Claude 3 Haiku costs $0.25 per 1,000,000 input tokens and $1.25 per 1,000,000 output tokens.
  A single test request (~50 input tokens, ~30 output tokens) costs ~$0.00005 (less than 1/100th of 1 cent).
"""

from __future__ import annotations

import json
import os
import sys
import time

# Ensure utf-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def verify_bedrock():
    print("=" * 70)
    print("DevFlow AI: Amazon Bedrock Converse API Verification")
    print("=" * 70)

    region = os.environ.get("AWS_REGION", "us-east-1")
    model_id = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

    print(f"Target Region : {region}")
    print(f"Target Model  : {model_id}")
    print("\n[Step 1/2] Verifying AWS Credentials in Environment / Profile...")

    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        print("  [-] Error: boto3 is not installed. Install via: pip install boto3")
        return False

    session = boto3.Session(region_name=region)
    sts = session.client("sts")

    try:
        caller = sts.get_caller_identity()
        account = caller.get("Account", "unknown")
        arn = caller.get("Arn", "unknown")
        # Mask account number for security: show only last 4 digits
        masked_account = f"***{account[-4:]}" if len(account) >= 4 else "***"
        print(f"  [+] AWS Credentials Authenticated Successfully!")
        print(f"      Account : {masked_account}")
        print(f"      IAM ARN : {arn}")
    except NoCredentialsError:
        print("  [-] NOT AUTHENTICATED: No AWS credentials found.")
        print("\n  [!] How to Configure AWS Credentials:")
        print("      Run in your PowerShell terminal:")
        print("          aws configure")
        print("      Enter your AWS Access Key ID, Secret Access Key, and default region (us-east-1).")
        print("      (Never paste access keys into chat or commit them to git repository).")
        return False
    except ClientError as e:
        print(f"  [-] STS Error: {e}")
        return False

    print("\n[Step 2/2] Testing Amazon Bedrock Converse API Call...")
    print("  Estimated Cost: ~$0.00005 (fraction of a single cent)")
    
    try:
        client = session.client("bedrock-runtime")
        prompt = "Reply in exactly one concise sentence: DevFlow AI Bedrock live test successful."
        
        start_t = time.perf_counter()
        response = client.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig={
                "maxTokens": 60,
                "temperature": 0.2,
            },
        )
        elapsed_ms = (time.perf_counter() - start_t) * 1000

        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])
        text_content = ""
        for block in content_blocks:
            if "text" in block:
                text_content += block["text"]

        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)

        # Calculate exact cost for Claude 3 Haiku
        cost = (input_tokens * 0.00000025) + (output_tokens * 0.00000125)

        print(f"  [+] BEDROCK LIVE CONVERSE CALL SUCCESSFUL!")
        print(f"      Response Latency : {elapsed_ms:.2f} ms")
        print(f"      Tokens Used      : {input_tokens} input, {output_tokens} output")
        print(f"      Actual Request Cost : ${cost:.6f} USD")
        print(f"\nModel Output:\n  \"{text_content.strip()}\"")
        print("=" * 70)
        return True

    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "Unknown")
        msg = e.response.get("Error", {}).get("Message", str(e))
        print(f"  [-] Bedrock ClientError [{code}]: {msg}")
        if "AccessDeniedException" in code or "AccessDenied" in code:
            print("\n  [!] Access Denied Action Plan:")
            print(f"      1. Open AWS Console -> Bedrock -> Model access (in {region}).")
            print(f"      2. Request/enable access for '{model_id}'.")
            print("      3. Ensure IAM user/role has permission 'bedrock:InvokeModel'.")
        return False
    except Exception as e:
        print(f"  [-] Unexpected Error: {e}")
        return False


if __name__ == "__main__":
    success = verify_bedrock()
    sys.exit(0 if success else 1)
