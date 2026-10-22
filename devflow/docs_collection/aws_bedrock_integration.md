# Amazon Bedrock Integration & Best Practices

## Overview
Amazon Bedrock is a fully managed service that offers choice of high-performing foundation models (FMs) from leading AI companies like Anthropic, Amazon, Meta, and Cohere along with a broad set of capabilities to build generative AI applications with security, privacy, and responsible AI.

## Unified Converse API
The modern standard for invoking Bedrock models using the AWS SDK (`boto3`) is the `converse` API:
```python
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")
response = client.converse(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    messages=[
        {"role": "user", "content": [{"text": "Explain this code snippet."}]}
    ],
    inferenceConfig={"maxTokens": 1024, "temperature": 0.2}
)
```

## Security & Authentication
- Always authenticate via IAM roles, AWS profiles (`~/.aws/credentials`), or temporary STS credentials.
- Never commit AWS access keys or secret keys to source control.
- Enforce least-privilege policies restricting Bedrock actions to `bedrock:InvokeModel` on necessary resource ARNs.

## Cost Management
- Use fast, low-cost models like Claude 3 Haiku or Amazon Nova Micro for task-specific explanations.
- Ground prompts with pre-filtered tool outputs to minimize prompt token counts.
