# Python Testing Best Practices with Pytest

## Structuring Test Suites
Maintain a strict separation between fast unit tests (isolated, mocked dependencies) and integration tests (validating inter-service communication):
- Unit tests run in under a few seconds and run reliably in CI environments without external hardware or cloud credentials.
- Integration tests should be clearly marked with `@pytest.mark.integration` or conditional skip decorators when hardware or credentials are not present.

## Mocking External Cloud APIs
When testing AWS Bedrock clients, use standard `unittest.mock` or `botocore.stub.Stubber` to mock responses:
```python
from unittest.mock import MagicMock, patch

@patch("boto3.client")
def test_bedrock_converse(mock_boto):
    mock_runtime = MagicMock()
    mock_boto.return_value = mock_runtime
    mock_runtime.converse.return_value = {
        "output": {"message": {"content": [{"text": "Mocked explanation."}]}},
        "usage": {"inputTokens": 50, "outputTokens": 20}
    }
```

## Assertion Standards
- Assert explicit expected values rather than generic truthiness.
- Use pytest's built-in diff reporting for dictionary and sequence comparisons.
