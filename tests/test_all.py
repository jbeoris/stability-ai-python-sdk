import pytest
import stability_ai

def test_client():
    client = stability_ai.default_client
    assert client is not None

def test_list_engines():
    result = stability_ai.v1.engines.list()
    print(result)
    assert result is not None
    assert len(result.engines) >= 1

def test_fetch_user_account():
    result = stability_ai.v1.user.account()
    print(result)
    assert result is not None
    assert isinstance(result.id, str)

def test_fetch_user_balance():
    result = stability_ai.v1.user.balance()
    print(result)
    assert result is not None
    assert isinstance(result.credits, float)