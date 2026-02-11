#!/usr/bin/env python3
"""
Unit tests for the worker module.
Can be run with: pytest tests/test_worker.py
"""
import pytest
import json
import asyncio
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.worker import get_sys_msg, get_usr_msg, load_config


def test_get_sys_msg():
    """Test that system message is generated correctly."""
    msg = get_sys_msg()
    assert isinstance(msg, str)
    assert len(msg) > 100
    assert "Steuerexperte" in msg
    assert "Konversation" in msg


def test_get_usr_msg():
    """Test that user message is generated correctly."""
    msg = get_usr_msg(
        text="Sample text",
        themengebiet="Steuerrecht",
        topic_filter="Einkommensteuer",
        titel="Test Title",
        autor="Test Author",
        volltext="This is the full text for testing."
    )
    
    assert isinstance(msg, str)
    assert "Steuerrecht" in msg
    assert "Einkommensteuer" in msg
    assert "Test Title" in msg
    assert "Test Author" in msg
    assert "This is the full text for testing." in msg


def test_get_usr_msg_with_empty_fields():
    """Test user message with empty optional fields."""
    msg = get_usr_msg(
        text="",
        themengebiet="",
        topic_filter="",
        titel="",
        autor="",
        volltext="Required volltext content"
    )
    
    assert isinstance(msg, str)
    assert "Required volltext content" in msg


def test_load_config():
    """Test configuration loading."""
    config_path = Path(__file__).parent / "config_test.yaml"
    config = load_config(str(config_path))
    
    assert isinstance(config, dict)
    assert "api" in config
    assert "worker" in config
    assert "data" in config
    assert config["api"]["base_url"] == "http://localhost:6000"
    assert config["worker"]["max_concurrent"] == 3


def test_test_data_format():
    """Test that test data has correct format."""
    test_data_path = Path(__file__).parent / "test_data_sample.json"
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert isinstance(data, list)
    assert len(data) > 0
    
    for record in data:
        assert "_recordid" in record
        assert "Volltext" in record
        assert isinstance(record["_recordid"], str)
        assert isinstance(record["Volltext"], str)
        assert len(record["Volltext"]) > 0


def test_conversation_json_parsing():
    """Test parsing of conversation JSON formats."""
    test_cases = [
        # Standard format
        '[{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]',
        # With markdown code block
        '```json\n[{"role": "user", "content": "Question"}]\n```',
    ]
    
    for test_json in test_cases:
        if "```json" in test_json:
            extracted = test_json.split("```json")[1].split("```")[0].strip()
            parsed = json.loads(extracted)
        else:
            parsed = json.loads(test_json)
        
        assert isinstance(parsed, list)
        if len(parsed) > 0:
            assert "role" in parsed[0]
            assert "content" in parsed[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
