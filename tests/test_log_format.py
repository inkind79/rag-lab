"""Verify LOG_FORMAT env var switches between text and structured JSON output."""

import importlib
import json
import logging
import sys

import pytest


def _reload_logger():
    for m in list(sys.modules):
        if m.startswith('src.utils.logger'):
            del sys.modules[m]
    return importlib.import_module('src.utils.logger')


@pytest.fixture
def fresh_record() -> logging.LogRecord:
    return logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='example.py',
        lineno=42,
        msg='hello world',
        args=None,
        exc_info=None,
    )


def test_default_format_is_text(monkeypatch, fresh_record):
    monkeypatch.delenv('LOG_FORMAT', raising=False)
    log_mod = _reload_logger()
    out = log_mod._build_formatter('console').format(fresh_record)
    assert 'hello world' in out
    assert out.startswith('[')


def test_json_format_emits_parseable_json(monkeypatch, fresh_record):
    monkeypatch.setenv('LOG_FORMAT', 'json')
    log_mod = _reload_logger()
    out = log_mod._build_formatter('console').format(fresh_record)
    parsed = json.loads(out)
    assert parsed['message'] == 'hello world'
    assert parsed['logger'] == 'test_logger'
    assert parsed['filename'] == 'example.py'
    assert parsed['lineno'] == 42
    assert parsed['level'] == 'INFO'
    assert 'timestamp' in parsed


def test_json_format_is_case_insensitive(monkeypatch, fresh_record):
    monkeypatch.setenv('LOG_FORMAT', 'JSON')
    log_mod = _reload_logger()
    parsed = json.loads(log_mod._build_formatter('console').format(fresh_record))
    assert parsed['message'] == 'hello world'


def test_unknown_format_falls_back_to_text(monkeypatch, fresh_record):
    monkeypatch.setenv('LOG_FORMAT', 'pretty-please')
    log_mod = _reload_logger()
    out = log_mod._build_formatter('console').format(fresh_record)
    # Falls back to bracketed text format.
    assert out.startswith('[')


def test_json_file_and_console_share_one_format(monkeypatch, fresh_record):
    """In JSON mode, file + console emit identical records (one schema for collectors)."""
    monkeypatch.setenv('LOG_FORMAT', 'json')
    log_mod = _reload_logger()
    console = json.loads(log_mod._build_formatter('console').format(fresh_record))
    file_ = json.loads(log_mod._build_formatter('file').format(fresh_record))
    assert console == file_
