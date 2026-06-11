"""Shared fixtures and env setup for backend tests.

Set dummy env vars *before* importing modules that read them at import time
(rnn_inference.py raises ValueError if GEMINI_API_KEY is missing).
"""
import os

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
os.environ.setdefault("GEMINI_MODEL", "gemini-1.5-flash")
os.environ.setdefault("GEMINI_TEMPERATURE", "0.7")
os.environ.setdefault("GEMINI_MAX_OUTPUT_TOKENS", "256")
