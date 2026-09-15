"""Baholash quruli (`evals/`) — docs/architecture-v2.md, 10-bo'lim.

Real chat kontenti bu paketda **saqlanmaydi**: `golden.jsonl` va `runs/`
`.gitignore`'da. Paket faqat kod — DB/LLM'ga faqat ishga tushirilganda
ulanadi, `pytest` yugurishida hech qanday tashqi chaqiruv bo'lmaydi
(`tests/test_evals.py` hammasini fake qiladi).
"""

from __future__ import annotations
