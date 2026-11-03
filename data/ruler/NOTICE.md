# Fixed RULER main-table evaluation data

This directory contains the exact fixed prompts and gold answers used for the
DAMI RULER-style main table: NIAH-1 and NIAH-2 at 1K/2K/4K/8K and MK-NIAH at
1K/2K/4K, with 500 examples per setting. It excludes NIAH-3 and all non-RULER tasks.

The upstream benchmark is NVIDIA RULER:
https://github.com/NVIDIA/RULER

Upstream RULER code is distributed under Apache-2.0:
https://github.com/NVIDIA/RULER/blob/main/LICENSE

The scoring convention follows the lm-evaluation-harness RULER utilities:
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/ruler/common_utils.py

The repository's MIT license for original DAMI code does not replace upstream
terms or attribution for benchmark material and background text.

The distributed JSONL files retain doc_id, task, length, prompt, and answer.
Original prompt strings and answer lists are unchanged. Obsolete score and
generation-metadata fields were omitted from input examples. The manifest gives
checksums for both the original sample files and these cleaned input files.
