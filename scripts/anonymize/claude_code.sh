#!/bin/sh
# Anonymize with Claude Code (CLI `claude -p`), no API key needed: uses the logged-in Claude Code account.
# Usage: sh scripts/anonymize/claude_code.sh [level] [model]   (model: haiku | sonnet | opus | full model id)
level=${1:-1}
model=${2:-haiku}
anon_methods="claude_code_basic, claude_code_full, claude_code_public, claude_code_rescriber, claude_code_clio"
timing_flag=1

python3 -m pii_benchmark.anonymize --results_path="benchmark/level_${level}.jsonl" \
                --data_path="benchmark/level_${level}.jsonl" \
                --anon_methods="$anon_methods" \
                --claude_code_version="$model" \
                --timing=$timing_flag \
                --scenario="random" \
                --level=$level
