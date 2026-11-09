#!/bin/sh
# Re-identification attack with Claude Code (CLI `claude -p`) as the attacker.
# Usage: sh scripts/attack/claude_code.sh [level] [model]   (model: haiku | sonnet | opus | full model id)
level=${1:-1}
model=${2:-opus}
anon_methods="pre_anon, claude_code_basic, claude_code_full, claude_code_public, claude_code_rescriber, claude_code_clio"

python3 main.py --data_path="benchmark/" \
                --results_path="benchmark/" \
                --level=$level \
                --attacker="claude_code" \
                --model_version="$model" \
                --anon_methods="$anon_methods" \
                --scenario="medical" \
                --uniqueness_results_folder="./results" \
                --anonymize="False" \
                --attack="True" \
                --only_correctness="False"
