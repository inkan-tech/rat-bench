from pii_benchmark.attackers.claude import ClaudeAttacker
from pii_benchmark.attackers.claude_code import ClaudeCodeAttacker
from pii_benchmark.attackers.deepseek import DeepSeekAttacker
from pii_benchmark.attackers.gemini import GeminiAttacker
from pii_benchmark.attackers.gpt import GPTAttacker
from pii_benchmark.attackers.llama import LlamaAttacker


def get_attacker(attacker, model_version):
    # Forward model_version only when the caller actually gave one, so a
    # missing --model_version falls back to each attacker's own default
    # instead of overriding it with an explicit None (which downstream
    # SDKs reject outright, e.g. Anthropic's "model: Input should be a
    # valid string").
    kwargs = {} if model_version is None else {"model_version": model_version}
    match attacker:
        case "deepseek":
            print("DeepSeek attacker")
            return DeepSeekAttacker(**kwargs)
        case "gemini":
            print("Gemini attacker")
            return GeminiAttacker(model_version)
        case "llama":
            print("Llama attacker")
            return LlamaAttacker(**kwargs)
        case "gpt":
            print("GPT attacker")
            return GPTAttacker(**kwargs)
        case "claude":
            print("Claude attacker")
            return ClaudeAttacker(**kwargs)
        case "claude_code":
            print("Claude Code attacker")
            return ClaudeCodeAttacker(**kwargs)
        case _:
            raise ValueError(f"Unknown attacker: {attacker!r}. Valid options: deepseek, gemini, llama, gpt, claude, claude_code")