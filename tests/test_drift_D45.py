"""D45: the English attacker JSON footer writes attribute keys without
quotes, as the paper-era (b8cface) code did (`sex :{ `); non-English runs
keep the quoted keys."""
from pii_benchmark.prompts import get_staab_prompt


def test_english_footer_keys_unquoted():
    prompt = get_staab_prompt(["SEX", "RAC2P"], "TEXT", scenario="Medical consultation", language="English")
    assert "\nsex :{ \nInference: Your detailed inference,\n" in prompt
    assert "\n race :{ \n" not in prompt
    assert "race :{ \nInference:" in prompt
    assert '"sex" :{' not in prompt
    assert '"race" :{' not in prompt


def test_default_language_footer_keys_unquoted():
    prompt = get_staab_prompt(["SEX"], "TEXT", scenario="Chatbot conversation")
    assert "sex :{ \n" in prompt
    assert '"sex" :{' not in prompt


def test_non_english_footer_keeps_quoted_keys():
    prompt = get_staab_prompt(["SEX"], "TEXT", scenario="Medical consultation", language="Spanish")
    assert '" :{\n' in prompt
