from armada.tools import calculator, execute_tool, kv_lookup


def test_calculator_basic():
    assert calculator("(1234 * 56) + 789") == "69893"
    assert calculator("2 ** 10") == "1024"
    assert calculator("200 * 15 / 100") == "30"


def test_calculator_division_by_zero():
    assert "error" in calculator("1 / 0")


def test_calculator_rejects_names_and_calls():
    # Security: tool arguments are untrusted LLM output; no code execution allowed.
    assert "error" in calculator("__import__('os').system('echo hi')")
    assert "error" in calculator("abs(-1)")
    assert "error" in calculator("x + 1")


def test_calculator_rejects_huge_exponent():
    assert "error" in calculator("2 ** 100000")


def test_kv_lookup():
    assert kv_lookup("capital_of_france") == "Paris"
    assert kv_lookup("Population_Of_Paris") == "2140000"
    assert "error" in kv_lookup("unknown_key")


def test_execute_tool_dispatch():
    assert execute_tool("calculator", {"expression": "1+1"}) == "2"
    assert "error" in execute_tool("does_not_exist", {})
    assert "error" in execute_tool("calculator", {"wrong": "arg"})
