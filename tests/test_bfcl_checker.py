"""Unit tests for the vendored BFCL AST checker (bfcl_checker.py)."""
from legit_edge.bfcl_checker import simple_function_checker, Language


# Minimal function description matching BFCL schema
_FUNC_DESC = {
    "name": "add",
    "parameters": {
        "type": "dict",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    },
}

# possible_answer wraps each param's accepted values in a list
_POSSIBLE_ANSWER = {"add": {"a": [2], "b": [3]}}


def test_bfcl_checker_accepts_correct_call():
    """Correct function name and matching args should return valid=True."""
    model_output = {"add": {"a": 2, "b": 3}}
    result = simple_function_checker(
        _FUNC_DESC, model_output, _POSSIBLE_ANSWER, Language.PYTHON, "legit-edge"
    )
    assert result["valid"] is True, f"Expected valid, got errors: {result['error']}"


def test_bfcl_checker_rejects_wrong_arg_value():
    """Correct function name but wrong arg value should return valid=False."""
    model_output = {"add": {"a": 99, "b": 3}}
    result = simple_function_checker(
        _FUNC_DESC, model_output, _POSSIBLE_ANSWER, Language.PYTHON, "legit-edge"
    )
    assert result["valid"] is False


def test_bfcl_checker_rejects_wrong_function_name():
    """Wrong function name should return valid=False with wrong_func_name error type."""
    model_output = {"subtract": {"a": 2, "b": 3}}
    result = simple_function_checker(
        _FUNC_DESC, model_output, _POSSIBLE_ANSWER, Language.PYTHON, "legit-edge"
    )
    assert result["valid"] is False
    assert "wrong_func_name" in result["error_type"]


def test_bfcl_checker_rejects_missing_required_param():
    """Missing required param should return valid=False."""
    model_output = {"add": {"a": 2}}  # missing b
    result = simple_function_checker(
        _FUNC_DESC, model_output, _POSSIBLE_ANSWER, Language.PYTHON, "legit-edge"
    )
    assert result["valid"] is False
