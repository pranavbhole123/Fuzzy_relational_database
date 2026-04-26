import pytest

print("TEST MODULE: tests/test_cli_parser.py loaded")

from cli.parser import parse, ParseError


def test_parse_load_command():
    cmd = parse("LOAD data/employees.csv AS employees")
    assert cmd["cmd"] == "load"
    assert cmd["path"] == "data/employees.csv"
    assert cmd["table"] == "employees"


def test_parse_show_commands():
    assert parse("SHOW TABLES")["cmd"] == "show_tables"
    cmd = parse("SHOW TERMS employees")
    assert cmd["cmd"] == "show_terms"
    assert cmd["table"] == "employees"


def test_parse_describe_command():
    cmd = parse("DESCRIBE employees")
    assert cmd["cmd"] == "describe"
    assert cmd["table"] == "employees"


def test_parse_define_term_commands():
    cmd = parse("DEFINE TERM employees.age AS young USING triangular(15, 25, 35)")
    assert cmd["cmd"] == "define"
    assert cmd["table"] == "employees"
    assert cmd["col"] == "age"
    assert cmd["term"] == "young"
    assert cmd["mf_type"] == "triangular"
    assert cmd["params"]["a"] == 15.0
    assert cmd["params"]["b"] == 25.0
    assert cmd["params"]["c"] == 35.0


def test_parse_select_and_hedges():
    cmd = parse("SELECT * FROM employees WHERE age IS very young THRESHOLD 0.3")
    assert cmd["cmd"] == "select"
    assert cmd["table"] == "employees"
    assert cmd["threshold"] == 0.3
    assert cmd["conditions"][0]["hedge"] == "very"
    assert cmd["conditions"][0]["term"] == "young"


def test_parse_invalid_commands_raise():
    assert parse("")["cmd"] == "noop"
    with pytest.raises(ParseError):
        parse("!!! not a command")
    with pytest.raises(ParseError):
        parse("DEFINE TERM employees.age AS young USING triangular(15, 25)")
