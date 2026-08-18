import ast
from pathlib import Path


def _symbols(path: str):
    tree = ast.parse(Path(path).read_text())
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }


def _without_docstrings(node):
    node = ast.fix_missing_locations(ast.parse(ast.unparse(node)).body[0])
    for child in ast.walk(node):
        body = getattr(child, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
    return ast.dump(node, include_attributes=False)


def test_model_split_matches_main_monolith():
    main = _symbols("examples/alphazero.py")
    split = _symbols("src/nanoalphazero/model.py")
    for name, node in split.items():
        if name in main:
            assert _without_docstrings(node) == _without_docstrings(main[name]), name


def test_builtin_configs_match_main_monolith():
    main = _symbols("examples/alphazero.py")
    split = _symbols("src/nanoalphazero/config.py")
    for name in (
        "get_ttt_config",
        "get_connect4_config",
        "get_hex_config",
        "get_chess_config",
        "get_go_config",
    ):
        assert _without_docstrings(split[name]) == _without_docstrings(main[name]), name


def test_training_host_functions_match_main_monolith():
    main = _symbols("examples/alphazero.py")
    split = _symbols("src/nanoalphazero/training.py")
    for name, node in split.items():
        if name in main:
            assert _without_docstrings(node) == _without_docstrings(main[name]), name
