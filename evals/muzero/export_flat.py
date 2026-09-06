"""Generate the readable standalone snapshot: uv run evals/muzero/export_flat.py.

This is a source export, not a module loader. Original comments and formatting
are retained. Only imports between included sections and a few filesystem/CLI
assumptions change. The package remains authoritative.
"""

import ast
import json
from pathlib import Path
import pprint
import re
import tomllib


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src/nanoalphazero"
SECTIONS = [
    ("config", None, "Environment and AlphaZero-compatible training defaults"),
    ("buffers", None, "Shared buffer primitives and packed chess legality"),
    ("model", None, "Shared KataGo network blocks and AlphaZero reference model"),
    ("mcts", None, "Fixed two-rung Gumbel search (unchanged production algorithm)"),
    ("checkpoint", None, "AlphaZero reference checkpoint format and path encoding"),
    ("core", {"WrappedEnv", "make_env"}, "Real environment adapter: actual transitions only"),
    ("eval.hex.perfect_play", None, "Opening reference tables: diagnostics only"),
    ("eval.hex.engine", None, "Optional external MoHex engine management"),
    ("eval.hex.runtime", None, "Real-game MoHex evaluation"),
    ("eval.hex.training", None, "Periodic MoHex evaluation and raw result logging"),
    ("training", {"_legal_mask_from_state", "all_opening_actions",
                  "_run_ttt_diagnostics", "_get_hex_perfect_play_values", "_run_hex_diagnostics",
                  "_get_connect4_perfect_play_values", "_run_connect4_diagnostics",
                  "_run_go_diagnostics"}, "Shared opening enumeration and value diagnostics"),
    ("research.muzero.model", None, "Vector MuZero: representation h, dynamics g, prediction f"),
    ("research.muzero.remat", None, "Optional activation recomputation to reduce training memory"),
    ("research.muzero.spatial", None, "Spatial MuZero: board-shaped latent and action planes"),
    ("research.muzero.search", None, "Latent search: root legality, learned hypothetical transitions"),
    ("research.muzero.replay", None, "Contiguous replay, signed returns, absorbing and padded targets"),
    ("research.muzero.staging", None, "Self-play staging -> consume/drain -> sequence replay"),
    ("research.muzero.learning", None, "Actual self-play and differentiable recurrent training"),
    ("research.muzero.checkpoint", None, "MuZero checkpoints: parameters, optimizer, replay, RNG and metadata"),
    ("research.muzero.charts", None, "Optional W&B charts"),
    ("research.muzero.metrics", None, "Training and replay metrics"),
    ("research.muzero.inspection", None, "Model diagnostics using real observations"),
    ("research.muzero.decisions", None, "Optional solver evaluation of actual decisions"),
    ("research.muzero.evaluation", None, "Held-out predictions and policy/search playing strength"),
    ("research.muzero.convergence", None, "Optional stopping gate for repeated MoHex opening coverage"),
    ("research.muzero.cli", None, "Four-device training loop and command line"),
]

SECTION_NOTES = {
    "research.muzero.model": (
        "h maps each root observation to [batch, width]. g concatenates the latent",
        "with a one-hot action and predicts an immediate reward and next latent.",
        "f predicts policy logits and side-to-move value. The three towers have",
        "separate parameters; their latent interface is trained end to end.",
    ),
    "research.muzero.spatial": (
        "The latent is [batch, board_height, board_width, channels]. Its cells",
        "are learned features, not reconstructed stones or simulator states.",
        "Action planes encode action coordinates/type, never future legality.",
    ),
    "research.muzero.search": (
        "Encode the root once. Both search rungs expand through g and f only.",
        "The simulator supplies root legality. Internal nodes use the full action",
        "vocabulary. Alternating-player backups are reward - discount * value.",
    ),
    "research.muzero.replay": (
        "At unroll index k: policy/value describe s_k; reward describes action",
        "a_k taking s_k to s_(k+1). Terminal absorbing targets differ from padding",
        "beyond a truncated boundary; masks keep those distinctions in the loss.",
    ),
    "research.muzero.learning": (
        "The collector may step the real environment; hypothetical search may not.",
        "Rewards use the acting player's perspective; signed discounts propagate",
        "returns between players. Recurrent gradients are scaled by 0.5, while",
        "forward latent values are unchanged. Unroll length is independent of",
        "network depth and the fixed two-rung search expansion budget.",
    ),
}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old!r}; exporter needs updating")
    return source.replace(old, new, 1)


def section_source(module, selected):
    path = SRC / (module.replace(".", "/") + ".py")
    source = path.read_text()
    if selected is not None:
        tree = ast.parse(source)
        # Shared helpers are extracted without pulling in the AlphaZero loop.
        nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                 or getattr(node, "name", None) in selected]
        lines = source.splitlines(keepends=True)
        source = "\n\n".join("".join(lines[
            min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1:
            node.end_lineno]) for node in nodes)
    lines = source.splitlines(keepends=True)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (
            node.module == "__future__" or (node.module or "").startswith("nanoalphazero")
        ):
            for index in range(node.lineno - 1, node.end_lineno):
                lines[index] = ""
    tree = ast.parse(source)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        for index in range(tree.body[0].lineno - 1, tree.body[0].end_lineno):
            lines[index] = ""
    source = "".join(lines)
    source = re.sub(r"^# ={5,}.*\n", "", source, flags=re.MULTILINE)
    if module == "eval.hex.engine":
        source = replace_once(source, 'DEFAULT_CONFIG = Path(__file__).with_name("mohex.cfg")',
                              '# The default engine configuration is embedded in this script.')
        source = replace_once(source, "return DEFAULT_CONFIG.resolve()", "return standalone_mohex_config()")
    if module == "eval.hex.runtime":
        source = source.replace("core.DATA_PARALLEL_SHARDING", "standalone_data_sharding()")
    if module == "research.muzero.evaluation":
        source = replace_once(source, "def main():", "def evaluation_main():")
    if module == "research.muzero.cli":
        source = replace_once(source, "def main():", "def training_main():")
        start = source.index('    parser = argparse.ArgumentParser', source.index('def training_main():'))
        end = source.index('    config = resolve(raw)', start)
        source = source[:start] + (
            '    args = standalone_train_parser().parse_args()\n'
            '    raw = standalone_train_settings(args)\n'
        ) + source[end:]
        source = replace_once(source,
            '    config = resolve(raw)\n    engine_pool = None',
            '    config = resolve(raw)\n'
            '    if args.print_config:\n'
            '        print(json.dumps(config, indent=2))\n'
            '        return\n'
            '    if args.output is None:\n'
            '        args.output = standalone_output(config)\n'
            '    print(f"Run output: {args.output}", flush=True)\n'
            '    engine_pool = None')
        source = source.replace("checkpoint.metadata(", "metadata(").replace("checkpoint.load(", "load(").replace("checkpoint.save(", "save(")
        source = replace_once(source,
            '"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),\n'
            '                "git_status": subprocess.check_output(["git", "status", "--short"], text=True)',
            '"standalone": True')
        start = source.index('    source_root = Path(__file__).parent')
        end = source.index('    model = build_model(config)', start)
        source = source[:start] + '    standalone_snapshot(args.output, manifest)\n' + source[end:]
    return source


def main():
    sections = []
    definitions = {}
    imports = {}
    for module, selected, title in SECTIONS:
        source = section_source(module, selected)
        lines = source.splitlines(keepends=True)
        for node in ast.parse(source).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    single = (ast.Import(names=[alias]) if isinstance(node, ast.Import)
                              else ast.ImportFrom(module=node.module, names=[alias], level=0))
                    imports[ast.unparse(single)] = None
                for index in range(node.lineno - 1, node.end_lineno):
                    lines[index] = ""
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name in definitions:
                    raise ValueError(f"Global collision: {node.name}: {definitions[node.name]} / {module}")
                definitions[node.name] = module
        source = "".join(lines)
        notes = "".join(f"# {line}\n" for line in SECTION_NOTES.get(module, ()))
        sections.append(f"\n\n# {title}\n{notes}\n{source.strip()}\n")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    metadata = ['# /// script', '# requires-python = ">=3.11"', '# dependencies = [']
    metadata.extend(f"#     {json.dumps(dep)}," for dep in config["project"]["dependencies"])
    metadata.extend(['# ]', '# [tool.uv.sources]'])
    for name, value in config["tool"]["uv"]["sources"].items():
        fields = ", ".join(f"{key} = {json.dumps(item)}" for key, item in value.items())
        metadata.append(f"# {name} = {{ {fields} }}")
    metadata.append('# ///')
    presets = {name: tomllib.loads((ROOT / f"evals/muzero/{name}.toml").read_text())
               for name in ("smoke-cpu", "smoke-tpu", "hex4-staged-smoke")}
    # Exercise the real staging/drain path in the built-in CPU smoke as well.
    presets["smoke-cpu"].update(data_pipeline="staged", staging_batches=2,
                               consume_size=8, replay_positions=128, replay_warmup_cycles=1)
    prelude = (ROOT / "evals/muzero/standalone_prelude.txt").read_text()
    prelude = prelude.replace("__EMBEDDED_PRESETS__", pprint.pformat(presets, width=100, sort_dicts=False))
    prelude = prelude.replace("__EMBEDDED_MOHEX_CONFIG__", repr((SRC / "eval/hex/mohex.cfg").read_text()))
    prelude_imports = {ast.unparse(node) for node in ast.parse(prelude).body
                       if isinstance(node, (ast.Import, ast.ImportFrom))}
    imports = [line for line in imports if line not in prelude_imports]
    license_notice = "\n".join(("# " + line).rstrip() for line in (ROOT / "LICENSE").read_text().splitlines())
    output = ("\n".join(metadata) + "\n\n" + prelude + "\n\n"
              + "\n".join(imports) + "\n" + "".join(sections))
    output += "\n\n" + license_notice + "\n"
    output += '\n\nif __name__ == "__main__":\n    standalone_main()\n'
    output = re.sub(r"\n{4,}", "\n\n\n", output)
    compile(output, "muzero.py", "exec")
    (ROOT / "muzero.py").write_text(output)
    print(f"Wrote muzero.py ({len(output.splitlines())} lines); no package imports or module loader.")


if __name__ == "__main__":
    main()
