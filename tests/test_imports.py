"""
Project N: AST-Based Forbidden Import Scanner (Invariant 10).
Scans Python source directories using Abstract Syntax Tree (AST) analysis
to strictly guarantee zero imports of external cloud AI SDKs or PyTorch.
"""

import ast
import sys
from pathlib import Path

FORBIDDEN_PREFIXES = (
    "openai",
    "anthropic",
    "vertexai",
    "google.generativeai",
    "torch",
    "torchvision",
    "torchaudio",
)

SCAN_DIRECTORIES = (
    "extraction",
    "models",
    "rag",
    "server",
    "storage",
    "training",
)


class ForbiddenImportVisitor(ast.NodeVisitor):
    def __init__(self, filename: Path) -> None:
        self.filename = filename
        self.violations: list[tuple[Path, int, str]] = []

    def _check_module(self, module_name: str | None, lineno: int) -> None:
        if not module_name:
            return
        parts = module_name.split(".")
        root_pkg = parts[0]
        full_module = module_name
        for forbidden in FORBIDDEN_PREFIXES:
            if root_pkg == forbidden or full_module.startswith(forbidden):
                self.violations.append((self.filename, lineno, full_module))
                break

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._check_module(node.module, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in ("__import__", "import_module") and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                self._check_module(first_arg.value, node.lineno)

        self.generic_visit(node)


def scan_directory(directory: Path) -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []
    for path in directory.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            visitor = ForbiddenImportVisitor(path)
            visitor.visit(tree)
            violations.extend(visitor.violations)
        except Exception as e:
            violations.append((path, 1, f"Failed to parse AST: {e}"))
    return violations


def verify_no_forbidden_imports(repo_root: Path) -> list[tuple[Path, int, str]]:
    all_violations: list[tuple[Path, int, str]] = []
    for dirname in SCAN_DIRECTORIES:
        dir_path = repo_root / dirname
        if dir_path.is_dir():
            all_violations.extend(scan_directory(dir_path))
    return all_violations


def test_ast_forbidden_imports() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    violations = verify_no_forbidden_imports(repo_root)
    assert not violations, f"Forbidden imports detected: {violations}"


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    violations = verify_no_forbidden_imports(repo_root)
    if violations:
        print("CRITICAL: Forbidden cloud AI or PyTorch imports detected:")
        for file_path, lineno, offending_module in violations:
            rel_path = file_path.relative_to(repo_root)
            print(f"  {rel_path}:{lineno} -> {offending_module}")
        sys.exit(1)
    else:
        print("AST Import Scan: 0 forbidden imports detected across all production directories.")
        sys.exit(0)
