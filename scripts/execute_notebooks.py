"""Execute local notebook cells and save their standard outputs.

This lightweight runner is used when Jupyter is unavailable. It supports the
project's plain Python cells, not arbitrary notebook magics or rich outputs.
"""

import contextlib
import io
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
sys.path.insert(0, str(ROOT))


def execute(path: Path) -> None:
    """Run code cells in one shared namespace and persist printed evidence."""
    data = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    count = 0
    for cell in data["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        output = io.StringIO()
        source = "".join(cell["source"])
        with contextlib.redirect_stdout(output):
            exec(compile(source, str(path), "exec"), namespace)
        cell["execution_count"] = count
        cell["outputs"] = ([{"output_type": "stream", "name": "stdout",
                             "text": output.getvalue().splitlines(True)}]
                           if output.getvalue() else [])
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(path.name, count, "code cells executed")


if __name__ == "__main__":
    names = sys.argv[1:] or ["01_extraction.ipynb", "02_chunking.ipynb",
                             "03_embeddings_retrieval.ipynb"]
    for name in names:
        execute(NOTEBOOKS / name)
