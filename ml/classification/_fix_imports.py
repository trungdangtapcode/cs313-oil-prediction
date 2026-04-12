"""Fix import paths for classification scripts."""
import os

folder = os.path.dirname(__file__)
for f in os.listdir(folder):
    if not f.endswith('.py') or f.startswith('_'):
        continue
    path = os.path.join(folder, f)
    with open(path, 'r', encoding='utf-8') as fh:
        content = fh.read()

    # Fix sys.path to go up to ml/
    content = content.replace(
        "sys.path.insert(0, os.path.dirname(__file__))",
        "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))"
    )

    # Fix OUT_DIR to local results/
    if "from config import" in content and "OUT_DIR" in content:
        content = content.replace(", OUT_DIR", "")
        content = content.replace("OUT_DIR", "os.path.join(os.path.dirname(__file__), 'results')")

    # Fix step4_improve import (for scripts that import add_technical_features)
    content = content.replace(
        "from step4_improve import",
        "from improve import"
    )

    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'Fixed: {f}')

