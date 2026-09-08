"""Build three local plans; never call a model."""
from pathlib import Path
from tempfile import TemporaryDirectory
from ctxfeed.cache_plan import CachePlan
import json
with TemporaryDirectory(prefix="ctxfeed-demo-") as folder:
    root = Path(folder)
    (root / "README.md").write_text("# Tiny service\nA local example.\n")
    (root / "app.py").write_text("def greet():\n    return 'hello'\n")
    with CachePlan.for_repo(root) as cp:
        rows = []
        for stage in ["first", "unchanged", "edited"]:
            if stage == "edited":
                (root / "app.py").write_text("def greet():\n    return 'hello again'\n")
            plan = cp.plan().plan
            rows.append({"stage": stage, "files": plan.files, "delta_hashes": len(plan.delta), "cached_fraction": plan.cache_hit_rate, "stable_prefix": [c.path for c in plan.stable_prefix]})
        print(json.dumps(rows, indent=2))
