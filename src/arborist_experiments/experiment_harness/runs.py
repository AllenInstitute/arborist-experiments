"""Run-directory plumbing shared by every driver.

`log` and `next_version_dir` were copy-pasted, byte-identical, into six drivers, and
`treemoco/probe_treemoco.py` already imported them out of a sibling driver's private
namespace -- which meant importing a whole benchmark module to get a two-line helper.

The mirror functions are parameterised rather than merged: the version dir under
scratch/ keeps the plain names the capsule's versioning convention expects, while
results/ is flat and shared with other pipelines, so each run prefixes its own
generic files on the way out and the prefix is what differed between the copies.
"""

import os
import shutil
import time

# The names every run writes and every other run would overwrite. Drivers that also
# emit a generic-sounding artifact (model_config.json, probe_metrics.csv) pass their
# own set, which is why this stays an argument.
DEFAULT_GENERIC_NAMES = frozenset({"desc.txt", "run_log.txt", "run_config.json"})


def log(fh, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def next_version_dir(base: str) -> str:
    os.makedirs(base, exist_ok=True)
    used = {int(d[1:]) for d in os.listdir(base) if d.startswith("v") and d[1:].isdigit()}
    out = os.path.join(base, f"v{max(used, default=0) + 1}")
    os.makedirs(out)
    return out


def mirror_dir(src_dir: str, dst_dir: str, prefix: str,
               generic_names=DEFAULT_GENERIC_NAMES) -> None:
    """Copy every file in the version dir into the flat results dir."""
    os.makedirs(dst_dir, exist_ok=True)
    for name in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, name)
        if os.path.isfile(src):
            dst = f"{prefix}{name}" if name in generic_names else name
            shutil.copy2(src, os.path.join(dst_dir, dst))


def mirror_file(src: str, dst_dir: str, prefix: str,
                generic_names=DEFAULT_GENERIC_NAMES) -> None:
    """Copy one file into the flat results dir.

    Chunked through a .partial rename rather than shutil.copy2, because the callers
    of this variant mirror multi-GB pretraining artifacts incrementally during a run
    and an interrupted copy must not leave a truncated file that looks complete.
    """
    os.makedirs(dst_dir, exist_ok=True)
    name = os.path.basename(src)
    if name in generic_names:
        name = f"{prefix}{name}"
    tmp = os.path.join(dst_dir, name + ".partial")
    with open(src, "rb") as fsrc, open(tmp, "wb") as fdst:
        while chunk := fsrc.read(8 << 20):
            fdst.write(chunk)
    os.replace(tmp, os.path.join(dst_dir, name))
