"""Exercise the actual bounded helpers in Docker before large downloads."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

from bounded import LimitAbort, blob, cgroup_sample, command, emit, save

sample = cgroup_sample()
assert sample["limit"] == 4 * 1024**3
assert 0 < sample["current"] <= sample["peak"] and sample["working_set"] >= 0
assert command([sys.executable, "-c", "print('fixture-ok')"], 5, 100) == b"fixture-ok\n"
assert command([sys.executable, "-c", "raise SystemExit(7)"], 5, 100, expected=7) == b""
for source, seconds, cap, exception in [
    ("raise SystemExit(9)", 5, 100, RuntimeError),
    ("import time; time.sleep(30)", .3, 100, TimeoutError),
    ("print('x'*1000)", 5, 20, OverflowError),
]:
    try:
        command([sys.executable, "-c", source], seconds, cap)
    except exception:
        pass
    else:
        raise AssertionError("expected failure not observed")
with tempfile.TemporaryDirectory() as raw:
    root = Path(raw)
    (root / "a" / "b").mkdir(parents=True)
    content = b"fixture\n"
    (root / "a" / "b" / "input").write_bytes(content)
    shutil.copytree(root / "a", root / "copy")
    assert (root / "copy" / "b" / "input").read_bytes() == content
    expected = hashlib.sha1(b"blob 8\0fixture\n").hexdigest()
    assert blob(content) == expected and blob(content + b"x") != expected
    save(root / "result.json", {"all_fixture_cases": "PASS"})
    assert json.loads((root / "result.json").read_text())["all_fixture_cases"] == "PASS"
    archive_path = root / "tiny.whl"
    with zipfile.ZipFile(archive_path, "w") as z:
        z.writestr("guard-1.data/scripts/uv", content)
    with zipfile.ZipFile(archive_path) as z:
        names = [n for n in z.namelist() if n.endswith(".data/scripts/uv")]
        assert len(names) == 1
        extracted = z.read(names[0])
    assert extracted == content and blob(extracted) == expected
    failure_log = root / "failed-command.log"
    try:
        command([sys.executable, "-c", "print('kept', flush=True); raise SystemExit(2)"], 5, 100, log=failure_log)
    except RuntimeError:
        assert failure_log.read_bytes() == b"kept\n"
    else:
        raise AssertionError("failure evidence fixture did not fail")
    pid_path = root / "child.pid"
    calls = 0
    def abort_guard():
        global calls
        calls += 1
        if calls == 2:
            wait_until = time.monotonic() + .8
            while (not pid_path.exists() or not pid_path.read_text().isdigit()) and time.monotonic() < wait_until:
                time.sleep(.01)
            assert pid_path.exists() and pid_path.read_text().isdigit(), "fixture child did not announce PID"
            raise LimitAbort("fixture global abort")
    start = time.monotonic()
    try:
        command([sys.executable, "-c", "import os,pathlib,time; pathlib.Path(" + repr(str(pid_path))
                 + ").write_text(str(os.getpid())); time.sleep(30)"], 5, 100, abort_guard)
    except LimitAbort:
        assert time.monotonic()-start < 3
    else:
        raise AssertionError("global abort did not propagate")
    pid = int(pid_path.read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("aborted child is still alive")
    emit(root)
print("HELPER_FIXTURE_PASS")
