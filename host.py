"""Run one prerequisite probe on an agent-controlled GitHub Linux VM only."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time

from bounded import blob, command, emit, limit, save

P = Path(__file__).resolve().parent
W = P.parent
M = json.loads((P / "manifest.json").read_text())
R = W / "runtime"
R.mkdir(exist_ok=True)
START = float(os.environ["PROBE_STARTED_EPOCH"])
END = START + M["absolute_seconds"]
RX = Path(os.environ["PROBE_RX_PATH"])
RX0 = int(os.environ["PROBE_RX_START"])
MODE = sys.argv[1]
assert MODE in ("fixture", "probe")
assert M["execution_enabled"] and os.environ.get("GITHUB_ACTIONS") == "true"
assert M["attempts_authorized"] == 1 and os.environ.get("GITHUB_RUN_ATTEMPT") == "1"
assert os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted" and platform.system() == "Linux"
assert platform.machine() == "x86_64"
RUN = os.environ["GITHUB_RUN_ID"]
assert RUN.isdigit()
NAME = "doubleml364-g0-" + RUN + "-" + MODE
DOCKER_ROOT = None
assert not (R / (MODE + "-result.json")).exists(), "NO_PHASE_REPLAY"


def guard():
    limit(time.time() < END, "ONE_JOB_DEADLINE")
    available = int(re.search(r"^MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text(), re.M)[1]) * 1024
    limit(available > 1536 * 1024**2, "HOST_MEMORY_PRESSURE")
    for path in (W, DOCKER_ROOT):
        if path:
            limit(shutil.disk_usage(path).free > 6 * 1024**3, "HOST_DISK_PRESSURE")
            fs = os.statvfs(path)
            limit(not fs.f_files or fs.f_favail / fs.f_files > .05, "HOST_INODE_PRESSURE")
    limit(int(RX.read_text()) - RX0 < M["network_bytes_cap"], "NETWORK_CAP")
    limit(sum(p.stat().st_size for p in R.rglob("*") if p.is_file()) < 16 * 1024**2, "RESULT_CAP")


def run(label, args, seconds=30, cap=2 * 1024**2):
    data = command(args, min(seconds, max(1, END-time.time())), cap, guard, log=R / (label + ".log"))
    return data


result = {"mode": MODE, "status": "FAILED", "full_validation": False, "start": START}
created = False
try:
    guard()
    info = json.loads(run("docker-info-" + MODE, ["docker", "info", "--format", "{{json .}}"], cap=2*1024**2))
    DOCKER_ROOT = info["DockerRootDir"]
    assert info["OSType"] == "linux"
    assert info["MemTotal"] >= 6 * 1024**3 and info["NCPU"] >= 2
    guard()
    if MODE == "probe":
        assert json.loads((R / "fixture-result.json").read_text())["status"] == "PASS"
        for kind in ("core", "docs"):
            source = W / kind
            assert run("head-" + kind, ["git", "-C", str(source), "rev-parse", "HEAD"]).decode().strip() == M[kind + "_pin"]
            assert run("clean-" + kind, ["git", "-C", str(source), "status", "--porcelain"]) == b""
        rows = json.loads((P / "notebooks.json").read_text())
        listed = run("notebook-paths", ["git", "-C", str(W / "docs"), "ls-files", "*.ipynb"]).decode().splitlines()
        assert set(listed) == {row["path"] for row in rows} and len(listed) == 42
        notebooks = []
        for row in rows:
            data = (W / "docs" / row["path"]).read_bytes()
            assert len(data) == row["bytes"] and blob(data) == row["oid"], row["path"]
            parsed = json.loads(data)
            hits = [{"cell": i, "type": c.get("cell_type"), "source": "".join(c.get("source", []))}
                    for i, c in enumerate(parsed["cells"])
                    if any(k in "".join(c.get("source", [])) for k in ("trimming_rule", "trimming_threshold"))]
            for hit in hits:
                hit["classification"] = ("R_PACKAGE_API_OUT_OF_SCOPE_KEEP" if Path(row["path"]).name.startswith("R_")
                    else "PYTHON_API_REVIEW")
                if hit["classification"] == "PYTHON_API_REVIEW" and "make_confounded_irm_data" in hit["source"]:
                    hit["classification"] = "PYTHON_REVIEW_INCLUDES_UNRELATED_DATASET_PARAMETER_KEEP"
                    hit["preserve"] = "make_confounded_irm_data(trimming_threshold=...) is unrelated; inspect any other calls in this same cell independently."
            notebooks.append(dict(row, hits=hits))
        assert len(notebooks) == 42
        save(R / "notebook-inventory.json", notebooks)
    image = M["bootstrap_image" if MODE == "fixture" else "docs_image"]
    run("pull-" + MODE, ["docker", "pull", "--platform", "linux/amd64", image], 600, 2*1024**2)
    inspected = json.loads(run("image-" + MODE, ["docker", "image", "inspect", image]))[0]
    assert inspected["Architecture"] == "amd64" and inspected["Os"] == "linux"
    assert not inspected["Config"].get("Volumes")
    args = ["docker", "create", "--name", NAME, "--label", "purpose=doubleml364-g0",
            "--platform", "linux/amd64", "--memory", "4g", "--memory-swap", "4g", "--cpus", "2",
            "--pids-limit", "512", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "1000:1000", "--workdir", "/work", "--mount", f"type=bind,src={P},dst=/packet,readonly",
            "--tmpfs", "/work:rw,exec,uid=1000,gid=1000,size=2g", "--tmpfs", "/tmp:rw,exec,uid=1000,gid=1000,size=256m",
            "--env", "HOME=/work", "--env", "UV_CACHE_DIR=/work/cache", "--env", "UV_HTTP_RETRIES=0",
            "--env", "UV_PYTHON_DOWNLOADS=never", "--env", "UV_NO_CONFIG=1",
            "--env", "OMP_NUM_THREADS=1", "--env", "OPENBLAS_NUM_THREADS=1"]
    if MODE == "fixture":
        args += ["--network", "none", "--entrypoint", "python", image, "/packet/fixture.py"]
    else:
        out = R / "probe"
        out.mkdir()
        os.chmod(out, 0o777)
        args += ["--mount", f"type=bind,src={W/'core'},dst=/core,readonly",
                 "--mount", f"type=bind,src={W/'docs'},dst=/docs,readonly",
                 "--mount", f"type=bind,src={out},dst=/results",
                 "--env", f"PROBE_END={END}", "--entrypoint", "/bin/sh", image,
                 "-c", "exec python /packet/probe.py"]
    run("create-" + MODE, args)
    created = True
    run("container-" + MODE, ["docker", "start", "--attach", NAME], 1350, 4*1024**2)
    state = json.loads(run("state-" + MODE, ["docker", "inspect", "--format", "{{json .State}}", NAME]))
    assert state["ExitCode"] == 0 and not state["OOMKilled"] and not state["Running"]
    if MODE == "fixture":
        assert b"HELPER_FIXTURE_PASS" in (R / "container-fixture.log").read_bytes()
    else:
        assert json.loads((R / "probe" / "summary.json").read_text())["status"] == "G0_PREREQUISITES_RESOLVED"
    result["status"] = "PASS"
except BaseException as exc:
    result["failure"] = type(exc).__name__ + ": " + str(exc)
    raise
finally:
    if created:
        # Stop only this job's exact owned container; no pruning or deletion.
        try:
            result["stop"] = command(["docker", "stop", "--time", "10", NAME], 20, 4096).decode().strip()
        except Exception as exc:
            result["stop_error"] = type(exc).__name__
    result.update(elapsed=time.time()-START, received_bytes=int(RX.read_text())-RX0)
    save(R / (MODE + "-result.json"), result)
    emit(R)
