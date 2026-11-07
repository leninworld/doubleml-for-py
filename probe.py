"""Inspect the real docs image and resolve prerequisite closures, without project tests."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import urllib.request
import zipfile

from bounded import LimitAbort, cgroup_sample, command, limit, save

P = Path("/packet")
R = Path("/results")
M = json.loads((P / "manifest.json").read_text())
END = float(os.environ["PROBE_END"])
SAMPLES = []
LAST = 0


def guard():
    global LAST
    limit(time.time() < END, "ONE_JOB_DEADLINE")
    if time.monotonic() - LAST >= 2:
        try:
            row = cgroup_sample()
        except Exception as exc:
            raise LimitAbort("CGROUP_OBSERVER_UNAVAILABLE") from exc
        SAMPLES.append(row)
        LAST = time.monotonic()
        limit(row["working_set"] < row["limit"] * .9375, "RESOLVER_MEMORY_PRESSURE")
        for path in ("/work", "/results"):
            fs = os.statvfs(path)
            limit(fs.f_bavail * fs.f_frsize > 128 * 1024**2, "INNER_DISK_PRESSURE")
            limit(not fs.f_files or fs.f_favail / fs.f_files > .05, "INNER_INODE_PRESSURE")
        limit(sum(p.stat().st_size for p in R.rglob("*") if p.is_file()) < 12 * 1024**2, "INNER_RESULT_CAP")


def run(label, args, seconds=120, cap=1024**2):
    value = command(args, min(seconds, max(1, END-time.time())), cap, guard, cwd="/work", log=R / (label + ".log"))
    return value


def marker_fixture(uv):
    """Test actual target-version marker selection using local synthetic wheels."""
    wheels = Path("/work/fixture-wheels")
    wheels.mkdir()
    for name in ("guard_old", "guard_new"):
        info = name + "-1.0.dist-info/"
        with zipfile.ZipFile(wheels / (name + "-1.0-py3-none-any.whl"), "w") as z:
            z.writestr(info + "METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
            z.writestr(info + "WHEEL", "Wheel-Version: 1.0\nGenerator: synthetic-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
            z.writestr(info + "RECORD", "")
    req = Path("/work/fixture.in")
    req.write_text("guard-old==1.0; python_version < '3.11'\nguard-new==1.0; python_version >= '3.11'\n")
    for version in M["python_versions"]:
        out = Path("/work/fixture-" + version + ".txt")
        run("marker-" + version, [str(uv), "pip", "compile", str(req), "--python-version", version,
            "--python-platform", "x86_64-manylinux_2_28", "--no-index", "--find-links", str(wheels),
            "--generate-hashes", "--no-header", "--no-annotate", "-o", str(out)], 30)
        pins = [line.split("==")[0] for line in out.read_text().splitlines() if "==" in line]
        assert pins == ["guard-old" if version == "3.10" else "guard-new"], pins


summary = {"status": "FAILED", "full_validation": False, "steps": {},
           "baseline": "NOT_RUN", "production_implementation": "NOT_STARTED",
           "native_mac_windows": "NOT_RUN", "R_current_main_upgrade": "NOT_PERFORMED",
           "resolution_platform": "x86_64-manylinux_2_28; glibc>=2.28 required for G1; may differ from glibc2.39 CI newest-wheel choices",
           "docs_resolution_limit": "Joint requirements resolve approximates CI's separate requirements install then editable core[rdd] install."}


def step(name, fn):
    guard()
    try:
        fn()
    except Exception as exc:
        summary["steps"][name] = {"status": "FAIL", "error": type(exc).__name__ + ": " + str(exc),
                                  "log": name + ".log"}
        if not (R / (name + ".log")).exists():
            (R / (name + ".log")).write_text(summary["steps"][name]["error"] + "\n")
        return False
    summary["steps"][name] = {"status": "PASS", "log": name + ".log"}
    if not (R / (name + ".log")).exists():
        (R / (name + ".log")).write_text("STEP_PASS; detailed metadata or marker logs retained alongside this receipt.\n")
    return True


def image_python():
    save(R / "image-python.json", {"python": sys.version, "platform": platform.platform(),
         "installed": sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()),
         "cache_locations": {s: Path(s).exists() for s in ("/home/ubuntu/.cache/pip", "/home/ubuntu/.cache/uv")}})
    assert sys.version_info[:2] == (3, 12), "ACTUAL_DOCS_IMAGE_PYTHON_NOT_312"


def jupyter_kernels():
    os.environ["JUPYTER_PATH"] = "/home/ubuntu/.local/share/jupyter"
    kernels = json.loads(run("jupyter-kernels", ["jupyter", "kernelspec", "list", "--json"], 60))
    assert any(k.get("spec", {}).get("language", "").lower() == "r" for k in kernels["kernelspecs"].values())


def acquire_uv():
    artifact = M["uv"]
    wheel = Path("/work") / artifact["name"]
    begin = time.monotonic()
    h = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(artifact["url"], timeout=20) as response, wheel.open("xb") as output:
        assert response.geturl().startswith("https://files.pythonhosted.org/")
        while True:
            guard()
            assert time.monotonic()-begin < 90, "UV_DOWNLOAD_DEADLINE"
            part = response.read(65536)
            if not part:
                break
            size += len(part)
            assert size <= artifact["bytes"]
            h.update(part)
            output.write(part)
    assert size == artifact["bytes"] and h.hexdigest() == artifact["sha256"]
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if n.endswith(".data/scripts/uv")]
        assert len(names) == 1 and archive.getinfo(names[0]).file_size < 80 * 1024**2
        uv.write_bytes(archive.read(names[0]))
    uv.chmod(0o755)
    assert run("uv-acquire", [str(uv), "--version"], 10).decode().startswith("uv 0.9.30")


def dependency_inputs():
    import tomllib
    config = tomllib.loads(Path("/core/pyproject.toml").read_text())
    declared = config["project"]["dependencies"] + config["project"]["optional-dependencies"]["rdd"]
    requirements = (declared + config["dependency-groups"]["dev"] + config["build-system"]["requires"]
                    + ["black==25.9.0", "ruff==0.14.4", "pre-commit-hooks==6.0.0"])
    assert all(isinstance(s, str) and "\n" not in s and " @ " not in s for s in requirements)
    core.write_text("\n".join(requirements) + "\n")
    (R / "core.in").write_bytes(core.read_bytes())
    raw = Path("/docs/requirements.txt").read_text().splitlines()
    assert sum(s.strip() == "DoubleML[rdd]" for s in raw) == 1
    docs.write_text("\n".join(s for s in raw if s.strip() != "DoubleML[rdd]") + "\n" + "\n".join(declared) + "\n")
    (R / "docs.in").write_bytes(docs.read_bytes())


uv = Path("/work/uv")
core = Path("/work/core.in")
docs = Path("/work/docs.in")
try:
    image_ok = step("image-python", image_python)
    rcode = '''cat(R.version.string, "\\n");
required <- c("DoubleML", "IRkernel", "remotes", "ggplot2", "xgboost", "hdm", "reshape2", "gridExtra", "igraph", "mlr3filters", "mlr3measures", "did")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly=TRUE)]
print(installed.packages()[,c("Package", "Version")]);
if(length(missing)) stop(paste("Missing R prerequisites:", paste(missing, collapse=", ")))
desc <- packageDescription("DoubleML"); print(desc[c("Version", "RemoteSha", "RemoteRepo", "RemoteUsername")])
'''
    step("R-prerequisites", lambda: run("R-prerequisites", ["Rscript", "--vanilla", "-e", rcode], 120))
    step("jupyter-kernels", jupyter_kernels)
    step("pandoc", lambda: run("pandoc", ["pandoc", "--version"], 15))
    step("graphviz", lambda: run("graphviz", ["dot", "-V"], 15))
    step("glibc", lambda: run("glibc", ["ldd", "--version"], 15))
    uv_ok = step("uv-acquire", acquire_uv)
    marker_ok = step("marker-fixture", lambda: marker_fixture(uv)) if uv_ok else False
    if not uv_ok:
        summary["steps"]["marker-fixture"] = {"status": "SKIPPED_PREREQUISITE", "requires": "uv-acquire"}
    inputs_ok = step("dependency-inputs", dependency_inputs)
    for version in M["python_versions"]:
        if not (uv_ok and marker_ok and inputs_ok):
            summary["steps"]["core-" + version] = {"status": "SKIPPED_PREREQUISITE", "requires": "uv-acquire, marker-fixture, dependency-inputs"}
            continue
        step("core-" + version, lambda version=version: run("core-" + version, [str(uv), "pip", "compile", str(core), "--python-version", version,
            "--python-platform", "x86_64-manylinux_2_28", "--index-url", "https://pypi.org/simple",
            "--only-binary", ":all:", "--generate-hashes", "--no-header", "--no-annotate",
            "-o", str(R / ("core-" + version + ".txt"))], 180))
    if uv_ok and marker_ok and inputs_ok and image_ok:
        step("docs-3.12", lambda: run("docs-3.12", [str(uv), "pip", "compile", str(docs), "--python-version", "3.12",
            "--python-platform", "x86_64-manylinux_2_28", "--index-url", "https://pypi.org/simple",
            "--generate-hashes", "--no-header", "--no-annotate", "-o", str(R / "docs-3.12.txt")], 240))
    else:
        summary["steps"]["docs-3.12"] = {"status": "SKIPPED_PREREQUISITE", "requires": "uv-acquire, marker-fixture, dependency-inputs, image-python"}
    summary["status"] = ("G0_PREREQUISITES_RESOLVED" if all(s["status"] == "PASS" for s in summary["steps"].values())
                         else "G0_PREREQUISITES_INCOMPLETE")
    summary["limits"] = "Wheel-route core closures and docs metadata resolved; installation, R freshness and all baseline/full-finish tests still require G1 approval."
except BaseException as exc:
    summary["failure"] = type(exc).__name__ + ": " + str(exc)
    raise
finally:
    save(R / "memory-samples.json", SAMPLES)
    save(R / "summary.json", summary)
