"""Standalone SSH worker bootstrap; standard library, Python 3.8+.

The launcher sends this source over SSH so setup works before NFS is mounted.
Keep it independent of package imports and accelerator initialization.
"""

import base64
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


def run(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, timeout=60, **kwargs).stdout.strip()


def environment(request):
    identity = hashlib.sha256(request["repo"].encode()).hexdigest()[:16]
    location = Path.home() / ".venvs" / "nanoalphazero" / identity
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(location), VIRTUAL_ENV="", UV_LINK_MODE="copy")
    # Do not inherit host-0's cache or virtualenv paths through SSH.
    env["UV_CACHE_DIR"] = str(Path.home() / ".cache" / "uv")
    # User-local build tools (e.g. npm for Trackio) also work in non-login SSH.
    env["PATH"] = str(Path.home() / ".local" / "bin") + os.pathsep + env.get("PATH", "")
    return location, env


def mount_source(share):
    result = subprocess.run(["findmnt", "-rn", "--mountpoint", share, "-o", "SOURCE,FSTYPE"],
                            text=True, capture_output=True, timeout=10)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def ensure_mount(request):
    if request["index"] == 0:
        return
    share = Path(request["share"])
    expected = request["hosts"][0] + ":" + str(share)
    mounted = mount_source(str(share))
    if mounted:
        if mounted.split()[0] != expected or mounted.split()[1] not in ("nfs", "nfs4"):
            raise RuntimeError("Unexpected mount on {}: {}".format(share, mounted))
        return
    if share.exists() and any(share.iterdir()):
        raise RuntimeError("Refusing to mount over nonempty directory {}".format(share))
    run(["sudo", "-n", "mkdir", "-p", str(share)])
    run(["sudo", "-n", "mount", "-t", "nfs", "-o", "retry=0", expected, str(share)])


def fingerprint(repo):
    paths = run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=repo).split("\0")
    digest = hashlib.sha256()
    for name in sorted(set(paths) - {""}):
        path = Path(repo) / name
        digest.update(name.encode() + b"\0")
        if not path.is_file():
            digest.update(b"<missing>")
            continue
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def dependencies(python):
    code = """import importlib.metadata as m, json, sys
items = sorted((d.metadata['Name'].lower(), d.version, d.read_text('direct_url.json')) for d in m.distributions())
print(json.dumps({'python': sys.version, 'packages': items}, sort_keys=True))
"""
    return hashlib.sha256(run([str(python), "-c", code]).encode()).hexdigest()


def tpu_owners():
    paths = ["/tmp/libtpu_lockfile"] + glob.glob("/dev/accel*") + glob.glob("/dev/vfio/[0-9]*")
    paths = [path for path in paths if os.path.exists(path)]
    if not paths:
        return {"available": False, "detail": "No TPU device or lock paths found"}
    result = subprocess.run(["sudo", "-n", "fuser", "-v", *paths],
                            text=True, capture_output=True, timeout=10)
    detail = (result.stdout + result.stderr).strip()
    return {"available": result.returncode == 1 and not detail,
            "detail": detail or ("No owners" if result.returncode == 1 else "Ownership check failed")}


def state_directory():
    return Path.home() / ".local" / "state" / "nanoalphazero" / "pod"


def process_start(pid):
    try:
        # comm can contain spaces or parentheses; field 22 follows the last ')'.
        return Path("/proc/{}/stat".format(pid)).read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def active_jobs():
    result = []
    for path in state_directory().glob("*.json"):
        try:
            job = json.loads(path.read_text())
            if job["status"] == "running" and process_start(job["pid"]) == job["start"]:
                result.append(job["launch"])
        except (OSError, ValueError, KeyError):
            continue
    return result


def inspect(request):
    repo = Path(request["repo"])
    location, _ = environment(request)
    python = location / "bin" / "python"
    report = {"host": os.uname().nodename, "mount": mount_source(request["share"]),
              "repo_exists": repo.is_dir(), "environment": str(location),
              "tpu": tpu_owners(), "jobs": active_jobs()}
    if repo.is_dir():
        report["code"] = fingerprint(str(repo))
        report["lock"] = hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest()
    if python.exists():
        report["dependencies"] = dependencies(python)
        marker = location / "pod-setup.json"
        report["setup"] = json.loads(marker.read_text()) if marker.exists() else None
    return report


def setup(request):
    ensure_mount(request)
    repo = Path(request["repo"])
    location, env = environment(request)
    if not repo.is_dir():
        raise RuntimeError("Shared checkout is missing: {}".format(repo))
    marker_path = location / "pod-setup.json"
    if marker_path.exists():
        marker_path.unlink()
    command = ["uv", "sync", "--locked", "--group", "dev"]
    if request.get("reinstall"):
        command.append("--reinstall")
    for extra in request.get("extras", []):
        command.extend(["--extra", extra])
    subprocess.run(command, cwd=str(repo), env=env, check=True, timeout=900)
    # Metadata alone cannot detect an interrupted install's missing files.
    # Force CPU for this import check, even when another job owns the TPU.
    try:
        subprocess.run([str(location / "bin" / "python"), "-c",
                        "import numpy, jax, flax, optax, pgx1, mctx, flashbax, zstandard, safetensors"],
                       env=dict(env, JAX_PLATFORMS="cpu"), check=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Dependency import check failed; run pod setup --reinstall") from exc
    marker = {"lock": hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest(),
              "extras": request.get("extras", [])}
    marker_path.write_text(json.dumps(marker))
    return inspect(request)


def verify_shared(request):
    probe = Path(request["probe"])
    if probe.read_text() != request["nonce"]:
        raise RuntimeError("Shared filesystem probe differs from host 0")
    reply = probe.with_name(probe.name + "." + str(request["index"]))
    reply.write_text(request["nonce"])


def terminate_group(child, grace=2):
    # Caller has not reaped the leader: its PID/PGID cannot have been reused.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            time.sleep(grace)
    return child.wait()


def supervise(request):
    location, env = environment(request)
    repo = request["repo"]
    if fingerprint(repo) != request["code"] or dependencies(location / "bin" / "python") != request["dependencies"]:
        raise RuntimeError("Code/dependencies changed after preflight; rerun setup")
    verify_shared(request)
    directory = state_directory()
    directory.mkdir(parents=True, exist_ok=True)
    # Serializes our TPU launches on this host without touching libtpu's lock.
    lock = (directory / "launch.lock").open("a+")
    if not request["cpu"]:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owners = tpu_owners()
        if not owners["available"]:
            raise RuntimeError("TPU occupied or unavailable: " + owners["detail"])
    else:
        env["JAX_PLATFORMS"] = "cpu"
    env.update(PYTHONUNBUFFERED="1", NANOAZ_POD_PROCESS_ID=str(request["index"]),
               NANOAZ_POD_PROCESS_COUNT=str(len(request["hosts"])),
               NANOAZ_POD_COORDINATOR=request["hosts"][0], NANOAZ_POD_LAUNCH=request["launch"])
    cancelled = False

    def stop(*_):
        nonlocal cancelled
        cancelled = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    child = subprocess.Popen(["uv", "run", "--no-sync", "--", *request["command"]],
                             cwd=repo, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
    record = {"launch": request["launch"], "pid": os.getpid(), "start": process_start(os.getpid()),
              "child": child.pid, "status": "running"}
    path = directory / (request["launch"] + ".json")
    heartbeat = time.monotonic()
    code = 1
    try:
        path.write_text(json.dumps(record))
        while not cancelled:
            # Keep the leader unreaped until all its descendants are stopped.
            if os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                break
            readable, _, _ = select.select([sys.stdin], [], [], 0.5)
            if readable:
                if not os.read(sys.stdin.fileno(), 4096):
                    cancelled = True
                    break
                heartbeat = time.monotonic()
            if time.monotonic() - heartbeat > 20:
                print("Launcher heartbeat lost; stopping this job", file=sys.stderr, flush=True)
                cancelled = True
    finally:
        code = terminate_group(child)
        record.update(status="cancelled" if cancelled else ("succeeded" if code == 0 else "failed"), code=code)
        path.write_text(json.dumps(record))
        lock.close()
    return code if not cancelled else 130


def main():
    request = json.loads(base64.urlsafe_b64decode(sys.argv[1]))
    action = request["action"]
    if action == "run":
        return supervise(request)
    if action == "setup":
        result = setup(request)
    elif action == "probe":
        ensure_mount(request)
        verify_shared(request)
        result = inspect(request)
    elif action == "health":
        verify_shared(request)
        result = {"shared_filesystem": "ok"}
    else:
        result = inspect(request)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("pod worker: {}".format(exc), file=sys.stderr, flush=True)
        sys.exit(1)
