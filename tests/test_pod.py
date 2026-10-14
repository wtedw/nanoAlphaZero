"""Launcher discovery, preflight, and real process-group cleanup tests."""

import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from nanoalphazero import pod, pod_worker


def test_metadata_worker_order_and_validation():
    assert pod.parse_workers("unknown:unknown:10.0.0.2,unknown:unknown:10.0.0.4") == ["10.0.0.2", "10.0.0.4"]
    for text in ("", "host", "x:y:host;echo bad", "x:y:10.0.0.2,x:y:10.0.0.2"):
        with pytest.raises(ValueError):
            pod.parse_workers(text)


def test_discovery_rejects_nonzero_worker_and_multislice(monkeypatch):
    values = {"worker-network-endpoints": "x:y:10.0.0.2", "agent-worker-number": "0", "tpu-env": ""}
    monkeypatch.setattr(pod, "metadata", values.__getitem__)
    assert pod.discover() == ["10.0.0.2"]
    values["agent-worker-number"] = "1"
    with pytest.raises(RuntimeError, match="worker 0"):
        pod.discover()
    values["agent-worker-number"] = "0"
    values["tpu-env"] = "MEGASCALE_NUM_SLICES: '2'"
    with pytest.raises(RuntimeError, match="one TPU slice"):
        pod.discover()


def test_ssh_preserves_literal_command_arguments():
    args = SimpleNamespace(hosts=["10.0.0.2"], ssh_option=[])
    command = ["python", "a file.py", "$(touch nope)", "'quoted'", "line\nbreak"]
    ssh = pod.ssh(args, 0, {"command": command})
    remote = shlex.split(ssh[-1])
    import base64
    assert remote[:3] == ["python3", "-u", "-c"]
    assert json.loads(base64.urlsafe_b64decode(remote[-1]))["command"] == command


def test_bootstrap_supports_system_python38():
    ast.parse(Path(pod_worker.__file__).read_text(), feature_version=(3, 8))


def test_fingerprint_includes_dirty_and_new_files(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "tracked.py").write_text("first")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    original = pod_worker.fingerprint(str(tmp_path))
    (tmp_path / "tracked.py").write_text("changed")
    changed = pod_worker.fingerprint(str(tmp_path))
    assert changed != original
    (tmp_path / "new.toml").write_text("value=1")
    assert pod_worker.fingerprint(str(tmp_path)) != changed


def test_mount_never_hides_existing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(pod_worker, "mount_source", lambda _: "")
    (tmp_path / "important").touch()
    with pytest.raises(RuntimeError, match="nonempty"):
        pod_worker.ensure_mount({"index": 1, "share": str(tmp_path), "hosts": ["10.0.0.2"]})


@pytest.mark.parametrize("key,value,match", [
    ("code", "different", "differ"),
    ("dependencies", "different", "differ"),
    ("lock", "different", "lockfile changed"),
    ("setup", None, "environment is missing"),
    ("tpu", {"available": False, "detail": "owned by another job"}, "owned by another job"),
])
def test_preflight_stops_on_inconsistent_or_occupied_worker(key, value, match):
    good = {"repo_exists": True, "dependencies": "same", "code": "same", "lock": "same",
            "setup": {"lock": "same"}, "tpu": {"available": True}}
    with pytest.raises(RuntimeError, match=match):
        pod.validate_reports(SimpleNamespace(hosts=["one", "two"]), [good, {**good, key: value}], cpu=False)


def wait_for_file(path, process):
    deadline = time.monotonic() + 10
    while not path.exists() and time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(process.communicate()[1])
        time.sleep(0.05)
    assert path.exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_foreground_output_and_interrupt(tmp_path, monkeypatch, capsys, interrupt):
    args = SimpleNamespace(hosts=["one", "two"], timeout=10)
    code = "import sys; print('worker output', flush=True); "
    if interrupt:
        code += "sys.stdin.read(); print('lease closed', flush=True)"
    monkeypatch.setattr(pod, "ssh", lambda *args: [sys.executable, "-c", code])
    if interrupt:
        sleep = time.sleep
        cancelled = False
        def cancel(seconds):
            nonlocal cancelled
            if not cancelled:
                cancelled = True
                raise KeyboardInterrupt
            sleep(seconds)
        monkeypatch.setattr(pod.time, "sleep", cancel)
    assert pod.launch(args, {}, tmp_path) == (130 if interrupt else 0)
    output = capsys.readouterr().out
    assert "[worker 0] worker output" in output
    assert "[worker 1] worker output" in output
    assert "Ctrl+C stops this launch" in output
    assert "tail -F" not in output
    for index in range(2):
        log = (tmp_path / f"worker-{index}.log").read_text()
        assert "worker output" in log
        if interrupt:
            assert "lease closed" in log
    assert json.loads((tmp_path / "result.json").read_text())["workers"] == [0, 0]


def test_lost_launcher_stops_only_its_process_group(tmp_path):
    """Kill a live job + grandchild by dropping stdin; an unrelated job survives."""
    stub = tmp_path / "uv"
    stub.write_text(f"#!{sys.executable}\nimport os, sys\nos.execvp(sys.argv[4], sys.argv[4:])\n")
    stub.chmod(0o755)
    pids_file = tmp_path / "pids.json"
    child_code = (
        "import os,sys,subprocess,time,json; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"open({str(pids_file)!r},'w').write(json.dumps([os.getpid(),child.pid])); time.sleep(60)"
    )
    request = {"repo": str(tmp_path), "code": "same", "dependencies": "same", "cpu": True,
               "index": 0, "hosts": ["localhost"], "launch": "test-lease", "command": [sys.executable, "-c", child_code]}
    bootstrap = (
        "import os,sys; from pathlib import Path; from nanoalphazero import pod_worker as w; "
        "w.environment=lambda r:(Path(sys.prefix),dict(os.environ)); "
        "w.fingerprint=lambda r:'same'; w.dependencies=lambda r:'same'; "
        "w.verify_shared=lambda r:None; "
        f"sys.exit(w.supervise({request!r}))"
    )
    env = {**os.environ, "HOME": str(tmp_path), "PATH": str(tmp_path) + ":" + os.environ["PATH"]}
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    supervisor = subprocess.Popen([sys.executable, "-c", bootstrap], env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        wait_for_file(pids_file, supervisor)
        supervisor.stdin.close()
        assert supervisor.wait(timeout=10) == 130
        assert unrelated.poll() is None
        for pid in json.loads(pids_file.read_text()):
            stat = Path(f"/proc/{pid}/stat")
            assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        status = json.loads((tmp_path / ".local/state/nanoalphazero/pod/test-lease.json").read_text())
        assert status["status"] == "cancelled"
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)
        if supervisor.poll() is None:
            supervisor.stdin.close()
            supervisor.wait(timeout=10)
