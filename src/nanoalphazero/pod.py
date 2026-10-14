"""Discover an existing TPU pod and launch commands from its shared checkout."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request
import uuid

from nanoalphazero import pod_worker


def metadata(name):
    # Metadata is link-local; external HTTP proxies must never receive it.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/" + name,
        headers={"Metadata-Flavor": "Google"},
    )
    with opener.open(request, timeout=5) as response:
        return response.read().decode().strip()


def parse_workers(value):
    hosts = []
    for endpoint in value.split(","):
        parts = endpoint.strip().split(":")
        if len(parts) != 3:
            raise ValueError("Unrecognized TPU worker endpoint: " + endpoint)
        hosts.append(str(ipaddress.IPv4Address(parts[2])))
    if not hosts or len(set(hosts)) != len(hosts):
        raise ValueError("TPU metadata contains missing or duplicate workers")
    return hosts


def discover():
    try:
        hosts = parse_workers(metadata("worker-network-endpoints"))
        if metadata("agent-worker-number") != "0":
            raise RuntimeError("Run pod commands from worker 0, which exports the shared checkout")
        tpu_env = metadata("tpu-env")
        for line in tpu_env.splitlines():
            if line.startswith("MEGASCALE_NUM_SLICES:") and int(line.split(":", 1)[1].strip(" '\"")) != 1:
                raise RuntimeError("This launcher currently supports one TPU slice")
        return hosts
    except (OSError, ValueError) as exc:
        raise RuntimeError("Could not discover a TPU pod from VM metadata; run on TPU worker 0: " + str(exc)) from exc


def ssh(args, index, request):
    payload = base64.urlsafe_b64encode(json.dumps({**request, "index": index}).encode()).decode()
    source = Path(pod_worker.__file__).read_text()
    remote = shlex.join(["python3", "-u", "-c", source, payload])
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
            *args.ssh_option, args.hosts[index], remote]


def request_all(args, request, timeout=90, logs=None):
    def one(index):
        try:
            if logs is None:
                result = subprocess.run(ssh(args, index, request), stdin=subprocess.DEVNULL,
                                        capture_output=True, text=True, timeout=timeout)
                error = result.stderr
            else:
                path = logs / f"worker-{index}.log"
                with path.open("w") as log:
                    result = subprocess.run(ssh(args, index, request), stdin=subprocess.DEVNULL,
                                            stdout=subprocess.PIPE, stderr=log, text=True, timeout=timeout)
                error = path.read_text()[-8000:]
                print(f"Worker {index} setup {'finished' if result.returncode == 0 else 'failed'}; {path}", flush=True)
            if result.returncode:
                raise RuntimeError(error.strip() or result.stdout.strip())
            # uv setup can print a short summary before the worker's JSON.
            return json.loads(result.stdout.splitlines()[-1])
        except (OSError, ValueError, subprocess.TimeoutExpired, RuntimeError) as exc:
            return {"error": str(exc), "worker": index}
    with ThreadPoolExecutor(max_workers=len(args.hosts)) as pool:
        reports = list(pool.map(one, range(len(args.hosts))))
    return reports


def require_success(args, reports):
    failures = [f"worker {i} ({args.hosts[i]}): {r['error']}" for i, r in enumerate(reports) if "error" in r]
    if failures:
        raise RuntimeError("\n".join(failures))


def configure_export(args):
    """Use an existing export, or create a narrow export to these workers."""
    share = Path(args.share)
    if not share.is_dir() or not Path(args.repo).is_relative_to(share):
        raise RuntimeError("Repository must be inside the shared directory: " + str(share))
    if any(char.isspace() for char in str(share)):
        raise RuntimeError("NFS share paths containing whitespace are unsupported")
    result = subprocess.run(["sudo", "-n", "exportfs", "-s"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("NFS server unavailable. Install nfs-kernel-server and ensure sudo access. " + result.stderr)
    exported = [line.split()[0] for line in result.stdout.splitlines() if line.strip()]
    if str(share) in exported:
        return
    identity = hashlib.sha256(str(share).encode()).hexdigest()[:12]
    content = str(share) + " " + " ".join(host + "(rw,sync,no_subtree_check)" for host in args.hosts[1:]) + "\n"
    subprocess.run(["sudo", "-n", "mkdir", "-p", "/etc/exports.d"], check=True)
    subprocess.run(["sudo", "-n", "tee", f"/etc/exports.d/nanoalphazero-{identity}.exports"],
                   input=content, text=True, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["sudo", "-n", "exportfs", "-ra"], check=True)
    subprocess.run(["sudo", "-n", "systemctl", "enable", "--now", "nfs-server"], check=True)


def validate_reports(args, reports, cpu):
    require_success(args, reports)
    for index, report in enumerate(reports):
        label = f"worker {index} ({args.hosts[index]})"
        if not report.get("repo_exists"):
            raise RuntimeError(label + ": shared checkout is missing; run pod setup")
        if not report.get("dependencies") or not report.get("setup"):
            raise RuntimeError(label + ": environment is missing; run pod setup")
        if report["setup"]["lock"] != report["lock"]:
            raise RuntimeError(label + ": lockfile changed; run pod setup")
        if any(report[key] != reports[0][key] for key in ("code", "dependencies", "lock")):
            raise RuntimeError(label + ": code/dependencies differ from worker 0; run pod setup")
        if not cpu and not report["tpu"]["available"]:
            raise RuntimeError(label + ": TPU occupied or unavailable:\n" + report["tpu"]["detail"])


def stream_logs(readers):
    """Copy bounded chunks to the controlling terminal, retaining worker logs."""
    for index, reader in enumerate(readers):
        for line in reader.read(65536).splitlines(keepends=True):
            print(f"[worker {index}] {line}", end="", flush=True)


def launch(args, request, directory):
    """SSH stdin is a renewable lease; losing it stops only that worker's job."""
    children, logs, readers = [], [], []
    started = heartbeat = time.monotonic()
    last_health = started
    result = 1
    try:
        for index in range(len(args.hosts)):
            log = (directory / f"worker-{index}.log").open("wb")
            logs.append(log)
            readers.append((directory / f"worker-{index}.log").open(errors="replace"))
            children.append(subprocess.Popen(ssh(args, index, {**request, "action": "run"}),
                                             stdin=subprocess.PIPE, stdout=log, stderr=log))
        print(f"Launched {len(children)} workers. Logs: {directory}", flush=True)
        print("Streaming worker output here. Ctrl+C stops this launch on every worker.", flush=True)
        while True:
            codes = [child.poll() for child in children]
            stream_logs(readers)
            failures = [(i, code) for i, code in enumerate(codes) if code not in (None, 0)]
            if failures:
                for index, code in failures:
                    print(f"Worker {index} ({args.hosts[index]}) failed: exit {code}; "
                          f"see {directory / f'worker-{index}.log'}", file=sys.stderr, flush=True)
                break
            if all(code is not None for code in codes):
                result = 0
                break
            now = time.monotonic()
            if args.timeout and now - started > args.timeout:
                print("Job timeout reached; stopping this launch", file=sys.stderr, flush=True)
                result = 124
                break
            if now - heartbeat >= 3:
                for child in children:
                    if child.poll() is None:
                        child.stdin.write(b".")
                        child.stdin.flush()
                heartbeat = now
            if now - last_health >= 30:
                # Filesystem health is checked through a separate bounded SSH
                # command. Never start persistent remount loops during training.
                health = request_all(args, {**request, "action": "health"}, timeout=10)
                require_success(args, health)
                last_health = time.monotonic()
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopping this launch", file=sys.stderr, flush=True)
        result = 130
    finally:
        for child in children:
            try:
                child.stdin.close()
            except BrokenPipeError:
                pass
        for child in children:
            try:
                child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                child.terminate()
                print("SSH cleanup timed out; worker lease expires after 20 seconds", file=sys.stderr)
        for log in logs:
            log.close()
        stream_logs(readers)
        for reader in readers:
            reader.close()
        (directory / "result.json").write_text(json.dumps({"exit_code": result,
            "workers": [child.poll() for child in children]}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--share", type=Path, help="Shared directory; defaults to /nfs_share when the repo is inside it")
    parser.add_argument("--ssh-option", action="append", default=[], help="Extra SSH argument (repeatable)")
    sub = parser.add_subparsers(dest="action", required=True)
    setup = sub.add_parser("setup", help="Mount the shared checkout and install each worker's local environment")
    setup.add_argument("--extra", action="append", default=[], help="Optional dependency extra, e.g. trackio")
    setup.add_argument("--reinstall", action="store_true", help="Repair an interrupted/corrupt dependency installation")
    status = sub.add_parser("status", help="Inspect workers without initializing JAX")
    status.add_argument("--json", action="store_true", help="Print full machine-readable worker reports")
    run = sub.add_parser("run", help="Run a command concurrently on every worker")
    run.add_argument("--cpu", action="store_true", help="Force CPU backend and skip TPU ownership requirement")
    run.add_argument("--timeout", type=float, help="Optional maximum job duration in seconds; no restart")
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    args.repo = str(args.repo.resolve())
    args.share = str((args.share or (Path("/nfs_share") if Path(args.repo).is_relative_to("/nfs_share") else Path(args.repo))).resolve())
    try:
        args.hosts = discover()
        request = {"action": args.action, "repo": args.repo, "share": args.share, "hosts": args.hosts}
        print(f"Discovered {len(args.hosts)} workers: {', '.join(args.hosts)}", flush=True)
        if args.action == "setup":
            configure_export(args)
            request["extras"] = args.extra
            request["reinstall"] = args.reinstall
            directory = Path(args.repo) / "logs" / "pod" / ("setup-" + uuid.uuid4().hex[:12])
            directory.mkdir(parents=True)
            print(f"Installing worker environments; tail -F {directory}/worker-*.log", flush=True)
            require_success(args, request_all(args, request, timeout=1100, logs=directory))
            reports = request_all(args, {**request, "action": "status"})
            validate_reports(args, reports, cpu=True)
            print("Shared checkout and local environments ready on every worker", flush=True)
            return 0
        if args.action == "status":
            reports = request_all(args, request)
            for index, report in enumerate(reports):
                label = f"worker {index} ({args.hosts[index]})"
                if args.json:
                    print(json.dumps({"worker": index, **report}, sort_keys=True))
                elif "error" in report:
                    print(label + ": " + report["error"])
                else:
                    mounted = "server" if index == 0 else ("mounted" if report["mount"] else "missing")
                    ready = report.get("setup", {}) or {}
                    environment = "ready" if ready.get("lock") == report.get("lock") and ready else "needs setup"
                    tpu = "free" if report["tpu"]["available"] else "occupied/unavailable"
                    print(f"{label}: NFS {mounted}; environment {environment}; TPU {tpu}; "
                          f"active jobs {len(report['jobs'])}")
            require_success(args, reports)
            return 0
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("Supply a command after run --")
        if args.timeout is not None and args.timeout <= 0:
            parser.error("--timeout must be positive")
        launch_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
        directory = Path(args.repo) / "logs" / "pod" / launch_id
        directory.mkdir(parents=True)
        probe = directory / "share-probe"
        nonce = uuid.uuid4().hex
        probe.write_text(nonce)
        request.update(probe=str(probe), nonce=nonce, cpu=args.cpu, launch=launch_id, command=command)
        reports = request_all(args, {**request, "action": "probe"})
        validate_reports(args, reports, args.cpu)
        for index in range(len(args.hosts)):
            if probe.with_name(probe.name + "." + str(index)).read_text() != nonce:
                raise RuntimeError(f"Worker {index} could not write to the shared filesystem")
        request.update(code=reports[0]["code"], dependencies=reports[0]["dependencies"])
        (directory / "launch.json").write_text(json.dumps(request, indent=2))
        return launch(args, request, directory)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print("pod: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
