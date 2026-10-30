"""Explicit, pinned, hash-verified model downloads. Importing does no I/O."""

import argparse, hashlib, json, os, re
from concurrent.futures import ThreadPoolExecutor
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "model_manifest.json"


def digest(path, progress=None, total=None):
    h = hashlib.sha256()
    size = 0
    if progress:
        progress({"phase": "verification", "bytes_done": 0, "bytes_total": total or path.stat().st_size})
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
            size += len(chunk)
            if progress:
                progress({"phase": "verification", "bytes_done": size, "bytes_total": total or path.stat().st_size})
    return h.hexdigest()


def checked_file(row, root):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", row.get("revision", "")):
        raise ValueError("Missing immutable revision")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", row.get("sha256") or ""):
        raise ValueError("Missing SHA256 for " + row.get("filename", "?"))
    if not isinstance(row.get("size_bytes"), int) or row["size_bytes"] <= 0:
        raise ValueError("Missing exact byte size")
    repo = row.get("repo_id", "")
    filename = row.get("filename", "")
    if (
        not re.fullmatch(r"[\w.-]+/[\w.-]+", repo)
        or not filename
        or ".." in Path(filename).parts
        or filename.startswith(("/", "\\"))
    ):
        raise ValueError("Invalid HF source")
    # Manifest destinations use POSIX separators on every host. Validate raw
    # input before native Path renders it with Windows backslashes.
    raw = row.get("destination", "")
    if (
        not isinstance(raw, str)
        or not raw
        or "\\" in raw
        or ":" in raw
        or "\x00" in raw
    ):
        raise ValueError("Invalid destination")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in raw.split("/")
    ):
        raise ValueError("Invalid destination")
    target = (root / Path(*relative.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Destination escapes models directory")
    url = f"https://huggingface.co/{repo}/resolve/{row['revision']}/{quote(filename, safe='/')}"
    return target, url


def verify(path, row, progress=None):
    return (
        path.is_file()
        and path.stat().st_size == row["size_bytes"]
        and digest(path, progress, row["size_bytes"]) == row["sha256"].lower()
    )


def install(row, root, progress=None, *, verify_only=False):
    target, url = checked_file(row, root)
    if verify(target, row, progress):
        return {"path": str(target), "status": "VERIFIED_EXISTING"}
    if verify_only:
        return {"path": str(target), "status": "MISSING_OR_MISMATCH"}
    if target.exists():
        raise ValueError("Existing file differs; refusing overwrite: " + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    # The immutable hash identifies resumable bytes across UI retries/restarts.
    temp = target.with_name(target.name + "." + row["sha256"][:12] + ".partial")
    h = hashlib.sha256()
    size = 0
    if temp.exists():
        with temp.open("rb") as existing:
            for block in iter(lambda: existing.read(8 * 1024**2), b""):
                h.update(block)
                size += len(block)
    if size > row["size_bytes"]:
        raise ValueError("Partial model is larger than the pinned file: " + str(temp))
    def fetch_range(start, end):
        data = bytearray()
        failures = 0
        while start + len(data) <= end:
            offset = start + len(data)
            headers = {"User-Agent": "kohaku-moss-verified-downloader/2",
                       "Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"}
            try:
                with urlopen(Request(url, headers=headers), timeout=60) as response:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if response.status != 206 or not match or tuple(map(int, match.groups())) != (offset, end, row["size_bytes"]):
                        raise ValueError("Model server returned an unexpected byte range")
                    while start + len(data) <= end:
                        try:
                            block = response.read(min(1024**2, end - start + 1 - len(data)))
                        except IncompleteRead as exc:
                            block = exc.partial
                        if not block:
                            raise ConnectionError("Model connection ended before its range completed")
                        data.extend(block)
            except HTTPError:
                raise
            except (OSError, URLError, IncompleteRead) as exc:
                failures += 1
                if failures >= 3:
                    raise OSError("下载连接中断；已保存的文件部分会在下次准备时继续，不会从头下载。" + str(exc)) from exc
        return data

    # Four bounded ranges overlap network latency; only contiguous verified-source
    # bytes are appended, so an interrupted process resumes at the file's length.
    chunk_size = 64 * 1024**2
    with temp.open("ab") as out, ThreadPoolExecutor(max_workers=4) as pool:
        while size < row["size_bytes"]:
            starts = range(size, min(size + 4 * chunk_size, row["size_bytes"]), chunk_size)
            futures = [pool.submit(fetch_range, start, min(start + chunk_size, row["size_bytes"]) - 1) for start in starts]
            for future in futures:
                block = future.result()
                out.write(block)
                out.flush()
                h.update(block)
                size += len(block)
                if progress:
                    progress({"phase": "download", "bytes_done": size, "bytes_total": row["size_bytes"]})
            futures.clear()
        os.fsync(out.fileno())
    if progress:
        progress({"phase": "verification", "bytes_done": size, "bytes_total": row["size_bytes"]})
    if h.hexdigest() != row["sha256"].lower():
        rejected = temp.with_suffix(".rejected")
        temp.replace(rejected)
        raise ValueError("模型完整 SHA256 不符，异常字节已保留为 .rejected；重新准备会下载新的文件。")
    if target.exists():
        if not verify(target, row):
            raise ValueError("Destination appeared during download; refusing overwrite")
    else:
        os.link(temp, target)
    temp.unlink()
    return {"path": str(target), "status": "DOWNLOADED_VERIFIED", "bytes": size}



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument(
        "--assets",
        nargs="+",
        help="Asset IDs, space or comma separated; defaults to Base INT8 + INT8 TE + original VAE",
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text())
    lookup = {x["id"]: x for x in manifest["assets"]}
    selected = [
        part
        for item in (a.assets or manifest["default_assets"])
        for part in item.split(",")
    ]
    plan = []
    # Validate ALL selected metadata before creating directories or fetching bytes.
    for id in dict.fromkeys(selected):
        asset = lookup.get(id)
        if not asset:
            ap.error("Unknown asset " + id)
        if asset["status"] != "public" or not asset["files"]:
            ap.error(id + " is local_only/publish_pending")
        for row in asset["files"]:
            target, url = checked_file(row, a.models_dir)
            plan.append((id, row, target, url))
    output = {
        "mode": "dry-run"
        if a.dry_run
        else "verify-only"
        if a.verify_only
        else "download",
        "total_selected_bytes": sum(x[1]["size_bytes"] for x in plan),
        "files": [],
    }
    for id, row, target, url in plan:
        result = (
            {
                "path": str(target),
                "url": url,
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "status": "PLANNED",
            }
            if a.dry_run
            else install(row, a.models_dir, verify_only=a.verify_only)
        )
        output["files"].append({"asset": id, **result})
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return (
        1 if any(x["status"] == "MISSING_OR_MISMATCH" for x in output["files"]) else 0
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
