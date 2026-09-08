"""Fetch the official SceneFlow-only checkpoint, without executing downloaded code."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

URL = "https://www.dropbox.com/scl/fi/5khx1bhz84dapi8vtwapg/models.zip?rlkey=ggddrn1du1iiq6mgc2dsdpmwi&dl=1"
SOURCE = "https://github.com/princeton-vl/RAFT-Stereo/blob/main/download_models.sh"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/models/raftstereo-sceneflow.pth")
    a = p.parse_args()
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = output.parent / "raft-upstream-models.zip"
    if not output.exists():
        if not archive.exists():
            partial = archive.with_suffix(".part")
            total = 0
            with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as destination:
                while blob := response.read(1024*1024):
                    total += len(blob)
                    if total > 300_000_000:
                        raise ValueError("Archive exceeds expected bounded download size")
                    destination.write(blob)
            partial.replace(archive)
        with zipfile.ZipFile(archive) as z:
            matches = [i for i in z.infolist() if Path(i.filename).name == "raftstereo-sceneflow.pth"]
            if len(matches) != 1 or matches[0].file_size > 100_000_000:
                raise ValueError("Unexpected official archive contents")
            # Write a selected file to an explicit destination; never extract archive paths.
            output.write_bytes(z.read(matches[0]))
    report = {"source": SOURCE, "download_url": URL, "checkpoint": output.name,
        "training_provenance": "Official RAFT-Stereo SceneFlow checkpoint; no ETH3D fine tuning by this project",
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "bytes": output.stat().st_size,
        "hash_note": "Observed download hash, not an independently supplied upstream checksum"}
    output.with_suffix(".source.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
