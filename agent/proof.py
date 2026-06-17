import hashlib
import json
from datetime import datetime


def hash_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_proof(csv_path: str, report_text: str) -> dict:
    return {
        "dataset_fingerprint": hash_file(csv_path),
        "report_hash": hash_text(report_text),
        "execution_timestamp": datetime.utcnow().isoformat() + "Z",
        "proof_type": "SHA256 dataset fingerprint + report hash"
    }


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
