import hashlib
import json
import re
from datetime import datetime


def hash_file(path: str) -> str:
    sha = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha.update(chunk)

    return sha.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_report(report_text: str) -> str:
    """
    Creates a stable version of the report for hashing.

    Generated timestamps and proof metadata are normalized so that the
    final report can contain its own proof values without creating a
    self-referential hash problem.
    """
    canonical = report_text.replace("\r\n", "\n")

    replacements = [
        (
            r"^\*\*Generated at:\*\* .*$",
            "**Generated at:** <normalized>",
        ),
        (
            r"^- Dataset fingerprint:.*$",
            "- Dataset fingerprint: <normalized>",
        ),
        (
            r"^- (?:Canonical )?[Rr]eport hash:.*$",
            "- Canonical report hash: <normalized>",
        ),
        (
            r"^- Execution timestamp:.*$",
            "- Execution timestamp: <normalized>",
        ),
    ]

    for pattern, replacement in replacements:
        canonical = re.sub(
            pattern,
            replacement,
            canonical,
            flags=re.MULTILINE,
        )

    return canonical.strip() + "\n"


def create_proof(csv_path: str, report_text: str) -> dict:
    canonical_report = canonicalize_report(report_text)

    return {
        "dataset_fingerprint": hash_file(csv_path),
        "report_hash": hash_text(canonical_report),
        "execution_timestamp": datetime.utcnow().isoformat() + "Z",
        "proof_type": "SHA256 dataset fingerprint + canonical report body hash",
        "canonicalization": (
            "Generated timestamp and proof metadata are normalized before "
            "the report hash is calculated."
        ),
    }


def verify_proof(csv_path: str, report_text: str, proof: dict) -> dict:
    actual_dataset_hash = hash_file(csv_path)
    actual_report_hash = hash_text(canonicalize_report(report_text))

    dataset_fingerprint_valid = (
        actual_dataset_hash == proof.get("dataset_fingerprint")
    )
    canonical_report_hash_valid = (
        actual_report_hash == proof.get("report_hash")
    )

    return {
        "dataset_fingerprint_valid": dataset_fingerprint_valid,
        "canonical_report_hash_valid": canonical_report_hash_valid,
        "valid": dataset_fingerprint_valid and canonical_report_hash_valid,
        "expected_dataset_fingerprint": proof.get("dataset_fingerprint"),
        "actual_dataset_fingerprint": actual_dataset_hash,
        "expected_report_hash": proof.get("report_hash"),
        "actual_report_hash": actual_report_hash,
    }


def save_json(data: dict, path: str):
    with open(path, "w") as file:
        json.dump(data, file, indent=2)
