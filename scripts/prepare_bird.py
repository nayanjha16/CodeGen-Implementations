"""Prepare BIRD dataset directory and run filtering.

This helper script will either copy files from a local `--source-dir` or
download and extract an archive from `--download-url` into `--dest-dir`
(default `data/bird`). After preparing the files it will run the existing
filtering logic from `schema_conversion.extract_schema` to create
`--filtered-dir/filtered.jsonl`.

Examples:
  python scripts/prepare_bird.py --source-dir /path/to/bird_raw
  python scripts/prepare_bird.py --download-url https://example.com/bird.zip

If the destination already exists, pass `--overwrite` to replace it.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable
import json

from schema_conversion import extract_schema as ext

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def copy_relevant_files(src: Path, dst: Path) -> None:
    """Copy SQL and JSON files from src into dst, preserving subdirs."""
    exts = {".sql", ".json", ".jsonl", ".nl", ".nlq", ".txt", ".question"}
    for root, _, files in os.walk(src):
        for f in files:
            if Path(f).suffix.lower() in exts:
                srcf = Path(root) / f
                # compute relative path and ensure parent exists
                rel = srcf.relative_to(src)
                destf = dst / rel
                destf.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(srcf, destf)


def download_and_extract(url: str, dst: Path) -> None:
    """Download an archive and extract into dst. Supports zip/tar.gz/tar."""
    tmpdir = Path(tempfile.mkdtemp(prefix="bird_dl_"))
    try:
        logging.info("Downloading %s ...", url)
        fname, _ = urllib.request.urlretrieve(url)
        logging.info("Downloaded to %s", fname)
        # try zip
        if zipfile.is_zipfile(fname):
            with zipfile.ZipFile(fname, 'r') as z:
                z.extractall(tmpdir)
        elif tarfile.is_tarfile(fname):
            with tarfile.open(fname, 'r:*') as t:
                t.extractall(tmpdir)
        else:
            # not an archive: save as-is into tmpdir
            dest = tmpdir / Path(url).name
            shutil.move(fname, dest)
        # copy relevant files from tmpdir to dst
        copy_relevant_files(tmpdir, dst)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def prepare_from_source(source: str, dst: Path, overwrite: bool = False) -> None:
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Source {source} does not exist")

    if dst.exists():
        if overwrite:
            shutil.rmtree(dst)
            dst.mkdir(parents=True, exist_ok=True)
        else:
            logging.info("Destination %s exists; adding files into it", dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)

    if src.is_file():
        # if file, try to treat as archive
        if zipfile.is_zipfile(src) or tarfile.is_tarfile(src):
            download_and_extract(str(src), dst)
        else:
            # assume a single SQL or JSON file; copy
            copy_relevant_files(src.parent, dst)
    else:
        copy_relevant_files(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BIRD dataset and run extractor to produce filtered data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-dir", help="Local directory containing raw BIRD files")
    group.add_argument("--download-url", help="URL to download BIRD archive (zip/tar.gz) or single file")
    group.add_argument("--hf-dataset", help="Hugging Face dataset id (e.g. sisinflab-ai/GradeSQL-training-dataset-bird-unbalanced)")
    parser.add_argument("--dest-dir", default="data/bird", help="Where to place raw BIRD files (default: data/bird)")
    parser.add_argument("--filtered-dir", default="data/bird_filtered", help="Where to write filtered output (default: data/bird_filtered)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it exists")
    args = parser.parse_args()

    dest = Path(args.dest_dir)
    if args.source_dir:
        logging.info("Preparing BIRD files from %s into %s", args.source_dir, dest)
        prepare_from_source(args.source_dir, dest, overwrite=args.overwrite)
    elif args.download_url:
        logging.info("Downloading and extracting %s into %s", args.download_url, dest)
        dest.mkdir(parents=True, exist_ok=True)
        download_and_extract(args.download_url, dest)
    else:
        # HF dataset path
        logging.info("Loading Hugging Face dataset %s into %s", args.hf_dataset, dest)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            from datasets import load_dataset

            ds = load_dataset(args.hf_dataset)
        except Exception as e:  # pragma: no cover - environment dependent
            logging.error("Failed to load HF dataset %s: %s", args.hf_dataset, e)
            return

        # For each split, write a JSONL file with db_id, nlq, sql
        for split, split_ds in ds.items():
            out_file = dest / f"hf_{split}.jsonl"
            logging.info("Writing %d examples to %s", len(split_ds), out_file)
            with out_file.open("w", encoding="utf-8") as fh:
                for rec in split_ds:
                    # Heuristic field extraction
                    sql = None
                    for key in ("sql", "query", "canonical_sql", "human_sql", "sql_query"):
                        if key in rec and rec[key]:
                            sql = rec[key]
                            break

                    nlq = None
                    for key in ("question", "nl", "nlq", "utterance", "text"):
                        if key in rec and rec[key]:
                            nlq = rec[key]
                            break

                    db_id = None
                    for key in ("db_id", "database_id", "database", "db"):
                        if key in rec and rec[key]:
                            db_id = rec[key]
                            break

                    if not sql:
                        # Try nested 'sql' inside 'query' objects
                        if isinstance(rec.get("query"), dict) and rec["query"].get("sql"):
                            sql = rec["query"]["sql"]

                    if sql:
                        obj = {"db_id": db_id or "", "nlq": nlq or "", "sql": sql}
                        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Now run the filtering logic (using the functions from extract_schema)
    logging.info("Running extractor on %s; output will be placed in %s", dest, args.filtered_dir)

    kept = []
    dropped = 0
    total = 0

    for item in ext.find_sql_and_nl_pairs(str(dest)):
        total += 1
        sql = item.get("sql", "")
        allowed, reason = ext.is_allowed_sql(sql)
        if allowed:
            item["sql"] = " ".join(item["sql"].split())
            kept.append({"db_id": item.get("db_id", ""), "nlq": item.get("nlq", ""), "sql": item.get("sql", "")})
        else:
            dropped += 1

    out_file = Path(args.filtered_dir) / "filtered.jsonl"
    ext.write_jsonl(kept, str(out_file))

    logging.info("Prepared BIRD: total=%d, kept=%d, dropped=%d", total, len(kept), dropped)
    logging.info("Filtered file written to %s", out_file)


if __name__ == "__main__":
    main()
