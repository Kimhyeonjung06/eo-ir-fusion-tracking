"""KAIST Multispectral Pedestrian 데이터 내려받기.

라이선스가 CC BY-NC-SA 4.0이므로 데이터는 이 저장소에 재배포하지 않는다.
원 배포처(https://github.com/SoonminHwang/rgbt-ped-detection)의 구글 드라이브 파일을 받는다.

주의: 배포 파일은 확장자가 .zip으로 안내되어 있으나 실제로는 gzip 압축 tar이다.
      이 스크립트는 매직 바이트를 보고 형식을 판별해 해제한다.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

PREVIEW_ID = "11nhHpmuh2FUjrLNfGs51R2Mqqy1GTjY8"   # 약 1.5GB
FULL_ID = "1sBcAmFqNJmNMBZdMtKmO2X4BRjKPyKMc"      # 약 36GB


def sniff(path: Path) -> str:
    with path.open("rb") as f:
        head = f.read(4)
    if head[:2] == b"\x1f\x8b":
        return "tar.gz"
    if head[:2] == b"PK":
        return "zip"
    return "unknown"


def extract(archive: Path, target: Path) -> None:
    kind = sniff(archive)
    target.mkdir(parents=True, exist_ok=True)
    if kind == "tar.gz":
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(target)
    elif kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    else:
        raise RuntimeError("알 수 없는 압축 형식: %s" % archive)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="전체 세트(약 36GB)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fid = FULL_ID if args.full else PREVIEW_ID
    archive = out / ("kaist_full.tgz" if args.full else "kaist_preview.tgz")

    if not archive.is_file():
        legacy = archive.with_suffix(".zip")
        if legacy.is_file():
            legacy.rename(archive)
        else:
            subprocess.run([sys.executable, "-m", "gdown", fid, "-O", str(archive)], check=True)

    if args.no_extract:
        print("다운로드 완료:", archive)
        return 0

    target = out / archive.stem
    if not target.is_dir():
        print("해제 중 ->", target)
        extract(archive, target)
    print("완료:", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
