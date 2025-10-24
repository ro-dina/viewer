# test_dicom_min.py
import argparse
from pydicom import dcmread
from pydicom.uid import UID
import pydicom
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="Compressed DICOM read test")
    ap.add_argument("path", help="Path to a DICOM file")
    args = ap.parse_args()

    ds = dcmread(args.path)

    ts = UID(ds.file_meta.TransferSyntaxUID)
    print(f"File: {args.path}")
    print(f"TransferSyntaxUID: {ts} ({ts.name})")
    if "SOPClassUID" in ds:
        print(f"SOPClassUID: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
    print(f"Has PixelData: {'PixelData' in ds}")

    # 利用可能なピクセルデータハンドラ一覧と、このTSをサポートするか
    print("\nPixel data handlers (availability / supports this TS):")
    for h in pydicom.config.pixel_data_handlers:
        name = h.__name__.split(".")[-1]
        avail = getattr(h, "is_available", lambda: False)()
        supports = getattr(h, "supports_transfer_syntax", lambda *_: False)(ts)
        print(f"  - {name:18}  available={avail!s:5}  supports_TS={supports!s:5}")

    # 実際にデコードしてみる
    try:
        arr = ds.pixel_array  # ここで解凍が走る（失敗すれば例外）
        print("\nDecoded pixel_array:")
        print(f"  shape={arr.shape}, dtype={arr.dtype}")
        # 数値の簡単な統計（巨大画像でも軽い範囲）
        flat = arr.ravel()
        sample = flat[: min(flat.size, 1_000_000)]  # 最大100万画素で統計
        print(f"  min={sample.min()}, max={sample.max()}, mean={float(sample.mean()):.3f}")
        # カラーかどうかの参考
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            print("  Looks like color image (last dim = 3 or 4).")
        print("\nSUCCESS: pixel_array decoded.")
    except Exception as e:
        print("\nFAILED to decode pixel_array.")
        print(type(e).__name__ + ":", e)

if __name__ == "__main__":
    main()