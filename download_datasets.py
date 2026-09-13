#!/usr/bin/env python3
"""
download_datasets.py — Script per scaricare e preparare i dataset.

Esegui su una macchina con accesso internet completo:
    python download_datasets.py --dataset coco --split val
    python download_datasets.py --dataset voc
    python download_datasets.py --dataset all

Requisiti: pip install gdown pycocotools fiftyone tqdm
"""

import os
import sys
import argparse
import zipfile
import tarfile
import json
from pathlib import Path
from tqdm import tqdm

try:
    import urllib.request
except ImportError:
    pass

# ─── Configuration ────────────────────────────────────────────────────────────

DATASETS = {
    "coco": {
        "val_images": {
            "url": "http://images.cocodataset.org/zips/val2017.zip",
            "size_gb": 1.0,
            "output": "data/coco/val2017.zip",
        },
        "annotations": {
            "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "size_gb": 0.25,
            "output": "data/coco/annotations_trainval2017.zip",
        },
    },
    "voc": {
        "trainval": {
            "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
            "size_gb": 2.0,
            "output": "data/voc/VOCtrainval_11-May-2012.tar",
        },
    },
}


def download_file(url: str, output_path: str, desc: str = ""):
    """Download con progress bar."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print(f"  [SKIP] {output_path} esiste già")
        return

    print(f"  Downloading {desc or url}...")

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

    print(f"  [OK] Salvato in {output_path}")


def extract_archive(archive_path: str, extract_dir: str):
    """Estrai zip o tar."""
    print(f"  Extracting {archive_path}...")
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as z:
            z.extractall(extract_dir)
    elif archive_path.endswith('.tar') or archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:*') as t:
            t.extractall(extract_dir)
    print(f"  [OK] Estratto in {extract_dir}")


def create_conformal_split(ann_file: str, output_dir: str, cal_ratio: float = 0.5, seed: int = 42):
    """
    Divide il validation set di COCO in calibration e test set
    per la Split Conformal Prediction.
    """
    import random
    random.seed(seed)

    print(f"\n  Creazione split conforme (cal={cal_ratio:.0%}, test={1-cal_ratio:.0%})...")

    with open(ann_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    image_ids = [img['id'] for img in coco['images']]
    random.shuffle(image_ids)

    n_cal = int(len(image_ids) * cal_ratio)
    cal_ids = set(image_ids[:n_cal])
    test_ids = set(image_ids[n_cal:])

    # Build calibration and test annotation files
    for split_name, split_ids in [("calibration", cal_ids), ("test", test_ids)]:
        split_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": [img for img in coco["images"] if img["id"] in split_ids],
            "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in split_ids],
        }

        out_path = os.path.join(output_dir, f"instances_val2017_{split_name}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(split_coco, f)

        n_imgs = len(split_coco["images"])
        n_anns = len(split_coco["annotations"])
        print(f"    {split_name}: {n_imgs} immagini, {n_anns} annotazioni → {out_path}")


def download_with_fiftyone(dataset_name: str, split: str = "validation"):
    """Alternativa: usa FiftyOne per scaricare (gestisce tutto automaticamente)."""
    try:
        import fiftyone.zoo as foz
        print(f"\n  Download via FiftyOne: {dataset_name} [{split}]...")
        dataset = foz.load_zoo_dataset(
            dataset_name,
            split=split,
            dataset_dir=f"data/{dataset_name.replace('-', '_')}",
        )
        print(f"  [OK] {len(dataset)} campioni caricati")
        return dataset
    except ImportError:
        print("  [WARN] FiftyOne non installato. Usa: pip install fiftyone")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download datasets per Conformal Calibration Pipeline")
    parser.add_argument("--dataset", choices=["coco", "voc", "all"], default="coco")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--method", choices=["direct", "fiftyone"], default="direct")
    parser.add_argument("--cal-ratio", type=float, default=0.5, help="Rapporto calibration/test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).parent
    os.chdir(project_root)

    datasets_to_download = ["coco", "voc"] if args.dataset == "all" else [args.dataset]

    for ds_name in datasets_to_download:
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds_name.upper()}")
        print(f"{'='*60}")

        if args.method == "fiftyone":
            download_with_fiftyone(
                "coco-2017" if ds_name == "coco" else "voc-2012",
                split="validation"
            )
        else:
            for part_name, part_info in DATASETS[ds_name].items():
                download_file(
                    part_info["url"],
                    part_info["output"],
                    desc=f"{ds_name}/{part_name} (~{part_info['size_gb']:.1f} GB)"
                )
                extract_dir = os.path.dirname(part_info["output"])
                extract_archive(part_info["output"], extract_dir)

        # Create conformal splits for COCO
        if ds_name == "coco":
            ann_file = "data/coco/annotations/instances_val2017.json"
            if os.path.exists(ann_file):
                create_conformal_split(
                    ann_file,
                    "data/coco/annotations/",
                    cal_ratio=args.cal_ratio,
                    seed=args.seed
                )

    print(f"\n{'='*60}")
    print("  DOWNLOAD COMPLETATO")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
