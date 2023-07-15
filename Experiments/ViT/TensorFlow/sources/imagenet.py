import os
import shutil
import tarfile
import tempfile
import argparse

import scipy.io as sio

from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple
from pathlib import Path


@contextmanager
def get_tmp_dir() -> Iterator[str]:
    tmp_dir = tempfile.mkdtemp()
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


def parse_meta_mat(devkit_root: str) -> Tuple[Dict[int, str], Dict[str, Tuple[str, ...]]]:
    metafile = os.path.join(devkit_root, "data", "meta.mat")
    meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
    nums_children = list(zip(*meta))[4]
    #for idx, num_children in enumerate(nums_children):
    #    print(f"{idx} {num_children}")
    meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(", ")) for clss in classes]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
    return idx_to_wnid, wnid_to_classes


def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
    file = os.path.join(devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt")
    with open(file) as txtfh:
        val_idcs = txtfh.readlines()
    return [int(val_idx) for val_idx in val_idcs]


def get_instances(val_path, class_to_idx):
    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(val_path, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index
                instances.append(item)

                if target_class not in available_classes:
                    available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        raise FileNotFoundError(msg)

    return instances


def load_imagenet_val(extract_root, dataset_root=None):
    if dataset_root is None:
        print("Ignore this argument \"dataset_root\" means that all data are extracted.")
    else:
        if not isinstance(dataset_root, Path):
            dataset_root = Path(dataset_root)
        assert dataset_root.exists(), f"provided ImageNet2012 path {dataset_root.name} does not exist:\n Under the DIR you specified, there should be 2 TARs, respectively.\n - 1. \"ILSVRC2012_devkit_t12.tar.gz\"\n - 2. \"ILSVRC2012_img_val.tar\""

    extract_root = Path(extract_root)
    kit_root = extract_root.joinpath('ILSVRC2012_devkit_t12')
    val_root = extract_root.joinpath('ILSVRC2012_img_val')
    if not extract_root.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
    else:
        assert kit_root.exists() and val_root.exists(), f"provided ImageNet2012 path {extract_root.name} does not valid:\n Under the DIR you specified, there should be 2 DIRs, respectively.\n - 1. \"ILSVRC2012_devkit_t12\"\n - 2. \"ILSVRC2012_img_val\""

    if dataset_root is not None:
        kit_tar_path = dataset_root.joinpath("ILSVRC2012_devkit_t12.tar.gz")
        val_tar_path = dataset_root.joinpath("ILSVRC2012_img_val.tar")

    if not kit_root.is_dir():
        print("Begin Extracting DevKit...")
        kit_tar = tarfile.open(kit_tar_path)
        for filename in kit_tar.getnames():
            kit_tar.extract(filename, path=extract_root)
            print(f"Extracted {filename};")

        print("Extraction Finished!")

    if not val_root.is_dir():
        print("Begin Extracting ImgVal...")
        val_tar = tarfile.open(val_tar_path)
        for index, filename in enumerate(val_tar.getnames()):
            val_tar.extract(filename, path=val_root)
            if (index + 1) % 100 == 0:
                print(f"Extracted Total {index+1} Images;")

        print("Extraction Finished!")

    print("Parsing Meta Matrix...")
    idx_to_wnid, wnid_to_classes = parse_meta_mat(kit_root)
    print("Parsing Val Ground Truth...")
    val_idcs = parse_val_groundtruth_txt(kit_root)
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

    images = sorted(image for image in val_root.iterdir() if image.is_file())

    if len(images) == 50000:
        print("Organizing Val Images...")
        for val_wnid in set(val_wnids):
            val_root.joinpath(val_wnid).mkdir()

        print("Moving Val Images...")
        for val_wnid, img_file in zip(val_wnids, images):
            shutil.move(img_file, val_root.joinpath(val_wnid, img_file.name))

    classes = sorted(entry.name for entry in val_root.iterdir() if entry.is_dir())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    samples = get_instances(val_root, class_to_idx)
    targets = [s[1] for s in samples]

    val_wnids = classes
    wnid_to_idx = class_to_idx
    classes = [wnid_to_classes[wnid] for wnid in val_wnids]
    class_to_idx = {cls: idx for idx, clss in enumerate(classes) for cls in clss}

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ImageNet Val2012")
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    args = parser.parse_args()
    samples = load_imagenet_val(args.save_dir, args.dataset_path)
    print(len(samples))
