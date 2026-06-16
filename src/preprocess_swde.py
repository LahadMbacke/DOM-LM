import argparse
import json
import re
import sys
from pathlib import Path
import tqdm
import pickle
import multiprocessing as mp

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from src.preprocess import extract_features, extract_features_ae_task
from src.domlm import DOMLMConfig


def build_domain_label2id(groundtruth_dir: Path, domain: str) -> dict:
    """Build {O: 0, attr1: 1, ...} from groundtruth .txt filenames for a domain."""
    label2id = {"O": 0}
    for f in sorted((groundtruth_dir / domain).glob("*.txt")):
        attr = f.stem.rsplit("-", 1)[-1]
        if attr not in label2id:
            label2id[attr] = len(label2id)
    return label2id


def extract_labels(label_files):
    label_info = {}
    for file in label_files:
        label = file.name.split('-')[-1].replace('.txt', '')
        with open(file, 'r') as f:
            content = f.readlines()
            for line in content[2:]:
                page_id = line.split('\t')[0]
                if page_id not in label_info:
                    label_info[page_id] = {}
                nums = line.split('\t')[1]
                value = line.split('\t')[2].strip()
                label_info[page_id][label] = {
                    'nums': nums,
                    'value': value,
                }
    return label_info


def _process_one_file(args):
    path, proc_path, domain, config = args
    try:
        with open(path, 'r', errors='replace') as f:
            html = f.read()
        html = html.lstrip('﻿')
        html = re.sub(r'<\?xml[^>]*\?>', '', html)
        features = extract_features(html, config)
        dir_name = proc_path / domain / path.parent.name
        dir_name.mkdir(parents=True, exist_ok=True)
        out_path = dir_name / path.with_suffix(".pkl").name
        with open(out_path, 'wb') as f:
            pickle.dump(features, f)
        return None
    except Exception as e:
        return (path, str(e))


def preprocess_swde(input_dir, config, output_dir, domains, num_workers=1):
    SWDE_PATH = Path(input_dir)
    PROC_PATH = Path(output_dir)

    config = DOMLMConfig.from_json_file(config)

    for domain in domains:
        files = sorted((SWDE_PATH / domain).glob("**/*.htm"))
        task_args = [(p, PROC_PATH, domain, config) for p in files]
        errors = []
        print(f"[{domain}] {len(files)} files, {num_workers} workers")
        if num_workers > 1:
            with mp.Pool(num_workers) as pool:
                for result in tqdm.tqdm(pool.imap_unordered(_process_one_file, task_args), total=len(files)):
                    if result is not None:
                        errors.append(result)
        else:
            for args in tqdm.tqdm(task_args):
                result = _process_one_file(args)
                if result is not None:
                    errors.append(result)
        print(f"[{domain}] Total errors: {len(errors)}")
        for path, err in errors[:10]:
            print(f"  {path}: {err}")

def preprocess_swde_attr_extract(input_dir, config_file, output_dir, domains):
    SWDE_PATH = Path(input_dir)
    LABEL_PATH = SWDE_PATH / 'groundtruth'
    # Handle case where py7zr extracted into a nested groundtruth/ subdirectory
    if (LABEL_PATH / 'groundtruth').exists():
        LABEL_PATH = LABEL_PATH / 'groundtruth'
    PROC_PATH = Path(output_dir)

    config = DOMLMConfig.from_json_file(config_file)

    for domain in domains:
        label2id = build_domain_label2id(LABEL_PATH, domain)
        domain_out = PROC_PATH / domain
        domain_out.mkdir(parents=True, exist_ok=True)
        with open(domain_out / 'label2id.json', 'w') as f:
            json.dump(label2id, f)
        print(f"[{domain}] labels: {label2id}")

        errors = []
        for website_dir in sorted((SWDE_PATH / domain).iterdir()):
            if not website_dir.is_dir():
                continue
            files = sorted(website_dir.glob("./*.htm"))
            website_name = website_dir.name.split('-')[1][:website_dir.name.split('-')[1].index('(')]
            label_files = sorted((LABEL_PATH / domain).glob(f'{domain}-{website_name}*'))
            label_infos = extract_labels(label_files)
            pbar = tqdm.tqdm(files, total=len(files))
            for path in pbar:
                pbar.set_description(f"Processing {path.relative_to(SWDE_PATH / domain)}")
                with open(path, 'r', errors='replace') as f:
                    html = f.read()
                html = html.lstrip('﻿')
                html = re.sub(r'<\?xml[^>]*\?>', '', html)
                try:
                    label2text = label_infos[path.name.split('.')[0]]
                    text2label = {v['value']: {'label': k, 'nums': v['nums']} for k, v in label2text.items()}
                    features = extract_features_ae_task(html, text2label, config, label2id=label2id)
                    dir_name = PROC_PATH / domain / path.parent.name
                    dir_name.mkdir(parents=True, exist_ok=True)
                    with open(dir_name / path.with_suffix(".pkl").name, 'wb') as f:
                        pickle.dump(features, f)
                except Exception as e:
                    print(e)
                    errors.append(path)
        print(f"[{domain}] Total errors: {len(errors)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='domlm', choices=['domlm', 'attr_extract'])
    parser.add_argument('--input_dir', type=str, default='data/swde_html')
    parser.add_argument('--config', type=str, default='domlm-config/config.json')
    parser.add_argument('--output_dir', type=str, default='data/swde_preprocessed')
    parser.add_argument('--domains', type=str, default='auto,book,camera,job,movie,nbaplayer,restaurant,university')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count())
    args = parser.parse_args()

    if args.task == 'domlm':
        preprocess_swde(args.input_dir, args.config, args.output_dir, args.domains.split(','), args.num_workers)
    elif args.task == 'attr_extract':
        preprocess_swde_attr_extract(args.input_dir, args.config, args.output_dir, args.domains.split(','))
