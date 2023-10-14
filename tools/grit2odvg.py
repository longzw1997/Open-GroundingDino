import jsonlines
from tqdm import tqdm
import random
import json
import os
from multiprocessing import Pool
from functools import partial
import emoji

import argparse

def clean_span(span):
    span = span.rstrip()
    span = span.replace('"', "'").replace('\"', "'").replace('“', "'").replace('”', "'")
    span = span.replace('‘', "'").replace('’', "'").replace('–', "—")
    if span.endswith('/') or span.endswith('.'):
        span = span[:-1]
    return span

def check_caption(cap):
    check_anno = cap["caption"].rstrip()[:-1]
    if not str.isascii(check_anno):
        return False
    # "The view is better from here 🦅 (Chouf" wtf??
    check_list = {"↙️", "-", ",", " ", "*", "/", "$", "[CLS]", "[SEP]", "?"}
    for ch in check_list:
        if ch in check_anno: 
            return False
    if '.' in check_anno[:-1]:
        return False
    if emoji.emoji_count(check_anno):
        print(check_anno)
        return False
    return True

def get_regions(nc, anno):
    h = anno["height"]
    w = anno["width"]
    phrase = clean_span(anno["caption"][int(nc[0]):int(nc[1])])
    bbox = [round(nc[2]*w,2), round(nc[3]*h,2), round(nc[4]*w,2), round(nc[5]*h,2)]
    return {
        "bbox": bbox,
        "phrase": phrase
    }


def prepare_list(file_name: str, random_samples):
    with open(file_name, "r") as f:
        metas = [line.strip() for line in f]
    num_of_files = len(metas)
    print(num_of_files)
    metas = random.sample(metas, random_samples)
    num_of_files = len(metas)
    print("after sample:", num_of_files)
    return metas, num_of_files


def process_item(file, args):
    with open(os.path.join(args.root, file)) as f:
        anno = json.load(f)
    if not check_caption(anno):
        return None
    noun_chunks = anno['noun_chunks']
    ref_exps = anno['ref_exps']
    regions = []
    random_num = random.random()
    if random_num > 0.5:
        for nc in noun_chunks:
            region = get_regions(nc, anno)
            if str.isascii(region["phrase"]):
                regions.append(region)
    else:
        for re in ref_exps:
            region = get_regions(re, anno)
            if str.isascii(region["phrase"]):
                regions.append(region)
    if len(regions) < args.min_phrase:
        return None
    odvg_anno = {
        "filename": f'{file.split(".")[0]}.jpg',
        "height": anno["height"],
        "width": anno["width"],
        "grounding": { 
            "caption": clean_span(anno["caption"]),
            "regions": regions
        }
    }
    return odvg_anno

if __name__ == "__main__":
    # jsons = "/share_data/mllm/kosmos-2/GRIT-20M/anno/14m_anno.list"
    # root = "/share_data/mllm/kosmos-2/GRIT-20M/data"
    # output_name = "./girt_14m_odvg.jsonl"
    parser = argparse.ArgumentParser(description="GRIT2ODVG List.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--root", type=str, default="", help="Source image root")
    parser.add_argument("--output_file", type=str, default="girt_14m_odvg.jsonl")
    parser.add_argument("--random_samples", type=int, default=200000)
    parser.add_argument("--chunk_or_ref", type=float, default=0.5)
    parser.add_argument("--min_phrase", type=int, default=6)
    parser.add_argument("--process_num", type=int, default=10, help="the number of processes")
    args = parser.parse_args()
    print(args)
    metas, metas_len = prepare_list(args.input_file, args.random_samples)
    odvg_anno = []
    func = partial(process_item, args=args)
    with Pool(processes=args.process_num) as pool:
        for result in tqdm(pool.imap(func=func, iterable=metas), total=len(metas)):
            odvg_anno.append(result)
    odvg_anno = list(filter(None, odvg_anno))  
    with jsonlines.open(args.output_file, mode="w") as fwriter:
        fwriter.write_all(odvg_anno)