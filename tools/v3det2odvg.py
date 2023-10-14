import argparse
import jsonlines
from tqdm import tqdm
import json
from pycocotools.coco import COCO

def dump_label_map(args):
    coco = COCO(args.input) 
    cats = coco.loadCats(coco.getCatIds())
    nms = {cat['id']-1:cat['name'] for cat in cats}
    with open(args.output,"w") as f:
        json.dump(nms, f)

def coco_to_xyxy(bbox):
    x, y, width, height = bbox
    x1 = round(x, 2) 
    y1 = round(y, 2)
    x2 = round(x + width, 2)
    y2 = round(y + height, 2)
    return [x1, y1, x2, y2]


def coco2odvg(args):
    coco = COCO(args.input) 
    cats = coco.loadCats(coco.getCatIds())
    nms = {cat['id']:cat['name'] for cat in cats}
    metas = []
    for img_id, img_info in tqdm(coco.imgs.items()):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        instance_list = []
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            bbox = ann['bbox']
            bbox_xyxy = coco_to_xyxy(bbox)
            label = ann['category_id']
            category = nms[label]
            instance_list.append({
                "bbox": bbox_xyxy,
                "label": label - 1,       # make sure start from 0
                "category": category
                }
            )
        metas.append(
            {
                "filename": img_info["file_name"],
                "height": img_info["height"],
                "width": img_info["width"],
                "detection": {
                    "instances": instance_list
                }
            }
        )
    print("  == dump meta ...")
    with jsonlines.open(args.output, mode="w") as writer:
        writer.write_all(metas)
    print("  == done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("coco to odvg format.", add_help=True)
    parser.add_argument("--input", '-i', required=True, type=str, help="input list name")
    parser.add_argument("--output", '-o', required=True, type=str, help="output list name")
    parser.add_argument("--output_label_map", '-olm', action="store_true", help="output label map or not")
    args = parser.parse_args()

    if args.output_label_map:
        dump_label_map(args)
    else:
        coco2odvg(args)