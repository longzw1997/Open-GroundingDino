import argparse

import jsonlines
from tqdm import tqdm


def entmoot2odvg(args):
    with jsonlines.open(args.input) as reader:
        with jsonlines.open(args.output, mode="w") as writer:
            for obj in tqdm(reader):
                phrase_bbox = []
                for region in obj["grounding"]["regions"]:
                    for bbox in region["bbox"]:
                        phrase_bbox.append({"phrase": region["phrase"], "bbox": bbox})
                obj["grounding"]["regions"] = phrase_bbox

                writer.write(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("entmoot to odvg format.", add_help=True)
    parser.add_argument("--input", "-i", required=True, type=str, help="input")
    parser.add_argument("--output", "-o", required=True, type=str, help="output")
    args = parser.parse_args()

    entmoot2odvg(args)
