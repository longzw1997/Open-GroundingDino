import xml.etree.ElementTree as ET
import jsonlines
import random
from tqdm import tqdm
import argparse
import os
import glob

def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      fn - full file path to the sentence file to parse
    
    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this 
                                    phrase belongs to

    """
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {'sentence' : ' '.join(words), 'phrases' : []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append({'first_word_index' : index,
                                             'phrase' : phrase,
                                             'phrase_id' : p_id,
                                             'phrase_type' : p_type})

        annotations.append(sentence_data)

    return annotations

def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      fn - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the 
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    filename = root.findall('filename')[0].text
    size_container = root.findall('size')[0]
    anno_info = {'filename': filename, 'boxes' : {}, 'scene' : [], 'nobox' : []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_id = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                if box_id not in anno_info['boxes']:
                    anno_info['boxes'][box_id] = []
                xmin = int(box_container[0].findall('xmin')[0].text) - 1
                ymin = int(box_container[0].findall('ymin')[0].text) - 1
                xmax = int(box_container[0].findall('xmax')[0].text) - 1
                ymax = int(box_container[0].findall('ymax')[0].text) - 1
                anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall('nobndbox')[0].text)
                if nobndbox > 0:
                    anno_info['nobox'].append(box_id)

                scene = int(object_container.findall('scene')[0].text)
                if scene > 0:
                    anno_info['scene'].append(box_id)

    return anno_info

def gen_record(sd, an):
    filename = an["filename"]
    caption = sd["sentence"]
    regions = []
    for ph in sd["phrases"]:
        if ph["phrase_id"] in an["boxes"]:
            for box in an["boxes"][ph["phrase_id"]]:
                regions.append(
                    {
                        "phrase": ph["phrase"],
                        "bbox": box
                    }
                )
    if len(regions) < 1:
        print("no phrase regions")
        return None
    return {
        "filename": filename,
        "height": an["height"],
        "width": an["width"],
        "grounding":{
            "caption": caption,
            "regions": regions
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="flickr30k entities to ODVG List.")
    parser.add_argument("--root", type=str, default="", help="Source anno root")
    parser.add_argument("--output_file", type=str, default="flickr30k_entities_odvg.jsonl")
    parser.add_argument("--osoi", action="store_true", default=False)
    args = parser.parse_args()
    print(args)

    odvg_anno = []
    sentence_list = os.path.join(args.root, "Sentences")
    annotation_list = os.path.join(args.root, "Annotations")
    sentence_list = sorted(glob.glob(sentence_list + "/*"))
    annotation_list = sorted(glob.glob(annotation_list + "/*"))
    len_anno = len(annotation_list)
    for idx in tqdm(range(len_anno)):
        sds = get_sentence_data(sentence_list[idx])
        an = get_annotations(annotation_list[idx])
        if args.osoi:
            sd = sds[random.randint(0, len(sds)-1)] 
            x = gen_record(sd, an)
            if x:
                odvg_anno.append(x)
        else:
            for sd in sds:
                x = gen_record(sd, an)
                if x:
                    odvg_anno.append(x)
    with jsonlines.open(args.output_file, mode="w") as fwriter:
        fwriter.write_all(odvg_anno)