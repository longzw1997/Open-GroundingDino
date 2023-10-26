
- [ODVG Dataset Format](#odvg-dataset-format)
  - [Label map](#label-map)
- [Config file](#config-file)

# ODVG Dataset Format

The files are in jsonl format, with one json object per line, as follows:
- Object Detection datasets utilize the ``detection`` field. If dealing with an Object Detection dataset, an additional ``label_map`` is required in the Dataset settings.
- Visual Grounding datasets employ the ``grounding`` field.  

You can refer to the [tools](./tools/) to convert other anno formats to ovdg format.
```json
{
  "filename": "image.jpg",
  "height": 693,
  "width": 1024,
  "detection": {
      "instances": [     
        {
          "bbox": [262,210,323,338],   # [x1,y1,x2,y2]
          "label": 0,
          "category": "dog"
        },
        {
          "bbox": [164,263,252,371],
          "label": 1,
          "category": "cat"
        },
        {
          "bbox": [4,243,66,373],
          "label": 2,
          "category": "apple"
        }
      ]
  },
  "grounding": { 
      "caption": "a wire hanger with a paper cover that reads we heart our customers", 
      "regions": [
        {
          "bbox": [20,215,985,665],   # [x1,y1,x2,y2]
          "phrase": "a paper cover that reads we heart our customers"
        },
        { 
          "bbox": [19,19,982,671],
          "phrase": "a wire hanger"
        }
      ]
    }
}
```

## Label map

- In order to align with VG data, we need to provide an additional mapping table for OD data.
- In dictionary form, indices start from "0" (it is essential to start from 0 to accommodate caption/grounding data). [Here](./config/coco2017_label_map.json) is an example for dataset:

```json
{"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane", "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light", "10": "fire hydrant", "11": "stop sign", "12": "parking meter", "13": "bench", "14": "bird", "15": "cat", "16": "dog", "17": "horse", "18": "sheep", "19": "cow", "20": "elephant", "21": "bear", "22": "zebra", "23": "giraffe", "24": "backpack", "25": "umbrella", "26": "handbag", "27": "tie", "28": "suitcase", "29": "frisbee", "30": "skis", "31": "snowboard", "32": "sports ball", "33": "kite", "34": "baseball bat", "35": "baseball glove", "36": "skateboard", "37": "surfboard", "38": "tennis racket", "39": "bottle", "40": "wine glass", "41": "cup", "42": "fork", "43": "knife", "44": "spoon", "45": "bowl", "46": "banana", "47": "apple", "48": "sandwich", "49": "orange", "50": "broccoli", "51": "carrot", "52": "hot dog", "53": "pizza", "54": "donut", "55": "cake", "56": "chair", "57": "couch", "58": "potted plant", "59": "bed", "60": "dining table", "61": "toilet", "62": "tv", "63": "laptop", "64": "mouse", "65": "remote", "66": "keyboard", "67": "cell phone", "68": "microwave", "69": "oven", "70": "toaster", "71": "sink", "72": "refrigerator", "73": "book", "74": "clock", "75": "vase", "76": "scissors", "77": "teddy bear", "78": "hair drier", "79": "toothbrush"}
```

# Config file

- config spec:
  - The ``train`` supports multiple datasets for simultaneous training, and ``dataset_model`` needs to be set to ``odvg``. 
  - The ``val``  only supports datasets in the COCO format now, so ``dataset_model`` should be set to ``coco``, and ``label_map`` should be set to null.
- config example:
  - [datasets_mixed_odvg.json](config/datasets_mixed_odvg.json)
  - [datasets_od_example.json](config/datasets_od_example.json)
  - [datasets_vg_example.json](config/datasets_vg_example.json)

```json
{
  "train": [
    {
      "root": "path/coco_2017/train2017/",
      "anno": "path/coco_2017/annotations/coco2017_train_odvg.jsonl",
      "label_map": "path/coco_2017/annotations/coco2017_label_map.json",
      "dataset_mode": "odvg"
    }, 
    {
      "root": "path/GRIT-20M/data/",
      "anno": "path/GRIT-20M/anno/grit_odvg_10k.jsonl",
      "dataset_mode": "odvg"
    }
  ],
  "val": [
    {
      "root": "path/coco_2017/val2017",
      "anno": "config/instances_val2017.json",
      "label_map": null,
      "dataset_mode": "coco"
    }
  ]
}
```