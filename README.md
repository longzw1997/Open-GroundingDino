<div align="center">
  <img src="figs/cute_dino.png" width="54%">
</div>


# Open GroundingDino

This is the third party implementation of the paper **[Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)** by [Zuwei Long]() and [Wei Li](https://github.com/bigballon)

**You can use this code to fine-tune a model on your own dataset, or start pretraining a model from scratch.**





***
# Supported Features

|                                | Official release version | The Version We Replicated |
| ------------------------------ | ------------------------ | ------------------------- |
| Inference                      | &#10004;                 | &#10004;                  |
| Train (Objecet Detection data) | &#10006;                 | &#10004;                  |
| Train (Grounding data)         | &#10006;                 | &#10004;                  |
| Slurm multi-machine support    | &#10006;                 | &#10004;                  |
| Training acceleration strategy | &#10006;                 | &#10004;                  |



# Setup
We test our models under ```python=3.7.11,pytorch=1.11.0,cuda=11.3```. Other versions might be available as well. 

1. Clone the GroundingDINO repository from GitHub.

```bash
git clone https://github.com/longzw1997/Open-GroundingDino.git
```

2. Change the current directory to the GroundingDINO folder.

```bash
cd Open-GroundingDino/
```

3. Install the required dependencies.

```bash
pip install -r requirements.txt 
cd models/GroundingDINO/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
```

4. Download pre-trained model weights.

```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

5.  Download BERT as the language model.


```bash
mkdir bert_weights
cd bert_weights
wget -q https://drive.google.com/drive/folders/1eM1HYf2K161YPzIcRDDMzE7S4WBGmDLM?usp=share_link
cd ..
```



# Dataset

- Dataset Format -> [data format](data_format.md)
- See [datasets_mixed_odvg.json](config/datasets_mixed_odvg.json) | [coco2odvg.py](./tools/coco2odvg.py) | [grit2odvg](./tools/grit2odvg.py) for more details




<details>
  <summary>mixed dataset</summary>
  </br>

``` json
{
  "train": [
    {
      "root": "path/V3Det/",
      "anno": "path/V3Det/annotations/v3det_2023_v1_all_odvg.jsonl",
      "label_map": "path/V3Det/annotations/v3det_label_map.json",
      "dataset_mode": "odvg"
    },
    {
      "root": "path/LVIS/train2017/",
      "anno": "path/LVIS/annotations/lvis_v1_train_odvg.jsonl",
      "label_map": "path/LVIS/annotations/lvis_v1_train_label_map.json",
      "dataset_mode": "odvg"
    },
    {
      "root": "path/Objects365/train/",
      "anno": "path/Objects365/objects365_train_odvg.json",
      "label_map": "path/Objects365/objects365_label_map.json",
      "dataset_mode": "odvg"
    },
    {
      "root": "path/coco_2017/train2017/",
      "anno": "path/coco_2017/annotations/coco2017_train_odvg.jsonl",
      "label_map": "path/coco_2017/annotations/coco2017_label_map.json",
      "dataset_mode": "odvg"
    },
    {
      "root": "path/GRIT-20M/data/",
      "anno": "path/GRIT-20M/anno/grit_odvg_620k.jsonl",
      "dataset_mode": "odvg"
    }, 
    {
      "root": "path/flickr30k/images/flickr30k_images/",
      "anno": "path/flickr30k/annotations/flickr30k_entities_odvg_158k.jsonl",
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

</details>

<details>
  <summary>example for odvg dataset</summary>
  </br>

``` json
{"filename": "000000391895.jpg", "height": 360, "width": 640, "detection": {"instances": [{"bbox": [359.17, 146.17, 471.62, 359.74], "label": 3, "category": "motorcycle"}, {"bbox": [339.88, 22.16, 493.76, 322.89], "label": 0, "category": "person"}, {"bbox": [471.64, 172.82, 507.56, 220.92], "label": 0, "category": "person"}, {"bbox": [486.01, 183.31, 516.64, 218.29], "label": 1, "category": "bicycle"}]}}
{"filename": "000000522418.jpg", "height": 480, "width": 640, "detection": {"instances": [{"bbox": [382.48, 0.0, 639.28, 474.31], "label": 0, "category": "person"}, {"bbox": [234.06, 406.61, 454.0, 449.28], "label": 43, "category": "knife"}, {"bbox": [0.0, 316.04, 406.65, 473.53], "label": 55, "category": "cake"}, {"bbox": [305.45, 172.05, 362.81, 249.35], "label": 71, "category": "sink"}]}}

{"filename": "014127544.jpg", "height": 400, "width": 600, "grounding": {"caption": "Homemade Raw Organic Cream Cheese for less than half the price of store bought! It's super easy and only takes 2 ingredients!", "regions": [{"bbox": [5.98, 2.91, 599.5, 396.55], "phrase": "Homemade Raw Organic Cream Cheese"}]}}
{"filename": "012378809.jpg", "height": 252, "width": 450, "grounding": {"caption": "naive : Heart graphics in a notebook background", "regions": [{"bbox": [93.8, 47.59, 126.19, 77.01], "phrase": "Heart graphics"}, {"bbox": [2.49, 1.44, 448.74, 251.1], "phrase": "a notebook background"}]}}
```

</details>

# Config

```
config/cfg_odvg.py                   # for backbone, batch size, LR, freeze layers, etc.
config/datasets_mixed_odvg.json      # support mixed dataset for both OD and VG
```

# Training

* Before starting the training, you need to modify the  ''config/datasets_vg_example.json'' according to ''data_format.md''
* The evaluation code defaults to using coco_val2017 for evaluation. If you are evaluating with your own test set, you need to convert the test data to coco format (not the ovdg format in data_format.md), and modify the config to set use_coco_eval = False. (The COCO dataset has 80 classes used for training but 90 categories in total, so there is a built-in mapping in the code.)


```  bash
#train on slrum：
bash train_multi_node.sh  ${PARTITION} ${GPU_NUM} ${CFG} ${DATASETS} ${OUTPUT_DIR}

# e.g.  check train_multi_node.sh for more details
# bash train_multi_node.sh v100_32g 32 config/cfg_odvg.py config/datasets_mixed_odvg.json ./logs
# bash train_multi_node.sh v100_32g 8 config/cfg_coco.py config/datasets_od_example.json ./logs
# bash train_multi_node.sh v100_32g 8 config/cfg_odvg.py config/datasets_vg_example.json ./logs



#train on dist：

bash train_dist.sh  ${GPU_NUM} ${CFG} ${DATASETS} ${OUTPUT_DIR}


# e.g.  check train_dist.sh for more details
# bash train_dist.sh  8 config/cfg_odvg.py config/datasets_mixed_odvg.json ./logs
# bash train_dist.sh  8 config/cfg_coco.py config/datasets_od_example.json ./logs
# bash train_dist.sh  8 config/cfg_odvg.py config/datasets_vg_example.json ./logs


#eval：
bash test.sh  ${PARTITION} ${GPU_NUM} ${CFG} ${DATASETS} ${OUTPUT_DIR}

# e.g.  check train_multi_node.sh for more details
# bash train_multi_node.sh v100_32g 32 config/cfg_odvg.py config/datasets_mixed_odvg.json ./logs
# bash train_multi_node.sh v100_32g 8 config/cfg_coco.py config/datasets_od_example.json ./logs
# bash train_multi_node.sh v100_32g 8 config/cfg_odvg.py config/datasets_vg_example.json ./logs
```



# Results and Models

<!-- insert a table -->
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Backbone</th>
      <th>Style</th>
      <th>Pretrain data</th>
      <th>mAP on COCO</th>
      <th>Checkpoint</th>
      <th>Config</th>
      <th>log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>GroundingDINO-T (offical)</td>
      <td>Swin-T</td>
      <td>zero-shot</td>
      <td>O365,GoldG,Cap4M</td>
      <td>48.4 (zero-shot) </td>
      <td><a href="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth">GitHub link</a> 
      <td>link</a></td>
      <td>link</a></td>
    </tr>
        <tr>
      <th>2</th>
      <td>GroundingDINO-T (finetune) </td>
      <td>Swin-T</td>
      <td>use coco finetune</td>
      <td>O365,GoldG,Cap4M</td>
      <td>57.3 (fine-tune)</td>
      <td><a href="https://drive.google.com/file/d/1H9xWCUr1vhrxM9lvENfJUxsXv44JaDee/view?usp=drive_link">GitHub link</a> 
      <td><a href="https://drive.google.com/file/d/1TJRAiBbVwj3AfxvQAoi1tmuRfXH1hLie/view?usp=drive_link">link</a></td>
      <td><a href="https://drive.google.com/file/d/1u8XyvBug56SrJY85UtMZFPKUIzV3oNV6/view?usp=drive_link">link</a></td>
    </tr>
    <tr>
      <th>3</th>
      <td>GroundingDINO-T (pretrain)</td>
      <td>Swin-T</td>
      <td>zero-shot</td>
      <td>COCO,Objects365,LIVS,V3Det,GRIT-200K,Flickr30k  (total 1.8M)</td>
      <td>55.1 (zero-shot)</td>
      <td><a href="https://drive.google.com/file/d/1ayAVNuIXXTGSJv7AyECdibFJVhlFjyux/view?usp=drive_link">GitHub link</a>  
      <td><a href='https://drive.google.com/file/d/1LwtkvBHkP1OkErKBsVfwjcedVXkyocA5/view?usp=drive_link'>link</a></td>
      <td><a href="https://drive.google.com/file/d/1kBEFk14OqcYHC7DPdA_BGtk2TBQkJtrL/view?usp=drive_link">link</a></td>
    </tr>
  </tbody>
</table>
GRIT-200K generated by GLIP and spaCy  



# Contact

- longzuwei at sensetime.com  
- liwei1 at sensetime.com  

Any discussions, suggestions and questions are welcome!
# Acknowledgments

Provided codes were adapted from:

- [microsoft/GLIP](https://github.com/microsoft/GLIP)
- [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO/)
- [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)





# Citation

```
@misc{Open Grounding Dino,
  author = {Zuwei Long,Wei Li},
  title = {Open Grounding Dino:The third party implementation of the paper Grounding DINO},
  howpublished = {\url{https://github.com/longzw1997/Open-GroundingDino}},
  year = {2023}
}
```

Feel free to contact me if you have any suggestions or questions, issues are welcome,
create a PR if you find any bugs or you want to contribute. 
