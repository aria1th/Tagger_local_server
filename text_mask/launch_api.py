from grounded_sam_demo import load_model, get_grounding_output
from segment_anything import SamPredictor, build_sam, build_sam_hq
# fastAPI
from fastapi import FastAPI

# pydantic
from pydantic import BaseModel

# uvicorn
import uvicorn

class Item(BaseModel):
    # image from numpy array of shape (H, W, 3)
    image: list[list[list[float]]]
    text_prompt: str
    box_threshold: float
    text_threshold: float
    return_json: bool = False

class ItemResponse(BaseModel):
    # returns mask as numpy array of shape (H, W)
    mask: list[list[int]]
    # optional json data
    json_data: list[dict] = []

model = None
predictor = None # SamPredictor
device = "cuda:0"
use_sam_hq = False

app = FastAPI()

def load_predictor(use_sam_hq, sam_checkpoint, sam_hq_checkpoint, device="cuda:0"):
    global predictor
    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

def reload_model(config_file, grounded_checkpoint, device="cuda:0"):
    global model
    model = load_model(config_file, grounded_checkpoint, device=device)

def check_model():
    return model is not None

# grounded_sam_demo.py functions
import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
# If everything is installed properly, you should be able to import all the classes below without any error
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def process_image(image_as_float_list):
    """
    Replacement for load_image if image is given as a list of floats
    image_as_float_list: list of float
    returns image_pil, image
    """
    # change to numpy array
    image_as_float = np.array(image_as_float_list)
    # convert to PIL image
    image_pil = Image.fromarray(image_as_float.astype(np.uint8))
    # convert to tensor
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def extract_mask(mask_list, box_list, label_list, return_json=False):
    """
    Given the output of get_grounding_output, merges the masks into one mask
    mask_list: list of torch.Tensor
    box_list: list of torch.Tensor
    label_list: list of str
    returns: {'mask': numpy array of shape (H, W) , 'json_data': list of dict}
    """
    value = 0  # 0 for background
    trues = 255 # 255 for foreground
    mask_img = torch.zeros(mask_list.shape[-2:], dtype=torch.uint8)
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = (trues if not return_json else value + idx + 1)
    if return_json:
        json_data = [{
            'value': value,
            'label': 'background'
        }]
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split('(')
            logit = logit[:-1] # the last is ')'
            json_data.append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.numpy().tolist(),
            })
        return {'mask': mask_img.numpy().tolist(), 'json_data': json_data}
    else:
        return {'mask': mask_img.numpy().tolist()}
            

def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
    return None


# root
@app.get("/")
def read_root():
    return {"Hello": "World"}

# fastAPI
@app.post("/predict/", response_model=ItemResponse)
def process(item: Item):
    image_arr, text_prompt, box_threshold, text_threshold, get_json = item.image, item.text_prompt, item.box_threshold, item.text_threshold, item.return_json
    global model, predictor
    assert model is not None and predictor is not None, "model and predictor must be loaded first"
    # load image
    image_pil, image = process_image(image_arr)
    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )
    # load from image arr
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    mask = extract_mask(masks, boxes_filt, pred_phrases, return_json=get_json)
    return mask


# def main(args):
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything API Server", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--port", type=int, default=10808, help="port number, default=10808")
    # ip
    parser.add_argument("--host", type=str, default="127.0.0.1", help="host ip, default=localhost")
    args = parser.parse_args()

    # load model
    reload_model(args.config, args.grounded_checkpoint, device=args.device)
    # load sam
    load_predictor(args.use_sam_hq, args.sam_checkpoint, args.sam_hq_checkpoint, device=args.device)
    # check model
    assert check_model(), "model must be loaded first"
    # run server
    uvicorn.run(app, host=args.host, port=args.port)

