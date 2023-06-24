# post to server to get mask
import requests
import os
import argparse
import numpy as np
import cv2

#class Item(BaseModel):
    # image from numpy array of shape (H, W, 3)
    #image: list[list[list[float]]]
    #text_prompt: str
    #box_threshold: float
    #text_threshold: float

def read_file(file_path):
    # image to np array
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def post_to_server(image_path, text_prompt, box_threshold, text_threshold, ip_address, port, get_json=False):
    assert box_threshold >= 0 and box_threshold <= 1, "box_threshold should be in [0, 1]"
    assert text_threshold >= 0 and text_threshold <= 1, "text_threshold should be in [0, 1]"
    assert os.path.exists(image_path), "image_path does not exist"
    assert isinstance(text_prompt, str), "text_prompt should be a string"
    assert isinstance(box_threshold, float), "box_threshold should be a float"
    assert isinstance(text_threshold, float), "text_threshold should be a float"
    assert isinstance(ip_address, str), "ip_address should be a string"
    # read image
    image = read_file(image_path)
    # image to list[list[list[float]]]
    image = image.tolist()
    # post to server
    url = f"http://{ip_address}:{port}/predict"
    data = {"image": image, "text_prompt": text_prompt, "box_threshold": box_threshold, "text_threshold": text_threshold, "get_json": get_json}
    response = requests.post(url, json=data)
    # get response, it has "mask" and "json"
    response = response.json()
    # response to list[list[list[float]]]
    mask = response["mask"]
    mask = np.array(mask)
    # mask to image
    return mask

def check_server(ip_address, port):
    url = f"http://{ip_address}:{port}/"
    response = requests.get(url)
    # check connection, it should return {"Hello": "World"}
    return response.json() == {"Hello": "World"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Mask Post", add_help=True)
    parser.add_argument("--image_path", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--ip_address", type=str, default="127.0.0.1", help="ip address")
    parser.add_argument("--port", type=int, default=10808, help="port")
    parser.add_argument("--get_json", action="store_true", help="get json")
    args = parser.parse_args()
    # example usage:
    # python post_get_mask.py --image_path ./data/1.jpg --text_prompt "a red apple" --box_threshold 0.3 --text_threshold 0.25
    # python post_get_mask.py --image_path "F:\\Github Desktop\\SDWebuiApiCommands\\text_mask\\test.png" --text_prompt "speech bubble. text. watermark" --box_threshold 0.3 --text_threshold 0.25
    # check server
    if not check_server(args.ip_address, args.port):
        print(f"server not running at {args.ip_address}:{args.port}")
        exit()
    image_path = args.image_path
    text_prompt = args.text_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    ip_address = args.ip_address
    port = args.port
    get_json = args.get_json
    mask = post_to_server(image_path, text_prompt, box_threshold, text_threshold, ip_address, port, get_json)
    # save mask to <image_path>.mask.png
    mask_path = image_path + ".mask.png"
    cv2.imwrite(mask_path, mask)
    print(f"mask saved to {mask_path}")

