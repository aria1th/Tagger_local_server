# using WD14 captioning to caption the image

from typing import List
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model as load_model_hf
from huggingface_hub import hf_hub_download
import torch
import os
import requests
import csv
import argparse

no_server = True
IMAGE_SIZE = 448
INTERPOLATION = cv2.INTER_LANCZOS4
REPOSITORY = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
PORT_NUMBER = 10080
IP = "localhost"
SERVER = None

# global params, models, general_tags, character_tags, rating_tags
model = None
general_tags:list|None = None
character_tags:list|None = None

def read_tags(base_path):
    """
    Reads tags from selected_tags.csv, and stores them in global variables
    base_path: base path to model (str)
    return: None
    """
    global general_tags, character_tags
    if general_tags is not None and character_tags is not None:
        return None
    with open(os.path.join(base_path, 'selected_tags.csv'), "r", encoding='utf-8') as f:
        reader = csv.reader(f)
        tags = list(reader)
        header = tags.pop(0)
        tags = tags[1:]
    assert header[0] == 'tag_id' and header[1] == 'name' and header[2] == 'category', f"header is not correct for {base_path} selected_tags.csv"
    # if category is 0, general, 4, character, else ignore
    general_tags = [tag[1] for tag in tags if tag[2] == '0']
    character_tags = [tag[1] for tag in tags if tag[2] == '4']
    return None

def preprocess_image(image) -> np.ndarray:
    global IMAGE_SIZE, INTERPOLATION
    image = np.array(image)
    image = image[:, :, ::-1].copy() # RGB to BGR
    
    # pad to square image
    target_size = [max(image.shape)] * 2
    # pad with 255 to make it white
    image_padded = 255 * np.ones((target_size[0], target_size[1], 3), dtype=np.uint8)
    dw = int((target_size[0] - image.shape[1]) / 2)
    dh = int((target_size[1] - image.shape[0]) / 2)
    image_padded[dh:image.shape[0]+dh, dw:image.shape[1]+dw, :] = image
    image = image_padded
    # assert
    assert image.shape[0] == image.shape[1]
    
    # resize
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=INTERPOLATION)
    image = image.astype(np.float32)
    return image

def download_model(repo_dir: str = REPOSITORY, save_dir: str = "./", force_download: bool = False):
    # tagger follows following files
    print("Downloading model")
    FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
    SUB_DIR = "variables"
    SUB_DIR_FILES = [f"{SUB_DIR}.data-00000-of-00001", f"{SUB_DIR}.index"]
    if os.path.exists(save_dir) and not force_download:
        return os.path.abspath(save_dir)
    # download
    for file in FILES:
        print(f"Downloading {file}")
        hf_hub_download(repo_dir, file, cache_dir = save_dir, force_download = force_download, force_filename = file)
    for file in SUB_DIR_FILES:
        print(f"Downloading {file}")
        hf_hub_download(repo_dir, (SUB_DIR+'/'+ file), cache_dir = os.path.join(save_dir, SUB_DIR), force_download = force_download, force_filename = file)
    return os.path.abspath(save_dir)

# check if model is already loaded in port 5050, if it is, use api
def check_model_loaded(port_number: int = PORT_NUMBER, ip: str = IP):
    """
    Checks if model is loaded in port_number
    port_number: port number to check (int)
    ip: ip address to check (str) default: localhost
    return: True if model is loaded, False otherwise (bool)
    """
    global no_server
    if no_server:
        # check if model is loaded
        return model is not None
    try:
        response = requests.get(f"http://f{IP}:{port_number}/")
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.ConnectionError:
        return False
    
def load_model(model_path: str = "", force_download: bool = False, port_number: int = PORT_NUMBER, ip: str = IP):
    """
    Loads model from model_path
    model_path: path to model (str)
    force_download: force download model (bool)
    port_number: port number to load model (int)
    return: None
    """
    if check_model_loaded(port_number, ip):
        return None
    if not model_path:
        raise ValueError("model_path is None")
    if not os.path.exists(model_path) or force_download:
        download_model(REPOSITORY, model_path, force_download = force_download)
    # load model
    global model
    print("Loading model")
    model = load_model_hf(model_path)
    return None

def predict_tags(prob_list:np.ndarray, threshold=0.5, model_path:str="./") -> List[str]:
    """
    Predicts tags from prob_list
    prob_list: list of probabilities, first 4 are ratings, rest are tags
    threshold: threshold for tags (float)
    model_path: path to model (str)
    return: list of tags (list of str)
    """
    global model, general_tags, character_tags
    probs = np.array(prob_list)
    #ratings = probs[:4] # first 4 are ratings
    #rating_index = np.argmax(ratings)
    tags = probs[4:] # rest are tags
    if general_tags is None or character_tags is None:
        read_tags(model_path)
    assert general_tags is not None and character_tags is not None, "general_tags and character_tags are not loaded"
    result = []
    for i, p in enumerate(tags):
        if i < len(general_tags) and p > threshold:
            tag_name = general_tags[i]
            # replace _ with space
            tag_name = tag_name.replace("_", " ")
            result.append(tag_name)
    return result

def predict_image(image: np.ndarray, model_path: str = "./") -> List[str]:
    """
    Predicts image from image
    image: image to predict (np.ndarray)
    model_path: path to model (str)
    return: list of tags (list of str)
    """
    global model
    if model is None:
        load_model(model_path)
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    probs = model.predict(image)[0]
    return predict_tags(probs, model_path=model_path)

def predict_images(images: List[np.ndarray], model_path: str = "./") -> List[List[str]]:
    """
    Predicts images from images
    images: images to predict (list of np.ndarray)
    model_path: path to model (str)
    return: list of tags (list of list of str)
    """
    global model
    if model is None:
        load_model(model_path)
    images = [preprocess_image(image) for image in images]
    images = np.array(images)
    probs = model.predict(images)
    return [predict_tags(prob, model_path=model_path) for prob in probs]

# for image paths, locally
def predict_images_from_path(image_paths: List[str], model_path: str = "./", port:int = PORT_NUMBER, ip:str = IP) -> List[List[str]]:
    """
    Predicts images from image_paths
    image_paths: paths to images (list of str)
    model_path: path to model (str)
    return: list of tags (list of list of str)
    """
    # check if model is loaded
    if not check_model_loaded(port, ip):
        global model
        if model is None:
            load_model(os.path.abspath(model_path))
    images = [Image.open(image_path) for image_path in image_paths]
    return predict_images(images, model_path=model_path)

def post_and_get(paths:list[str], ip, port, save:bool=False):
    # post to server and get
    global no_server
    if no_server:
        # no server, use predict_images_from_path
        return predict_images_from_path(paths, port=port, ip=ip)
    results = []
    for path in paths:
        image = cv2.imread(path)
        # send, preprocessing will be done in server
        response = requests.post(f"http://{ip}:{port}/predict", json={"image": image.tolist()})
        if response.status_code == 200:
            results.append(response.json()["tags"])
        else:
            results.append([])
    if save:
        # save to filename.txt
        for files in zip(paths, results):
            pure_filename = os.path.splitext(os.path.basename(files[0]))[0]
            with open(f"{pure_filename}.txt", "w") as f:
                f.write(",".join(files[1]))
    return results

# using glob, get all images in a folder then request
def predict_local_path(path:str, model_path:str="./", port:int=PORT_NUMBER, ip:str=IP, save:bool=False) -> List[List[str]]:
    """
    Predicts images from path
    path: path to folder (str)
    model_path: path to model (str)
    return: list of tags (list of list of str)
    """
    # get all images in path
    import glob
    paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        paths.extend(glob.glob(os.path.join(path, ext)))
    # post and get
    result = post_and_get(paths, ip, port)
    if save:
        # save to filename.txt
        for files in zip(paths, result):
            pure_filename = os.path.splitext(os.path.basename(files[0]))[0]
            with open(f"{pure_filename}.txt", "w") as f:
                f.write(",".join(files[1]))
    return result

def shutdown_local_server():
    if os.path.exists("server.pid"):
        with open("server.pid", "r") as f:
            pid = f.read()
        import subprocess
        # if os is windows, use taskkill
        if os.name == "nt":
            subprocess.Popen(["taskkill", "/F", "/PID", pid])
        else:
            subprocess.Popen(["kill", "-9", pid])
        os.remove("server.pid")
    global SERVER
    if SERVER is not None:
        import subprocess
        # SERVER is pid of subprocess
        subprocess.Popen(["kill", "-9", str(SERVER)])
        SERVER = None

def start_local_server(model_path:str="./", port:int=PORT_NUMBER, ip:str=IP):
    """
    Starts local server
    model_path: path to model (str)
    port: port number to start server (int)
    ip: ip address to start server (str)
    return: None
    """
    print("Starting local server")
    import subprocess
    import time
    # check if model is loaded
    if not check_model_loaded(port, ip):
        # call api.py with subprocess and store as global variable
        # might be shut down further
        # call "python", "api.py", "--model_path", model_path, "--port", str(port), "--ip", ip with background
        new_proc = subprocess.Popen(["python", "api.py", "--model_path", model_path, "--port", str(port), "--ip", ip])
        # get pid
        pid = new_proc.pid
        # save to file to shut down later
        with open("server.pid", "w") as f:
            f.write(str(pid))
        # store in global variable
        global SERVER
        SERVER = pid
        # wait for 5 seconds
        time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--port", type=int, default=10080)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--shutdown", action="store_true")
    parser.add_argument("--force_download", action="store_true")
    # override no_server
    parser.add_argument("--server", action="store_true")
    # start server with     start_local_server(model_path, port, ip) if start_local_server is True
    parser.add_argument("--start_local_server", action="store_true")
    parser.add_argument("--save", action="store_true")
    if parser.parse_args().server:
        no_server = False
    if parser.parse_args().start_local_server:
        start_local_server(parser.parse_args().model_path, parser.parse_args().port, parser.parse_args().ip)
    # example : python tag_image.py --image_path ./test.jpg --save
    # save if image_path / image_folder is not None

    args = parser.parse_args()
    if args.shutdown:
        shutdown_local_server()
    elif args.image_path:
        # predict image using post_and_get
        if args.save:
            post_and_get([args.image_path], args.ip, args.port, save=True)
        else:
            print(post_and_get([args.image_path], args.ip, args.port)[0])
    elif args.image_folder:
        # predict folder
        if args.save:
            print(predict_local_path(args.image_folder, args.model_path, args.port, args.ip, save=True))
        else:
            print(predict_local_path(args.image_folder, args.model_path, args.port, args.ip))
    else:
        # test
        image = Image.open("./test.jpg")
        print(predict_image(image, os.path.abspath(args.model_path)))