from grounded_sam_demo import load_model

model = None

def reload_model(config_file, grounded_checkpoint, device="cuda:0"):
    global model
    model = load_model(config_file, grounded_checkpoint, device=device)

def check_model():
    return model is not None