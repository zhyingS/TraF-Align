import torch
import numpy as np

def map2det(head, preds, cfg):

    outputs = head.predict(preds, cfg['post_processing'])
    detections = []
    for i,output in enumerate(outputs):
        a={}
        for k, v in output.items():
            if k != "token":
                output[k] = v.to(torch.device("cpu"))
        detections.append(output)
        
    return detections
    