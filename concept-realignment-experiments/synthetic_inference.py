
import sys
import os
from pathlib import Path

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory of the current file
current_file_dir = os.path.dirname(current_file_path)

# Add the path to the directory containing your script to sys.path
sys.path.insert(0, os.path.join(current_file_dir, '..'))
sys.path.insert(0, os.path.join(current_file_dir, '..', 'experiments'))
sys.path.insert(0, os.path.join(current_file_dir, '..', 'concept-realignment-experiments'))


import torch
import yaml
import os
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from cem.models.construction import construct_model
from cem.data.CUB200.cub_loader import SELECTED_CONCEPTS
from collections import defaultdict
from typing import Callable
from torch.utils.data import DataLoader
from tqdm import tqdm

class CUBDataset(Dataset):
    def __init__(self, root: str, split: str, preprocess: Callable):
        self.root = root
        self.split = split
        self.preprocess = preprocess
        self.class_id_to_name = dict()
        
        # Load the image files
        self.images = dict()
        with open(os.path.join(root, "images.txt"), "r") as f:
            for line in f:
                image_id, image_path = line.strip().split()
                image_id = int(image_id)
                self.images[image_id] = os.path.join(root, "images", image_path)
        
        # Load the splits
        self.splits = dict()
        with open(os.path.join(root, "train_test_split.txt"), "r") as f:
            for line in f:
                image_id, split = line.strip().split()
                image_id, split = int(image_id), int(split)
                split = "train" if split == 1 else "test"
                self.splits[image_id] = split
        
        assert len(self.images) == len(self.splits)

        # Load the labels
        self.labels = dict()
        with open(os.path.join(root, "image_class_labels.txt"), "r") as f:
            for line in f:
                image_id, class_id = line.strip().split()
                image_id, class_id = int(image_id), int(class_id)
                self.labels[image_id] = class_id - 1
        
        assert len(self.images) == len(self.labels)
        
        # Load the label names
        self.class_id_to_name = dict()
        with open(os.path.join(root, "classes.txt"), "r") as f:
            for line in f:
                class_id, class_name = line.strip().split()
                class_id = int(class_id) - 1
                self.class_id_to_name[class_id] = class_name
        
        # Load the bird attributes
        self.attributes = defaultdict(dict)
        all_attribute_ids = set()
        with open(os.path.join(root, "attributes","image_attribute_labels_clean.txt"), "r") as f:
            for line in f:
                image_id, attribute_id, is_present, _, _ = line.strip().split()
                image_id, attribute_id = int(image_id), int(attribute_id)
                is_present = bool(int(is_present))
                self.attributes[image_id][attribute_id] = is_present
                all_attribute_ids.add(attribute_id)
        
        self.attributes = {image_id: [self.attributes[image_id][attribute_id] for attribute_id in sorted(all_attribute_ids)] for image_id in self.images}

        # Make a list containing the image ids
        self.image_ids = list([image_id for image_id in self.images if self.splits[image_id] == self.split])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = Image.open(self.images[image_id]).convert("RGB")
        image = self.preprocess(image)
        label = self.labels[image_id]
        attrs = torch.tensor(self.attributes[image_id])[SELECTED_CONCEPTS]
        return image, label, attrs
    
    def get_all_labels(self):
        class_ids = list(sorted(self.class_id_to_name.keys()))
        class_names = [self.class_id_to_name[class_id] for class_id in class_ids]
        return class_names
    
    def collate_fn(self, batch):
        images, labels, attrs = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels).long()
        attrs = torch.stack(attrs)  
        return images, labels, attrs

def load_model(config: str, checkpoint: str):
    # Load config
    with open(config, 'r') as f:
        loaded_config = yaml.load(f, Loader=yaml.FullLoader)
    loaded_config.update(loaded_config["runs"][0])
    loaded_config.update(loaded_config["shared_params"])

    # Construct model
    model = construct_model(112, 200, loaded_config)
    model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=False)
    return model


def load_transform(resol=299):
    transform = transforms.Compose(
        [
            transforms.CenterCrop(resol),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ]
    )
    return transform


DEFAUT_CKPT = "/dss/dsshome1/04/go25kod3/projects/concept_realignment/concept-realignment-experiments/checkpoints/trained_model/epoch=89-step=3420.ckpt"
DEFAULT_CONFIG = "/dss/dsshome1/04/go25kod3/projects/concept_realignment/concept-realignment-experiments/results_old/experiment_2025_03_05_17_41_config.yaml"
DEFAULT_CUB_DIR = "/dss/dssmcmlfs01/pn39yu/pn39yu-dss-0000/datasets/CUB_200_2011"

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=str, default=DEFAUT_CKPT)
    parser.add_argument("--cub-dir", type=str, default=DEFAULT_CUB_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(args.config, args.checkpoint)
    model.to(device)
    transform = load_transform()
    dataset = CUBDataset(args.cub_dir, "test", transform)
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn, num_workers=0)

    # Run inference
    model.eval()

    predicted_attrs, true_attrs = [], []
    predicted_labels, true_labels = [], []
    with torch.no_grad():
        for images, labels, attrs in tqdm(test_loader):
            outputs = model(images.to(device), c=attrs.to(device), y=labels.to(device))
            c_sem, _, y_pred = outputs
            predicted_attrs.extend(c_sem.cpu().numpy())
            predicted_labels.extend(y_pred.cpu().numpy())
            true_attrs.extend(attrs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    predicted_attrs = np.concatenate(predicted_attrs, axis=0)
    true_attrs = np.concatenate(true_attrs, axis=0)
    predicted_labels = np.concatenate(predicted_labels, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # Save the results
    np.savez("synthetic_inference_results.npz", predicted_attrs=predicted_attrs, true_attrs=true_attrs, predicted_labels=predicted_labels, true_labels=true_labels)

