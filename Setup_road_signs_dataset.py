import os
import torch
import random

import time
import copy
import numpy as np

import shutil

import wget
from zipfile import ZipFile

import pandas as pd

import cv2

from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
# ======================================================================
# == Check GPU is connected
# ======================================================================

print("======================")
print("Check GPU is info")
print("======================")
print("How many GPUs are there? Answer:",torch.cuda.device_count())
print("The Current GPU:",torch.cuda.current_device())
print("The Name Of The Current GPU",torch.cuda.get_device_name(torch.cuda.current_device()))
# Is PyTorch using a GPU?
print("Is Pytorch using GPU? Answer:",torch.cuda.is_available())
print("======================")

# switch to False to use CPU
use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");

# =====================================================
# == Set random seeds
# =====================================================
def set_random_seeds(random_seed=0):

   torch.manual_seed(random_seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   np.random.seed(random_seed)
   random.seed(random_seed)

# =====================================================
# == Load and normalize road signs dataset
# =====================================================

# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):
   root = ET.parse(xml_file).getroot()

   # Initialise the info dict 
   info_dict = {}
   info_dict['bboxes'] = []

   # Parse the XML Tree
   for elem in root:
      # Get the file name 
      if elem.tag == "filename":
         info_dict['filename'] = elem.text
         
      # Get the image size
      elif elem.tag == "size":
         image_size = []
         for subelem in elem:
            image_size.append(int(subelem.text))
            
         info_dict['image_size'] = tuple(image_size)
     
      # Get details of the bounding box 
      elif elem.tag == "object":
         bbox = {}
         for subelem in elem:
            if subelem.tag == "name":
               bbox["class"] = subelem.text
                 
            elif subelem.tag == "bndbox":
               for subsubelem in subelem:
                  bbox[subsubelem.tag] = int(subsubelem.text)            
         info_dict['bboxes'].append(bbox)

   return info_dict

# Dictionary that maps class names to IDs
class_name_to_id_mapping = {"trafficlight": 0,
                           "stop": 1,
                           "speedlimit": 2,
                           "crosswalk": 3}

# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict):
   print_buffer = []

   # For each bounding box
   for b in info_dict["bboxes"]:
      try:
         class_id = class_name_to_id_mapping[b["class"]]
      except KeyError:
         print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
     
      # Transform the bbox co-ordinates as per the format required by YOLO v5
      b_center_x = (b["xmin"] + b["xmax"]) / 2 
      b_center_y = (b["ymin"] + b["ymax"]) / 2
      b_width    = (b["xmax"] - b["xmin"])
      b_height   = (b["ymax"] - b["ymin"])

      # Normalise the co-ordinates by the dimensions of the image
      image_w, image_h, image_c = info_dict["image_size"]  
      b_center_x /= image_w 
      b_center_y /= image_h 
      b_width    /= image_w 
      b_height   /= image_h 

      #Write the bbox details to the file 
      print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
   # Name of the file which we have to save 
   save_file_name = os.path.join("datasets/road_signs/annotations", info_dict["filename"].replace("png", "txt"))
    
   # Save the annotation to disk
   print("\n".join(print_buffer), file= open(save_file_name, "w"))

# # =====================================================
# # == Get YOLO model
# # =====================================================
# def get_yolo_model():
#    # Load a pretrained model from Pytorch.
#    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#    return model
# =====================================================
# == Main
# =====================================================

random_seed = 0
num_classes = 200
cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu:0")

set_random_seeds(random_seed=random_seed)


# model = get_yolo_model()
print(extract_info_from_xml('datasets/road_signs/annotations/road4.xml'))

# Get the annotations
annotations = [os.path.join('datasets/road_signs/annotations', x) for x in os.listdir('datasets/road_signs/annotations') if x[-3:] == "xml"]
annotations.sort()

# Convert and save the annotations
for ann in tqdm(annotations):
   info_dict = extract_info_from_xml(ann)
   convert_to_yolov5(info_dict)
annotations = [os.path.join('datasets/road_signs/annotations', x) for x in os.listdir('datasets/road_signs/annotations') if x[-3:] == "txt"]

class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.savefig("temp.png")

# Get any random annotation file 
annotation_file = random.choice(annotations)
with open(annotation_file, "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x ] for x in annotation_list]

#Get the corresponding image file
image_file = annotation_file.replace("annotations", "images").replace("txt", "png")
assert os.path.exists(image_file)

#Load the image
image = Image.open(image_file)

#Plot the Bounding Box
plot_bounding_box(image, annotation_list)


# Read images and annotations
images = [os.path.join('datasets/road_signs/images', x) for x in os.listdir('datasets/road_signs/images')]
annotations = [os.path.join('datasets/road_signs/annotations', x) for x in os.listdir('datasets/road_signs/annotations') if x[-3:] == "txt"]

images.sort()
annotations.sort()

# Split the dataset into train-valid-test splits 
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

temp_paths = ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]
for temp_path in temp_paths:
   temp_path = "datasets/road_signs/"+ temp_path
   if not os.path.isdir(temp_path):
           os.makedirs(temp_path)

#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
move_files_to_folder(train_images, 'datasets/road_signs/images/train')
move_files_to_folder(val_images, 'datasets/road_signs/images/val/')
move_files_to_folder(test_images, 'datasets/road_signs/images/test/')
move_files_to_folder(train_annotations, 'datasets/road_signs/labels/train/')
move_files_to_folder(val_annotations, 'datasets/road_signs/labels/val/')
move_files_to_folder(test_annotations, 'datasets/road_signs/labels/test/')