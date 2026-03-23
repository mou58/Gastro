import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import os
import pandas as pd
import cv2
import numpy as np
from PIL import Image

def dataframe(root, save_csv=False):
    """"This function makes a dataframe that has two columns. The 1st column
    stores image locations, and the 2nd column stores corresponding image classes"""
    
    # Get folder names in the root directory
    folders = os.listdir(root)
    
    image_dir_list = [] # list for image directories
    class_list = [] # list for class values
    
    # Iterate over the folders
    for class_val, folder in enumerate(folders):
        
        # Read all image names
        images = os.listdir(os.path.join(root, folder))
        
        # Get the full image directory and class value
        for image in images:
            image_dir_list.append(os.path.join(root, folder, image))
            class_list.append(class_val)
    
    # Create dataframe to save image dir and class values            
    df = pd.DataFrame(
            {
             "Image": image_dir_list,
             "Class": class_list,     
             }    
        )
    
    # Save dataframe in csv
    if save_csv: df.to_csv('df_class.csv', index=False)
    
    return df


class MultiClassDataset(Dataset):
    def __init__(self, dataframe, n_classes:int=None, 
                 transform=None, one_hot:bool=None, 
                 mode:str=None, default_img=None, default_mask=None):
        self.dataframe = dataframe
        self.n_classes = n_classes
        self.transform = transform
        self.one_hot = one_hot 
        self.mode = mode
        self.default_img = default_img
        self.default_mask = default_mask
            
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        im_loc = self.dataframe["Image"][index] # image location
        
        try:
            im = Image.open(im_loc)
            im = np.array(im)
            # im = cv2.imread(im_loc)[:,:,::-1] # read image
            y = self.dataframe["Class"][index] # classes
            
        except Exception as e:
            print(f"Error loading image: {im_loc}. Skipping this iteration. ", end="")
            im = self.default_img
            y = self.default_mask
            print("Loaded default image and mask.")
            # return None, None # one for image, another for label
        
        if self.transform:
            im_pil = Image.fromarray(im.astype(np.uint8))
            im = self.transform(im_pil)
            im = np.array(im) # Convert to PIL Image to numpy array
        
        if self.one_hot:
            # y = F.one_hot(torch.tensor(y), num_classes=self.n_classes) # y is  a tensor now
            # y = y.float()
            
            y = np.eye(self.n_classes)[y]
        
        # Normalize 
        im = im/255.

        # Convert to Ch x H x W
        im = im.transpose(2, 0, 1).astype('float32')
        
        if self.mode == 'test':
            return im, y.astype('float32'), im_loc
        
        return im, y.astype('float32')



