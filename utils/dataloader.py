"""
Created on Thu Jul  6 13:18:44 2023

@author: Mou Deb
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import os
import pandas as pd
import cv2
import emd
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

def emd2d(im, normalize:bool=True):
    """This function generates 2d IMFs from an image"""
    im_1d = np.reshape(im, -1) # convert to 1D
    
    # Calculate IMFs
    all_imfs = emd.sift.sift(im_1d, sift_thresh=1e-08) # shape: length of each IMF vector x no. of IMFs (e.g.786432 x 14)
    
    imfs_list = []
    
    n_imfs = all_imfs.shape[1] # no. of IMFs
    
    for i in range(n_imfs): 
        current_imf_1d = all_imfs[:,i]
    
        current_imf_2d = np.reshape(current_imf_1d, im.shape) # Convert to 2D
        
        # Normalize current_imf_2d
        if normalize:
            current_imf_2d = (current_imf_2d - np.min(current_imf_2d))/(np.max(current_imf_2d) - np.min(current_imf_2d))
        
        # Store the imf
        imfs_list.append(current_imf_2d)
        
    return imfs_list
    
    
def imfs2im(im, normalize:bool=True, imf_index_list:list=None, to_gray:bool=False):
    """This function generates an image from IMFs"""
    
    # Get all IMFs in a list
    imfs_list = emd2d(im, normalize)
    
    indexed_imfs = [] # it stores imfs shown in index list
    
    # Filter IMFs according to the index list
    for idx in imf_index_list:
        imf = imfs_list[idx] # get the imf according to the index, size = H x W x Ch=3
        
        if to_gray: # convert to gray
            imf = cv2.cvtColor(imf.astype("float32"), cv2.COLOR_RGB2GRAY) # size: H x W
            imf = np.expand_dims(imf, axis=-1) # size: H x W x Ch=1
            
        indexed_imfs.append(imf)
        
    # Concatenate all indexed IMFs to generate the final image
    if len(imf_index_list) == 1: 
        image = imf # only one IMF
    else: 
        image = np.concatenate(indexed_imfs, axis=-1) # Multiple IMFs
    
    return image
        

class MultiClassDataset(Dataset):
    def __init__(self, dataframe, n_classes:int=None, emd_dict:dict=None,
                 transform=None, one_hot:bool=None, array=None,
                 mode:str=None, default_img=None, default_mask=None):
        self.dataframe = dataframe
        self.n_classes = n_classes
        self.emd_dict = emd_dict
        self.transform = transform
        self.one_hot = one_hot 
        self.array = array 
        self.mode = mode
        self.default_img = default_img
        self.default_mask = default_mask
        
        """Structure of emd_dict:
            emd_dict = {
                "to_emd":Bool, # if True, then performs emd
                "imf_index": List, # list of indices of desired imfs
                "to_gray": Bool, # if True, converts imfs to grayscale
                "normalize": Bool, # if True, normalizes the imfs
                "subtract": Bool, # if True, then perform IM - IMF
                }
        
        """
            
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        im_loc = self.dataframe["Image"][index] # image location
        
        try:
            if self.array is not None: # array structure = [[data, label], ...]
                im = self.array[index][0]
                y = self.array[index][1]
                # print('**Entered to Array version**', im.shape, '-----', y, '-------', index)
            else:
                im = cv2.imread(im_loc)[:,:,::-1] # read image
                y = self.dataframe["Class"][index] # classes
                # print('**Entered to None version**', im.shape, '-----', y, '-------', index)
            
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
        
        if self.emd_dict["to_emd"]:
            if self.emd_dict["subtract"]:
                # Uncomment to subtract only single imf from the original image             
                # imfs = imfs2im(im, normalize=self.emd_dict["normalize"], 
                #              imf_index_list=self.emd_dict["imf_index"], 
                #              to_gray=self.emd_dict["to_gray"])
                # im = im - imfs
                
                # Uncomment to subtract single/multiple imfs from the original image 
                for i in range(len(self.emd_dict["imf_index"])):
                    imf = imfs2im(im, normalize=self.emd_dict["normalize"], 
                             imf_index_list=[self.emd_dict["imf_index"][i]], # ith imf index value
                             to_gray=self.emd_dict["to_gray"])
                    im = im - imf
                
            else:
                im = imfs2im(im, normalize=self.emd_dict["normalize"], 
                             imf_index_list=self.emd_dict["imf_index"], 
                             to_gray=self.emd_dict["to_gray"])
        
        if self.one_hot:
            # y = F.one_hot(torch.tensor(y), num_classes=self.n_classes) # y is  a tensor now
            # y = y.float()
            
            y = np.eye(self.n_classes)[y]
        
        # Normalize if emd is false
        if not self.emd_dict["to_emd"]:
            im = im/255.

        # Convert to Ch x H x W
        im = im.transpose(2, 0, 1).astype('float32')
        
        if self.mode == 'test':
            return im, y.astype('float32'), im_loc
        
        return im, y.astype('float32')



#%%

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # root = r'C:\Study\SDSMT_MS\Research\Dataset\kvasir-dataset-v2'
    
    root = r'D:\Mou\kvasir-dataset'
    
    RESIZE = (64, 64)
    
    train_transform = transforms.Compose([
        transforms.Resize(RESIZE),
    ])
    
    seed = np.random.seed(42)
    
    df = dataframe(root, save_csv=False)
    
    df_shuffled = df.sample(frac = 1, random_state=seed)
    
    n_train_val = int(len(df_shuffled) * 0.60)
    # n_test = int(len(df_shuffled) * 0.40)
    n_train = int(n_train_val * 0.70)
    n_val = int(n_train_val * 0.30)
    
    train_val_df = df_shuffled[0:n_train_val]
    train_df = train_val_df[0:n_train]
    train_df = train_df.reset_index(drop=True) # reset indices starting from 0
    val_df = train_val_df[n_train:]
    val_df = val_df.reset_index(drop=True) # reset indices starting from 0
    test_df = df_shuffled[n_train_val:]
    test_df = test_df.reset_index(drop=True) # reset indices starting from 0
    
    
    emd_dict = {
        "to_emd":True, # if True, then performs emd
        "imf_index": [1], # list of indices of desired imfs
        "to_gray": False, # if True, converts imfs to grayscale
        "normalize": True, # if True, normalizes the imfs
        }
    
    
    train_dataset = MultiClassDataset(train_df, 
                                      n_classes=8, 
                                      emd_dict = emd_dict,
                                      transform=train_transform, 
                                      one_hot=True)
        
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    itr = iter(train_loader)
    
    #%%
    data, label = next(itr)    
    
    data_s = torch.squeeze(data)
    
    data_p = torch.permute(data_s, (1, 2, 0))
       
    plt.figure()
    plt.imshow(data_p)
    print(data.shape)
    print("Label:", label)

