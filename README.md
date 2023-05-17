# Data_Augmentation
In this repository you will be able to find codes for Data Augmentation in python using different libraries.

import albumentations as A        #For data augmentation
import cv2                        #For Loading images
import matplotlib.pyplot as plt   #For plotting images
import numpy as np

#defining a function to load a image
def load_img(path):
    image=cv2.imread(path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

#loading Image
image=load_img('iron_man.jpg')
#image.shape

# Define an augmentation pipeline
transform = A.Compose([
    A.Blur(blur_limit=(20,30),p=1),
    A.RandomBrightnessContrast(p=1),
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    A.RandomCrop(width=150, height=150),
    A.Normalize()
])

#Apply the transformation
transformed_img=transform(image=image)['image']
#transformed_img.shape  #use this if wish to see the output


#Apply each transformation separately and save the transformed images
for i, t in enumerate(transform.transforms):
    transformed_img = t(image=image)['image']
    plt.imshow(transformed_img)
    plt.savefig(f"transformed_image_{i}.jpg")
