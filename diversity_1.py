import numpy as np
import sklearn
import os
import pdb
import matplotlib.pyplot as plt
from scipy.stats import moment
import json
import math
from PIL import Image
import pandas as pd
# Define the class labels and their corresponding IDs
# This dictionary maps class IDs to class names

# Repeat the same for ground truth flux-> animate data matrices -> /archive/projects/palgo/syn_data_animate


CLASS_DICT = {
    0: 'cow',
    1: 'sheep',
    2: 'bird',
    3: 'person',
    4: 'cat',
    5: 'dog',
    6: 'horse',
    7: 'aeroplane',
    8: 'motorbike',
    9: 'bicycle',
    10: 'pottedplant',
}
colors = ['red', 'blue', 'green', 'orange']

cow_part_labels = {0: 'head', 1: 'left horn', 2: 'right horn', 3: 'torso', 4: 'neck', 5: 'left front upper leg', 6: 'left front lower leg', 7: 'right front upper leg', 8: 'right front lower leg', 9: 'left back upper leg', 10: 'left back lower leg', 11: 'right back upper leg', 12: 'right back lower leg', 13: 'tail'}

bird_part_labels = {0: 'head', 1: 'torso', 2: 'neck', 3: 'left wing', 4: 'right wing', 5: 'left leg', 6: 'left foot', 7: 'right leg', 8: 'right foot', 9: 'tail'}

cat_part_labels = {0: 'head', 1: 'torso', 2: 'neck', 3: 'left front leg', 4: 'left front paw', 5: 'right front leg', 6: 'right front paw', 7: 'left back leg', 8: 'left back paw', 9: 'right back leg', 10: 'right back paw', 11: 'tail'}

dog_part_labels = {0: 'head', 1: 'torso', 2: 'neck', 3: 'left front leg', 4: 'left front paw', 5: 'right front leg', 6: 'right front paw', 7: 'left back leg', 8: 'left back paw', 9: 'right back leg', 10: 'right back paw', 11: 'tail', 12: 'muzzle'} 


person_part_labels = {
    'head': 0, 'torso': 1, 'neck': 2, 'left lower arm': 3, 'left upper arm': 4,
    'left hand': 5, 'right lower arm': 6, 'right upper arm': 7,
    'right hand': 8, 'left lower leg': 9, 'left upper leg': 10,
    'left foot': 11, 'right lower leg': 12, 'right upper leg': 13,
    'right foot': 14
}

horse_part_labels = {
    'head': 0, 'left front hoof': 1, 'right front hoof': 2, 'torso': 3,
    'neck': 4, 'left front upper leg': 5, 'left front lower leg': 6,
    'right front upper leg': 7, 'right front lower leg': 8,
    'left back upper leg': 9, 'left back lower leg': 10,
    'right back upper leg': 11, 'right back lower leg': 12, 'tail': 13,
    'left back hoof': 14, 'right back hoof': 15}

aeroplane_part_labels = {0:'body', 1:'left wing', 2:'right wing', 3:'stern', 4:'tail'}
for i in range(0, 6):
    aeroplane_part_labels[5+i] = f'engine'
for i in range(0,8):
    aeroplane_part_labels[11+i] = f"wheel"


bicycle_part_labels = {0:'body', 1:'front wheel', 2:'back wheel', 3:'chainwheel', 4:'handlebar', 5:'saddle', 6:'headlight'}

motorbike_part_labels = {0:'body', 1:'front wheel', 2:'back wheel', 3:'saddle', 4:'handlebar', 5:'headlight', 6:'headlight', 7:'headlight'}

pottedplant_part_labels = {0:'body', 1:'pot', 2:'plant'}

parts_array= [
    ['head', 'torso'],  
    ['torso', 'right front upper leg'],  # cow
    ['torso', 'left front upper leg'],   # cow
    ['torso', 'right back upper leg'],   # cow
    ['torso', 'left back upper leg'],    # cow
]
class_dict_reverse = {v: k for k, v in CLASS_DICT.items()}

def to_one_hot(num, num_classes):
    """Convert a number to one-hot encoding"""
    one_hot = np.zeros(num_classes)
    one_hot[num] = 1
    return one_hot

def coefficient_of_variation(data):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    return std / abs(mean) if mean != 0 else np.inf

def get_x_y_overlap(head_bbx, torso_bbx):
    # print(head_bbx)
    # print(torso_bbx)
    x1_min, x1_max, y1_min, y1_max = head_bbx
    x2_min, x2_max, y2_min, y2_max = torso_bbx

    # Compute overlap in x and y directions - only magnitudes
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    
    
    head_bbx = (x1_min, x1_max, y1_min, y1_max)
    torso_bbx = (x2_min, x2_max, y2_min, y2_max)
    

    # Only return overlap if both axes have overlap
    if x_overlap > 0 and y_overlap > 0:
        return (x_overlap, y_overlap)
    elif is_fully_inside(head_bbx, torso_bbx) or is_fully_inside(torso_bbx, head_bbx):
        return 0,0
    else:
        return 0, 0
    
    


import numpy as np


def get_row_theta(head_bbx, torso_bbx):
    """
    Calculate the magnitude and angle between head and torso bounding box centers.

    Parameters:
    - head_bbx: tuple (xmin, xmax, ymin, ymax) in pixel coordinates
    - torso_bbx: tuple (xmin, xmax, ymin, ymax) in pixel coordinates

    Returns:
    - magnitude: Euclidean distance between centers
    - angle_deg: angle (in degrees) from head to torso, wrt X-axis (top-left origin)
    """
    print("Calculating row theta for head and torso bounding boxes")

    # Unpack bounding boxes
    x1_min, x1_max, y1_min, y1_max = head_bbx
    x2_min, x2_max, y2_min, y2_max = torso_bbx

    # Midpoints
    head_x = (x1_min + x1_max) / 2
    head_y = (y1_min + y1_max) / 2
    torso_x = (x2_min + x2_max) / 2
    torso_y = (y2_min + y2_max) / 2

    # Vector from head to torso
    dx = torso_x - head_x
    dy = torso_y - head_y


    # Vector from head to torso
    # Magnitude of the vector
    magnitude = math.hypot(dx, dy)

    # Angle with respect to the horizontal (top-left origin, y increases downward)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    return magnitude, angle_deg

def is_fully_inside(inner, outer):
    x1_min, x1_max, y1_min, y1_max = inner
    x2_min, x2_max, y2_min, y2_max = outer

    return (x1_min >= x2_min and x1_max <= x2_max and
            y1_min >= y2_min and y1_max <= y2_max)

    # canvas size is 660
    
def get_clusters(head_bbx, torso_bbx):
    x1_min, x1_max, y1_min, y1_max = head_bbx
    x2_min, x2_max, y2_min, y2_max = torso_bbx

    # Compute midpoints
    midpoint_head_x = (x1_min + x1_max) / 2
    midpoint_head_y = (y1_min + y1_max) / 2
    midpoint_torso_x = (x2_min + x2_max) / 2
    midpoint_torso_y = (y2_min + y2_max) / 2
    return midpoint_head_x, midpoint_head_y, midpoint_torso_x, midpoint_torso_y
    
    

def coefficient_of_variation(data):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    return std / abs(mean) if mean != 0 else np.inf


def cluster(bounding_box):
    x1_min, x1_max, y1_min, y1_max = bounding_box


    # Compute midpoints
    midpoint_1_x = (x1_min + x1_max) / 2
    midpoint_1_y = (y1_min + y1_max) / 2
    return midpoint_1_x, midpoint_1_y

    
def compute_stats(points, n=3):
    """
    Computes standard deviation and n-th central moment for x and y coordinates.

    Args:
        points (list of tuples): List of (x, y) coordinates.
        n (int): Order of the central moment to compute (default is 3).

    Returns:
        dict: {
            'std_x': ..., 'std_y': ...,
            'moment_x': ..., 'moment_y': ...
        }
    """
    x_vals, y_vals = zip(*points)  # Unpack coordinates 
    x = np.array(x_vals)
    y = np.array(y_vals)
    

    return {
        'std_x': np.std(x),
        'std_y': np.std(y),
        'moment_x': moment(x, moment=n),
        'moment_y': moment(y, moment=n),
        'dispersion_x': coefficient_of_variation(x),
        'dispersion_y': coefficient_of_variation(y)
    }
    
    


import sys
sys.path.append("/home/saksham.gupta/inference/Palgo_pipeline/Palgo-main/src/ImagegenDataProcessing/utils/")
data_root ="/ssd_scratch/saksham.gupta/data/"
#import data set matrices
# import constants
CANVAS_SIZE = 660


path = "/ssd_scratch/saksham.gupta/data_playgen/class_v_test_combined_mask_data.np"
path_1 = "/ssd_scratch/saksham.gupta/data_playgen/layout_pred.npy"

import yaml
 # either tak frm yaml or termianl input (argparse)
with open("/home/saksham.gupta/inference/diversity/config_diversity.yml", "r") as f:
    config = yaml.safe_load(f)
    
bruh_1 = ['train', 'test', 'val', 'playgen']
#bruh_1 = ['train']
NET_ARRAY=[]
NET_IMAGE=[]

for variable_2 in range(0,5):
    '''
    part combinations : head and torso
                      : torso and right upper leg / right leg
                      : torso and right upper leg / right leg
                      : torso and right upper leg / right leg
                      : torso and right upper leg / right leg
                      
                    
    '''
    print("This loop runs 5 times (0 to 4)")
    print(variable_2)
    print("="*70)
    
    SPLIT_ARRAY= []
    SPLIT_IMAGE = []
    
#  I want a per split, per class list of dictionaries (2)
    for variable_1 in bruh_1:
        plt.clf() # top is from writing the points ion teh same canvas
        print(f"Running for {variable_1} split")
        if variable_1 == 'train' or variable_1 == 'test' or variable_1 == 'val':
            
            x_train_path = f"X_{variable_1}_combined_mask_data.np"
            obj_class_train_path = f"class_v_{variable_1}_combined_mask_data.np"
            images_train_path = f"img_{variable_1}_combined_mask_data.np"
            x_train = np.load(os.path.join(data_root, x_train_path), allow_pickle=True)
            obj_class_train = np.load(os.path.join(data_root, obj_class_train_path),allow_pickle=True)
            images_train = np.load(os.path.join(data_root, images_train_path), allow_pickle=True)
            print("So there 4577 images for train")
            print(f"considering the images of the {variable_1} split")
            print(f"Some shape analysis")
            print("I dont know how these matrices are created so I don't know what their shape entails")
            # image_array = images_train[0][:, :, 0]  # Extract the first channel (grayscale)

            # # Convert to uint8 if necessary (PIL expects pixel values in [0, 255])
            # image_array = (image_array).astype(np.uint8)  # Scale if values are in [0, 1]

            # # Convert NumPy array to PIL Image
            # image = Image.fromarray(image_array)

            # # Save the image
            # image.save("/home/saksham.gupta/inference/diversity_verify/output_image.png")


        # Maybe consider test and train combined too
        elif variable_1 == 'playgen':
            x_train = np.load(path_1, allow_pickle=True)
            obj_class_train = np.load(path, allow_pickle=True)



        # print(x_train.shape)
        # print(x_train[0, :, 1:].shape)
        train_bbxs = [x_train]

        x_train[:, :, 1:] *=CANVAS_SIZE # except where the part checker


        # print(x_train[0, :, 1:])

        # print(to_one_hot(1, 10))
        # only cow

        #input ='bird'
        bruh=['cow', 'bird', 'cat', 'dog','sheep','horse'] # for playgen # EXCLUDING PERSON CLASS FOR NOW - ONE HOT VECTOR ERROR IS OUT OF BOUNDS
        # bruh =['cow']
        # some problem with horse and person
        # aeroplane, motorbike, bicycle 
        # no human and horse for now

        # for class
        CLASS_ARRAY = []
        IMAGE_CLASS_ARRAY=[]
        
        for input in bruh: #only iterating thorugh cow for now
            
            print(f" Only considereing the {input} for now")
            
            label = class_dict_reverse.get(input)

            if variable_1 == 'playgen':
                # the numeber of available classes for playgen are different
                one_hot_vector = [i for i in range(len(obj_class_train)) if np.all(obj_class_train[i] == to_one_hot(label, 7))]
                print(f"Shape of X_train {x_train.shape}")

            # for Test,Train,Val
            else:
                one_hot_vector = [i for i in range(len(obj_class_train)) if np.all(obj_class_train[i] == to_one_hot(label, 11)) ]
                
#           Making sure they ar all cows: images of the same object category
            image_cow = images_train[one_hot_vector]  # Shape should be (height, width, 3)
            print("So ya you can consider them now: all the cow images along with bounding box matrixa")
            # for i,j in enumerate(image_cow):
            #     image = Image.fromarray(j)
            #     image.save(f"/home/saksham.gupta/inference/diversity_verify/output_image_{i}.png")

            cow_part_labels_reverse = {v: k for k, v in cow_part_labels.items()}
            bird_part_labels_reverse = {v: k for k, v in bird_part_labels.items()}
            cat_part_labels_reverse = {v: k for k, v in cat_part_labels.items()}    
            dog_part_labels_reverse = {v: k for k, v in dog_part_labels.items()}
            horse_part_labels_reverse = {v: k for k, v in horse_part_labels.items()}
            sheep_part_labels_reverse = {v: k for k, v in cow_part_labels.items()}
            # print("For some reason sheep doesn't work")
            person_part_labels_reverse = {v: k for k, v in person_part_labels.items()}

            if input == 'cow':
                part_labels_reverse = cow_part_labels_reverse
            elif input == 'bird':
                part_labels_reverse = bird_part_labels_reverse
            elif input == 'cat':
                part_labels_reverse = cat_part_labels_reverse
            elif input == 'dog':
                part_labels_reverse = dog_part_labels_reverse
            elif input == 'horse':
                part_labels_reverse = horse_part_labels
            elif input == 'sheep':
                part_labels_reverse = cow_part_labels_reverse
            elif input == 'person':
                part_labels_reverse = person_part_labels
            elif input == 'aeroplane':
                part_labels_reverse = aeroplane_part_labels
            elif input == 'motorbike':
                part_labels_reverse = motorbike_part_labels
            elif input == 'bicycle':
                part_labels_reverse = bicycle_part_labels


            #parts =['head','torso']
            # threw an error here, when I changed from torso to body
            
            #Parts Hyperparameter
            
            # Combinations for cow - upper leg
            # parts = ['torso', 'right front upper leg'] # for cow
            # parts= ['torso', 'left front upper leg'] # for cow
            # parts= ['torso', 'right back upper leg'] # for cow
            # parts= ['torso', 'left back upper leg'] # for cow
            # parts= ['head', 'torso'] # for cow
            
            # For some classes,
            # cow , dog, horse, sheep, bird, cat, person
            if input == 'cow' or input == 'sheep':
                part_array = [
                    ['head', 'torso'],
                    ['torso', 'right front upper leg'],
                    ['torso', 'left front upper leg'],
                    ['torso', 'right back upper leg'],
                    ['torso', 'left back upper leg'],
                ]
                parts = part_array[variable_2]
                print("These are the parts")
                print(parts)


            elif input == 'bird':
                part_array = [
                    ['head', 'torso'],
                    ['torso', 'right wing'],
                    ['torso', 'left wing'],
                    ['torso', 'left leg'],
                    ['torso', 'right leg']
                ]
                parts = part_array[variable_2]

            elif input == 'cat' or input == 'dog':
                part_array = [
                    ['head', 'torso'],
                    ['torso', 'right front leg'],
                    ['torso', 'left front leg'],
                    ['torso', 'left back leg'],
                    ['torso', 'right back leg']
                ]
                parts = part_array[variable_2]

            elif input == 'horse':
                part_array = [
                    ['head', 'torso'],
                    ['torso', 'right front upper leg'],
                    ['torso', 'left front upper leg'],
                    ['torso', 'left back upper leg'],
                    ['torso', 'right back upper leg']
                ]
                parts = part_array[variable_2]

            elif input == 'person':
                part_array = [
                    ['head', 'torso'],
                    ['torso', 'right upper arm'],
                    ['torso', 'left upper arm'],
                    ['torso', 'left upper leg'],
                    ['torso', 'right upper leg']
                ]
                parts = part_array[variable_2]

            # Only works for cow, bird, cat, dog,  sheep,
            # doesn't work for  person, horse
            
            row=[]
            print("size of row array will always be 2")
            for part in parts:
                row.append(part_labels_reverse.get(part))
            print("Row:", row)
            print("test this by iterating thorugh differnt parts") # Outer most loop

            # # Only for the limimted parts generation
            # if variable_1 == 'playgen':
            #     all_rows_except= np.array([i for i in range(16) if i!= row[0] and i!= row[1]])
            # else:
            #     all_rows_except = np.array([i for i in range(18) if i!= row[0] and i!= row[1]])


        # this is for playgen inanimate data

            x_train_1 = x_train[one_hot_vector]
            # Corresponding Images?
            #isolating cases where there are only head and torso
            
            # images/ layouts where only these parts are present
            #  Playgen
            
            # You don't pass x_train in this? 
            # if config['parts'] == 'limited':
            #     L = []
            #     for i in range(len(x_train)):
            #         shape_like = x_train[i][0]  # Pick any part to get the correct shape
            #         Zero = np.zeros_like(shape_like)
            #         true_counter = True  

            #         for j in all_rows_except:
            #             if not np.all(x_train[i][j] == Zero):
            #                 true_counter = False
            #                 break  # No need to check further if one is non-zero

            #         if true_counter:
            #             L.append(i)
            #         else:
            #             print(f"Example {i}: No cases found with only head and torso bounding boxes.")


            #     x_train_cow = x_train[L]  # Filter to only those with head and torso

                    
            # Bounding boxes for head and torso
            
            bounding_box_1 = {'xmin': [], 'xmax': [], 'ymin': [], 'ymax': []}
            bounding_box_2 = {'xmin': [], 'xmax': [], 'ymin': [], 'ymax': []}

            actually_present =[] # list of indices present
            for j,i in enumerate(x_train_1):
                # print("This the bounding box for head and right front upper leg",i[row[0]],i[row[1]])
                # if both the parts under consideration are present
                part_present_1,_,_,_,_=i[row[0]]
                part_present_2,_,_,_,_=i[row[1]]

                if part_present_1 == 0 or part_present_2 == 0:
                    continue
                else:
                    part_present_1,xmin,ymin,xmax,ymax=i[row[0]]
                # Based on the parts - you choose which dictionary to append it to
                    bounding_box_1['xmin'].append(xmin)
                    bounding_box_1['xmax'].append(xmax)
                    bounding_box_1['ymin'].append(ymin)
                    bounding_box_1['ymax'].append(ymax)

                    part_present_2,xmin,ymin,xmax,ymax=i[row[1]]
                    bounding_box_2['xmin'].append(xmin)
                    bounding_box_2['xmax'].append(xmax)
                    bounding_box_2['ymin'].append(ymin)
                    bounding_box_2['ymax'].append(ymax)
                
            # train split - cow - normal
                    actually_present.append(j)
                # ARE THERE CASES WHERE THE FRONT UPPER LEG IS PRESENT BUT THE FRONT 
            # print("Checking to see if the bounding boxes are right across differnt classes, train cases and part configurations")
            # if input=='sheep' and variable_1 =='train':
            #     print(len(actually_present))
            
            #     print(f"{variable_2} and {variable_1} and {row}")
            #     print(cow_part_labels)
            #     print("="*70)
            #     print(bounding_box_1['xmin'][:5])
            #     print(bounding_box_2['xmin'][:5])
            #     print("="*70)
                
            #     print(" After this I'll print them on the image")
            #     print("then i'll move on to make sure they're saved correctly in the final array, if it wokrs for one class it'll work for all")
            #     pdb.set_trace()
            interm_tuple = (bounding_box_1, bounding_box_2) 
            CLASS_ARRAY.append(interm_tuple)
            
            # print(f"parts_actually present {variable_2} {variable_1} {input} {len(actually_present)}")
            # print("Class array:", len(CLASS_ARRAY))
            x_train_present = x_train_1[actually_present]
            corresponding_img = image_cow[actually_present]
            print("both of them will have the same shape")
            IMAGE_CLASS_ARRAY.append(corresponding_img)  # Store the corresponding images
            
            

            # print("checking class array: you could just calculate the diverisuty here and add it to a table: based on differnt variabel in the loop")
            print('Checking class array')
            

        SPLIT_ARRAY.append(CLASS_ARRAY)
        SPLIT_IMAGE.append(IMAGE_CLASS_ARRAY)
        # print("Split array:", len(SPLIT_ARRAY))
    NET_ARRAY.append(SPLIT_ARRAY)
    NET_IMAGE.append(SPLIT_IMAGE)



# The below function uses
 
def combination_function(split,object_class, parts_array_number, configuration,compare_with=None):
    '''
    Docstring contains part array number correspoinding parts
    ['head', 'torso'],
    ['torso', 'right front upper leg'],
    ['torso', 'left front upper leg'],
    ['torso', 'right back upper leg'],
    ['torso', 'left back upper leg'],
    '''
    
    arr = NET_ARRAY[parts_array_number]
    arr_image = NET_IMAGE[parts_array_number]

    if split =='train':
        Arr_1 = arr[0]
        image_array = arr_image[0]
    if split == 'playgen':
        Arr_1 = arr[-1]
        image_array = arr_image[-1]

    arr_1 = Arr_1[bruh.index(object_class)]
    image_1 = image_array[bruh.index(object_class)]
    image_1 = np.array(image_1)
    # print("this is the shape of the image array", np.array(image_1).shape)
    

    if configuration == 'overlap':
        # One graph
        # instead of a tuple make a dictionary
        overlap=[]
        plt.clf()
        #  Just two dicts at a time - LATER
        dict_1 =arr_1[0] # - PART -1
        dict_2 =arr_1[1] # - PART -2
        for i in range(len(dict_1['xmin'])):
            overlap.append(get_x_y_overlap((dict_1['xmin'][i],dict_1['xmax'][i],dict_1['ymin'][i],dict_1['ymax'][i]),(dict_2['xmin'][i],dict_2['xmax'][i],dict_2['ymin'][i],dict_2['ymax'][i])))
            
            head_bbx = (dict_1['xmin'][i], dict_1['xmax'][i], dict_1['ymin'][i], dict_1['ymax'][i])
            
            torso_bbx = (dict_2['xmin'][i], dict_2['xmax'][i], dict_2['ymin'][i], dict_2['ymax'][i])
                
        #plot graph:
        overlap_1 = [pt for pt in overlap if pt != (0, 0)]  # Filter out zero overlaps
            # for some reason it makes individual graphs for each class as well - got it
            # this loop runs per class
        x_1_vals = [pt[0] for pt in overlap_1]
        y_1_vals = [pt[1] for pt in overlap_1]
        # checking complete overap cases


        num_points = len(overlap_1)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_1_vals, x_1_vals, color='red', label='Overlap Points')  # Note: x and y swapped

        plt.xlabel("X ")
        plt.ylabel("Y)")
        coeff_var_x = coefficient_of_variation(x_1_vals)
        coeff_var_y = coefficient_of_variation(y_1_vals)

        
        # Add title with point count
        # num_points = len(dict_1['xmin'])
        plt.title(f"Overlap plot - {num_points} points - {coeff_var_x:.2f} (X), {coeff_var_y:.2f} (Y)")
        


        os.makedirs('/home/saksham.gupta/inference/diversity_verify/overlap/', exist_ok=True)
        filename = f"/home/saksham.gupta/inference/diversity_verify/overlap/{parts_array_number}_overlap_{split}_{object_class}_{parts_array[parts_array_number]}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        plt.clf()
        #  returns two numebrs for each case
        
        # image_1 - These are the images that correpsonding to the points in the graphs
        
        x_1_vals = [pt[0] for pt in overlap_1]
        y_1_vals = [pt[1] for pt in overlap_1]
        

        # Not the same length

        # print(len(image_1))
        # print(len(overlap))
        if len(overlap)== 0:

            print("No overlap found")
        else:
            print("Max index")
            max_index = max(range(len(overlap)), key=lambda i: overlap[i][0])
            print(max_index)

        
        # pdb.set_trace()
        # print("suppose you want the image, correpsoing to the corners, justt tak ethe argmax adn fetch the index")
        # print("from here you can check which image correponds to which point on the graph")
    
        print("="*70)
        
        return (f"{coeff_var_x:.2f}", f"{coeff_var_y:.2f}")

    if configuration == 'geometric':
        geometry=[]
        # one graph
        #  Just two dicts at a time - LATER
        dict_1 =arr_1[0] # - PART -1
        dict_2 =arr_1[1] # - PART -2
        for i in range(len(dict_1['xmin'])):
            geometry.append(get_row_theta((dict_1['xmin'][i],dict_1['xmax'][i],dict_1['ymin'][i],dict_1['ymax'][i]),(dict_2['xmin'][i],dict_2['xmax'][i],dict_2['ymin'][i],dict_2['ymax'][i])))
        #plot graph:
        

            # for some reason it makes individual graphs for each class as well - got it
            # this loop runs per class
        x_1_vals = [pt[0] for pt in geometry]
        y_1_vals = [pt[1] for pt in geometry]
        
        coeff_var_x = coefficient_of_variation(x_1_vals)
        coeff_var_y = coefficient_of_variation(y_1_vals)


        
        plt.figure(figsize=(6, 6))
        plt.scatter(x_1_vals, y_1_vals, color='red', label='Geometric Points')  # Note: x and y swapped

        
        plt.xlabel("Magnitude (increasing →)")
        plt.ylabel("Angle ")

        
        
        
        num_points = len(dict_1['xmin'])
 
        plt.title(f"Geometric Plot - {num_points} -{coeff_var_x:.2f}, {coeff_var_y:.2f} ")

        # Save figure
        os.makedirs('/home/saksham.gupta/inference/diversity_verify/geometric/', exist_ok=True)
        
        filename = f"/home/saksham.gupta/inference/diversity_verify/geometric/{parts_array_number}_geometric_{split}_{object_class}_{parts_array[parts_array_number]}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        plt.clf()
        print("below have the same length")

        print(len(image_1))
        print(len(geometry))
        print("="*70)
        geometry_negative_indices = [i for i, point in enumerate(geometry) if point[1] < 0]
        # corresponding image
        plt.clf()
        os.makedirs(f'/home/saksham.gupta/inference/diversity_verify/geometric_images/{split}/{object_class}/{part_array[parts_array_number]}/', exist_ok=True)
        image_filename = f"/home/saksham.gupta/inference/diversity_verify/geometric_images/{split}/{object_class}/{part_array[parts_array_number]}/{parts_array_number}_geometric_{split}_{object_class}_{parts_array[parts_array_number]}"
        image_negative = image_1[geometry_negative_indices]  # Get images corresponding to negative angles
        for s,i in enumerate(image_negative):
            plt.imshow(i)  # Display the image corresponding to the max overlap
            plt.axis('off')  # Hide axes
            plt.savefig(f"{image_filename}_{s}.png", bbox_inches='tight')
            plt.close()
            plt.clf()
        # print("Negative angles found at indices:", geometry_negative_indices)
        
        
        return (f"{coeff_var_x:.2f}", f"{coeff_var_y:.2f}")
    
    # if configuration == 'cluster':
    #     # Two graphs
    #     cluster_list_1 = []
    #     cluster_list_2 = []
    #     # one graph
    #     #  Just two dicts at a time - LATER
    #     print(type(arr_1))
    #     print(len(arr_1))
    #     dict_1 =arr_1[0] # - PART -1
    #     dict_2 =arr_1[1] # - PART -2
    #     plt.clf()
 
    #     for i in range(len(dict_1['xmin'])):
    #     # Only takes one seth of bounding box coordinates
    #         cluster_list_1.append(cluster((dict_1['xmin'][i],dict_1['xmax'][i],dict_1['ymin'][i],dict_1['ymax'][i])))
    #     # plot graph:
    #         cluster_list_2.append(cluster((dict_2['xmin'][i],dict_2['xmax'][i],dict_2['ymin'][i],dict_2['ymax'][i])))
            
    #     plt.clf()
    #     colors = ['red', 'blue', 'green', 'orange']
    #         # for some reason it makes individual graphs for each class as well - got it
    #         # this loop runs per class
    #     x_1_vals = [pt[0] for pt in cluster_list_1]
    #     y_1_vals = [pt[1] for pt in cluster_list_1]
    #     plt.figure(figsize=(6, 6))
    #     plt.scatter(x_1_vals, y_1_vals, color='red', label='Cluster Points')  # Note: x and y swapped

    #     # Invert axes to match desired orientation
    #     plt.gca().invert_yaxis()  # Makes x increase down
    #     plt.gca().xaxis.tick_top()  # Moves x-axis to top
    #     plt.gca().xaxis.set_label_position('top')
    #     plt.xlabel("x (increasing →)")
    #     plt.ylabel("y (increasing ↓)")

    #     # Add title with point count
    #     num_points = len(dict_1['xmin'])
    #     plt.title(f"Cluster plot - {num_points} points")

    #     # Save figure
    #     filename = f"/home/saksham.gupta/inference/diversity/cluster_{split}_{object_class}_{parts_array[parts_array_number]}_0.png"
    #     plt.savefig(filename, bbox_inches='tight')
    #     plt.close()
    #     plt.clf()
        
        
    #         # for some reason it makes individual graphs for each class as well - got it
    #         # this loop runs per class
    #     x_1_vals = [pt[0] for pt in cluster_list_2]
    #     y_1_vals = [pt[1] for pt in cluster_list_2]
    #     plt.figure(figsize=(6, 6))
    #     plt.scatter(x_1_vals, y_1_vals, color='red', label='Cluster Points')  # Note: x and y swapped
    #     # Invert axes to match desired orientation
    #     plt.gca().invert_yaxis()  # Makes x increase down
    #     plt.gca().xaxis.tick_top()  # Moves x-axis to top
    #     plt.gca().xaxis.set_label_position('top')
    #     plt.xlabel("Y (increasing →)")
    #     plt.ylabel("X (increasing ↓)")

    #     # Add title with point count
    #     num_points = len(dict_1['xmin'])
    #     plt.title(f"Cluster plot - {num_points} points")

    #     # Save figure
    #     os.makedirs('/home/saksham.gupta/inference/diversity/cluster', exist_ok=True)
        
    #     filename = f"/home/saksham.gupta/inference/diversity/cluster/cluster_{split}_{object_class}_{parts_array[parts_array_number]}_1.png"
    #     plt.savefig(filename, bbox_inches='tight')
    #     plt.close()
    #     plt.clf()
                    
    
    
    
    
        

        # Cluster
# for i in ['train','playgen']:
#     for j in ['cow', 'bird', 'cat', 'dog','sheep', 'person', 'horse',]:
#         for k in range(0,5):
#             combination_function(i, j, k, 'cluster')

        #  Overlap - pull up the correpsonfing bounding box images too
        # check for the containement cases for the torso and upper legs
import pandas as pd

# list of dictionaries
results_overlap = []

print("=" * 90)

for i in ['train', 'playgen']:
    for j in ['cow', 'bird', 'cat', 'dog', 'horse']:
        for k in range(0, 5):
            print(f"Running overlap for {i}, {j}, {k}")
            x_coeff, y_coeff = combination_function(i, j, k, 'overlap')
            # Append results as a dictionary
            results_overlap.append({
                'split': i,
                'class': j,
                'index': k,
                'x_coeff': x_coeff,
                'y_coeff': y_coeff
            })

# Convert to DataFrame
overlap_dataframe = pd.DataFrame(results_overlap)

overlap_dataframe.to_csv('/home/saksham.gupta/inference/diversity_verify/overlap_results.csv', index=False)


results_geometric = []

print("=" * 90)

for i in ['train', 'playgen']:
    for j in ['cow', 'bird', 'cat', 'dog', 'sheep', 'horse']:
        for k in range(0, 5):
            print(f"Running overlap for {i}, {j}, {k}")
            x_coeff, y_coeff = combination_function(i, j, k, 'geometric')
            # Append results as a dictionary
            results_geometric.append({
                'split': i,
                'class': j,
                'index': k,
                'x_coeff': x_coeff,
                'y_coeff': y_coeff
            })

# Convert to DataFrame
geometric_dataframe = pd.DataFrame(results_geometric)

geometric_dataframe.to_csv('/home/saksham.gupta/inference/diversity_verify/geometric_results.csv', index=False)
# for i in ['train','playgen']:
#     for j in ['cow', 'bird', 'cat', 'dog','sheep', 'horse']:
#         for k in range(0,5):
#             combination_function(i, j, k, 'geometric')

# convert to pandas dataframe:

# import pandas as pd

# # Sample placeholder names, you can replace these with actual ones if available
# name_ids = [0, 1, 2, 3, 4]
# splits = ['train', 'test', 'val', 'playgen']
# classes = ['cow', 'bird', 'cat', 'dog', 'sheep', 'horse']

# import pandas as pd
# # REPLACE YOUR ENTIRE DATAFRAME CREATION SECTION WITH THIS:
# # STEP 1: DEBUGGING - Add this right before your dataframe creation
# print("DEBUG: About to start dataframe creation...")
# print(f"NET_ARRAY shape: {len(NET_ARRAY)} x {len(NET_ARRAY[0])} x {len(NET_ARRAY[0][0])}")

# # STEP 2: COMPLETELY REPLACE your dataframe creation section with this:
# records = []
# data = NET_ARRAY

# # Add counter to track record creation
# record_counter = 0

# for name_idx, name_data in enumerate(data):
#     print(f"Processing name_idx: {name_idx}")
    
#     for split_idx, split_data in enumerate(name_data):
#         print(f"  Processing split_idx: {split_idx} ({splits[split_idx]})")
        
#         for class_idx, class_data in enumerate(split_data):
#             print(f"    Processing class_idx: {class_idx} ({classes[class_idx]})")
            
#             # CRITICAL: Only process each configuration ONCE per class
#             for config in ['overlap', 'geometric']:
#                 print(f"      Processing config: {config}")
                
#                 config_points = []
                
#                 # Get both boxes directly - NO ITERATION over boxes
#                 if len(class_data) != 2:
#                     print(f"ERROR: Expected 2 boxes, got {len(class_data)}")
#                     continue
                    
#                 head_box = class_data[0]
#                 torso_box = class_data[1]
                
#                 # Check if boxes have data
#                 if not head_box['xmin'] or not torso_box['xmin']:
#                     print(f"        Skipping - empty boxes")
#                     continue
                
#                 num_instances = len(head_box['xmin'])
#                 print(f"        Processing {num_instances} instances")
                
#                 # Process all instances for this configuration
#                 for i in range(num_instances):
#                     head_bbx = (head_box['xmin'][i], head_box['xmax'][i], 
#                                head_box['ymin'][i], head_box['ymax'][i])
#                     torso_bbx = (torso_box['xmin'][i], torso_box['xmax'][i], 
#                                 torso_box['ymin'][i], torso_box['ymax'][i])
                    
#                     if config == 'overlap':
#                         a, b = get_x_y_overlap(head_bbx, torso_bbx)
#                         config_points.append((a, b))
#                     elif config == 'geometric':
#                         a, b = get_row_theta(head_bbx, torso_bbx)
#                         config_points.append((a, b))
                
#                 # Apply filtering for overlap (match your plotting code)
#                 if config == 'overlap':
#                     original_count = len(config_points)
#                     config_points = [pt for pt in config_points if pt != (0, 0)]
#                     print(f"        Filtered from {original_count} to {len(config_points)} points")
                
#                 # Calculate coefficients
#                 if config_points:
#                     coeff_x = coefficient_of_variation([pt[0] for pt in config_points])
#                     coeff_y = coefficient_of_variation([pt[1] for pt in config_points])
#                 else:
#                     coeff_x = 0
#                     coeff_y = 0
                
#                 # Create exactly ONE record per configuration
#                 record = {
#                     'name_id': name_ids[name_idx],
#                     'split': splits[split_idx],
#                     'class': classes[class_idx],
#                     'configuration': config,
#                     'coeff_x': coeff_x,
#                     'coeff_y': coeff_y,
#                     'num_points': len(config_points)
#                 }
                
#                 records.append(record)
#                 record_counter += 1
                
#                 print(f"        Created record #{record_counter}: {name_ids[name_idx]}-{splits[split_idx]}-{classes[class_idx]}-{config}")

# print(f"Total records created: {len(records)}")

# # STEP 3: Check for duplicates before creating DataFrame
# print("\nChecking for duplicates...")
# seen_combinations = set()
# duplicates_found = []

# for i, record in enumerate(records):
#     key = (record['name_id'], record['split'], record['class'], record['configuration'])
#     if key in seen_combinations:
#         duplicates_found.append((i, key))
#         print(f"DUPLICATE at index {i}: {key}")
#     else:
#         seen_combinations.add(key)

# if duplicates_found:
#     print(f"Found {len(duplicates_found)} duplicates!")
#     # Remove duplicates
#     unique_records = []
#     seen = set()
#     for record in records:
#         key = (record['name_id'], record['split'], record['class'], record['configuration'])
#         if key not in seen:
#             unique_records.append(record)
#             seen.add(key)
    
#     print(f"Reduced from {len(records)} to {len(unique_records)} records")
#     records = unique_records
# else:
#     print("No duplicates found!")

# # Create DataFrame
# df = pd.DataFrame(records)

# # Final verification
# print(f"\nFinal DataFrame shape: {df.shape}")
# print("Sample records:")
# for i in range(min(5, len(df))):
#     row = df.iloc[i]
#     print(f"{row['name_id']}{row['split']}{row['class']}{row['configuration']}{row['coeff_x']:.3f}{row['coeff_y']:.3f}")

# # Save
# df.to_csv('/home/saksham.gupta/inference/diversity/diversity_data.csv', index=False)
# print(f"Saved {len(df)} unique records")
# Assuming `data` is your actual 5x4x7x2 structure
# records = []


# data = NET_ARRAY  # This should be the actual data structure you have
# for name_idx, name_data in enumerate(data):
#     for split_idx, split_data in enumerate(name_data):
#         for class_idx, class_data in enumerate(split_data):
#             for config in ['overlap', 'geometric', 'cluster']:
#                 for box_index, box in enumerate(class_data):  # 0 or 1
#                     # for each instance, they'll be three configurations
#                     # part - split - class - box_index - but configuration needs two instances
#                     config_points=[]
#                     if config =='overlap':
#                         print('overlap')
#                         if box_index==0:
#                             # I need head_bbx to be saved, but I don't know how
#                             head_bbx = box
#                         else:
#                             print("Both boxes have been iterated")
#                             for i in range(len(box['xmin'])):

#                                 torso_bbx_1 = box['xmin'][i], box['xmax'][i], box['ymin'][i], box['ymax'][i]
#                                 head_bbx_1 = head_bbx['xmin'][i], head_bbx['xmax'][i], head_bbx['ymin'][i], head_bbx['ymax'][i]
#                                 # removing isntances where there is no overlap between bounding boxes-or of there is full containment

#                                 a,b = get_x_y_overlap(head_bbx_1,torso_bbx_1)
#                                 config_points.append((a,b))
#                     elif config =='geometric':
#                         print('geometric')
#                         if box_index==0:
#                             # I need head_bbx to be saved, but I don't know how
#                             head_bbx = box
#                         else:
#                             print("Both boxes have been iterated")
#                             for i in range(len(record['xmin'])):

#                                 torso_bbx_1 = box['xmin'][i], box['xmax'][i], box['ymin'][i], box['ymax'][i]
#                                 head_bbx_1 = head_bbx['xmin'][i], head_bbx['xmax'][i], head_bbx['ymin'][i], head_bbx['ymax'][i]

#                                 a,b = get_row_theta(head_bbx_1,torso_bbx_1)
#                                 config_points.append((a,b))
#                     elif config =='cluster':
#                         print('cluster')
#                         if box_index==0:
#                             # I need head_bbx to be saved, but I don't know how
#                             head_bbx = box
#                         else:
#                             print("Both boxes have been iterated")
#                             for i in range(len(record['xmin'])):

#                                 torso_bbx_1 = box['xmin'][i], box['xmax'][i], box['ymin'][i], box['ymax'][i]
#                                 head_bbx_1 = head_bbx['xmin'][i], head_bbx['xmax'][i], head_bbx['ymin'][i], head_bbx['ymax'][i]

#                                 # a,b = get_clusters(head_bbx_1,torso_bbx_1)
#                                 config_points.append((a,b))
#                                 a,b=0,0
                        
                                
#                         # config points is a list of tuples
#                     config_points=[pt for pt in config_points if pt != (0, 0)]
                    
#                     coefficient_of_variation_x = coefficient_of_variation([pt[0] for pt in config_points])
#                     coefficient_of_variation_y = coefficient_of_variation([pt[1] for pt in config_points])
#                     a = coefficient_of_variation_x
#                     b = coefficient_of_variation_y
#                     record = {
#                         'name_id': name_ids[name_idx],
#                         'split': splits[split_idx],
#                         'class': classes[class_idx],
#                         'box_index': box_index,
#                         'xmin': box.get('xmin', None),
#                         'xmax': box.get('xmax', None),
#                         'ymin': box.get('ymin', None),
#                         'ymax': box.get('ymax', None),
#                         'configuration': config , # or 'geometric' or 'cluster' based on your context
#                         'coeff_x': a,
#                         'coeff_y': b

#                     }
#                     records.append(record)

# df = pd.DataFrame(records)

# for i in range(len(df)):
#     print(df.iloc[i])
# df.drop(columns=['xmin', 'xmax', 'ymin', 'ymax'], inplace=True)  # Drop bounding box columns if not needed
# new_df = df[df['configuration']!= 'cluster']
# new_df.to_csv('/home/saksham.gupta/inference/diversity/diversity_data.csv', index=False) 

    
    # print(df.iloc[i]['xmin'])
    # print(df.iloc[i]['xmax'])
    # print(df.iloc[i]['ymin'])
    # print(df.iloc[i]['ymax'])



# Code for clustering

# what kind of input does it expect







# tranining an autoencoder to capture diversity
# So far we have been VaeS for inference- but now the data set is all the bounding boxes and the grounf trith is the divesity.

# can a combination of autoencoders be used
# some latent space representation of the bounding boxes
# graphs and latent space representations are weird

import pandas as pd

rows = []


