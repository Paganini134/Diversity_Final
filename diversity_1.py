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
# /home/saksham.gupta/inference/diversity_verify/geometric/diversity_better.py
bruh=['cow', 'bird', 'cat', 'dog','sheep','horse'] 

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
    x_vals, y_vals = zip(*points)  
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
    
    




#import data set matrices
# import constants
CANVAS_SIZE = 660

#path for playgen matrices
path = "/ssd_scratch/saksham.gupta/data_playgen/class_v_test_combined_mask_data.np"
path_1 = "/ssd_scratch/saksham.gupta/data_playgen/layout_pred.npy"


def generate_matrix(data_root):
    bruh_1 = ['train', 'test', 'val', 'playgen']

    NET_ARRAY=[]
    NET_IMAGE=[]

    for variable_2 in range(0,5): 
        # 0: head and torso, 1: torso and right leg, 2: torso and left leg, 3: torso and right arm, 4: torso and left arm

        SPLIT_ARRAY= []
        SPLIT_IMAGE = []
        
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


            elif variable_1 == 'playgen':
                x_train = np.load(path_1, allow_pickle=True)
                obj_class_train = np.load(path, allow_pickle=True)

            train_bbxs = [x_train]

            x_train[:, :, 1:] *=CANVAS_SIZE 



            # For class
            CLASS_ARRAY = []
            IMAGE_CLASS_ARRAY=[]
            
            for input in bruh: #only iterating thorugh cow for now
                
                label = class_dict_reverse.get(input)

                if variable_1 == 'playgen':
                    # the numeber of available classes for playgen are different
                    one_hot_vector = [i for i in range(len(obj_class_train)) if np.all(obj_class_train[i] == to_one_hot(label, 7))]
                    print(f"Shape of X_train {x_train.shape}")

                # for Test,Train,Val
                else:
                    one_hot_vector = [i for i in range(len(obj_class_train)) if np.all(obj_class_train[i] == to_one_hot(label, 11)) ]
                    

                image_cow = images_train[one_hot_vector]  


                cow_part_labels_reverse = {v: k for k, v in cow_part_labels.items()}
                bird_part_labels_reverse = {v: k for k, v in bird_part_labels.items()}
                cat_part_labels_reverse = {v: k for k, v in cat_part_labels.items()}    
                dog_part_labels_reverse = {v: k for k, v in dog_part_labels.items()}
                horse_part_labels_reverse = {v: k for k, v in horse_part_labels.items()}
                sheep_part_labels_reverse = {v: k for k, v in cow_part_labels.items()}
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


                
                row=[]

                for part in parts:
                    row.append(part_labels_reverse.get(part))


                x_train_1 = x_train[one_hot_vector]
        
                bounding_box_1 = {'xmin': [], 'xmax': [], 'ymin': [], 'ymax': []}
                bounding_box_2 = {'xmin': [], 'xmax': [], 'ymin': [], 'ymax': []}



                actually_present =[] # list of indices present
                for j,i in enumerate(x_train_1):
                
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
                    

                        actually_present.append(j)
            
                interm_tuple = (bounding_box_1, bounding_box_2) 
                CLASS_ARRAY.append(interm_tuple)

                x_train_present = x_train_1[actually_present]
                corresponding_img = image_cow[actually_present]
                IMAGE_CLASS_ARRAY.append(corresponding_img)  # Store the corresponding images

            SPLIT_ARRAY.append(CLASS_ARRAY)
            SPLIT_IMAGE.append(IMAGE_CLASS_ARRAY)
            
            
        NET_ARRAY.append(SPLIT_ARRAY)
        NET_IMAGE.append(SPLIT_IMAGE)
    return NET_ARRAY, NET_IMAGE





    




def combination_function(NET_ARRAY,NET_IMAGE,split,object_class, parts_array_number, configuration,compare_with=None):
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

    

    if configuration == 'overlap':
        overlap=[]
        plt.clf()
        #  Just two dicts at a time - LATER
        dict_1 =arr_1[0] # - PART -1
        dict_2 =arr_1[1] # - PART -2
        for i in range(len(dict_1['xmin'])):
            overlap.append(get_x_y_overlap((dict_1['xmin'][i],dict_1['xmax'][i],dict_1['ymin'][i],dict_1['ymax'][i]),(dict_2['xmin'][i],dict_2['xmax'][i],dict_2['ymin'][i],dict_2['ymax'][i])))
            
            head_bbx = (dict_1['xmin'][i], dict_1['xmax'][i], dict_1['ymin'][i], dict_1['ymax'][i])
            
            torso_bbx = (dict_2['xmin'][i], dict_2['xmax'][i], dict_2['ymin'][i], dict_2['ymax'][i])
                

        overlap_1 = [pt for pt in overlap if pt != (0, 0)]  # Filter out zero overlaps
     
        x_1_vals = [pt[0] for pt in overlap_1]
        y_1_vals = [pt[1] for pt in overlap_1]



        num_points = len(overlap_1)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_1_vals, x_1_vals, color='red', label='Overlap Points')  

        plt.xlabel("X ")
        plt.ylabel("Y)")
        coeff_var_x = coefficient_of_variation(x_1_vals)
        coeff_var_y = coefficient_of_variation(y_1_vals)

    
        plt.title(f"Overlap plot - {num_points} points - {coeff_var_x:.2f} (X), {coeff_var_y:.2f} (Y)")
        


        os.makedirs('/home/saksham.gupta/inference/diversity_verify/overlap/', exist_ok=True)
        filename = f"/home/saksham.gupta/inference/diversity_verify/overlap/{parts_array_number}_overlap_{split}_{object_class}_{parts_array[parts_array_number]}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        plt.clf()

        
        x_1_vals = [pt[0] for pt in overlap_1]
        y_1_vals = [pt[1] for pt in overlap_1]
        


        if len(overlap)== 0:

            print("No overlap found")
        else:
            print("Max index")
            max_index = max(range(len(overlap)), key=lambda i: overlap[i][0])
            print(max_index)
    
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
        #images corresponding to the Negative values of angle in the geometric data
        os.makedirs(f'/home/saksham.gupta/inference/diversity_verify/geometric_images/{split}/{object_class}/{parts_array[parts_array_number]}/', exist_ok=True)
        image_filename = f"/home/saksham.gupta/inference/diversity_verify/geometric_images/{split}/{object_class}/{parts_array[parts_array_number]}/{parts_array_number}_geometric_{split}_{object_class}_{parts_array[parts_array_number]}"
        image_negative = image_1[geometry_negative_indices]  # Get images corresponding to negative angles
        for s,i in enumerate(image_negative):
            plt.imshow(i)  # Display the image corresponding to the max overlap
            plt.axis('off')  # Hide axes
            plt.savefig(f"{image_filename}_{s}.png", bbox_inches='tight')
            plt.close()
            plt.clf()
        # print("Negative angles found at indices:", geometry_negative_indices)
        
        
        return (f"{coeff_var_x:.2f}", f"{coeff_var_y:.2f}")
    
    
    
    

if __name__ == "__main__":
    # list of dictionaries
    
    
    data_root ="/ssd_scratch/saksham.gupta/data/"
    NET_ARRAY,NET_IMAGE= generate_matrix(data_root)
    print("Checking the NET ARRAY")

    pdb.set_trace()
    results_overlap = []

    print("=" * 90)

    for i in ['train', 'playgen']:
        for j in ['cow', 'bird', 'cat', 'dog', 'horse']:
            for k in range(0, 5):
                print(f"Running overlap for {i}, {j}, {k}")
                x_coeff, y_coeff = combination_function(NET_ARRAY,NET_IMAGE,i, j, k, 'overlap')
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
                x_coeff, y_coeff = combination_function(NET_ARRAY,NET_IMAGE,i, j, k, 'geometric')
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


# Apply this to the ground truth - playgen
# Run this across training for old trained playgen model
