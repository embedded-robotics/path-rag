from PIL import Image
import numpy as np
import os
import json
import pickle
import sys
sys.path.append(os.path.join(os.getcwd(), 'histocartography'))

from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder

PVQA_DATA_PATH = "pvqa"
HISTO_PATCH_SAVE_PATH = "histo_image_patch"
ACRH_DATA_PATH = "arch"

# Cell Graph Generation Definitions
nuclei_detector = NucleiExtractor()
feats_extractor = DeepFeatureExtractor(architecture='resnet34', patch_size=72, resize_size=224)
knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)

# PathVQA Dataset Processing
def get_path_vqa_open_images(data_path : str = "pvqa"):
    
    # Reading the PathVQA dataset
    img_train_path = os.path.join(data_path, "images", "test")
    qas_train_path = os.path.join(data_path, "qas", "test", "test_qa.pkl")
    with open(qas_train_path, 'rb') as file:
        pvqa_qas = pickle.load(file)
    
    # Getting all open-ended images
    qas_general = [qas for qas in pvqa_qas if qas['answer'] != 'yes' and qas['answer'] != 'no']
    img_general = [qas['image']  for qas in qas_general]
    img_general = list(set(img_general))
    img_general = sorted(img_general, key=str)
    img_general_path = [img_train_path + img_name + '.jpg' for img_name in img_general]
    
    return img_general, img_general_path

# PathVQA Dataset Processing
def get_arch_open_images(data_path : str = "arch"):

    # Extract the image data for PubMed dataset
    pubmed_path_images = os.path.join(data_path, "pubmed_set", "images")
    pubmed_img_uuid = os.listdir(pubmed_path_images)
    pubmed_img_uuid = [uuid.split('.')[0] for uuid in pubmed_img_uuid]
    pubmed_img_uuid_path = [os.path.join(pubmed_path_images, img_uuid + '.jpg') for img_uuid in pubmed_img_uuid]

    # Extract the image data for Books dataset
    books_path_images = os.path.join(data_path, "books_set", "images")
    books_img_uuid = os.listdir(books_path_images)
    books_img_uuid = [uuid.split('.')[0] for uuid in books_img_uuid]
    books_img_uuid_path = [os.path.join(books_path_images, img_uuid + '.png') for img_uuid in books_img_uuid]    

    return pubmed_img_uuid, pubmed_img_uuid_path, books_img_uuid, books_img_uuid_path

# Save top patches using histocartography
def save_histocartography_top_patches(img_general : list, img_general_path: list):
    for image_idx in range(0, len(img_general)):
    
        print(f"{image_idx}: Started ")
        query_img = Image.open(img_general_path[image_idx]).convert(mode="RGB")
        image = np.array(query_img)
        nuclei_map, nuclei_centers = nuclei_detector.process(image)

        # Only consider if more than 5 nuclei are detected since knn needs to form a graph using 5 neighbors.
        # If less than 5 nuclei are present, most of the images are not pathology related
        if nuclei_centers.shape[0] > 5:
            print(f"{image_idx}: Patches ")
            
            # Get the Features
            features = feats_extractor.process(image, nuclei_map)
            
            # Make Cell Graph
            cell_graph = knn_graph_builder.process(nuclei_map, features)
            
            # Make calculations to extract patches and the overlap images
            width, height = query_img.size
            width_range = np.linspace(0, width, 4, dtype=int)
            height_range = np.linspace(0, height, 4, dtype=int)

            overlap_percent = 20
            width_overlap = int((overlap_percent/100) * width)
            height_overlap = int((overlap_percent/100) * height)
            
            # Extract the patches
            image_patches = []
            patch_nuclei_centers = []
            for i in range(len(width_range)-1):
                for j in range(len(height_range)-1):
                    # Consider the overlap width from second patch only
                    if i != 0:
                        start_width = width_range[i] - width_overlap
                    else:
                        start_width = width_range[i]

                    # Consider the overlap height from second patch only
                    if j != 0:
                        start_height = height_range[j] - height_overlap
                    else:
                        start_height = height_range[j]
                    
                    # List out the patch ranges
                    left = start_width
                    upper = start_height
                    right = width_range[i+1]
                    lower = height_range[j+1]
                    
                    center_list = []
                    for center in nuclei_centers:
                        if ((center[0] >= left) and (center[0] <=right) and 
                            (center[1] >= upper) and (center[1] <=lower)):
                            center_list.append(center)

                    image_patches.append(query_img.crop((left, upper, right, lower)))
                    patch_nuclei_centers.append(center_list)

            # Calculate the length of nuclei in each patch
            patch_center_length = []
            for center in patch_nuclei_centers:
                patch_center_length.append(len(center))
            
            # Sort the patch indices based on maximum number of nuclei
            sorted_indices_desc = np.flip(np.argsort(patch_center_length))
            
            # Create a directory to store all the patches of the image
            save_directory = os.path.join(HISTO_PATCH_SAVE_PATH, img_general[image_idx])
            if not os.path.isdir(save_directory):
                os.mkdir(save_directory)
            
            # Store all the image patches into the newly created directory
            for patch_index in range(0,6):
                save_file_path = os.path.join(save_directory, str(patch_index+1) + ".png")
                image_patches[sorted_indices_desc[patch_index]].save(save_file_path)
        
        print(f"{image_idx}: Ended ")
        print(".........")

# Save top patches using histocartography
def save_histocartography_top_patches_arch(img_uuid : list, img_uuid_path: list, books_pubmed_class: str):
    for image_idx in range(0, len(img_uuid)):
    
        print(f"{image_idx}/{len(img_uuid)}: Started ")
        
        query_img = Image.open(img_uuid_path[image_idx]).convert(mode="RGB")
        image = np.array(query_img)
        nuclei_map, nuclei_centers = nuclei_detector.process(image)

        # Only consider if more than 5 nuclei are detected since knn needs to form a graph using 5 neighbors.
        # If less than 5 nuclei are present, most of the images are not pathology related
        if nuclei_centers.shape[0] > 5:
            print(f"{image_idx}: Patches ")
            
            # Get the Features
            features = feats_extractor.process(image, nuclei_map)
            
            # Make Cell Graph
            cell_graph = knn_graph_builder.process(nuclei_map, features)
            
            # Make calculations to extract patches and the overlap images
            width, height = query_img.size
            width_range = np.linspace(0, width, 4, dtype=int)
            height_range = np.linspace(0, height, 4, dtype=int)

            overlap_percent = 20
            width_overlap = int((overlap_percent/100) * width)
            height_overlap = int((overlap_percent/100) * height)
            
            # Extract the patches
            image_patches = []
            patch_nuclei_centers = []
            for i in range(len(width_range)-1):
                for j in range(len(height_range)-1):
                    # Consider the overlap width from second patch only
                    if i != 0:
                        start_width = width_range[i] - width_overlap
                    else:
                        start_width = width_range[i]

                    # Consider the overlap height from second patch only
                    if j != 0:
                        start_height = height_range[j] - height_overlap
                    else:
                        start_height = height_range[j]
                    
                    # List out the patch ranges
                    left = start_width
                    upper = start_height
                    right = width_range[i+1]
                    lower = height_range[j+1]
                    
                    center_list = []
                    for center in nuclei_centers:
                        if ((center[0] >= left) and (center[0] <=right) and 
                            (center[1] >= upper) and (center[1] <=lower)):
                            center_list.append(center)

                    image_patches.append(query_img.crop((left, upper, right, lower)))
                    patch_nuclei_centers.append(center_list)

            # Calculate the length of nuclei in each patch
            patch_center_length = []
            for center in patch_nuclei_centers:
                patch_center_length.append(len(center))
            
            # Sort the patch indices based on maximum number of nuclei
            sorted_indices_desc = np.flip(np.argsort(patch_center_length))
            
            # Create a directory to store all the patches of the image
            save_directory = os.path.join(os.getcwd(), HISTO_PATCH_SAVE_PATH, ACRH_DATA_PATH, books_pubmed_class, img_uuid[image_idx])
            if not os.path.isdir(save_directory):
                os.mkdir(save_directory)
            
            # Store all the image patches into the newly created directory
            for patch_index in range(0,6):
                save_file_path = os.path.join(save_directory, str(patch_index+1) + ".png")
                image_patches[sorted_indices_desc[patch_index]].save(save_file_path)
        # except:
        #     # Create an empty directory
        #     save_directory = os.path.join(os.getcwd(), HISTO_PATCH_SAVE_PATH, ACRH_DATA_PATH, books_pubmed_class, img_uuid[image_idx])
        #     if not os.path.isdir(save_directory):
        #         os.mkdir(save_directory)

        print(f"{image_idx}/{len(img_uuid)}: Ended ")
        print(".........")


if __name__ == "__main__":
    
    # # Get all the open-ended images of PathVQA
    # img_general, img_general_path = get_path_vqa_open_images(PVQA_DATA_PATH)
    
    # # Generate the top patches using histocartography and save them
    # save_histocartography_top_patches(img_general, img_general_path)
    
    pubmed_img_uuid, pubmed_img_uuid_path, books_img_uuid, books_img_uuid_path = get_arch_open_images(ACRH_DATA_PATH)
    
    save_histocartography_top_patches_arch(pubmed_img_uuid, pubmed_img_uuid_path, "pubmed")
    
    save_histocartography_top_patches_arch(books_img_uuid, books_img_uuid_path, "books")
    