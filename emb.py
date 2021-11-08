########################################################################
# import default libraries
########################################################################
import os
import sys
import gc
########################################################################


########################################################################
# import additional libraries
########################################################################
import numpy as np
import scipy.stats
# from import
from tqdm import tqdm
try:
    from sklearn.externals import joblib
except:
    import joblib
# original lib
import common as com
import torch
import dataloader_dcase
from torch.utils.data import DataLoader
import torchvision.models as models
from model import MobileNetV2Cus
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################


########################################################################


########################################################################
# get data from the list for file paths
########################################################################
def file_list_to_data(file_list,
                      msg="calc...",
                      n_mels=64,
                      n_frames=5,
                      n_hop_frames=1,
                      n_fft=1024,
                      hop_length=512,
                      power=2.0,
                      linear=0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        if linear == 0:
            vectors = com.file_to_vectors(file_list[idx],
                                                n_mels=n_mels,
                                                n_frames=n_frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        else:
            vectors = com.file_to_vectors_linear(file_list[idx],
                                                n_mels=n_mels,
                                                n_frames=n_frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        vectors = vectors[: : n_hop_frames, :]
        if idx == 0:
            data = np.zeros((len(file_list) * vectors.shape[0], dims), float)
        data[vectors.shape[0] * idx : vectors.shape[0] * (idx + 1), :] = vectors

    return data


########################################################################


########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)
        
    # make output directory
    model_path = os.path.join(param["exp_directory"], param["model_directory"])
    os.makedirs(model_path, exist_ok=True)

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.pt".format(model=model_path,
                                                                     machine_type=machine_type)
      
        # pickle file for storing section names
        section_names_file_path = "{model}/section_names_{machine_type}.pkl".format(model=model_path,
                                                                                    machine_type=machine_type)
        
        # get section names from wave file names
        section_names = com.get_section_names(target_dir, dir_name="train")
        unique_section_names = np.unique(section_names)
        n_sections = unique_section_names.shape[0]
        print(n_sections)

        # model = models.mobilenet_v2(pretrained=False, progress=True, num_classes=n_sections)
        model = MobileNetV2Cus(n_sections)
        model.load_state_dict(torch.load(model_file_path))
        model.to(device)
        model.eval()
        print(model)
        
        # make condition dictionary
        joblib.dump(unique_section_names, section_names_file_path)

        # generate dataset
        print("============== DATASET_GENERATOR ==============")
        # number of wave files in each section
        # required for calculating y_pred for each wave file
        # n_files_ea_section = []
        
        # data = np.empty((0, param["feature"]["n_frames"] * param["feature"]["n_mels"]), float)
        
        for section_idx, section_name in enumerate(unique_section_names):
            if section_idx == 3 or section_idx == 4 or section_idx == 5:
                continue
            
            emb_path = os.path.join(param["exp_directory"], param["emb_directory"], "{}_{}".format(machine_type, section_idx))
            if not os.path.exists(emb_path):
                os.makedirs(emb_path)

            # get file list for each section
            # all values of y_true are zero in training
            files, y_true = com.file_list_generator(target_dir=target_dir,
                                                    section_name=section_name,
                                                    dir_name="train",
                                                    mode=mode)

            n_files_ea_section = len(files)

            data_ea_section = file_list_to_data(files,
                                                msg="generate train_dataset",
                                                n_mels=param["feature"]["n_mels"],
                                                n_frames=param["feature"]["n_frames"],
                                                n_hop_frames=param["feature"]["n_hop_frames"],
                                                n_fft=param["feature"]["n_fft"],
                                                hop_length=param["feature"]["hop_length"],
                                                power=param["feature"]["power"],
                                                linear=param["feature"]["linear"])

            # data = np.append(data, data_ea_section, axis=0)

            # number of all files
            n_all_files = n_files_ea_section
            # number of vectors for each wave file
            n_vectors_ea_file = int(data_ea_section.shape[0] / n_all_files)

            # make one-hot vector for conditioning
            condition = np.full((data_ea_section.shape[0]), section_idx, float)
            # domain = np.zeros((data.shape[0]), float)
            # start_idx = 0
            # for section_idx in range(n_sections):
            #     n_vectors = n_vectors_ea_file * n_files_ea_section[section_idx]
            #     condition[start_idx : start_idx + n_vectors] = section_idx
            #     start_idx += n_vectors

                # source_vectors = n_vectors_ea_file * 1000
                # target_vectors = n_vectors_ea_file * 3
                # domain[start_idx : start_idx + source_vectors] = section_idx
                # start_idx += source_vectors
                # domain[start_idx : start_idx + target_vectors] = section_idx
                # start_idx += target_vectors

            # 1D vector to 2D image
            data = data_ea_section.reshape(data_ea_section.shape[0], 1, param["feature"]["n_frames"], param["feature"]["n_mels"])
            # data = np.concatenate((data, data, data), 1)
            print(data.shape)

            training_data = dataloader_dcase.CustomDataset(data, condition)
            train_dataloader = DataLoader(training_data, param["fit"]["batch_size"], shuffle=False)


            # embed
            print("============== MODEL EMBEDDING ==============")

            for step, (x, _) in enumerate(tqdm(train_dataloader)):
                x = torch.cat((x, x, x), 1)
                x = x.to(device=device, dtype=torch.float)
                # emb = model.features(x)
                # emb = torch.nn.functional.adaptive_avg_pool2d(emb, (1, 1))
                # emb = torch.squeeze(emb)
                emb = model.emb(x)
                np.save(os.path.join(emb_path, "emb_{}.npy".format(step)), emb.detach().cpu().numpy())


            del data
            del data_ea_section
            del condition
            del training_data
            del train_dataloader
            gc.collect()

        del model
        gc.collect()
