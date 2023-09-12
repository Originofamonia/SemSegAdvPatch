#---------------------------------------------------------------------
#---------------------------------------------------------------------
# patch optimization with scene-specific attack
#
#---------------------------------------------------------------------
#---------------------------------------------------------------------


import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import time

from torch.utils import data
# from tqdm import tqdm

# import torch.nn as nn

# import scipy.misc as misc

import patch_utils as patch_utils
import test_patch as test_patch
from ptsemseg.utils import convert_state_dict
from ptsemseg.models import get_model
from torch.autograd import Variable
from ptsemseg.loss import get_loss_function, get_patch_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.utils import get_model_state
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter


def optimize_patch(cfg):

    cfg_patch_opt = cfg['adv_patch']['optimization']
    cfg_patch_attr = cfg['adv_patch']['attr']

    device = torch.device("cuda")
    torch.cuda.set_device(cfg["device"]["gpu"])

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup Augmentations TODO NONE
    augmentations = None #cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]


    train_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["train_split"],
        version= cfg["data"]["version"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        img_norm = cfg["data"]["img_norm"],
        bgr = cfg["data"]["bgr"], 
        std_version = cfg["data"]["std_version"],
        bottom_crop = 0
    )

    num_train_samples = cfg_patch_opt['num_opt_samples']
    if num_train_samples is not None:
        opt_loader, _ = torch.utils.data.random_split(
            train_loader, 
            [num_train_samples, len(train_loader)-num_train_samples])
        
    else:
        opt_loader = train_loader
    print("num optimization images (from training set): " + str(len(opt_loader)))


    validation_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        version= cfg["data"]["version"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        img_norm = cfg["data"]["img_norm"],
        bgr = cfg["data"]["bgr"], 
        std_version = cfg["data"]["std_version"],
        bottom_crop = 0
    )

    n_classes = train_loader.n_classes
    batch_size = cfg_patch_opt["batch_size"]
    batch_size_val = cfg_patch_opt["batch_size_val"]
    
    trainloader = data.DataLoader(
        opt_loader,
        batch_size=batch_size ,
        num_workers= cfg["device"]["n_workers"],
        shuffle=True
    )

    valloader = data.DataLoader(
        validation_loader, 
        batch_size=batch_size_val, 
        num_workers=cfg["device"]["n_workers"],
        shuffle=False
    )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model and patch
    model_file_name = os.path.split(cfg["model"]["path"])[1]
    model_name = model_file_name[: model_file_name.find("_")]
    print(model_name)
    model_dict = {"arch": cfg["model"]["arch"]} #model_dict = {"arch": model_name}
    model = get_model(model_dict, n_classes, version=cfg["data"]["dataset"])
    state = torch.load(cfg["model"]["path"], map_location = 'cpu')
    state = get_model_state(state, model_name)
    model.load_state_dict(state)

    # Setup experiment folders
    cfg_path = cfg["adv_patch"]["path"]
    exp_folder = cfg_path["out_dir"]
    exp_name = cfg_path["exp_name"]
    exp_root = os.path.join(exp_folder, exp_name)
    patches_folder = os.path.join(exp_root, "patches")
    if exp_name in os.listdir(exp_folder):
        input("The folder %s already exists. If you want to overwrite it, press Enter. Otherwise ctrl-c to exit" % exp_root)
        input("Are you sure? Do you really want to overwrite folder %s?" % exp_root)
        os.popen("rm -r %s" % exp_root)
        print("Folder %s deleted!" % exp_root)
        time.sleep(1) # required to complete rm request
    os.mkdir(exp_root)
    
    if "patches" not in os.listdir(exp_root):
        os.mkdir(patches_folder)
        
    with open(os.path.join(exp_root, "config.yaml"), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
        
    resume_path = cfg_path["resume"]
    if resume_path is not None:
        print(os.path.basename(resume_path)) # in os.listdir(os.path.join(os.path.split[:-1])):
        print("Resuming optimization from %s" % resume_path)
        seed_patch = patch_utils.get_patch_from_img(resume_path, set_loader=train_loader)
#         raise("Resuming patch optimization still not implemented")
    else:
        seed_patch = patch_utils.get_random_patch(cfg_patch_attr, set_loader=train_loader)

        
    # initialize the patch into the model 
    patch_utils.init_model_patch(model = model, mode = "train", seed_patch = seed_patch)

    if cfg_patch_opt['use_multiple_outputs'] == True:
        patch_utils.set_multiple_output(model = model)

    p_w, real_width, offset = cfg['adv_patch']['attr']['width'], cfg['adv_patch']['attr']['world_width'], cfg['adv_patch']['attr']['offset']
    block_width, rescale = cfg['adv_patch']['attr']['block_width'], cfg['adv_patch']['attr']['rescale']
    pixel_width = real_width / p_w
        
    print(device)
    model = model.to(device)

    # Setup optimizer and loss function
    optimizer_cls = get_optimizer(cfg_patch_opt["optimizer"])
    learning_rate = cfg_patch_opt["optimizer"]["lr"]
#     momentum = cfg_patch_opt["optimizer"]["momentum"]
    optimizer = optimizer_cls([model.patch], lr=learning_rate)
    print(optimizer)
    (loss_fn, smooth_loss_fn, NPS_fn), weights = get_patch_loss_function(cfg_patch_opt)

    epoch_loss = []
    epoch_gradnorm = []
    epoch_CE_loss = []
    epoch_gamma = []
    i = 0


    use_transformations = cfg_patch_opt['use_transformations']
    print("use_transformations: " + str(use_transformations))

    online_test1_results, online_test2_results = [],[]

    clipper = patch_utils.PatchConstraints(set_loader=train_loader)

    patch_params = patch_utils.patch_params(
        noise_magn_percent = 0.1,
        set_loader = train_loader
    )
    
#     
    
    while i <= cfg_patch_opt["opt_iters"]:

        #------------------------------------------------------------------------------------------------------
        # ONLINE TEST 
        #------------------------------------------------------------------------------------------------------
        if i % cfg_patch_opt["test_log"] == 0:
            
            #--------------------------------------------------------------------------------------------------
            # Test 1 (w/o transformations)
            #--------------------------------------------------------------------------------------------------
            if cfg_patch_opt["opt_validation_log1"] is True:
#                 print("=======", torch.max(model.patch))
                score, class_iou = test_patch.test_specific_patch(
                                                                cfg = cfg,
                                                                loader = valloader,
                                                                n_classes = n_classes,
                                                                patch = model.patch.clone(), 
                                                                patch_params=patch_params,
                                                                output_file = None,
                                                                use_transformations = False, 
                                                                )
            
                print("---------------TEST LOG 1----------------------")
                print("Mean IoU:")
                print(score["Mean IoU : \t"])
                print("Mean Acc:")
                print(score["Mean Acc : \t"])
                print("-----------------------------------------------")


                online_test1_results.append({"score": score, "class_iou": class_iou, "iteration_count": str(i)})
                
                test_results_filename = "result_test1_optimization_%d.pkl" % i
                patch_utils.save_obj(os.path.join(exp_root, test_results_filename), online_test1_results)
#                 print("=======", torch.max(model.patch))
                # Get example images
                clear_images, adv_images, clear_outputs, adv_outputs = test_patch.get_specific_examples(cfg, 
                                                                                                        valloader, 
                                                                                                        n_classes, 
                                                                                                        patch = model.patch.clone(),
                                                                                                        patch_params=patch_params,
                                                                                                        output_file = None, 
                                                                                                        num_batch = 1)
#                 print("= ======", torch.max(model.patch))
                ex_ind = 0 #np.random.randint(batch_size_val)
                patch_utils.save_summary_img(
                        tensor_list = [adv_images[ex_ind], clear_images[ex_ind], adv_outputs[ex_ind:ex_ind+1], clear_outputs[ex_ind:ex_ind+1]], 
                        path = exp_root,
                        set_loader = train_loader,
                        model_name = model_name,
                        orig_size =(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
                        count=i, 
                        img_num=0)
                
                # save the perturbed image without reducing quality (helpful to visualize the patch in the whole image)
                #patch_utils.save_tensor_png(ex_adv_image, cfg_patch_opt["out_fig"]+"perturbed_image"+str(i)+'.png', bgr=True, img_norm=False, mean = train_loader.mean)

#                 del ex_clear_image, ex_adv_image, ex_clear_out, ex_adv_out 

            
        
        #------------------------------------------------------------------------------------------------------
        # export the current patch as pkl or png
        if i % cfg_patch_opt["checkpoint_patch"] == 0 and cfg_path["save_patch"] is True:
            patch_png = "patch_%d.png" % i
            patch_pkl = "patch_%d.pkl" % i
            patch_utils.save_patch_numpy(model.patch, os.path.join(patches_folder, patch_pkl))
            patch_utils.save_patch_png(model.patch, os.path.join(patches_folder, patch_png), set_loader=train_loader)

        #------------------------------------------------------------------------------------------------------
        # END ONLINE TEST 
        #------------------------------------------------------------------------------------------------------
        
        model.eval() 
        model.to(device)

        # log var
        epoch_samples = 0
        epoch_loss.append([0, 0, 0, 0])
        epoch_gradnorm.append(0)
        epoch_gamma.append(0)
        
        for (images, labels) in trainloader:

            images = images.to(device)
            if isinstance(labels, list):
                labels, extrinsic, intrinsic = labels
                extrinsic, intrinsic = extrinsic.to(device), intrinsic.to(device)
                
            labels = labels.to(device)

            #------------------------------------------------------------------------------------------------------------------
            # extract the predicted labels that are used in the optimization loss function
#             prediction_labels = None
#             with torch.no_grad():
#                 clear_prediction = model(images)
                
#                 if not isinstance(clear_prediction, tuple):
#                     aus = clear_prediction.transpose(1, 2).transpose(2, 3).contiguous().view(-1, n_classes)
#                     prediction_labels = torch.argmax(aus, dim=1).to(device)
#                 else:
#                     prediction_labels = []
#                     for _, pred in enumerate(clear_prediction) :
#                         aus = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, n_classes)
#                         prediction_labels.append(torch.argmax(aus, dim=1).to(device))
#                     prediction_labels = tuple(prediction_labels)
            #------------------------------------------------------------------------------------------------------------------


            #------------------------------------------------------------------------------------------------------------------
            # batch optimization step

            # add_patch to the inputs using multiple transformations (wrt EOT)
            perturbed_images, patch_masks = patch_utils.project_patch_batch(images, model.patch, extrinsic, intrinsic, 
                                                                   pixel_dim=pixel_width, offset=offset, 
                                                                   rescale=rescale, device=device, patch_params=patch_params)
            model.patch.requires_grad = True

            # pre-clipping
            model.apply(clipper) 

            perturbed_images = perturbed_images.to(device)

            optimizer.zero_grad()
            outputs = model(perturbed_images)


            # compute losses
#             loss_no_misc, loss_misc, gamma = loss_fn(input=outputs, target=prediction_labels, patch_mask=patch_masks) 
#             print(outputs.shape, labels.shape)
            loss_no_misc, loss_misc, gamma = loss_fn(input=outputs, target=labels, patch_mask=patch_masks) 
            smooth_loss = smooth_loss_fn(model.patch)  
            NPS_loss = NPS_fn(model.patch, patch_params=patch_params)


            # log info
            epoch_loss[i][0]    += loss_no_misc.data
            epoch_loss[i][1]    += loss_misc.data
            epoch_loss[i][2]    += smooth_loss.data
            epoch_loss[i][3]    += NPS_loss.data
            epoch_gamma[i]      += gamma 
            epoch_samples       += cfg_patch_opt['batch_size']

            
            
            
            other_losses = [loss_no_misc, loss_misc, smooth_loss, NPS_loss]
            # retain_graph needed for multiple adv_loss computed on the same model output
            retain_graph_bool = [True, False, False, False]
            norm_grad_losses = [None, None, None, None]

            def norm(v):
                n = torch.norm(v, p=float('2'))
                return (v/n) if n > 0 else v # to avoid NaN error 
                
            def sign(v):
                v = torch.sign(v)
                return v


            # compute and normalize each loss
            for count, l in enumerate(other_losses):
                optimizer.zero_grad()
                l.backward(retain_graph=retain_graph_bool[count])
                grad_loss = model.patch.grad.data.clone().to(device)
                norm_grad_losses[count] = norm(grad_loss)
                #norm_grad_losses[count] = sign(grad_loss)
                

            # weighted sum of all the gradient losses
            final_grad_adv = gamma * (-norm_grad_losses[0]) + (1-gamma) * (-norm_grad_losses[1])
            final_grad_adv = norm(final_grad_adv)
            final_grad_smooth = norm_grad_losses[2]

            # update patch variable gradient with the overall formulation
            model.patch.grad.data = final_grad_adv * weights[0] + \
                norm_grad_losses[2] * weights[1] + norm_grad_losses[3] * weights[2]

            # step  
            optimizer.step()
            
            # post-clipping
            model.apply(clipper) 

            # cleaning
            torch.cuda.empty_cache()
#             del prediction_labels
#             del clear_prediction
            del norm_grad_losses
            #------------------------------------------------------------------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------------
 
        fmt_str = "Epochs [{:d}/{:d}]  Mean Losses: adv no misc {:.4f}, adv misc {:.4f}, Smoothing: {:.4f}, NPS {:.4f} (on {:d} training samples)  | gamma: {:.4f} |Mean value patch: {:.4f} "
        

        print_str = fmt_str.format(
                i + 1,
                cfg_patch_opt["opt_iters"],
                epoch_loss[i][0]/epoch_samples,
                epoch_loss[i][1]/epoch_samples,
                epoch_loss[i][2]/epoch_samples,
                epoch_loss[i][3]/epoch_samples,
                epoch_samples, 
                epoch_gamma[i]/(epoch_samples/cfg_patch_opt['batch_size']),
                torch.mean(torch.abs(model.patch.data)).item() 
            )
        print(print_str)

       
        i += 1

    

    # save test1 and test2 results
    if cfg_patch_opt["opt_validation_log1"] is True:
        test_results_filename = "result_test1_optimization_final_%d.pkl" % i
        patch_utils.save_obj(os.path.join(exp_root, test_results_filename), online_test1_results)
    print("online results file saved")

    # save the final patch
    if cfg_path["save_patch"] is True:
        patch_png = "patch_final_%d.png" % i
        patch_pkl = "patch_final_%d.pkl" % i
        patch_utils.save_patch_numpy(model.patch, os.path.join(patches_folder, patch_pkl))
        patch_utils.save_patch_png(model.patch, os.path.join(patches_folder, patch_png), set_loader=train_loader)
        
    print("Final patch saved")






if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/icnet_patch.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    optimize_patch(cfg)



