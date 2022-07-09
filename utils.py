

# --- Imports --- #
import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import metrics
import pdb
import cv2

def calc_psnr(im1, im2):
   
   im1_y = cv2.cvtColor(im1, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
   im2_y = cv2.cvtColor(im2, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
   ans = metrics.peak_signal_noise_ratio(im1_y, im2_y)
   #ans = measure.compare_psnr(im1, im2)
   #pdb.set_trace()
   return ans

def calc_ssim(im1, im2):
   
   # pdb.set_trace()

   im1_y = cv2.cvtColor(im1, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
   im2_y = cv2.cvtColor(im2, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
   ans = metrics.structural_similarity(im1_y, im2_y)
   #ans = metrics.structural_similarity(im1, im2, multichannel=True)
   return ans

def to_metrics(pred_image, gt):
    pred_image_list = torch.split(pred_image, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    psnr_list = [calc_psnr(pred_image_list_np[ind],  gt_list_np[ind]) for ind in range(len(pred_image_list))]
    ssim_list = [calc_ssim(pred_image_list_np[ind],  gt_list_np[ind]) for ind in range(len(pred_image_list))]

    return psnr_list,ssim_list


def validation(net, val_data_loader, device, category, exp_name, save_tag=False):
    """
    :param net: Gatepred_imageNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: derain or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt, image_name = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            #print(input_im.shape)
            pred_out = net(input_im)
            pred_image,zy_in = pred_out[0],pred_out[-1]

        im_psnr,im_ssim = to_metrics(pred_image, gt)
        # --- Calculate the average PSNR --- #
        psnr_list.extend(im_psnr)

        # --- Calculate the average SSIM --- #
        ssim_list.extend(im_ssim)

        # --- Save image --- #
        if save_tag:
            save_image(pred_image, image_name, category, exp_name)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def save_image(pred_image, image_name, category, exp_name):
    pred_image_images = torch.split(pred_image, 1, dim=0)
    batch_num = len(pred_image_images)
    
    for ind in range(batch_num):
        image_name_1 = image_name[ind].split('/')[-1]
        utils.save_image(pred_image_images[ind], './{}_results/{}/{}'.format(category, exp_name, image_name_1[:-3] + 'png'))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category, exp_name):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    with open('./training_log/{}_{}_log.txt'.format(category, exp_name), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)



def adjust_learning_rate(optimizer, epoch, category, lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 40 if category == 'derain' else 2

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))
