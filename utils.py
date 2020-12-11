from skimage import draw
import numpy as np
import torch
import cv2
from torchvision import models
from torch import nn

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def numpy2tensor(np_array):
    if len(np_array.shape) == 2:
        tensor = torch.from_numpy(np_array).unsqueeze(0).float().to(device)
    else:
        tensor = torch.from_numpy(np_array).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(device)
    return tensor

def make_canvas_mask(x_start, y_start, target_img, mask):
    canvas_mask = np.zeros((target_img.shape[0], target_img.shape[1]))
    canvas_mask[int(x_start-mask.shape[0]*0.5):int(x_start+mask.shape[0]*0.5), int(y_start-mask.shape[1]*0.5):int(y_start+mask.shape[1]*0.5)] = mask
    return canvas_mask

def laplacian_filter_tensor(img_tensor):

    laplacian_filter = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])
    laplacian_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    laplacian_conv.weight = nn.Parameter(torch.from_numpy(laplacian_filter).float().unsqueeze(0).unsqueeze(0).to(device))
    
    for param in laplacian_conv.parameters():
        param.requires_grad = False
    
    red_img_tensor = img_tensor[:,0,:,:].unsqueeze(1)
    green_img_tensor = img_tensor[:,1,:,:].unsqueeze(1)
    blue_img_tensor = img_tensor[:,2,:,:].unsqueeze(1)
    
    red_gradient_tensor = laplacian_conv(red_img_tensor).squeeze(1) 
    green_gradient_tensor = laplacian_conv(green_img_tensor).squeeze(1) 
    blue_gradient_tensor = laplacian_conv(blue_img_tensor).squeeze(1)
    return red_gradient_tensor, green_gradient_tensor, blue_gradient_tensor

def compute_gt_gradient(x_start, y_start, source_img, target_img, mask):
    
    # compute source image gradient
    source_img_tensor = torch.from_numpy(source_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(device)
    red_source_gradient_tensor, green_source_gradient_tensor, blue_source_gradient_tenosr = laplacian_filter_tensor(source_img_tensor)    
    red_source_gradient = red_source_gradient_tensor.cpu().data.numpy()[0]
    green_source_gradient = green_source_gradient_tensor.cpu().data.numpy()[0]
    blue_source_gradient = blue_source_gradient_tenosr.cpu().data.numpy()[0]
    
    # compute target image gradient
    target_img_tensor = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(device)
    red_target_gradient_tensor, green_target_gradient_tensor, blue_target_gradient_tenosr = laplacian_filter_tensor(target_img_tensor)    
    red_target_gradient = red_target_gradient_tensor.cpu().data.numpy()[0]
    green_target_gradient = green_target_gradient_tensor.cpu().data.numpy()[0]
    blue_target_gradient = blue_target_gradient_tenosr.cpu().data.numpy()[0]    
    
    # mask and canvas mask
    canvas_mask = np.zeros((target_img.shape[0], target_img.shape[1]))
    canvas_mask[int(x_start-source_img.shape[0]*0.5):int(x_start+source_img.shape[0]*0.5), int(y_start-source_img.shape[1]*0.5):int(y_start+source_img.shape[1]*0.5)] = mask
    
    # foreground gradient
    red_source_gradient = red_source_gradient * mask
    green_source_gradient = green_source_gradient * mask
    blue_source_gradient = blue_source_gradient * mask
    red_foreground_gradient = np.zeros((canvas_mask.shape))
    red_foreground_gradient[int(x_start-source_img.shape[0]*0.5):int(x_start+source_img.shape[0]*0.5), int(y_start-source_img.shape[1]*0.5):int(y_start+source_img.shape[1]*0.5)] = red_source_gradient
    green_foreground_gradient = np.zeros((canvas_mask.shape))
    green_foreground_gradient[int(x_start-source_img.shape[0]*0.5):int(x_start+source_img.shape[0]*0.5), int(y_start-source_img.shape[1]*0.5):int(y_start+source_img.shape[1]*0.5)] = green_source_gradient
    blue_foreground_gradient = np.zeros((canvas_mask.shape))
    blue_foreground_gradient[int(x_start-source_img.shape[0]*0.5):int(x_start+source_img.shape[0]*0.5), int(y_start-source_img.shape[1]*0.5):int(y_start+source_img.shape[1]*0.5)] = blue_source_gradient
    
    # background gradient
    red_background_gradient = red_target_gradient * (canvas_mask - 1) * (-1)
    green_background_gradient = green_target_gradient * (canvas_mask - 1) * (-1)
    blue_background_gradient = blue_target_gradient * (canvas_mask - 1) * (-1)
    
    # add up foreground and background gradient
    gt_red_gradient = red_foreground_gradient + red_background_gradient
    gt_green_gradient = green_foreground_gradient + green_background_gradient
    gt_blue_gradient = blue_foreground_gradient + blue_background_gradient
    
    gt_red_gradient = numpy2tensor(gt_red_gradient)
    gt_green_gradient = numpy2tensor(gt_green_gradient)  
    gt_blue_gradient = numpy2tensor(gt_blue_gradient)
    
    gt_gradient = [gt_red_gradient, gt_green_gradient, gt_blue_gradient]
    return gt_gradient


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def specify_bottom_center(img):
    print("If it doesn't get you to the drawing mode, then rerun this function again. Also, make sure the object fill fit into the background image. Otherwise it will crash")
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    fig.set_label('Choose target bottom-center location')
    plt.axis('off')
    target_loc = np.zeros(2, dtype=int)

    def on_mouse_pressed(event):
        target_loc[0] = int(event.xdata)
        target_loc[1] = int(event.ydata)
        
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    return target_loc

def align_source(object_img, mask, background_img, bottom_center):
    ys, xs = np.where(mask == 1)
    (h,w,_) = object_img.shape
    y1 = x1 = 0
    y2, x2 = h, w
    object_img2 = np.zeros(background_img.shape)
    yind = np.arange(y1,y2)
    yind2 = yind - int(max(ys)) + bottom_center[1]
    xind = np.arange(x1,x2)
    xind2 = xind - int(round(np.mean(xs))) + bottom_center[0]

    ys = ys - int(max(ys)) + bottom_center[1]
    xs = xs - int(round(np.mean(xs))) + bottom_center[0]
    mask2 = np.zeros(background_img.shape[:2], dtype=bool)
    for i in range(len(xs)):
        mask2[int(ys[i]), int(xs[i])] = True
    for i in range(len(yind)):
        for j in range(len(xind)):
            object_img2[yind2[i], xind2[j], :] = object_img[yind[i], xind[j], :]
    mask3 = np.zeros([mask2.shape[0], mask2.shape[1], 3])
    for i in range(3):
        mask3[:,:, i] = mask2
    background_img  = object_img2 * mask3 + (1-mask3) * background_img
    plt.figure()
    plt.imshow(background_img)
    return object_img2, mask2


def specify_mask(img):
    # get mask
    print("If it doesn't get you to the drawing mode, then rerun this function again.")
    fig = plt.figure()
    fig.set_label('Draw polygon around source object')
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    xs = []
    ys = []
    clicked = []

    def on_mouse_pressed(event):
        x = event.xdata
        y = event.ydata
        xs.append(x)
        ys.append(y)
        plt.plot(x, y, 'r+')

    def onclose(event):
        clicked.append(xs)
        clicked.append(ys)
    # Create an hard reference to the callback not to be cleared by the garbage
    # collector
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    fig.canvas.mpl_connect('close_event', onclose)
    return clicked

def get_mask(ys, xs, img):
    mask = poly2mask(ys, xs, img.shape[:2]).astype(int)
    fig = plt.figure()
    plt.imshow(mask, cmap='gray')
    return mask

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

class MeanShift(nn.Conv2d):
    def __init__(self):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        rgb_range=1
        rgb_mean=(0.4488, 0.4371, 0.4040)
        rgb_std=(1.0, 1.0, 1.0)
        sign=-1
        std = torch.Tensor(rgb_std).to(device)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1).to(device) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean).to(device) / std
        for p in self.parameters():
            p.requires_grad = False

def get_matched_features_pytorch(blended_features, target_features):
    matched_features = blended_features.new_full(size=blended_features.size(), fill_value=0, requires_grad=False).to(blended_features.device)
    for filter in range(0, blended_features.size(1)):
        matched_filter = hist_match_pytorch(blended_features[0, filter, :, :], target_features[0, filter, :, :])
        matched_features[0, filter, :, :] = matched_filter
    return matched_features


def hist_match_pytorch(source, template):

    oldshape = source.size()
    source = source.view(-1)
    template = template.view(-1)

    max_val = max(source.max().item(), template.max().item())
    min_val = min(source.min().item(), template.min().item())

    num_bins = 400
    hist_step = (max_val - min_val) / num_bins

    if hist_step == 0:
        return source.reshape(oldshape)

    hist_bin_centers = torch.arange(start=min_val, end=max_val, step=hist_step).to(source.device)
    hist_bin_centers = hist_bin_centers + hist_step / 2.0

    source_hist = torch.histc(input=source, min=min_val, max=max_val, bins=num_bins)
    template_hist = torch.histc(input=template, min=min_val, max=max_val, bins=num_bins)

    source_quantiles = torch.cumsum(input=source_hist, dim=0)
    source_quantiles = source_quantiles / source_quantiles[-1]

    template_quantiles = torch.cumsum(input=template_hist, dim=0)
    template_quantiles = template_quantiles / template_quantiles[-1]

    nearest_indices = torch.argmin(torch.abs(template_quantiles.repeat(len(source_quantiles), 1) - source_quantiles.view(-1, 1).repeat(1, len(template_quantiles))), dim=1)

    source_bin_index = torch.clamp(input=torch.round(source / hist_step), min=0, max=num_bins - 1).long()

    mapped_indices = torch.gather(input=nearest_indices, dim=0, index=source_bin_index)
    matched_source = torch.gather(input=hist_bin_centers, dim=0, index=mapped_indices)

    return matched_source.reshape(oldshape)