import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, dataset
import os, json
import cv2

class MediumShotCountingDataset(Dataset):
    def __init__(self, data_file, transform, n_samples, n_support=None):
        
        self.n_samples = n_samples
        self.n_support = n_support
        with open(data_file) as f:
            self.meta = json.load(f)
        self.cls = self.meta['class_names']
        self.transform = transform
    
    def __len__(self):
        return len(self.cls)

    def __getitem__(self, idx):
        """
        idx: class name index
        return data of idx class with size n_samples ==> task dataset
        """
        class_name = self.cls[idx]
        images_path = self.meta['image_names'][class_name]
        gt_densities_path = self.meta['image_labels'][class_name]
        annotation_path = self.meta['annotation_names'][idx]
        # print(f'class_name {class_name}')
        # print(f'images_path {images_path}')
        assert self.cls[idx] in annotation_path
        assert self.cls[idx] in images_path[0]
        assert self.cls[idx] in gt_densities_path[0]
        with open(annotation_path) as f:
            annotations = json.load(f)
        # print(f'anno {annotations}')
        sample_indices = np.random.choice(np.arange(len(images_path)), size=self.n_samples, replace=False)
        samples = {'image': [], 'boxes': [], 'gt_density': [], 'gt_count': []}
        for im_id in sample_indices:

            assert class_name in images_path[im_id]
            assert class_name in gt_densities_path[im_id]
            im_name = os.path.basename(images_path[im_id])
            # print(im_name)
            anno = annotations[im_name]
            bboxes = anno['box_examples_coordinates']
            dots = np.array(anno['points'])
            rects = list()
            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                rects.append([y1, x1, y2, x2])
            image = Image.open('{}'.format(images_path[im_id])).convert('RGB')
            image.load()
            # density_path = gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"
            density = np.load(gt_densities_path[im_id]).astype('float32')    
            sample = {'image':image,'lines_boxes':rects,'gt_density':density}
            sample = self.transform(sample)
            image, boxes, gt_density, gt_count = sample['image'].cuda(), sample['boxes'].cuda(),\
                                                            sample['gt_density'].cuda(), sample['gt_count'].cuda()
                                                            
            samples['image'].append(image)
            samples['gt_density'].append(gt_density)
            samples['boxes'].append(boxes)
            samples['gt_count'].append(gt_count)
            # print(f'key image {image.shape}, key gt_density {gt_density.shape}, key boxes {boxes.shape}, key gt_count {gt_count.shape}')
        
        for key in samples.keys():
            # print(f'key {key}, {len(samples[key])}')
            if key != 'boxes':
                samples[key] = torch.stack(samples[key])
            # print(f'after, key {key}, {samples[key].shape}')
      
        return samples  #image, boxes, gt_density


class resizeImageWithGT(object):
    """
    Resize image along with graound truth
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    Modified by: Viresh
    """
    
    def __init__(self, MAX_HW=84):
        self.max_hw = MAX_HW
        IM_NORM_MEAN = [0.485, 0.456, 0.406]
        IM_NORM_STD = [0.229, 0.224, 0.225]
        self.Normalize = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])

    def __call__(self, sample):

        image,lines_boxes,density = sample['image'], sample['lines_boxes'], sample['gt_density']
        
        W, H = image.size
        # print(H)
        if W > self.max_hw or H > self.max_hw:
            
            scale_factor_H = float(self.max_hw)/ H
            scale_factor_W = float(self.max_hw)/ W
            # print(f'{W}, {H}, {scale_factor_W}, {scale_factor_H}')
            new_H = int(np.round(H*scale_factor_H)) #8*int(H*scale_factor/8)
            new_W = int(np.round(W*scale_factor_W))

            # print(f'new_W {new_W}, new_H {new_H}')
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)

            if new_count > 0: resized_density = resized_density * (orig_count / new_count)
            
            
        else:    
            scale_factor_W = 1
            scale_factor_H = 1
            # print(f'\*{scale_factor}')
            resized_image = image
            resized_density = density
        boxes = list()
        for box in lines_boxes:
            # print(f'box1 {box}')
            # box2 = [int(k*scale_factor) for k in box]
            # print(f'box2 {box2}')
            # y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            y1, x1, y2, x2 = box[0]*scale_factor_H, box[1] * scale_factor_W, box[2] * scale_factor_H, box[3] * scale_factor_W
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = self.Normalize(resized_image)
        # resized_image = transforms.ToTensor()(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        
        # print(f'shape { resized_density.shape} gt count {torch.round(resized_density.sum())}')
        sample = {'image':resized_image,'boxes':boxes, 'gt_density':resized_density, 
                        'gt_count': torch.round(resized_density.sum())}
        
        return sample


def get_batch(data_file, n_samples):

    transform = resizeImageWithGT(84)

    dataset = MediumShotCountingDataset(data_file=data_file, n_samples=n_samples, transform=transform)
    task_indices  = np.arange(len(dataset))
    task_indices = np.random.permutation(task_indices)
    for i in task_indices:

        yield dataset[i]

def denormalize(tensor):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    """Reverses the normalisation on a tensor.
    Performs a reverse operation on a tensor, so the pixel value range is
    between 0 and 1. Useful for when plotting a tensor into an image.
    Normalisation: (image - mean) / std
    Denormalisation: image * std + mean
    Args:
        tensor (torch.Tensor, dtype=torch.float32): Normalized image tensor
    Shape:
        Input: :math:`(N, C, H, W)`
        Output: :math:`(N, C, H, W)` (same shape as input)
    Return:
        torch.Tensor (torch.float32): Demornalised image tensor with pixel
            values between [0, 1]
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """

    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)

    return denormalized
