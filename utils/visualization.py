import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import math
from typing import Tuple
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def color_pro(pro, img=None, mode='hwc'):
	H, W = pro.shape
	pro_255 = (pro*255).astype(np.uint8)
	pro_255 = np.expand_dims(pro_255,axis=2)
	color = cv2.applyColorMap(pro_255,cv2.COLORMAP_JET)
	color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


	if img is not None:
		rate = 0.5
		if mode == 'hwc':
			assert img.shape[0] == H and img.shape[1] == W
			color = cv2.addWeighted(img,rate,color,1-rate,0)
		elif mode == 'chw':
			assert img.shape[1] == H and img.shape[2] == W
			img = np.transpose(img,(1,2,0))
			color = cv2.addWeighted(img,rate,color,1-rate,0)
			color = np.transpose(color,(2,0,1))
	else:
		if mode == 'chw':
			color = np.transpose(color,(2,0,1))

	return color

def max_norm(p, e=1e-5):
	N, C, H, W = p.size()
	p = F.relu(p)
	max_v = torch.max(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
	min_v = torch.min(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
	p = F.relu(p - min_v - e) / (max_v - min_v + e)
	return p

def ColorCAM(prob, img, mode='hwc'):
	assert prob.ndim == 3
	C, H, W = prob.shape
	colorlist = []
	for i in range(C):
		colorlist.append(color_pro(prob[i, :, :], img, mode=mode))

	CAM = np.array(colorlist)/225.0
	CAM = (np.clip(CAM, 0, 1) * 255).astype(np.uint8)
	# print('CAM:', CAM.shape)
	return CAM

def visualize_original_image(image, mode='chw'):
	if mode == 'chw':
		image = image.transpose((1, 2, 0))

	# Display the image
	plt.imshow(image)
	plt.axis('off')
	plt.title('Original Image')
	plt.show()

def display_CAM(CAM, prob, mode='hwc', category=None):
	# print(prob.shape)
	H = prob.shape[1]
	W = prob.shape[2]
	if mode == 'chw':
		num_classes, _, height, width = CAM.shape
	elif mode == 'hwc':
		num_classes, height, width, _ = CAM.shape
	fig, axes = plt.subplots(1, num_classes, figsize=(20, 5))

	for i in range(num_classes):
		ax = axes[i]
		if mode == 'chw':
			cam_data = CAM[i].transpose(1, 2, 0)
			ax.imshow(cam_data)
		elif mode == 'hwc':
			cam_data = CAM[i]
			ax.imshow(cam_data)

		ax.axis('off')
		ax.set_title(category[i])

		prob_data = prob[i]

	plt.tight_layout()
	plt.show()


def visualize(image, attention_maps, class_names, idx):
	N, C, H, W = image.size()
	if attention_maps.dim() == 3:
		B, num_classes, area = attention_maps.shape
		h = w = math.floor(math.sqrt(area))
		attention_maps = attention_maps.reshape(B, num_classes, h, w)

	attention_maps = F.interpolate(attention_maps, (H, W), mode='bilinear', align_corners=True)

	# normalization
	cam = max_norm(attention_maps)

	# visualization for training process
	img = image[idx].cpu().numpy().transpose((1, 2, 0))
	img = np.ascontiguousarray(img)
	mean = (0.485, 0.456, 0.406)
	std = (0.229, 0.224, 0.225)
	img[:, :, 0] = img[:, :, 0] * std[0] + mean[0]
	img[:, :, 1] = img[:, :, 1] * std[1] + mean[1]
	img[:, :, 2] = img[:, :, 2] * std[2] + mean[2]
	img = np.clip(img, 0, 1)
	img = (img * 255).astype(np.uint8)

	input_image = img.transpose((2, 0, 1))
	visualize_original_image(input_image, mode='chw')

	p = cam[idx].detach().cpu().numpy()
	# print(p)
	CAM = ColorCAM(p, img=img, mode='hwc')
	# print('CAM:', CAM.shape)
	display_CAM(CAM, p, category=class_names, mode='hwc')
	return img, CAM

def visualize_cam(image, attention_maps, class_names, idx):
	N, C, H, W = image.size()
	if attention_maps.dim() == 3:
		B, num_classes, area = attention_maps.shape
		h = w = math.floor(math.sqrt(area))
		attention_maps = attention_maps.reshape(B, num_classes, h, w)

	cam = F.interpolate(attention_maps, (H, W), mode='bilinear', align_corners=True)

	img = image[idx].cpu().numpy().transpose((1, 2, 0))
	img = np.ascontiguousarray(img)
	mean = (0.485, 0.456, 0.406)
	std = (0.229, 0.224, 0.225)
	img[:, :, 0] = img[:, :, 0] * std[0] + mean[0]
	img[:, :, 1] = img[:, :, 1] * std[1] + mean[1]
	img[:, :, 2] = img[:, :, 2] * std[2] + mean[2]
	img = np.clip(img, 0, 1)
	img = (img * 255).astype(np.uint8)

	input_image = img.transpose((2, 0, 1))
	# visualize_original_image(input_image, mode='chw')

	p = cam[idx].detach().cpu().numpy()
	CAM = ColorCAM(p, img=img, mode='hwc')
	# display_CAM(CAM, p, category=class_names, mode='hwc')
	return img, CAM

def get_norm_CAM(CAM, mask):
	result_tensor = torch.zeros_like(CAM)
	CAM = CAM.clone().cpu()

	# Apply softmax only to the masked positions
	masked_tensor = CAM * mask.float()
	if CAM.dim() == 1:
		sum_masked = masked_tensor.sum() + 1e-5
	else:
		sum_masked = masked_tensor.sum(dim=1, keepdim=True) + 1e-5
	custom_softmax_values = masked_tensor / sum_masked

	# Combine the softmax values with the original CAM values where mask is False
	result_tensor = custom_softmax_values * mask.float()

	return result_tensor


def get_norm_logits(logits, mask):
	logits = logits.clone().cpu()
	masked_logits = logits.masked_fill(mask == 0, float('-inf'))

	softmax_values = torch.softmax(masked_logits, dim=1)

	result = softmax_values * mask

	return result


def softmax_and_normalize(logits, mask):
	result = torch.zeros_like(logits)

	for i in range(logits.size(0)):
		current_logits = logits[i, :]
		current_mask = mask[i, :]

		masked_logits = current_logits[current_mask == 1]

		if masked_logits.numel() > 0:  
			softmax_values = torch.softmax(masked_logits, dim=0)

			print(softmax_values)
			min_val = softmax_values.min()
			max_val = softmax_values.max()
			normalized_values = (softmax_values - min_val) / (max_val - min_val)

			result[i, current_mask == 1] = normalized_values

	return result


def count_connected_components(cam: np.ndarray,
							   threshold_type: str = 'global',
							   block_size: int = 5,
							   C_param: int = 5) -> Tuple[int, np.ndarray]:
	cam_8bit = (cam * 255).astype(np.uint8)

	if threshold_type == 'global':
		_, thr_mask = cv2.threshold(
			src=cam_8bit,
			thresh=0,
			maxval=255,
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
		)
	elif threshold_type == 'adaptive':
		thr_mask = cv2.adaptiveThreshold(
			src=cam_8bit,
			maxValue=255,
			adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  
			thresholdType=cv2.THRESH_BINARY,
			blockSize=block_size,
			C=C_param
		)
	else:
		raise ValueError("Invalid threshold_type. Must be 'global' or 'adaptive'.")

	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
		image=thr_mask,
		connectivity=8
	)
	binary_mask = (thr_mask > 0).astype(np.uint8)
	activated_intensities = cam * binary_mask

	activated_pixel_count = np.sum(binary_mask)

	if activated_pixel_count == 0:
		mean_activated_intensity = 0.0
	else:
		mean_activated_intensity = np.sum(activated_intensities) / activated_pixel_count

	if num_labels <= 1:
		return 0, str(round(float(mean_activated_intensity), 2))
	else:
		return num_labels-1, str(round(float(mean_activated_intensity), 2))

def calculate_cam_metrics(cam: np.ndarray,
							   threshold: float = 0.4,
							   threshold_type: str = 'adaptive',
							   block_size: int = 5,
							   C_param: int = 5):
	num_components, _ = count_connected_components(cam, threshold, threshold_type, block_size, C_param)

	mean_intensity = np.mean(cam)

	return {
		'conn': num_components,
		'mean_intensity': mean_intensity
	}

def calculate_cam_metrics_enhanced(cam: np.ndarray,
								   threshold: float = 0.4,
								   threshold_type: str = 'adaptive',
								   block_size: int = 5,
								   C_param: int = 5):

	cam_8bit = (cam * 255).astype(np.uint8)
	if threshold_type == 'adaptive':
		thr_mask = cv2.adaptiveThreshold(
			src=cam_8bit,
			maxValue=255,
			adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
			thresholdType=cv2.THRESH_BINARY,
			blockSize=block_size,
			C=C_param
		)
	total_activated_pixels = np.sum(thr_mask > 0)
	MIN_ACTIVATED_PIXELS = 10
	if total_activated_pixels < MIN_ACTIVATED_PIXELS or np.mean(cam) < 1e-4:
		return {
			'conn': 0,
			'area_ratio': 0.0,
			'mean_intensity': np.mean(cam)
		}

	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
		image=thr_mask,
		connectivity=8
	)

	num_components = max(0, num_labels - 1)

	max_component_area = 0
	total_activated_area = 0
	area_ratio = 0.0

	if num_labels > 1:
		all_component_areas = stats[1:, cv2.CC_STAT_AREA]

		max_component_area = np.max(all_component_areas)
		total_activated_area = np.sum(all_component_areas)
		area_ratio = max_component_area / (total_activated_area + 1e-8)

	return {
		'conn': num_components,
		'area_ratio': area_ratio,
		'mean_intensity': np.mean(cam)
	}

def get_binary_mask(cam: np.ndarray,
                    threshold_type: str = 'global',
                    block_size: int = 5,
                    C_param: int = 5) -> np.ndarray:
    cam_8bit = (cam * 255).astype(np.uint8)

    if threshold_type == 'adaptive':
        thr_mask = cv2.adaptiveThreshold(
            src=cam_8bit,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=C_param
        )
    elif threshold_type == 'global':
        _, thr_mask = cv2.threshold(
            src=cam_8bit,
            thresh=0,  
            maxval=255,
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        raise ValueError("Invalid threshold_type. Must be 'adaptive' or 'global'.")

    binary_mask = (thr_mask > 0).astype(np.uint8)

    return binary_mask

def scoremap2bbox(scoremap, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=0,
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)
