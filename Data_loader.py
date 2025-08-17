from __future__ import print_function, division
import torch
from skimage import io, transform, color
import numpy as np
from torch.utils.data import Dataset
import random

#==========================dataset load==========================
class Rescale(object):
	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		image, label = sample['image'],sample['label']

		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'image':img,'label':lbl}

class Rotate(object):
	def __init__(self,flag=1):
		self.flag = flag

	def __call__(self,sample):
		image, label = sample['image'],sample['label']

		if self.flag ==1: # 随机旋转 0==0, 1==90, 2==180, 3==270
			degree = random.randint(0, 3)
			if degree==0:
				img = transform.rotate(image, 0)
				lbl = transform.rotate(label, 0)
			elif degree==1:
				img = transform.rotate(image, 90)
				lbl = transform.rotate(label, 90)
			elif degree==2:
				img = transform.rotate(image, 180)
				lbl = transform.rotate(label, 180)
			elif degree==3:
				img = transform.rotate(image, 270)
				lbl = transform.rotate(label, 270)
		else:
			img = transform.rotate(image, 0)
			lbl = transform.rotate(label, 0)

		return {'image':img,'label':lbl}

class ToTensor(object):
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		image, label = sample['image'], sample['label']

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		if self.flag == 2:
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1:
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else:
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		tmpImg = tmpImg.transpose((2, 0, 1)) # H W C --> C H W
		tmpLbl = label.transpose((2, 0, 1)) # H W C --> C H W

		return {'image': torch.from_numpy(tmpImg),
			'label': torch.from_numpy(tmpLbl)}


def weather_type2label(type):
	if type=='clean':
		return 0
	elif type=='rain':
		return 1
	elif type=='snow':
		return 2
	elif type=='fog':
		return 3
	elif type=='dark':
		return 4
	elif type=='light':
		return 5
	elif type=='rainafog':
		return 6
	elif type=='snowafog':
		return 7
	elif type=='rainasnow':
		return 8
	else:
		print('ERROR: Unknow Weather type')


# 返回：输入RGB_input、GT-mask、weather-label
class SalObjDataset_train(Dataset):
	def __init__(self,img_name_list, lbl_name_list, transform=None):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):
		# 加载RGB图片
		image = io.imread(self.image_name_list[idx])

		# 获取图片对应的天气类别
		image_name = self.image_name_list[idx].split("/")[-1] # 例如 0001_light.jpg
		image_number = image_name.split('.')[0]
		weather_type = image_number.split('_')[-1] # light\rain\rainafog\......
		weather_label = weather_type2label(weather_type) # 把字符串转换为数字，方便计算loss,[0-8]

		# 加载RGB图片对应的GT
		if (0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])

		label = np.zeros(label_3.shape[0:2])
		if (3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif (2==len(label_3.shape)):
			label = label_3

		if (3==len(image.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif (2==len(image.shape) and 2==len(label.shape)):
			image = image[:,:,np.newaxis]
			label = label[:,:,np.newaxis]

		sample = {'image':image, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample, weather_label


# 返回：输入RGB、GT-mask
class SalObjDataset_test(Dataset):
	def __init__(self,img_name_list,lbl_name_list,transform=None):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):
		# 加载RGB图片
		image = io.imread(self.image_name_list[idx])

		# 获取图片对应的天气类别
		image_name = self.image_name_list[idx].split("/")[-1] # 例如 0001_light.jpg
		image_number = image_name.split('.')[0]
		weather_type = image_number.split('_')[-1] # light\rain\rainafog\......
		weather_label = weather_type2label(weather_type) # 把字符串转换为数字，方便计算loss,[1-9]

		# 加载RGB图片对应的GT
		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])

		label = np.zeros(label_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3

		if(3==len(image.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif(2==len(image.shape) and 2==len(label.shape)):
			image = image[:,:,np.newaxis]
			label = label[:,:,np.newaxis]
		sample = {'image':image, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample, weather_label