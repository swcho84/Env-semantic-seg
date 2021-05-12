import os,time,cv2, sys, math, time
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder

import keras as K

# parsing params
parser = argparse.ArgumentParser()
parser.add_argument('--rawimgfolder', type=str, default="/SemSeg_handler/raw/", required=False, help='The imgfolder you want to predict on. ')
parser.add_argument('--resimgfolder', type=str, default="/SemSeg_handler/res/", required=False, help='The imgfolder you want to save on. ')
parser.add_argument('--rawimage', type=str, default="raw_img.png", required=False, help='The image you want to predict on. ')
parser.add_argument('--resimage', type=str, default="res_img.png", required=False, help='The image you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/latest_model_MobileUNet_kariDB.ckpt", required=False, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=480, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default="MobileUNet", required=False, help='The model you are using')
parser.add_argument('--dataset', type=str, default="kariDB", required=False, help='The dataset you are using')
parser.add_argument('--duration', type=float, default=1.5, required=False, help='')
args = parser.parse_args()
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "classes.csv"))
num_classes = len(label_values)

# generating the folder path
rawImgFilePath = os.environ['HOME'] + args.rawimgfolder + args.rawimage
resImgFilePath = os.environ['HOME'] + args.resimgfolder + args.resimage

# for debugging
print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("RawImageFilePath -->", rawImgFilePath)
print("ResImageFilePath -->", resImgFilePath)
print("Duration -->", args.duration)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])
network, _ = model_builder.build_model(args.model, net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=args.crop_width,
                                        crop_height=args.crop_height,
                                        is_training=False)
sess.run(tf.global_variables_initializer())

# loading model, ckpt type
print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

# for debugging
print("Target: " + rawImgFilePath)

# main loop for inferencing the raw image
while True:
	if os.path.isfile(rawImgFilePath):
		loaded_image = utils.load_image(rawImgFilePath)
		resized_image = cv2.resize(loaded_image, (args.crop_width, args.crop_height))
		input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

		st = time.time()
		output_image = sess.run(network, feed_dict={net_input:input_image})
		output_image = np.array(output_image[0,:,:,:])
		output_image = helpers.reverse_one_hot(output_image)

		out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
		run_time = time.time()-st
		cv2.imwrite(resImgFilePath, cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

		print("Finished! %f [sec]"%run_time)
		print("Wrote image:", resImgFilePath)	
	else:
		print("Raw image file is not exist..please check...")

	time.sleep(args.duration)
