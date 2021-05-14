import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import glob

from utils import utils, helpers
from builders import model_builder

import keras as K

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default="./test/src/test_9.jpg", required=False, help='The image you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/latest_model_MobileUNet_kariDB.ckpt", required=False, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=480, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default="MobileUNet", required=False, help='The model you are using')
parser.add_argument('--dataset', type=str, default="kariDB", required=False, help='The dataset you are using')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "classes.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Image -->", args.image)

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

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

ind = 0
run_time = 0.0
st = time.time()

# using file browsing
for filename in glob.glob('./test/src/*.png'): 
  loaded_image = utils.load_image(filename)
  resized_image =cv2.resize(loaded_image, (args.crop_width, args.crop_height))
  input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

  output_image = sess.run(network, feed_dict={net_input:input_image})

  # saving pbtxt file
  #tf.io.write_graph(sess.graph_def, './checkpoints', "latest_model_predict_"+ args.model + "_" + args.dataset + ".pb", as_text=False)
  #tf.io.write_graph(sess.graph_def, './checkpoints', "latest_model_predict_"+ args.model + "_" + args.dataset + ".pbtxt", as_text=True)

  output_image = np.array(output_image[0,:,:,:])
  output_image = helpers.reverse_one_hot(output_image)

  out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

  file_name_with_ext = os.path.basename(filename)
  file_name = os.path.splitext(file_name_with_ext)[0]
  # print("Wrote image " + "%s_pred.png"%("./test/dst/" + file_name))
  os.system('clear')
  print("Wrote image[ind]:", ind)
  ind = ind + 1
  cv2.imwrite("%s_pred.png"%("./test/dst/" + file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

print("")
run_time = time.time()-st
print("Finished! %f [sec]"%run_time)

