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

class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

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

train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

# Which validation images do we want
test_indices = []
num_tests = len(test_input_names)
test_indices=range(0,len(test_input_names))

scores_list = []
class_scores_list = []
precision_list = []
recall_list = []
f1_list = []
iou_list = []
ind = 0

target = open("./test/dst_testDB/test_scores.csv",'w')
target.write("test_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))

# using file browsing /home/drswchornd/Env-semantic-seg/
st = time.time()
for filename in test_input_names:
  loaded_image = utils.load_image(filename)
  resized_image =cv2.resize(loaded_image, (args.crop_width, args.crop_height))
  input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

  output_image = sess.run(network, feed_dict={net_input:input_image})
  output_image = np.array(output_image[0,:,:,:])
  output_image = helpers.reverse_one_hot(output_image)
  out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

  gt = utils.load_image(test_output_names[ind])[:args.crop_height, :args.crop_width]
  gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

  accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

  file_name = utils.filepath_to_name(test_output_names[ind])
  target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
  for item in class_accuracies:
      target.write(", %f"%(item))
  target.write("\n")

  scores_list.append(accuracy)
  class_scores_list.append(class_accuracies)
  precision_list.append(prec)
  recall_list.append(rec)
  f1_list.append(f1)
  iou_list.append(iou)
  ind = ind + 1

  file_name_with_ext = os.path.basename(filename)
  file_name = os.path.splitext(file_name_with_ext)[0]
  print("Wrote image " + "%s_pred.png"%("./test/dst_testDB/" + file_name))
  cv2.imwrite("%s_pred.png"%("./test/dst_testDB/" + file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

avg_score = np.mean(scores_list)
class_avg_scores = np.mean(class_scores_list, axis=0)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_iou = np.mean(iou_list)

for index, item in enumerate(class_avg_scores):
    print("%s = %f" % (class_names_list[index], item))
print("Test precision = ", avg_precision)
print("Test recall = ", avg_recall)
print("Test F1 score = ", avg_f1)
print("Test IoU score = ", avg_iou)

print("")
run_time = time.time()-st
print("Finished! %f [sec]"%run_time)

