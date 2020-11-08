import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt_path', 
                    '/home/drswchornd/Env-semantic-seg/checkpoints/latest_model_MobileUNet_kariDB.ckpt', 
                    'path to ckpt files')

def main(argv):
    ckpt_path = FLAGS.ckpt_path

    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(ckpt_path + '.meta')
    g = tf.get_default_graph()
    input_graph_def = g.as_graph_def()
    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)
        writer = tf.summary.FileWriter('/home/drswchornd/Env-semantic-seg/tensorboard_graphs')
        writer.add_graph(sess.graph)

        i = 0
        for n in tf.get_default_graph().as_graph_def().node:
            print(n.name,i);    
            i += 1
        #end for
        print("total:",i);
      
if __name__ == '__main__':
  app.run(main)

