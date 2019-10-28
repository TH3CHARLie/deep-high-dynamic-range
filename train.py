import numpy as np
import os
# test training only on 1 GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import cv2
from model import create_model_and_loss
from config import Config
from data import read_training_examples
import util
import pathlib
import sys
from datetime import datetime


def train_main(config: Config):
    model_type = sys.argv[1]

    model, loss_function = create_model_and_loss(model_type)
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists(config.SAVE_PATH):
        pathlib.Path(config.SAVE_PATH + time).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(config.TENSORBOARD_PATH):
        pathlib.Path(config.TENSORBOARD_PATH + time).mkdir(parents=True, exist_ok=True)
    
    checkpoint = tf.train.Checkpoint(myModel=model)
    summary_writer = tf.summary.create_file_writer(config.TENSORBOARD_PATH + time + "/") 
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.ADAM_LEARNING_RATE,
        beta_1=config.ADAM_BETA1,
        beta_2=config.ADAM_BETA2)
    paths = util.read_dir(config.TRAINING_DATA_PATH, folder_only=False)
    
    # load tf record files into tf dataseat
    dataset = read_training_examples(paths)
    # transform dataset
    dataset = dataset.shuffle(120).batch(config.BATCH_SIZE)
    
    global_step = 0
    for epoch in range(3):
        print('Start of epoch %d' % (epoch, ))

        for step, (inputs_batch, label_batch) in enumerate(dataset):
            with tf.GradientTape() as tape:
                outputs = model(inputs_batch)
                loss_value = loss_function(label_batch, outputs)
                with summary_writer.as_default():
                    tf.summary.scalar("loss", loss_value, step=global_step) 
            
            grads = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if step % 1000 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * config.BATCH_SIZE))
                print('Saving the models')
                checkpoint.save(config.SAVE_PATH + time + "/model.ckpt")
            global_step += 1

# train_direct
# train_WE
# train_WIE TODO:

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    cfg = Config()
    train_main(cfg)
