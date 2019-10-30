from datetime import datetime
import sys
import pathlib
import util
from data import read_training_examples, read_test_examples
from config import *
from model import create_model_and_loss, tf_compute_PSNR
import cv2
import tensorflow as tf
import numpy as np
import os
from random import shuffle


def train_main():
    model_type = sys.argv[1]

    model, loss_function_generator, output_function = create_model_and_loss(
        model_type)
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    folder = time + '-' + model_type
    if not os.path.exists(SAVE_PATH):
        pathlib.Path(
            SAVE_PATH +
            folder).mkdir(
            parents=True,
            exist_ok=True)
    if not os.path.exists(TENSORBOARD_PATH):
        pathlib.Path(
            TENSORBOARD_PATH +
            folder).mkdir(
            parents=True,
            exist_ok=True)

    checkpoint = tf.train.Checkpoint(myModel=model)
    summary_writer = tf.summary.create_file_writer(
        TENSORBOARD_PATH + folder + "/")
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=ADAM_LEARNING_RATE,
        beta_1=ADAM_BETA1,
        beta_2=ADAM_BETA2)
    # training dataset
    paths = util.read_dir(TRAINING_DATA_PATH, folder_only=False)
    training_dataset = read_training_examples(paths)
    training_dataset = training_dataset.shuffle(120).batch(BATCH_SIZE)
    # test dataset
    test_paths = util.read_dir(TEST_DATA_PATH, folder_only=False)
    test_dataset = read_test_examples(test_paths)
    test_dataset = test_dataset.batch(1)
    global_step = 0
    for epoch in range(3):
        print('Start of epoch %d' % (epoch, ))

        for step, (inputs_batch, label_batch) in enumerate(training_dataset):
            with tf.GradientTape() as tape:
                outputs = model(inputs_batch)
                loss_function = loss_function_generator(inputs_batch)
                loss_value = loss_function(label_batch, outputs)
                with summary_writer.as_default():
                    tf.summary.scalar("loss", loss_value, step=global_step)

            grads = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if step != 0 and step % 1000 == 0:
                scene_cnt = 0
                sum_psnr = 0.0
                for (inputs, label) in test_dataset:
                    outputs = model(inputs)
                    output_image = output_function(inputs, outputs)
                    psnr = tf_compute_PSNR(output_image, label)
                    sum_psnr += psnr.numpy()
                    scene_cnt += 1
                print(
                    'Training loss (for one batch) at step %s: %s' %
                    (step, float(loss_value)))
                print(
                    'Seen so far: %s samples' %
                    ((step + 1) * BATCH_SIZE))
                print('Test PSNR at step %s %s' %
                      (step, float(sum_psnr / scene_cnt)))
                print('Saving the models')
                with summary_writer.as_default():
                    tf.summary.scalar(
                        "test_PSNR", sum_psnr / scene_cnt, step=global_step)
                checkpoint.save(SAVE_PATH + folder + "/model.ckpt")
            global_step += 1


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    train_main()
