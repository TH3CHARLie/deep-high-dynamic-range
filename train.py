from datetime import datetime
import sys
import pathlib
import util
from data import read_training_examples
from config import Config
from model import create_model_and_loss
import cv2
import tensorflow as tf
import numpy as np
import os
from random import shuffle


def train_main(config: Config):
    model_type = sys.argv[1]

    model, loss_function_generator = create_model_and_loss(model_type)
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists(config.SAVE_PATH):
        pathlib.Path(
            config.SAVE_PATH +
            time).mkdir(
            parents=True,
            exist_ok=True)
    if not os.path.exists(config.TENSORBOARD_PATH):
        pathlib.Path(
            config.TENSORBOARD_PATH +
            time).mkdir(
            parents=True,
            exist_ok=True)

    checkpoint = tf.train.Checkpoint(myModel=model)
    summary_writer = tf.summary.create_file_writer(
        config.TENSORBOARD_PATH + time + "/")
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.ADAM_LEARNING_RATE,
        beta_1=config.ADAM_BETA1,
        beta_2=config.ADAM_BETA2)
    paths = util.read_dir(config.TRAINING_DATA_PATH, folder_only=False)
    

    global_step = 0
    for epoch in range(3):
        shuffle(paths)
        paths_len = len(paths)
        evaluation_paths = paths[0:int(0.005 * paths_len) + 1]
        training_paths = paths[int(0.005 * paths_len) + 1:]
        # load tf record files into tf dataseat
        evaluation_dataset = read_training_examples(evaluation_paths)
        training_dataset = read_training_examples(training_paths)
        # transform dataset
        evaluation_dataset = evaluation_dataset.batch(config.BATCH_SIZE * 2)
        training_dataset = training_dataset.shuffle(120).batch(config.BATCH_SIZE)
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
                eval_cnt = 0
                eval_sum_loss = 0.0
                for eval_step, (eval_inputs_batch, eval_label_batch) in enumerate(
                        evaluation_dataset):
                    eval_outputs = model(eval_inputs_batch)
                    eval_loss_function = loss_function_generator(
                        eval_inputs_batch)
                    eval_loss_value = eval_loss_function(
                        eval_label_batch, eval_outputs)
                    eval_sum_loss += eval_loss_value
                    eval_cnt += 1
                eval_avg_loss = eval_sum_loss / eval_cnt
                print(
                    'Training loss (for one batch) at step %s: %s' %
                    (step, float(loss_value)))
                print(
                    'Evaluation loss(for %s batches) at step %s: %s' %
                    (eval_cnt, step, float(eval_avg_loss)))
                print(
                    'Seen so far: %s samples' %
                    ((step + 1) * config.BATCH_SIZE))
                print('Saving the models')
                with summary_writer.as_default():
                    tf.summary.scalar(
                        "eval_loss", eval_avg_loss, step=global_step)
                checkpoint.save(config.SAVE_PATH + time + "/model.ckpt")
            global_step += 1


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    device = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"]= device
    cfg = Config()
    train_main(cfg)
