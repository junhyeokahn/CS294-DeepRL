#!/usr/bin/env python

import os
import sys
import numpy as np
import tensorflow as tf
import logging
import gym
from bc_data import Data
from bc_net import BehaviorCloneNetwork

def config_logging(log_file):
    if os.path.exists(log_file):
        os.remove(log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def create_model(session, logger, opt, lr, restore):
    model = BehaviorCloneNetwork(optimizer=opt, lr=lr)
    ckpt = tf.train.latest_checkpoint("models")

    if restore:
        logger.info("[Restore Model Parameters from %s]" % ckpt)
        model.saver.restore(session, ckpt)
    else:
        logger.info("[Create Model with New Parameters]")
        session.run(tf.global_variables_initializer())

    return model

def train(params):
    logger = config_logging(params.log_dir + "train_out.log")
    data = Data(params.data_file, train_ratio=0.9, val_ratio=0.05)
    training_set = data.train

    num_train = len(training_set["observations"])
    batch_size, num_epochs = params.batch_size, params.num_epochs
    num_batch_per_epoch = int((num_train - 1) / batch_size) + 1
    avg_loss, min_val_loss = 0, sys.maxsize


    with tf.Session() as sess:
        model = create_model(sess, logger, params.optimizer, params.lr,
                params.restore)

        batches = data.batch_iter(training_set, params.batch_size, params.num_epochs)
        for i, (batch_x, batch_y) in enumerate(batches):
            pred, loss = model.step(sess, batch_x, batch_y)
            avg_loss += loss
            num_epoch = i / num_batch_per_epoch
            if i % 100 == 0:
                logger.debug("Epoch %04d step %08d loss %04f" % (num_epoch, i, loss))

            if i>0 and (i+1) % num_batch_per_epoch == 0:
                avg_loss /= num_batch_per_epoch
                logger.debug("###################################")
                logger.info("Finished epoch %d, average training loss = %f"
                        % (num_epoch, avg_loss))
                if params.val:
                    min_val_loss = validate(
                            model, logger, sess, data, num_epoch, batch_size, min_val_loss,
                            params.ckpt_dir)
                logger.debug("###################################")
                avg_loss = 0

def validate(model, logger, sess, data, num_epoch, batch_size, min_loss, ckpt_dir):
    batches = data.batch_iter(data.val, batch_size, 1)
    avg_loss = []
    for i, (batch_x, batch_y) in enumerate(batches):
        pred, loss = model.step(sess, batch_x, batch_y, is_train=False)
        avg_loss.append(loss)
    new_loss = sum(avg_loss) / len(avg_loss)
    logger.info("Finished epoch %d, average validation loss = %f" % (num_epoch, new_loss))
    if new_loss < min_loss:
        model.saver.save(sess, ckpt_dir + "model_bc.ckpt")
        logger.info("Model saved!")
        min_loss = new_loss
    return min_loss

def run_bc(params):
    logger = config_logging(params.log_dir + "test_out.log")

    with tf.Session() as sess:
        model = create_model(sess, logger, params.optimizer, params.lr, restore=True)
        env = gym.make(params.envname)
        max_steps = env.spec.timestep_limit

        returns, observations, actions = [], [], []
        for i in range(params.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            total_reward = 0.
            steps = 0
            while not done:
                action = model.step(sess, obs[None, :], is_train=False)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                total_reward += r
                steps += 1
                env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(total_reward)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/hopper_data.pkl")
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--ckpt_dir", type=str, default="models/")
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--restore", type=bool, default=False)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--val", type=bool, default=True)
    parser.add_argument("--overfit", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=True)
    parser.add_argument('--envname', type=str, default="Hopper-v1")
    parser.add_argument('--num_rollouts', type=int, default=20)
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.test:
        run_bc(args)

if __name__ == '__main__':
    main()
