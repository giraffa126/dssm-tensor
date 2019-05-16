# coding: utf8
import sys
import os
import numpy as np
import tensorflow as tf
import argparse
from model import DSSMNet 
from reader import DataReader

def train(args):
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    tf.reset_default_graph()
    model = DSSMNet(vocab_size=args.vocab_size)
    # optimizer
    train_step = tf.contrib.opt.LazyAdamOptimizer(learning_rate=args.learning_rate).minimize(model.loss)
    saver = tf.train.Saver()
    loss_summary = tf.summary.scalar("train_loss", model.loss)
    init = tf.group(tf.global_variables_initializer(), 
            tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        # feeding embedding
        _writer = tf.summary.FileWriter(args.logdir, sess.graph)

        # summary
        summary_op = tf.summary.merge([loss_summary])
        step = 0
        for epoch in range(args.epochs):
            train_reader = DataReader(args.vocab_path, args.train_data_path, 
                    args.vocab_size, args.batch_size)
            for train_batch in train_reader.batch_generator():
                query, pos, neg = train_batch
                _, _loss, _summary = sess.run([train_step, model.loss, summary_op],
                        feed_dict={model.query_in: query, model.pos_in: pos, model.neg_in: neg})
                _writer.add_summary(_summary, step)
                step += 1

                # test
                sum_loss = 0.0
                iters = 0
                summary = tf.Summary()
                if step % args.eval_interval == 0:
                    print("Epochs: {}, Step: {}, Train Loss: {}".format(epoch, step, _loss))

                    test_reader = DataReader(args.vocab_path, args.test_data_path, 
                            args.vocab_size, args.batch_size)
                    for test_batch in test_reader.batch_generator():
                        query, pos, neg = test_batch
                        _loss = sess.run(model.loss,
                                feed_dict={model.query_in: query, model.pos_in: pos, model.neg_in: neg})
                        sum_loss += _loss
                        iters += 1
                    avg_loss = sum_loss / iters
                    summary.value.add(tag="test_loss", simple_value=avg_loss)
                    _writer.add_summary(summary, step)
                    print("Epochs: {}, Step: {}, Test Loss: {}".format(epoch, step, sum_loss / iters))
                if step % args.save_interval == 0:
                    save_path = saver.save(sess, "{}/model.ckpt".format(args.model_path), global_step=step)
                    print("Model save to path: {}/model.ckpt".format(args.model_path))


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--train_data_path", type=str, default="./data/train.txt")
    parser.add_argument("--test_data_path", type=str, default="./data/test.txt")
    parser.add_argument("--vocab_path", type=str, default="./data/vocab.pkl")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--save_interval", type=int, default="100")
    parser.add_argument("--eval_interval", type=int, default="100")
    args = parser.parse_args()
    train(args)
    
