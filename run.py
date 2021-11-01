import time
import os
import numpy as np
import pickle as pkl

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import optimizer
import warnings
from models import RelationalGraphConvModel
from data_utils import load_data
from utils import row_normalize, accuracy, get_splits
from params import args
from pytorchtools import EarlyStopping

warnings.filterwarnings("ignore")


class Train:
    def __init__(self, args):
        self.args = args
        self.best_val = 0
        # Load data
        self.A, self.y, self.train_idx, self.test_idx = self.input_data()
        self.num_nodes = self.A[0].shape[0]
        self.num_rel = len(self.A)
        self.labels = paddle.to_tensor(np.array(np.argmax(self.y, axis=-1)).squeeze())

        # Get dataset splits
        (
            self.y_train,
            self.y_val,
            self.y_test,
            self.idx_train,
            self.idx_val,
            self.idx_test,
        ) = get_splits(self.y, self.train_idx, self.test_idx, self.args.validation)

        # Adjacency matrix normalization
        self.A = row_normalize(self.A)

        # Create Model
        self.model = RelationalGraphConvModel(
            input_size=self.num_nodes,
            hidden_size=self.args.hidden,
            output_size=self.y_train.shape[1],
            num_bases=self.args.bases,
            num_rel=self.num_rel,
            num_layer=2,
            dropout=self.args.drop,
            featureless=True,
            cuda=self.args.using_cuda,
        )
        print(
            "Loaded %s dataset with %d entities, %d relations and %d classes"
            % (self.args.data, self.num_nodes, self.num_rel, self.y_train.shape[1])
        )

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.model_optimizer = optimizer.Adam(
            learning_rate=self.args.lr, parameters=self.model.parameters()
        )

        # initialize the early_stopping object
        if self.args.validation:
            self.early_stopping = EarlyStopping(patience=10, verbose=True)

        if self.args.using_cuda:
            print("Using the GPU")
            self.model.cuda()
            self.labels = self.labels.cuda()

    def input_data(self, dirname="./data"):
        data = None
        if os.path.isfile(
            dirname + "/" + self.args.data + "_" + str(self.args.hop) + ".pickle"
        ):
            with open(
                dirname + "/" + self.args.data + "_" + str(self.args.hop) + ".pickle",
                "rb",
            ) as f:
                data = pkl.load(f)
        else:
            with open(
                dirname + "/" + self.args.data + "_" + str(self.args.hop) + ".pickle",
                "wb",
            ) as f:
                # Data Loading...
                (
                    A,
                    X,
                    y,
                    labeled_nodes_idx,
                    train_idx,
                    test_idx,
                    rel_dict,
                    train_names,
                    test_names,
                ) = load_data(self.args.data, self.args.hop)
                data = {
                    "A": A,
                    "y": y,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
                pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
        return data["A"], data["y"], data["train_idx"], data["test_idx"]

    def train(self, epoch):
        t = time.time()
        X = None  # featureless
        # Start training
        self.model.train()
        emb_train = self.model(A=self.A, X=None)
        sit = paddle.to_tensor(self.idx_train)
        et = paddle.gather(emb_train, sit, axis=0)
        slt = paddle.gather(self.labels, sit, axis=0)
        loss = self.criterion(et, slt)
        # Backward and optimize
        self.model_optimizer.clear_grad()
        loss.backward()
        self.model_optimizer.step()

        print(
            "Epoch: {epoch}, Training Loss on {num} training data: {loss}".format(
                epoch=epoch, num=len(self.idx_train), loss=str(loss.item())
            )
        )

        if self.args.validation:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            with paddle.no_grad():
                self.model.eval()
                emb_valid = self.model(A=self.A, X=None)
                siv = paddle.to_tensor(self.idx_val)
                ev = paddle.gather(emb_valid, siv, axis=0)
                slv = paddle.gather(self.labels, siv, axis=0)
                loss_val = self.criterion(
                    ev, slv
                )
                acc_val = accuracy(ev, slv)
                if acc_val >= self.best_val:
                    self.best_val = acc_val
                    self.model_state = {
                        "state_dict": self.model.state_dict(),
                        "best_val": acc_val,
                        "best_epoch": epoch,
                        "model_optimizer": self.model_optimizer.state_dict(),
                    }
                print(
                    "loss_val: {:.4f}".format(loss_val.item()),
                    "acc_val: {:.4f}".format(acc_val.item()),
                    "time: {:.4f}s".format(time.time() - t),
                )
                print("\n")

                self.early_stopping(loss_val, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    self.model_state = {
                        "state_dict": self.model.state_dict(),
                        "best_val": acc_val,
                        "best_epoch": epoch,
                        "model_optimizer": self.model_optimizer.state_dict(),
                    }
                    return False
        return True

    def test(self):
        with paddle.no_grad():
            self.model.eval()
            emb_test = self.model(A=self.A, X=None)
            sit = paddle.to_tensor(self.idx_test)
            et = paddle.gather(emb_test, sit, axis=0)
            slt = paddle.gather(self.labels, sit, axis=0)
            loss_test = self.criterion(
                et, slt
            )
            acc_test = accuracy(et, slt)
            print(
                "Accuracy of the network on the {num} test data: {acc} %, loss: {loss}".format(
                    num=len(self.idx_test), acc=acc_test.item() * 100, loss=loss_test.item()
                )
            )

    def save_checkpoint(self, filename="./.checkpoints/" + args.name):
        print("Save model...")
        if not os.path.exists(".checkpoints"):
            os.makedirs(".checkpoints")
        paddle.save(self.model_state, filename)
        print("Successfully saved model\n...")

    def load_checkpoint(self, filename="./.checkpoints/" + args.name, ts="teacher"):
        print("Load model...")
        load_state = paddle.load(filename)
        self.model.set_state_dict(load_state["state_dict"])
        self.model_optimizer.set_state_dict(load_state["model_optimizer"])
        print("Successfully Loaded model\n...")
        print("Best Epoch:", load_state["best_epoch"])
        print("Best acc_val:", load_state["best_val"].item())


if __name__ == "__main__":
    train = Train(args)
    # train or pdparams
    if args.train:
        for epoch in range(args.epochs):
            if train.train(epoch) is False:
                break
        paddle.save(train.model.state_dict(), 'rgn_paddlepaddle.pdparams')
    else:
        train.model.load_dict(paddle.load('rgcn_paddlepaddle.pdparams'))
    # test
    train.test()
