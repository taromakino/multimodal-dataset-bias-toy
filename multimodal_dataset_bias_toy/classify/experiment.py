import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from data import make_data
from model import MultimodalClassifier, UnimodalClassifier
from utils.file import save_file
from utils.nn_utils import make_trainer


def main(config):
    seed = config.__dict__.pop("seed")
    save_file(config, os.path.join(config.dpath, "args.pkl"))
    os.makedirs(config.dpath, exist_ok=True)
    pl.seed_everything(seed)
    data_train, data_val, data_test = make_data(seed, config.n_examples, config.input_dim,
        config.origin_offset, config.temperature, True, False, config.batch_size, config.n_workers)
    multimodal_model = MultimodalClassifier(config.input_dim, config.hidden_dims, config.lr)
    unimodal_model = UnimodalClassifier(config.input_dim, config.hidden_dims, config.lr)
    multimodal_trainer = make_trainer(os.path.join(config.dpath, "multimodal"), seed, config.n_epochs,
        config.early_stop_ratio, config.n_gpus)
    unimodal_trainer = make_trainer(os.path.join(config.dpath, "unimodal"), seed, config.n_epochs,
        config.early_stop_ratio, config.n_gpus)
    multimodal_trainer.fit(multimodal_model, data_train, data_val)
    multimodal_trainer.test(multimodal_model, data_test, ckpt_path="best")
    unimodal_trainer.fit(unimodal_model, data_train, data_val)
    unimodal_trainer.test(unimodal_model, data_test, ckpt_path="best")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_examples", nargs="+", type=int, default=[60, 20, 20])
    parser.add_argument("--input_dim", type=int, default=16)
    parser.add_argument("--origin_offset", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=100)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--early_stop_ratio", type=float, default=0.1)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=20)
    args = parser.parse_args()
    main(args)