import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from utils.file import save_file
from utils.nn_utils import make_trainer
from data import make_data
from model import UnimodalEnsemble, Multimodal

def main(config):
    seed = config.__dict__.pop("seed")
    save_file(config, os.path.join(config.dpath, "args.pkl"))
    os.makedirs(config.dpath, exist_ok=True)
    pl.seed_everything(seed)
    data_train, data_val, data_test = make_data(seed, config.n_examples, config.s_shift, config.data_dim,
        config.batch_size, config.n_workers)
    model_class = Multimodal if config.is_multimodal else UnimodalEnsemble
    model = model_class(seed, config.dpath, config.data_dim, config.hidden_dims, config.lr)
    trainer = make_trainer(config.dpath, seed, config.n_epochs, config.patience)
    trainer.fit(model, data_train, data_val)
    trainer.test(model, data_test)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpath", type=str, required=True)
    parser.add_argument("--n_examples", nargs="+", type=int, default=[1000, 1000, 1000])
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--s_shift", type=float, default=0)
    parser.add_argument("--data_dim", type=int, default=1)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_workers", type=int, default=20)
    parser.add_argument("--is_multimodal", action="store_true")
    main(parser.parse_args())