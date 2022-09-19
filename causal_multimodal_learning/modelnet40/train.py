import pytorch_lightning as pl
from argparse import ArgumentParser
from utils.framework import make_trainer
from modelnet40.data import make_data
from modelnet40.model import SemiSupervisedVae

def main(args):
    pl.seed_everything(args.seed)
    model = SemiSupervisedVae(args.lr, args.clip_name, args.latent_dim)
    data_train, data_val, data_test = make_data(args.batch_size, args.n_workers, model.encoder.clip_transforms)
    trainer = make_trainer(args.name, args.seed, args.n_epochs, args.patience)
    trainer.fit(model, data_train, data_val)
    trainer.test(model, data_test)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="modelnet40")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip_name", type=str, default="ViT-B/32")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--n_workers", type=int, default=20)
    parser.add_argument("--latent_dim", type=int, default=100)
    main(parser.parse_args())