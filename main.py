# main.py
import argparse
from src.baselines.train import train_pytorch_baseline
from src.baselines.train_xgboost import train_xgboost
from src.gnn.train import train_gnn

def main():
    parser = argparse.ArgumentParser(description="Run Landslide Prediction Models")
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        choices=['mlp', 'gru', 'xgboost', 'gnn'],
        help="Model to train: 'mlp', 'gru', 'xgboost', or 'gnn'"
    )
    args = parser.parse_args()

    if args.model == 'mlp':
        train_pytorch_baseline('mlp')
    elif args.model == 'gru':
        train_pytorch_baseline('gru')
    elif args.model == 'xgboost':
        train_xgboost()
    elif args.model == 'gnn':
        train_gnn()

if __name__ == "__main__":
    main()