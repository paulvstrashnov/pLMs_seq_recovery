# main.py
import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from models import ESM2, ProtBERTModel, Electra, ProtT5Model
from utils import load_fasta_file
from evaluation import evaluate_batch
from transformers import logging as transformers_logging
from tqdm import tqdm

transformers_logging.set_verbosity_error()

def setup_logging(log_file):
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

from models import ESM2, ESM2dec, ProtBERTModel, Electra, ProtT5Model

def create_model(model_key, device="cpu"):
    model_classes = {
        "ESM2_15B": (ESM2, 'facebook/esm2_t48_15B_UR50D'),   # Embedding size: 5120
        "ESM2_3B": (ESM2, 'facebook/esm2_t36_3B_UR50D'),     # Embedding size: 2560
        "ESM2_650M": (ESM2, 'facebook/esm2_t33_650M_UR50D'), # Embedding size: 1280
        "ESM2_150M": (ESM2, 'facebook/esm2_t30_150M_UR50D'), # Embedding size: 640
        "ESM2_35M": (ESM2, 'facebook/esm2_t12_35M_UR50D'),   # Embedding size: 480
        "ESM2_8M": (ESM2, 'facebook/esm2_t6_8M_UR50D'),      # Embedding size: 320

        "ESM2_15B_dec": (ESM2dec, 'facebook/esm2_t48_15B_UR50D'),
        "ESM2_3B_dec": (ESM2dec, 'facebook/esm2_t36_3B_UR50D'),
        "ESM2_650M_dec": (ESM2dec, 'facebook/esm2_t33_650M_UR50D'),
        "ESM2_150M_dec": (ESM2dec, 'facebook/esm2_t30_150M_UR50D'),
        "ESM2_35M_dec": (ESM2dec, 'facebook/esm2_t12_35M_UR50D'),
        "ESM2_8M_dec": (ESM2dec, 'facebook/esm2_t6_8M_UR50D'),

        "ProtBERT": (ProtBERTModel, 'Rostlab/prot_bert_bfd'),
        "Electra": (Electra, 'Rostlab/prot_electra_generator_bfd'),
        "ProtT5": (ProtT5Model, 'Rostlab/prot_t5_xl_uniref50'),
    }

    if model_key in model_classes:
        model_class, model_params = model_classes[model_key]
        return model_class(model_params, device=device)
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def main(args):
    setup_logging(args.log_file)

    # Load and preprocess the dataset
    fasta_file_path = args.data_path
    dataset = load_fasta_file(fasta_file_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 3. Create a list of model names
    model_names = args.models.split(',')

    # 4. Run the evaluation loop
    results = {}
    if not args.device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logging.info(f"Using {device}")

    for model_name in model_names:
        # Create the model instance
        model = create_model(model_name, device=device)

        total_distance = 0
        total_score = 0
        total_normalized_distance = 0
        num_samples = 0
        # print(dataset[0])
        # encoded = model.encode(dataset[0])
        # print(encoded.shape)
        # break

        logging.info(f"Starting {model_name} evaluation")

        for batch in tqdm(dataloader):
            mean_distance, mean_normalized_distance, mean_score = evaluate_batch(batch, model, device)
            total_distance += mean_distance * len(batch)
            total_normalized_distance += mean_normalized_distance * len(batch)
            total_score += mean_score * len(batch)
            num_samples += len(batch)
            # logging.info(f"Evaluated samples so far: {num_samples}")

        if num_samples == 0:
            logging.error(f"Model {model_name} did not produce any results.")
            continue
        average_distance = total_distance / num_samples
        average_normalized_distance = total_normalized_distance / num_samples
        average_score = total_score / num_samples
        results[model_name] = (average_distance, average_score)
        logging.info(f"{model_name}: Average Levenshtein distance: {average_distance:.2f}")
        logging.info(f"{model_name}: Average Normalized Levenshtein distance: {average_normalized_distance:.2f}")
        logging.info(f"{model_name}: Average (NW - NW_self) score: {average_score:.2f}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate protein language models.")
    parser.add_argument("--models", type=str, help="Comma-separated list of model names to evaluate (e.g., 'ESM2_650M,ProtBERT,ProtT5').")
    parser.add_argument("--data_path", type=str, default="data/test.fasta", help="Path to the FASTA file containing the test dataset.")
    parser.add_argument("--device", type=str, help="Device to use for evaluation (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use for evaluation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for evaluation.")
    parser.add_argument("--log_file", type=str, default="logs.log", help="Path to the log file.")
    args = parser.parse_args()
    
    
    main(args)
