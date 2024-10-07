import os
import sys
import logging
import argparse
import torch
from transformers import AutoTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead

from src.config import get_ppo_config
from src.data import build_dataset, build_dataset_standard
from src.models import save_model
from src.training.rlsf_trainer import RLSFTrainer
from src.training.standard_trainer import StandardTrainer
from src.training.utils import ModelEvaluator
from src.environment import SymbolicEnvironment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_trl():
    """Clone and install the TRL repository."""
    try:
        os.system("git clone https://github.com/lvwerra/trl.git")
        os.chdir("trl")
        os.system("pip install -e .")
        os.chdir("..")
        logger.info("TRL repository cloned and installed successfully.")
    except Exception as e:
        logger.error(f"Error setting up TRL: {str(e)}")
        sys.exit(1)

def setup_models_and_tokenizers(config):
    """Initialize models, tokenizers, and datasets."""
    try:
        rlsf_dataset, rlsf_tokenizer = build_dataset(config)
        standard_dataset, standard_tokenizer = build_dataset_standard(config)

        rlsf_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
        standard_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rlsf_model.to(device)
        standard_model.to(device)
        ref_model.to(device)

        logger.info(f"Models and tokenizers set up successfully. Using device: {device}")
        return rlsf_model, standard_model, ref_model, rlsf_tokenizer, standard_tokenizer, rlsf_dataset, standard_dataset
    except Exception as e:
        logger.error(f"Error setting up models and tokenizers: {str(e)}")
        sys.exit(1)

def train_and_evaluate(args):
    """Main function to train and evaluate models."""
    setup_trl()

    config = get_ppo_config()
    rlsf_model, standard_model, ref_model, rlsf_tokenizer, standard_tokenizer, rlsf_dataset, standard_dataset = setup_models_and_tokenizers(config)

    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=rlsf_model.device)
    symbolic_env = SymbolicEnvironment(sentiment_pipe, rlsf_tokenizer)

    # Train RLSF model
    logger.info("Starting RLSF model training...")
    rlsf_trainer = RLSFTrainer(config, rlsf_model, ref_model, rlsf_tokenizer, rlsf_dataset, symbolic_env)
    trained_rlsf_model = rlsf_trainer.train(num_epochs=args.epochs)
    save_model(trained_rlsf_model, "rlsf_model")
    logger.info("RLSF model training completed and model saved.")

    # Train standard model
    logger.info("Starting standard model training...")
    standard_trainer = StandardTrainer(config, standard_model, ref_model, standard_tokenizer, standard_dataset)
    trained_standard_model = standard_trainer.train(num_epochs=args.epochs, accumulation_steps=args.accumulation_steps)
    save_model(trained_standard_model, "standard_model")
    logger.info("Standard model training completed and model saved.")

    # Evaluate models
    logger.info("Evaluating models...")
    rlsf_evaluator = ModelEvaluator(trained_rlsf_model, rlsf_dataset, rlsf_tokenizer, sentiment_pipe)
    standard_evaluator = ModelEvaluator(trained_standard_model, standard_dataset, standard_tokenizer, sentiment_pipe)

    rlsf_reward = rlsf_evaluator.evaluate(num_samples=args.eval_samples)
    standard_reward = standard_evaluator.evaluate(num_samples=args.eval_samples)

    logger.info(f"RLSF Model Average Reward: {rlsf_reward}")
    logger.info(f"Standard Model Average Reward: {standard_reward}")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate RLSF and standard models for sentiment analysis.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--eval_samples", type=int, default=1000, help="Number of samples to use for evaluation")
    args = parser.parse_args()

    try:
        train_and_evaluate(args)
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()