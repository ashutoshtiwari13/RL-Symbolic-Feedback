import os
import json
from typing import Dict, Any, List
import torch
import wandb
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel, pipeline
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler
from datasets import Dataset

class StandardTrainer:
    def __init__(
        self,
        config: PPOConfig,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        checkpoint_dir: str = "standard_checkpoints"
    ):
        """
        Initialize the Standard Trainer.

        Args:
            config (PPOConfig): Configuration for PPO training.
            model (PreTrainedModel): The model to be trained.
            ref_model (PreTrainedModel): The reference model.
            tokenizer (PreTrainedTokenizer): The tokenizer.
            dataset (Dataset): The preprocessed dataset.
            checkpoint_dir (str): Directory to save checkpoints.
        """
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.checkpoint_dir = checkpoint_dir
        
        self.ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset)
        self.output_length_sampler = LengthSampler(4, 16)
        
        self.generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        self.device = self._get_device()
        self.sentiment_pipe_std = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=self.device)

    def train(self, num_epochs: int = 8, accumulation_steps: int = 4) -> PreTrainedModel:
        """
        Train the model using standard PPO.

        Args:
            num_epochs (int): Number of training epochs.
            accumulation_steps (int): Number of steps to accumulate gradients.

        Returns:
            PreTrainedModel: The trained model.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.json")

        wandb.init(project="standard-sentiment-analysis", name="standard-run")

        start_epoch = self._load_checkpoint(checkpoint_path)

        for epoch in range(start_epoch, num_epochs):
            epoch_rewards = self._train_epoch(accumulation_steps)
            self._log_epoch_stats(epoch, epoch_rewards)
            self._save_checkpoint(epoch, checkpoint_path)

        wandb.finish()
        return self.model

    def _train_epoch(self, accumulation_steps: int) -> List[float]:
        """Train for one epoch and return the rewards."""
        epoch_rewards = []
        for i, batch in enumerate(tqdm(self.ppo_trainer.dataloader)):
            query_tensors = batch["input_ids"]
            response_tensors = self._generate_responses(query_tensors)
            batch["response"] = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]

            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = self.sentiment_pipe_std(texts, top_k=None, function_to_apply="none", batch_size=16)
            rewards = [torch.tensor(output[0]["score"] if output[0]["label"] == "POSITIVE" else -output[0]["score"]) for output in pipe_outputs]

            epoch_rewards.extend([r.item() for r in rewards])

            stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
            self.ppo_trainer.log_stats(stats, batch, rewards)

        return epoch_rewards

    def _generate_responses(self, query_tensors: torch.Tensor) -> List[torch.Tensor]:
        """Generate responses for the given queries."""
        response_tensors = []
        for query in query_tensors:
            gen_len = self.output_length_sampler()
            self.generation_kwargs["max_new_tokens"] = gen_len
            query_response = self.ppo_trainer.generate(query, **self.generation_kwargs).squeeze()
            response_len = len(query_response) - len(query)
            response_tensors.append(query_response[-response_len:])
        return response_tensors

    def _log_epoch_stats(self, epoch: int, epoch_rewards: List[float]):
        """Log epoch statistics to Weights & Biases."""
        log_dict = {
            "epoch": epoch + 1,
            "mean_reward": np.mean(epoch_rewards),
            "median_reward": np.median(epoch_rewards),
            "min_reward": np.min(epoch_rewards),
            "max_reward": np.max(epoch_rewards),
            "reward_distribution": wandb.Histogram(epoch_rewards),
        }
        wandb.log(log_dict)

    def _save_checkpoint(self, epoch: int, checkpoint_path: str):
        """Save a checkpoint of the model and training progress."""
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "model.pth"))
        with open(checkpoint_path, "w") as f:
            json.dump({"epoch": epoch + 1}, f)

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load a checkpoint if it exists and return the starting epoch."""
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
            start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "model.pth")))
            print(f"Resuming training from epoch {start_epoch}")
            return start_epoch
        return 0

    def _get_device(self) -> torch.device:
        """Determine the appropriate device for training."""
        if self.ppo_trainer.accelerator.num_processes == 1:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.ppo_trainer.accelerator.device