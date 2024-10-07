import os
import json
from typing import Dict, Any
import torch
import numpy as np
import wandb
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel, get_linear_schedule_with_warmup
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler
from datasets import Dataset

class RLSFTrainer:
    def __init__(
        self,
        config: PPOConfig,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        symbolic_env: Any,
        checkpoint_dir: str = "rlsf_checkpoints"
    ):
        """
        Initialize the RLSF Trainer.

        Args:
            config (PPOConfig): Configuration for PPO training.
            model (PreTrainedModel): The model to be trained.
            ref_model (PreTrainedModel): The reference model.
            tokenizer (PreTrainedTokenizer): The tokenizer.
            dataset (Dataset): The preprocessed dataset.
            symbolic_env (Any): The SymbolicEnvironment object.
            checkpoint_dir (str): Directory to save checkpoints.
        """
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.symbolic_env = symbolic_env
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

    def train(self, num_epochs: int = 8) -> PreTrainedModel:
        """
        Train the model using Reinforcement Learning with Symbolic Feedback (RLSF).

        Args:
            num_epochs (int): Number of training epochs.

        Returns:
            PreTrainedModel: The trained model.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.json")

        wandb.init(project="rlsf-sentiment-analysis", name="rlsf-run")

        start_epoch = self._load_checkpoint(checkpoint_path)

        num_training_steps = len(self.ppo_trainer.dataloader) * num_epochs
        lr_scheduler = get_linear_schedule_with_warmup(
            self.ppo_trainer.optimizer,
            num_warmup_steps=100,
            num_training_steps=num_training_steps
        )

        for epoch in range(start_epoch, num_epochs):
            epoch_rewards = self._train_epoch(lr_scheduler)
            self._log_epoch_stats(epoch, epoch_rewards)
            self._save_checkpoint(epoch, checkpoint_path)

        wandb.finish()
        return self.model

    def _train_epoch(self, lr_scheduler) -> List[float]:
        """Train for one epoch and return the rewards."""
        epoch_rewards = []
        for batch in tqdm(self.ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]
            response_tensors = self._generate_responses(query_tensors)
            batch["response"] = [self.tokenizer.decode(r) for r in response_tensors]
            
            rewards = self.symbolic_env.get_vectorized_reward_batch(batch["query"], batch["response"])
            rewards = [r.squeeze() for r in rewards]
            epoch_rewards.extend([r.item() for r in rewards])

            stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
            self.ppo_trainer.log_stats(stats, batch, rewards)
            lr_scheduler.step()
        
        return epoch_rewards

    def _generate_responses(self, query_tensors: torch.Tensor) -> List[torch.Tensor]:
        """Generate responses for the given queries."""
        response_tensors = []
        gen_len = self.output_length_sampler()
        self.generation_kwargs["max_new_tokens"] = gen_len

        for query in query_tensors:
            response = self.ppo_trainer.generate(
                query_tensor=query,
                return_prompt=False,
                **self.generation_kwargs
            )
            response_tensors.append(response.squeeze())
        
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