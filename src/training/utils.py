import torch
from typing import List
from transformers import PreTrainedTokenizer, PreTrainedModel, Pipeline
from datasets import Dataset

class ModelEvaluator:
    def __init__(self, model: PreTrainedModel, dataset: Dataset, tokenizer: PreTrainedTokenizer, sentiment_pipe: Pipeline):
        """
        Initialize the ModelEvaluator.

        Args:
            model (PreTrainedModel): Model to evaluate.
            dataset (Dataset): Dataset for evaluation.
            tokenizer (PreTrainedTokenizer): Tokenizer for processing text.
            sentiment_pipe (Pipeline): Sentiment analysis pipeline.
        """
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sentiment_pipe = sentiment_pipe
        self.device = next(model.parameters()).device

    def evaluate(self, num_samples: int = 1000) -> float:
        """
        Evaluate the model on a dataset.

        Args:
            num_samples (int): Number of samples to evaluate.

        Returns:
            float: Average reward across samples.
        """
        self.model.eval()
        total_reward = 0
        samples = self.dataset.select(range(min(num_samples, len(self.dataset))))

        for sample in samples:
            reward = self._evaluate_sample(sample["query"])
            total_reward += reward

        return total_reward / len(samples)

    def _evaluate_sample(self, query: str) -> float:
        """
        Evaluate a single sample.

        Args:
            query (str): Input query.

        Returns:
            float: Reward for the generated text.
        """
        input_ids = self.tokenizer.encode(query, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=input_ids.shape[1] + 50, num_return_sequences=1)

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        generated_text = generated_text[len(query):].strip()

        sentiment_output = self.sentiment_pipe([generated_text], **{"top_k": None, "function_to_apply": "none"})[0]
        reward = sentiment_output[0]["score"] if sentiment_output[0]["label"] == "POSITIVE" else -sentiment_output[0]["score"]

        return reward

def evaluate_model(model: PreTrainedModel, dataset: Dataset, tokenizer: PreTrainedTokenizer, sentiment_pipe: Pipeline, num_samples: int = 1000) -> float:
    """
    Convenience function to evaluate a model using ModelEvaluator.

    Args:
        model (PreTrainedModel): Model to evaluate.
        dataset (Dataset): Dataset for evaluation.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text.
        sentiment_pipe (Pipeline): Sentiment analysis pipeline.
        num_samples (int): Number of samples to evaluate.

    Returns:
        float: Average reward across samples.
    """
    evaluator = ModelEvaluator(model, dataset, tokenizer, sentiment_pipe)
    return evaluator.evaluate(num_samples)