from typing import List
import torch
from transformers import PreTrainedTokenizer, Pipeline

class SymbolicEnvironment:
    def __init__(self, tokenizer: PreTrainedTokenizer, sentiment_analyzer: Pipeline):
        """
        Initialize the SymbolicEnvironment.
        
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer for processing text.
            sentiment_analyzer (Pipeline): Sentiment analysis pipeline.
        """
        self.tokenizer = tokenizer
        self.sentiment_analyzer = sentiment_analyzer

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment for a single text.
        
        Args:
            text (str): Text to analyze.
        
        Returns:
            float: Sentiment score.
        """
        try:
            output = self.sentiment_analyzer([text], top_k=None, function_to_apply="none")[0]
            return output[0]["score"] if output[0]["label"] == "POSITIVE" else -output[0]["score"]
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return 0.0

    def get_vectorized_feedback(self, prompt: str, response: str) -> List[float]:
        """
        Get vectorized feedback for a single prompt-response pair.
        
        Args:
            prompt (str): Input prompt.
            response (str): Generated response.
        
        Returns:
            List[float]: Vectorized feedback.
        """
        full_text = prompt + response
        tokens = self.tokenizer.tokenize(full_text)
        sentiment_score = self.analyze_sentiment(full_text)
        
        feedback = [sentiment_score if sentiment_score > 0.5 else 1 - sentiment_score] * len(tokens)
        return feedback

    def get_vectorized_reward_batch(self, queries: List[str], responses: List[str]) -> List[torch.Tensor]:
        """
        Get vectorized rewards for a batch of query-response pairs.
        
        Args:
            queries (List[str]): List of input queries.
            responses (List[str]): List of generated responses.
        
        Returns:
            List[torch.Tensor]: List of vectorized rewards.
        """
        vectorized_rewards = []
        for query, response in zip(queries, responses):
            feedback = self.get_vectorized_feedback(query, response)
            reward = torch.tensor(feedback, device=self.sentiment_analyzer.device)
            vectorized_rewards.append(reward)

        return vectorized_rewards

    def analyze_sentiment_batch(self, texts: List[str]) -> List[float]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (List[str]): List of texts to analyze.
        
        Returns:
            List[float]: List of sentiment scores.
        """
        try:
            outputs = self.sentiment_analyzer(texts, top_k=None, function_to_apply="none", batch_size=16)
            return [output[0]["score"] if output[0]["label"] == "POSITIVE" else -output[0]["score"] for output in outputs]
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return [0.0] * len(texts)