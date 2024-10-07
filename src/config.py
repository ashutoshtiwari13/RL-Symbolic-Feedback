from trl import PPOConfig

def get_ppo_config():
    """
    Returns the PPO configuration for training.
    
    Returns:
        PPOConfig: Configuration object for PPO training.
    """
    return PPOConfig(
        model_name="lvwerra/gpt2-imdb",
        learning_rate=1.41e-5,
        log_with="wandb",
    )