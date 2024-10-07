from trl import AutoModelForCausalLMWithValueHead

def load_model(model_name):
    """
    Load a pre-trained model.
    
    Args:
        model_name (str): Name or path of the pre-trained model.
    
    Returns:
        AutoModelForCausalLMWithValueHead: Loaded model.
    """
    return AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

def save_model(model, path):
    """
    Save a model to the specified path.
    
    Args:
        model (AutoModelForCausalLMWithValueHead): Model to save.
        path (str): Path to save the model.
    """
    model.save_pretrained(path)