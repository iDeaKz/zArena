import torch # Assuming the healing model uses PyTorch
import random

def mutate_healing_model_parameters(model, mutation_rate=0.05, mutation_strength=0.1):
    """
    Mutates the parameters of a PyTorch model (e.g., a HealingModule).

    Args:
        model (torch.nn.Module): The PyTorch model to mutate.
        mutation_rate (float): The probability that any given parameter tensor will be mutated.
        mutation_strength (float): The standard deviation of the Gaussian noise added to parameters.
    """
    if not isinstance(model, torch.nn.Module):
        print("Warning: mutate_healing_model_parameters expects a torch.nn.Module.")
        return

    with torch.no_grad(): # Ensure mutations are not tracked by autograd
        for param in model.parameters():
            if random.random() < mutation_rate:
                noise = torch.randn_like(param) * mutation_strength
                param.add_(noise)
    print(f"Model parameters mutated with rate {mutation_rate} and strength {mutation_strength}.")

# Example usage (assuming you have a HealingModule instance that is a torch.nn.Module):
# from healing_module import HealingModule # your actual module
# healing_model_instance = HealingModule.load_from_checkpoint("trained_healing_function.pt")
# mutate_healing_model_parameters(healing_model_instance)