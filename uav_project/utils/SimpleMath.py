import torch

def tand(theta):
    angle = torch.tensor(theta * torch.pi / 180, dtype=torch.float32)
    return torch.tan(angle).item()

def sind(theta):
    angle = torch.tensor(theta * torch.pi / 180, dtype=torch.float32)
    return torch.sin(angle).item()

def cosd(theta):
    angle = torch.tensor(theta * torch.pi / 180, dtype=torch.float32)
    return torch.cos(angle).item()