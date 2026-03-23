import torch


def distance(z, z_public):
    """
    Compute the distance between two distributions
    """

    p_z = torch.softmax(z, dim=-1)
    p_z_public = torch.softmax(z_public, dim=-1)

    difference = p_z - p_z_public

    mean = difference.mean(dim=0)

    distance = mean.norm(p=1)

    return distance


def clip_logits(z, c):
    """
    Applies clipping between -c and c
    """

    max_z = z.max(dim=-1, keepdim=True).values

    shifted_z = z - max_z + c

    clipped_z = torch.max(torch.tensor(-c, device=z.device, dtype=z.dtype), shifted_z)

    return clipped_z
