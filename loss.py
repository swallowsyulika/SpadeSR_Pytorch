import torch
import torch.nn as nn
import torch.nn.functional as F


def d_source_loss(real_labels, fake_labels, mode="hinge"):
    if mode == 'wgan':
        return torch.mean(fake_labels) - torch.mean(real_labels)
    elif mode == 'hinge':
        return torch.mean(F.relu(1.0 - real_labels)) + torch.mean(F.relu(1.0 + fake_labels))


def calc_error(arr1, arr2, mode='l1', hinge_thr=0.0):
    if mode == 'l1':
        return torch.mean(torch.abs(arr1 - arr2))
    elif mode == 'l2':
        return torch.mean((arr1 - arr2)**2)
    elif mode == 'hinge':
        return torch.mean(F.relu(torch.abs(arr1 - arr2) - hinge_thr))


def gradient_penalty(real_images, fake_images, discriminator):
    alpha = torch.randn(real_images[0], 1, 1, 1, dtype=torch.float64, mean=0.0, std=1.0)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff
    interpolated.requires_grad = True

    # need to check again
    with torch.autograd.record():
        label, rec = discriminator(interpolated, training=True)
        label.backward()

    grads = interpolated.grad
    norm = torch.sqrt(torch.sum(grads**2, dim=[1, 2, 3]))
    gp = torch.mean(norm**2)
    return gp


def r1_penalty(images, discriminator):
    images.requires_grad = True

    # need to check again
    with torch.autograd.record():
        label, rec = discriminator(images, training=True)
        label.backward()

    grads = images.grad
    norm = torch.sqrt(torch.sum(grads**2, dim=[1, 2, 3]))
    gp = 0.5 * torch.mean(norm**2)
    return gp


def g_source_loss(fake_labels):
    return -torch.mean(fake_labels)