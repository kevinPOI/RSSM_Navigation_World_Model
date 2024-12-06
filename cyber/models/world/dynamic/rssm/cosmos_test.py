import torch
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from cosmos_tokenizer.image_lib import ImageTokenizer

from cyber.models.world.dynamic import RecurrentStateSpaceModel
from cyber.models.world import DynamicModel
from omegaconf import OmegaConf
import torch
from action_image_dataset import ActionImageDataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import os



images_folder = "train_set/images"
actions_file = "train_set/actions.npy"

# Hyperparameters
batch_size = 1
num_epochs = 10
learning_rate = 0.001
sequence_length = 5

# Data transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images
    transforms.ToTensor(),          # Convert to tensor
])

# Dataset and DataLoader
dataset = ActionImageDataset(images_folder, actions_file, sequence_length, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)



model_name = "Cosmos-Tokenizer-DI16x16"
input_tensor = torch.randn(1, 3, 512, 512).to('cuda').to(torch.bfloat16)  # [B, C, T, H, W]
encoder = ImageTokenizer(checkpoint_enc=f'cyber/models/world/dynamic/rssm/pretrained_ckpts/{model_name}/encoder.jit')
l1, l2 = encoder.encode(input_tensor)
#torch.testing.assert_close(latent.shape, (1, 16, 3, 64, 64))

# The input tensor can be reconstructed by the decoder as:
decoder = ImageTokenizer(checkpoint_dec=f'cyber/models/world/dynamic/rssm/pretrained_ckpts/{model_name}/decoder.jit')
# reconstructed_tensor1 = decoder.decode(l1)
# reconstructed_tensor1 = decoder.decode(l2)

for images, actions in dataloader:
    images, actions = images.cuda(), actions.cuda()  # Move to GPU if available
    input = {
        "actions": actions,
        "observations": images
    }
            # Zero the parameter gradients
    # Compute the loss
    img = images[:,0,...]
    l1, l2 = encoder.encode(img)
    random_tensor = torch.randint(-3, 3, l1.shape, dtype=l1.dtype, device = l1.device)
    l1 += random_tensor
    reconstructed_tensor1 = decoder.decode(l1)
    # reconstructed_tensor2 = decoder.decode(l2)
    second_image_prompt = img[0].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    recon1 = reconstructed_tensor1[0].to(dtype=torch.float32, device='cpu').detach().numpy().transpose(1, 2, 0)
    # recon2 = reconstructed_tensor2.cpu().detach().numpy().transpose(1, 2, 0)

    # Rescale images from [0, 1] to [0, 255] for OpenCV
    # second_image_prompt = (second_image_prompt * 255).astype('uint8')
    # second_image_prediction = (second_image_prediction * 255).astype('uint8')
    

    # Display the images
    cv2.imshow("Second Image in Prompt", second_image_prompt)
    cv2.imshow("Second Image in Prediction", recon1)
    # cv2.imshow("Second Image in Prediction2", recon2)
    cv2.waitKey(0)


# import torch
# from cosmos_tokenizer.video_lib import CausalVideoTokenizer

# model_name = "Cosmos-Tokenizer-CV4x8x8"
# input_tensor = torch.randn(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)  # [B, C, T, H, W]
# encoder = CausalVideoTokenizer(checkpoint_enc=f'cyber/models/world/dynamic/rssm/pretrained_ckpts/{model_name}/encoder.jit')
# (latent,) = encoder.encode(input_tensor)
# torch.testing.assert_close(latent.shape, (1, 16, 3, 64, 64))

# # The input tensor can be reconstructed by the decoder as:
# decoder = CausalVideoTokenizer(checkpoint_dec=f'cyber/models/world/dynamic/rssm/pretrained_ckpts/{model_name}/decoder.jit')
# reconstructed_tensor = decoder.decode(latent)
# torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)