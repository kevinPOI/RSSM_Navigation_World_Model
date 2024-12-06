from rssm2 import RecurrentStateSpaceModel2
from rssm import RecurrentStateSpaceModel
from cyber.models.world import DynamicModel
from omegaconf import OmegaConf
import torch
from action_image_dataset import ActionImageDataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import os
from cosmos_tokenizer.image_lib import ImageTokenizer
# pred_steps = 5
# input = {
#             "initial": torch.randn(1, rssm_conf.hidden_size+rssm_conf.latent_size),
#             "actions": torch.randn(1, pred_steps, rssm_conf.action_size),
#             "observations": torch.randn(1, pred_steps, 3, 64, 64)
#         }
# model.train()
# loss = model(input)
USECOSMOS = False

def train_model_v2(model, dataloader, optimizer,encoder, decoder, num_epochs=10, weight_path = "weights_w_cosmos.pt" ):
    # if os.path.exists(weight_path):
    #     print(f"Loading weights from {weight_path}")
    #     model.load_state_dict(torch.load(weight_path))
    model.train()  # Set model to training mode
    pred_steps = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, actions in dataloader:
            images, actions = images.cuda(), actions.cuda()  # Move to GPU if available
            images,_ = encoder.encode(images[0])
            oi = images.clone()
            images = images.reshape([pred_steps, -1])[torch.newaxis,...]
            images = images.to(torch.float32) / 65535
            images.require_grad = True
            actions.require_grad = True
            input = {
            "actions": actions,
            "observations": images
        }
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Compute the loss
            predictions, loss = model.compute_training_loss_output(input)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(loss)
            # images = reshape_embeddings(images)
            # images = (images[0,...]*65535).to(torch.int32)
            if epoch >= 1:
                images = oi
                predictions = reshape_embeddings(predictions)
                predictions = (predictions[:,0,...]*65535).to(torch.int32)
                decoded_imgs = decoder.decode(images)
                pred_imgs = decoder.decode(predictions)
                display_images(decoded_imgs, pred_imgs)
            
        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")
        torch.save(model.state_dict(), weight_path)
def train_model(model, dataloader, optimizer, num_epochs=10, weight_path = "weights.pt"):
    # if os.path.exists(weight_path):
    #     print(f"Loading weights from {weight_path}")
    #     model.load_state_dict(torch.load(weight_path))
    model.train()  # Set model to training mode
    pred_steps = 5
    counter = 0
    for epoch in range(num_epochs):
        
        running_loss = 0.0
        for images, actions in dataloader:
            images, actions = images.cuda(), actions.cuda()  # Move to GPU if available
            input = {
            "actions": actions,
            "observations": images
        }
            # Zero the parameter gradients
            

            # Compute the loss
            predictions, loss = model.compute_training_loss_output(input)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(loss)
            counter += 1
            if counter % 100 == 0:
                predictions = torch.stack(predictions).reshape([-1, 3, 128, 128])
                display_images(images[0], predictions)
            
        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")
        torch.save(model.state_dict(), weight_path)
def reshape_embeddings(images, sz=16):
    batch, stp = images.shape[0:2]
    images = images.reshape([batch, stp, sz, sz])
    return images
def display_images(images, predictions):
    for i in range(4):
        second_image_prompt = images[i+1].to(dtype=torch.float32, device='cpu').numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        second_image_prediction = predictions[i].to(dtype=torch.float32, device='cpu').detach().numpy().transpose(1, 2, 0)
        # Display the images
        cv2.imshow("Second Image in Prompt", second_image_prompt)
        cv2.imshow("Second Image in Prediction", second_image_prediction)
        cv2.waitKey(0)
if __name__ == "__main__":
    # Paths
    images_folder = "train_set/images"
    actions_file = "train_set/actions.npy"

    # Hyperparameters
    batch_size = 1
    num_epochs = 10
    learning_rate = 0.002
    sequence_length = 5

    
    # Data transforms
    if USECOSMOS:
        model_name = "Cosmos-Tokenizer-DI16x16"
        encoder = ImageTokenizer(checkpoint_enc=f'cyber/models/world/dynamic/rssm/pretrained_ckpts/{model_name}/encoder.jit')
        decoder = ImageTokenizer(checkpoint_dec=f'cyber/models/world/dynamic/rssm/pretrained_ckpts/{model_name}/decoder.jit')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize images
            transforms.ToTensor(),          # Convert to tensor
        ])
        rssm_conf = OmegaConf.load("experiments/configs/models/world/rssm2.yaml")
        model = RecurrentStateSpaceModel2(**rssm_conf)
    else:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize images
            transforms.ToTensor(),          # Convert to tensor
        ])
        rssm_conf = OmegaConf.load("experiments/configs/models/world/rssm.yaml")
        model = RecurrentStateSpaceModel(**rssm_conf)

    # Dataset and DataLoader
    dataset = ActionImageDataset(images_folder, actions_file, sequence_length, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device=device)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    if USECOSMOS:
        train_model_v2(model, dataloader, optimizer, encoder, decoder, num_epochs)
    else:
        train_model(model, dataloader, optimizer,  num_epochs)
