import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import os

from autoencoder import Autoencoder
from data_loader import Dataload

def setup_inference (model_path, dataloader):
    model = Autoencoder()

    # load model weights
    model.load_state_dict(torch.load(model_path, weights_only=True))

    model.eval() # set model to eval mode

    data_iter = iter(dataloader) # get first batch of images

    imgs, _ = next(data_iter)

    return model, imgs # return model and one batch of images

def calculate_difference_map (model, images):
    with torch.no_grad():
        recons = model(images) # get model's reconstructions

        diff_map = torch.abs(images - recons) # calculate differene between reconstructions and images

        return recons, diff_map

def plot_triplets(images, reconstructions, difference_maps, num_to_plot=3):
    fig, axes = plt.subplots(nrows=num_to_plot, ncols=3, figsize=(12, 4 * num_to_plot))

    for i in range(num_to_plot):
        image = images[i]
        recon = reconstructions[i]
        diff_map = difference_maps[i]

        image_np = image.detach().numpy()
        recon_np = recon.detach().numpy()
        diff_map_np = diff_map.detach().numpy()

        # remove channel dimension
        image_np = image_np.squeeze(0)
        recon_np = recon_np.squeeze(0)
        diff_map_np = diff_map_np.squeeze(0)

        axes[i, 0].imshow(image_np, cmap='viridis', vmin=0.0, vmax=3.0)
        axes[i, 0].axis('off')
        axes[i, 0].set_title("Original")

        axes[i, 1].imshow(recon_np, cmap='viridis', vmin=0.0, vmax=3.0)
        axes[i, 1].axis('off')
        axes[i, 1].set_title("Reconstruction")

        axes[i, 2].imshow(diff_map_np, cmap='hot', vmin=0.0, vmax=1.0)
        axes[i, 2].axis('off')
        axes[i, 2].set_title("Difference")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    processed_dir = "/content/drive/MyDrive/Phytoplankton_Project/data/processed"
    model_weights_file = "/content/APS360_Project/CAE_model.pth"

    if not os.path.exists(processed_dir):
        print ("Directory not found")
        exit()
               
    dataset = Dataload(processed_dir)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    model, imgs = setup_inference(model_weights_file, dataloader)

    recons, diff_maps = calculate_difference_map(model, imgs)

    plot_triplets(imgs, recons, diff_maps, 3)

