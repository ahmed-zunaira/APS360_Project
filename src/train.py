import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os

from data_loader import Dataload
from autoencoder import Autoencoder

torch.manual_seed(15)

def evaluate (model, loader, criterion):
    total_loss = 0.0
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            img, _ = data
            output = model(img)
            loss = criterion (output, img)
        
            total_loss += loss.item()
    
        return total_loss / (i+1)

def plot (batch_size=32, learning_rate=0.001, epochs=30):
    train_loss = np.loadtxt("CAE_bs{}_lr{}_epoch{}_train_loss.csv".format(batch_size,learning_rate,epochs))
    val_loss = np.loadtxt("CAE_bs{}_lr{}_epoch{}_val_loss.csv".format(batch_size,learning_rate,epochs))

    plt.title ("Train vs. Validation Loss")
    plt.plot(range(1, len(train_loss)+1), train_loss, label="Train")
    plt.plot(range(1, len(val_loss)+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend(loc='best')
    plt.show()

def train (model, train_data, val_data, batch_size=32, learning_rate=0.001, epochs=30, patience=15):
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)

    min_val_loss = float('inf')
    track_bad_epochs = 0
    curr_epochs = epochs

    start_time = time.time()
    for epoch in range(epochs):

        total_loss = 0.0
        model.train()

        for data in train_loader:

            img, _ = data
            
            output = model(img)
            loss = criterion (output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss[epoch] = total_loss / len(train_loader)
        val_loss[epoch] = evaluate(model, val_loader, criterion)
        print (("Epoch {}: Train Loss: {}, Validation Loss: {}").format(epoch+1, train_loss[epoch], val_loss[epoch]))

        if val_loss[epoch] < min_val_loss:
            min_val_loss = val_loss[epoch]
            track_bad_epochs = 0
            torch.save(model.state_dict(), "CAE_model.pth")
        else:
            track_bad_epochs += 1
            if track_bad_epochs >= patience:
                print (f"Stopping at epoch {epoch+1}. Best validation loss was: {min_val_loss:.6f}")
                curr_epochs = epoch+1
                break

    print ("\nTraining completed.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print ("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    train_loss = train_loss[:curr_epochs]
    val_loss = val_loss[:curr_epochs]

    np.savetxt("CAE_bs{}_lr{}_epoch{}_train_loss.csv".format(batch_size,learning_rate,epochs), train_loss)
    np.savetxt("CAE_bs{}_lr{}_epoch{}_val_loss.csv".format(batch_size,learning_rate,epochs), val_loss)

if __name__ == "__main__":
    processed_dir = "/content/drive/MyDrive/Phytoplankton_Project/data/processed"
    if not os.path.exists(processed_dir):
        print ("Directory not found")
               
    dataset = Dataload(processed_dir)

    train_data, val_data = random_split(dataset, [0.8, 0.2])

    model = Autoencoder()

    train(model, train_data, val_data, 16, 0.001, 100)

    plot(16, 0.001, 100)