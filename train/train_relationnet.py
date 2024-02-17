import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import sys
sys.path.append('.')
import argparse
import yaml
from datasets.ade_relations import ADE_relations
from model.relation_net.relation_net import RelationNet

def train(configs):
    # Initialize the dataset
    data_file = configs['data_file']
    dataset = ADE_relations(data_file=data_file)

    # Define the size of train and test datasets
    total_data = len(dataset)
    train_size = int(0.8 * total_data)
    test_size = total_data - train_size

    # Perform the random split
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Define your dataloaders for train and test datasets
    batch_size = configs['batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = RelationNet(configs)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=configs['learning_rate'])

    num_epochs = configs['num_epochs']

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_predictions = 0

        # Training loop
        model.train()
        for relations, embeddings in train_dataloader:
            # Forward pass
            outputs = model(embeddings.float())

            # Calculate loss
            loss = criterion(outputs, relations)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track total loss and correct predictions
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == relations).sum().item()

        # Print epoch statistics
        average_loss = total_loss / len(train_dataloader)
        accuracy = correct_predictions / len(train_dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss:.4f}, Training Accuracy: {accuracy:.4f}")

        # Validation loop
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            correct_predictions = 0
            for relations, embeddings in test_dataloader:
                outputs = model(embeddings.float())
                # print(outputs)
                loss = criterion(outputs, relations)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == relations).sum().item()

            average_loss = total_loss / len(test_dataloader)
            accuracy = correct_predictions / len(test_dataloader.dataset)
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {average_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

    # Save the trained model if needed
    # torch.save(model.state_dict(), "trained_model.pth")

def main():
    parser = argparse.ArgumentParser(description="Training AirLoc")
    parser.add_argument("-c", "--config_file", dest="config_file", type=str, default="configs/train_relationnet.yaml")
    parser.add_argument("-g", "--gpu", dest="gpu", type=int, default=[1])
    args = parser.parse_args()

    config_file = args.config_file
    f = open(config_file, 'r', encoding='utf-8')
    configs = f.read()
    configs = yaml.safe_load(configs)
    configs['num_gpu'] = args.gpu

    train(configs)

if __name__ == "__main__":
    main()
