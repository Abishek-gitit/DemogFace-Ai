import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders
from model import AgeGenderModel

def train_epoch(model, dataloader, optimizer, criterion_age, criterion_gender, device, age_weight=0.05):
    model.train()
    running_loss = 0.0
    running_age_loss = 0.0
    running_gender_loss = 0.0
    correct_gender = 0
    total_samples = 0
    
    for images, ages, genders in dataloader:
        images = images.to(device)
        ages = ages.to(device)
        genders = genders.to(device)
        
        optimizer.zero_grad()
        
        age_pred, gender_pred = model(images)
        
        age_loss = criterion_age(age_pred, ages)
        gender_loss = criterion_gender(gender_pred, genders)
        
        loss = gender_loss + age_weight * age_loss
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        running_age_loss += age_loss.item() * images.size(0)
        running_gender_loss += gender_loss.item() * images.size(0)
        
        _, predicted_gender = torch.max(gender_pred, 1)
        correct_gender += (predicted_gender == genders).sum().item()
        total_samples += images.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_age_mae = running_age_loss / total_samples
    epoch_gender_acc = (correct_gender / total_samples) * 100
    
    return epoch_loss, epoch_age_mae, epoch_gender_acc

def evaluate(model, dataloader, criterion_age, criterion_gender, device, age_weight=0.05):
    model.eval()
    running_loss = 0.0
    running_age_loss = 0.0
    running_gender_loss = 0.0
    correct_gender = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, ages, genders in dataloader:
            images = images.to(device)
            ages = ages.to(device)
            genders = genders.to(device)
            
            age_pred, gender_pred = model(images)
            
            age_loss = criterion_age(age_pred, ages)
            gender_loss = criterion_gender(gender_pred, genders)
            
            loss = gender_loss + age_weight * age_loss
            
            running_loss += loss.item() * images.size(0)
            running_age_loss += age_loss.item() * images.size(0)
            running_gender_loss += gender_loss.item() * images.size(0)
            
            _, predicted_gender = torch.max(gender_pred, 1)
            correct_gender += (predicted_gender == genders).sum().item()
            total_samples += images.size(0)
            
    val_loss = running_loss / total_samples
    val_age_mae = running_age_loss / total_samples
    val_gender_acc = (correct_gender / total_samples) * 100
    
    return val_loss, val_age_mae, val_gender_acc

def main():
    parser = argparse.ArgumentParser(description="Train Age & Gender Estimation Model on UTKFace")
    parser.add_argument("--data_dir", type=str, default="data/UTKFace", help="Path to UTKFace dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small"], help="Backbone model")
    parser.add_argument("--img_size", type=int, default=128, help="Resize dimension for faces")
    parser.add_argument("--age_weight", type=float, default=0.05, help="Weight factor for age regression loss")
    args = parser.parse_args()
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Device: CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU")
        
    print(f"Loading dataset from: {args.data_dir}")
    try:
        train_loader, val_loader, num_train, num_val = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            img_size=args.img_size
        )
        print(f"Dataset loaded. Train samples: {num_train}, Val samples: {num_val}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check if the data directory exists and has files in the format age_gender_race_date.jpg")
        return
        
    model = AgeGenderModel(backbone=args.backbone, pretrained=True).to(device)
    
    criterion_age = nn.L1Loss()
    criterion_gender = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    os.makedirs("models", exist_ok=True)
    checkpoint_path = f"models/best_age_gender_{args.backbone}.pth"
    
    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        train_loss, train_age_mae, train_gender_acc = train_epoch(
            model, train_loader, optimizer, criterion_age, criterion_gender, device, args.age_weight
        )
        val_loss, val_age_mae, val_gender_acc = evaluate(
            model, val_loader, criterion_age, criterion_gender, device, args.age_weight
        )
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} (Age MAE: {train_age_mae:.2f}, Gender Acc: {train_gender_acc:.2f}%) | "
              f"Val Loss: {val_loss:.4f} (Age MAE: {val_age_mae:.2f}, Gender Acc: {val_gender_acc:.2f}%)")
              
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_age_mae': val_age_mae,
                'val_gender_acc': val_gender_acc,
                'backbone': args.backbone,
                'img_size': args.img_size
            }, checkpoint_path)
            print(f" => Saved best model checkpoint to {checkpoint_path}")
            
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
