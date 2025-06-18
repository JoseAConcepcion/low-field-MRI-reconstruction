import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from torchvision.io import read_image, ImageReadMode

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pytorch_wavelets import DWTForward, DWTInverse
from piq import SSIMLoss  


TRAIN_DIR = 'drive/MyDrive/MRI/Train'
CHECKPOINT_DIR = 'drive/MyDrive/MRI/Checkpoints_SSIM'
RESULTS_DIR = 'drive/MyDrive/MRI/Results_SSIM'


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class WaveletLayer(nn.Module):
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
        self.dwt = DWTForward(J=1, wave=wavelet, mode='symmetric')
    
    def forward(self, x):
        yl, yh = self.dwt(x)
        cA = yl.unsqueeze(2)
        cH, cV, cD = torch.unbind(yh[0], dim=2)
        combined = torch.cat([cA, cH.unsqueeze(2), cV.unsqueeze(2), cD.unsqueeze(2)], dim=2)
        batch, channels, _, h, w = combined.shape
        return combined.view(batch, channels * 4, h, w)


class InverseWaveletLayer(nn.Module):
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
        self.idwt = DWTInverse(wave=wavelet, mode='symmetric')
    
    def forward(self, x):
        batch, total_channels, h, w = x.shape
        channels = total_channels // 4
        x = x.view(batch, channels, 4, h, w)
        yl = x[:, :, 0]
        yh = [x[:, :, 1:]]
        return self.idwt((yl, yh))


class DeepCNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64 if i > 0 else 4, 64, 3, padding=1),
                nn.ReLU(inplace=True))
            for i in range(10)
        ])
        
        
        self.conv11 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True))
        
        
        self.conv13 = nn.Sequential(
            nn.Conv2d(64, 4, 1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        
        x0 = x  
        x1 = self.conv_layers[0](x)  
        x2 = self.conv_layers[1](x1) 
        
        
        out = x2
        for i in range(2, 10):
            out = self.conv_layers[i](out)
        
        
        out = self.conv11(out + x2)  
        out = self.conv12(out + x1)  
        out = self.conv13(out) + x0  
        
        return out


class SubpixelUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.relu(x)



class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)) 
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)) 
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)) 
        
        
        self.dec1_upsample = SubpixelUpsample(128, 64) 
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True))
        
        self.dec2_upsample = SubpixelUpsample(64, 32) 
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),   
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True))
            
        
        self.dec3_upsample = SubpixelUpsample(32, 16) 
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(20, 16, 3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True))
        
        
        self.final_conv = nn.Conv2d(16, 4, 1) 

    def forward(self, x):
        
        x_wave_input = x 
        
        
        enc1_out = self.enc1(x_wave_input)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        
        
        dec1 = self.dec1_upsample(enc3_out)
        dec1 = torch.cat([dec1, enc2_out], dim=1)
        dec1 = self.dec1_conv(dec1)
        
        dec2 = self.dec2_upsample(dec1)
        dec2 = torch.cat([dec2, enc1_out], dim=1)
        dec2 = self.dec2_conv(dec2)
        
        
        dec3 = self.dec3_upsample(dec2)
        dec3 = torch.cat([dec3, x_wave_input], dim=1) 
        dec3 = self.dec3_conv(dec3)
        
        
        return self.final_conv(dec3)


class WaCAEN(nn.Module):
    def __init__(self, in_channels=1, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
        self.wavelet_layer = WaveletLayer(wavelet)
        self.inverse_wavelet_layer = InverseWaveletLayer(wavelet)
        
        
        self.branch1 = DeepCNNBranch()
        self.branch2 = ConvAutoencoder()

    def forward(self, x):
        
        x_wave = self.wavelet_layer(x)
        
        
        branch1_out = self.branch1(x_wave)
        branch2_out = self.branch2(x_wave)
        
        
        combined = branch1_out + branch2_out
        
        
        reconstructed_image = self.inverse_wavelet_layer(combined)
        
        
        
        return torch.sigmoid(reconstructed_image)


class MRIDataset(Dataset):
    def __init__(self, source_dir, target_dir, input_size=(128, 128)):
        self.source_files = sorted(glob.glob(os.path.join(source_dir, '*.png')))
        self.target_files = sorted(glob.glob(os.path.join(target_dir, '*.png')))
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Lambda(lambda x: x.float() / 255.0)
        ])
        
        
        min_files = min(len(self.source_files), len(self.target_files))
        self.source_files = self.source_files[:min_files]
        self.target_files = self.target_files[:min_files]
        
        print(f"Found {min_files} image pairs")

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):
        source_img = read_image(self.source_files[idx], ImageReadMode.GRAY)
        target_img = read_image(self.target_files[idx], ImageReadMode.GRAY)
        
        
        source_img = self.transform(source_img)
        target_img = self.transform(target_img)
        
        return source_img, target_img


class RobustCheckpointCallback:
    def __init__(self, checkpoint_dir, save_freq=5, max_checkpoints=10):
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.max_checkpoints = max_checkpoints
        self.best_loss = float('inf')
        self.checkpoint_info_file = os.path.join(checkpoint_dir, 'checkpoint_info.json')

    def on_epoch_end(self, epoch, model, optimizer, val_loss):
        current_loss = val_loss
        
        
        should_save = (epoch + 1) % self.save_freq == 0 or current_loss < self.best_loss
        
        if should_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                checkpoint_name = f"best_model_epoch_{epoch+1}_loss_{current_loss:.4f}_{timestamp}.pt"
                print(f"\n¡Nuevo mejor modelo! Loss: {current_loss:.4f}")
            else:
                checkpoint_name = f"checkpoint_epoch_{epoch+1}_{timestamp}.pt"
            
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
            }, checkpoint_path)
            
            
            checkpoint_info = {
                'epoch': epoch + 1,
                'loss': current_loss,
                'timestamp': timestamp,
                'model_path': checkpoint_path,
                'is_best': current_loss < self.best_loss
            }
            
            with open(self.checkpoint_info_file, 'w') as f:
                json.dump(checkpoint_info, f, indent=2, default=str)
            
            print(f"Checkpoint guardado: {checkpoint_name}")
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Mantener solo los últimos max_checkpoints checkpoints (excepto el mejor)"""
        try:
            checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt"))
            checkpoint_files.sort(key=os.path.getmtime)
            
            if len(checkpoint_files) > self.max_checkpoints:
                files_to_remove = checkpoint_files[:-self.max_checkpoints]
                for file_path in files_to_remove:
                    if 'best_model' not in file_path:  
                        os.remove(file_path)
        except Exception as e:
            print(f"Error limpiando checkpoints: {e}")


def load_latest_checkpoint(checkpoint_dir, model, optimizer, device):
    checkpoint_info_file = os.path.join(checkpoint_dir, 'checkpoint_info.json')
    
    if not os.path.exists(checkpoint_info_file):
        print("No se encontraron checkpoints previos")
        return 0, float('inf')
    
    try:
        with open(checkpoint_info_file, 'r') as f:
            checkpoint_info = json.load(f)
        
        model_path = checkpoint_info['model_path']
        if os.path.exists(model_path):
            print(f"Cargando checkpoint desde época {checkpoint_info['epoch']}")
            print(f"Loss anterior: {checkpoint_info['loss']:.4f}")
            
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            return checkpoint['epoch'], checkpoint['loss']
        else:
            print("Archivo de modelo no encontrado")
            return 0, float('inf')
    
    except Exception as e:
        print(f"Error cargando checkpoint: {e}")
        return 0, float('inf')

def create_data_loaders(dataset, batch_size, train_ratio=0.65, test_ratio=0.20, val_ratio=0.15):
    
    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6, "Los ratios deben sumar 1"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = int(test_ratio * total_size)
    val_size = total_size - train_size - test_size
    
    
    train_dataset, test_dataset, val_dataset = random_split(
        dataset,
        [train_size, test_size, val_size],
        generator=torch.Generator().manual_seed(42)  
    )
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset dividido: {len(train_dataset)} entrenamiento, {len(test_dataset)} prueba, {len(val_dataset)} validación")
    
    return train_loader, test_loader, val_loader

def train_mri_restoration(degradation_type='source_a', input_size=(128, 128),
                         batch_size=8, epochs=100, learning_rate=1e-4):
    print(f"Iniciando entrenamiento para {degradation_type}")
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    
    source_dir = os.path.join(TRAIN_DIR, degradation_type)
    target_dir = os.path.join(TRAIN_DIR, 'target')
    dataset = MRIDataset(source_dir, target_dir, input_size)
    
    
    total_size = len(dataset)
    train_size = int(0.65 * total_size)
    test_size = int(0.20 * total_size)
    val_size = total_size - train_size - test_size
    
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset, val_dataset = random_split(
        dataset, 
        [train_size, test_size, val_size],
        generator=generator
    )
    
    print(f"Dataset dividido: {train_size} entrenamiento, {test_size} prueba, {val_size} validación")
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    model = WaCAEN(in_channels=1, wavelet='haar').to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    
    criterion = SSIMLoss(data_range=1.0)  
    print("Utilizando SSIMLoss como función de pérdida")
    
    
    start_epoch, best_loss = load_latest_checkpoint(CHECKPOINT_DIR, model, optimizer, device)
    
    
    checkpoint_callback = RobustCheckpointCallback(CHECKPOINT_DIR)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-7)
    
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': None
    }
    
    
    for epoch in range(start_epoch, epochs):
        
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        
        scheduler.step(val_loss)
        
        
        checkpoint_callback.on_epoch_end(epoch, model, optimizer, val_loss)
        
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
    
    test_loss /= len(test_loader.dataset)
    history['test_loss'] = test_loss
    print(f'\nEvaluación Final - Test Loss: {test_loss:.6f}')
    
    
    history_path = os.path.join(RESULTS_DIR, f'training_history_{degradation_type}.npy')
    np.save(history_path, history)
    
    
    final_model_path = os.path.join(CHECKPOINT_DIR, f'final_model_{degradation_type}.pt')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_loss': test_loss,
    }, final_model_path)
    
    return model, history, test_loader


def visualize_results(model, val_loader, degradation_type, device, n_samples=5):
    model.eval()
    inputs, targets = next(iter(val_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    with torch.no_grad():
        predictions = model(inputs[:n_samples])
    
    inputs = inputs.cpu().numpy()
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    plt.figure(figsize=(15, 10))
    
    for i in range(min(n_samples, len(inputs))):
        
        plt.subplot(3, n_samples, i + 1)
        plt.imshow(inputs[i].squeeze(), cmap='gray')
        plt.title(f'{"Aliasing" if degradation_type == "source_a" else "Blur"}')
        plt.axis('off')
        
        
        plt.subplot(3, n_samples, i + 1 + n_samples)
        plt.imshow(predictions[i].squeeze(), cmap='gray')
        plt.title('Restaurada')
        plt.axis('off')
        
        
        plt.subplot(3, n_samples, i + 1 + 2 * n_samples)
        plt.imshow(targets[i].squeeze(), cmap='gray')
        plt.title('Target')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'results_{degradation_type}.png'), dpi=150, bbox_inches='tight')
    plt.show()


def main_training_torch():
    
    required_folders = ['source_a', 'source_b', 'target']
    missing_folders = []
    
    for folder in required_folders:
        folder_path = os.path.join(TRAIN_DIR, folder)
        if os.path.exists(folder_path):
            png_files = glob.glob(os.path.join(folder_path, '*.png'))
            print(f"✓ {folder}: {len(png_files)} archivos PNG")
            if len(png_files) == 0:
                print(f"  ⚠️ No se encontraron archivos PNG en {folder}")
        else:
            missing_folders.append(folder)
            print(f"✗ {folder}: Carpeta no encontrada")
    
    if missing_folders:
        print(f"\n❌ Faltan las siguientes carpetas: {missing_folders}")
        print(f"Estructura esperada en {TRAIN_DIR}:")
        print("├── source_a/  (imágenes con aliasing)")
        print("├── source_b/  (imágenes con blur)")
        print("└── target/    (imágenes target)")
        return
    
    print("\n" + "="*60)
    print("=== ENTRENAMIENTO PARA ALIASING (source_a) ===")
    print("="*60)
    
    #! the amount of epoch is fixed after some experimentations for this particular training set
    #! in a particular dataset experimentations are strongly sugested with a learning rate implented
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_aliasing, history_aliasing, val_loader_a = train_mri_restoration(
            degradation_type='source_a',
            input_size=(256, 256),
            batch_size=12,
            epochs=150,
            learning_rate=1e-4
        )
        
        print("\n=== VISUALIZACIÓN RESULTADOS ALIASING ===")
        visualize_results(model_aliasing, val_loader_a, 'source_a', device)
    
    except Exception as e:
        print(f"❌ Error en entrenamiento para aliasing: {e}")
        print("Continuando con blur...")
    
    print("\n" + "="*60)
    print("=== ENTRENAMIENTO PARA BLUR (source_b) ===")
    print("="*60)
    

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_blur, history_blur, val_loader_b = train_mri_restoration(
            degradation_type='source_b',
            input_size=(256, 256),
            batch_size=12,
            epochs=300,
            learning_rate=1e-4
        )
        
        print("\n=== VISUALIZACIÓN RESULTADOS BLUR ===")
        visualize_results(model_blur, val_loader_b, 'source_b', device)
    
    except Exception as e:
        print(f"❌ Error en entrenamiento para blur: {e}")
    
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO!")
    print("="*60)
    print("Modelos guardados en:", CHECKPOINT_DIR)
    print("Resultados guardados en:", RESULTS_DIR)
