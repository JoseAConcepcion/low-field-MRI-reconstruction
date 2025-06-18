import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward, DWTInverse
import math


class WaveletLayer(nn.Module):
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wavelet, mode='symmetric')
    
    def forward(self, x):
        yl, yh = self.dwt(x)
        cA = yl
        cH, cV, cD = torch.unbind(yh[0], dim=2)  
        return cA, cH, cV, cD


class InverseWaveletLayer(nn.Module):
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.idwt = DWTInverse(wave=wavelet, mode='symmetric')
    
    def forward(self, cA, cH, cV, cD):
        yh = [torch.stack([cH, cV, cD], dim=2)]
        return self.idwt((cA, yh))


def calculate_psnr(original, reconstructed, max_pixel_value=1.0):
    original_np = original.squeeze().detach().cpu().numpy()
    reconstructed_np = reconstructed.squeeze().detach().cpu().numpy()

    mse = ((original_np - reconstructed_np) ** 2).mean()

    if mse == 0:
        return float('inf')
    
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)
    return psnr


def main(image_path, target_image_path, output_filename_full='wavelet_decomposition_full.png', 
         output_filename_experiment='wavelet_experiment_details.png'):
    
    
    image = read_image(image_path).float() / 255.0  
    if image.shape[0] != 1 and image.shape[0] != 3: 
        image = image.permute(2, 0, 1) 
    if image.shape[0] == 3:
        image = transforms.Grayscale()(image)
    original_image_for_psnr = image.clone().unsqueeze(0) 
    image = image.unsqueeze(0)  

    
    target_image = read_image(target_image_path).float() / 255.0
    if target_image.shape[0] != 1 and target_image.shape[0] != 3:
        target_image = target_image.permute(2, 0, 1)
    if target_image.shape[0] == 3:
        target_image = transforms.Grayscale()(target_image)
    target_image_for_psnr = target_image.clone().unsqueeze(0) 


    
    wavelet_layer = WaveletLayer()
    inverse_wavelet_layer = InverseWaveletLayer()

    
    cA, cH, cV, cD = wavelet_layer(image)

    
    reconstructed_full = inverse_wavelet_layer(cA, cH, cV, cD)

    
    psnr_full_vs_target = calculate_psnr(target_image_for_psnr, reconstructed_full)
    print(f"PSNR (Reconstrucción Completa vs Target): {psnr_full_vs_target:.2f} dB")

    
    plt.figure(figsize=(15, 10))

    
    plt.subplot(2, 3, 1)
    plt.title("Imagen Original")
    plt.imshow(image.squeeze().detach().cpu().numpy(), cmap='gray') 
    plt.axis('off')

    
    plt.subplot(2, 3, 2)
    plt.title("Aproximación (cA)")
    plt.imshow(cA.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.axis('off')

    
    plt.subplot(2, 3, 3)
    plt.title("Detalle Horizontal (cH)")
    plt.imshow(cH.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.axis('off')

    
    plt.subplot(2, 3, 4)
    plt.title("Detalle Vertical (cV)")
    plt.imshow(cV.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.axis('off')

    
    plt.subplot(2, 3, 5)
    plt.title("Detalle Diagonal (cD)")
    plt.imshow(cD.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.axis('off')

    
    plt.subplot(2, 3, 6)
    plt.title(f"Imagen Reconstruida (PSNR vs Target: {psnr_full_vs_target:.2f} dB)")
    plt.imshow(reconstructed_full.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_filename_full) 
    plt.close() 
    print(f"Figura de descomposición y reconstrucción completa guardada como '{output_filename_full}'")

    
    print("\n--- Iniciando Experimento de Eliminación de Detalles ---")

    
    cH_zero = torch.zeros_like(cH)
    reconstructed_no_H = inverse_wavelet_layer(cA, cH_zero, cV, cD)
    psnr_no_H_vs_target = calculate_psnr(target_image_for_psnr, reconstructed_no_H)
    print(f"PSNR (Sin Detalle Horizontal - cH vs Target): {psnr_no_H_vs_target:.2f} dB")

    
    cV_zero = torch.zeros_like(cV)
    reconstructed_no_V = inverse_wavelet_layer(cA, cH, cV_zero, cD)
    psnr_no_V_vs_target = calculate_psnr(target_image_for_psnr, reconstructed_no_V)
    print(f"PSNR (Sin Detalle Vertical - cV vs Target): {psnr_no_V_vs_target:.2f} dB")

    
    cD_zero = torch.zeros_like(cD)
    reconstructed_no_D = inverse_wavelet_layer(cA, cH, cV, cD_zero)
    psnr_no_D_vs_target = calculate_psnr(target_image_for_psnr, reconstructed_no_D)
    print(f"PSNR (Sin Detalle Diagonal - cD vs Target): {psnr_no_D_vs_target:.2f} dB")

    
    cA_zero = torch.zeros_like(cA)
    reconstructed_only_details = inverse_wavelet_layer(cA_zero, cH, cV, cD)
    psnr_only_details_vs_target = calculate_psnr(target_image_for_psnr, reconstructed_only_details)
    print(f"PSNR (Solo Detalles - sin cA vs Target): {psnr_only_details_vs_target:.2f} dB")


    
    plt.figure(figsize=(15, 7))

    plt.subplot(2, 2, 1)
    plt.title(f"Reconstrucción sin Detalle H (PSNR vs Target: {psnr_no_H_vs_target:.2f} dB)")
    plt.imshow(reconstructed_no_H.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title(f"Reconstrucción sin Detalle V (PSNR vs Target: {psnr_no_V_vs_target:.2f} dB)")
    plt.imshow(reconstructed_no_V.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title(f"Reconstrucción sin Detalle D (PSNR vs Target: {psnr_no_D_vs_target:.2f} dB)")
    plt.imshow(reconstructed_no_D.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title(f"Reconstrucción solo con Detalles (PSNR vs Target: {psnr_only_details_vs_target:.2f} dB)")
    plt.imshow(reconstructed_only_details.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_filename_experiment)
    plt.close()
    print(f"Figura de experimentación guardada como '{output_filename_experiment}'")

if __name__ == "__main__":
    image_path = 'IM000000.png' 
    target_image_path = 'IM000000t.png' 
    
    try:
        main(image_path, target_image_path)
    except FileNotFoundError:
        print(f"Error: Una de las imágenes no se encontró.")
        print(f"Asegúrate de que '{image_path}' y '{target_image_path}' existan y las rutas sean correctas.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")