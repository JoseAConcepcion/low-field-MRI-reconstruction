import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward, DWTInverse

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

def main(image_path, output_filename='wavelet_decomposition.png'):
    image = read_image(image_path).float() / 255.0  
    
    if image.shape[0] != 1 and image.shape[0] != 3: 
        image = image.permute(2, 0, 1) 

    if image.shape[0] == 3:
        image = transforms.Grayscale()(image)

    image = image.unsqueeze(0)  

    wavelet_layer = WaveletLayer()
    inverse_wavelet_layer = InverseWaveletLayer()

    cA, cH, cV, cD = wavelet_layer(image)

    reconstructed_image = inverse_wavelet_layer(cA, cH, cV, cD)

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
    plt.title("Imagen Reconstruida")
    plt.imshow(reconstructed_image.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    
    
    plt.savefig(output_filename) 
    plt.close() 
    print(f"Figura guardada como '{output_filename}'")
    

if __name__ == "__main__":
    
    image_path = 'IM000000.png'
    
    
    output_figure_name = 'wavelet_decomposition_results.png' 

    try:
        main(image_path, output_figure_name)
    except FileNotFoundError:
        print(f"Error: La imagen no se encontró en la ruta: {image_path}")
        print("Por favor, asegúrate de que la ruta de la imagen sea correcta.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")