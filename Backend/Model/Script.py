import torch
import torch.nn as nn
import nibabel as nib
from scipy.ndimage import zoom
from torchvision.transforms import Normalize
import logging
import gc
import numpy as np  # Import numpy for dtype and calculations

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlzheimerNet(nn.Module):
    def __init__(self, printtoggle=False):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=7, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=7, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=7, stride=1, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=7, stride=1, padding=1)
        self.conv5 = nn.Conv3d(256, 128, kernel_size=7, stride=1, padding=1)
        self.fc_input_size = 128 * 4 * 4 * 4
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc_age_sex = nn.Linear(2, 16)
        self.fc_combined = nn.Linear(128 + 16, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 3)
        self.print = printtoggle

    def forward(self, x, age_sex):
        if self.print: print(f'Input: {list(x.shape)}')
        x = torch.nn.functional.gelu(torch.nn.functional.max_pool3d(self.conv1(x), 2))
        if self.print: print(f'Layer conv1/pool1: {x.shape}')
        x = torch.nn.functional.gelu(torch.nn.functional.max_pool3d(self.conv2(x), 2))
        if self.print: print(f'Layer conv2/pool2: {x.shape}')
        x = torch.nn.functional.gelu(torch.nn.functional.max_pool3d(self.conv3(x), 2))
        if self.print: print(f'Layer conv3/pool3: {x.shape}')
        x = torch.nn.functional.gelu(torch.nn.functional.max_pool3d(self.conv4(x), 2))
        if self.print: print(f'Layer conv4/pool4: {x.shape}')
        x = torch.nn.functional.gelu(torch.nn.functional.max_pool3d(self.conv5(x), 2))
        if self.print: print(f'Layer conv5/pool5: {x.shape}')
        x = x.view(x.size(0), -1)
        if self.print: print(f'Flattened: {x.shape}')
        x = torch.nn.functional.gelu(self.fc1(x))
        if self.print: print(f'Layer fc1: {x.shape}')
        x = torch.nn.functional.gelu(self.fc2(x))
        if self.print: print(f'Layer fc2: {x.shape}')
        age_sex = torch.nn.functional.gelu(self.fc_age_sex(age_sex))
        if self.print: print(f'Age and Sex branch: {age_sex.shape}')
        x = torch.cat((x, age_sex), dim=1)
        if self.print: print(f'Combined features: {x.shape}')
        x = torch.nn.functional.gelu(self.fc_combined(x))
        if self.print: print(f'Layer fc_combined: {x.shape}')
        x = torch.nn.functional.gelu(self.fc3(x))
        if self.print: print(f'Layer fc3: {x.shape}')
        x = torch.nn.functional.gelu(self.fc4(x))
        if self.print: print(f'Layer fc4: {x.shape}')
        return self.out(x)

def load_model(model_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AlzheimerNet()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully.")
        return model, device
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def preprocess_image(image_path, target_shape=(256, 256, 256)):
    try:
        img = nib.load(image_path).get_fdata(dtype=np.float32)
        logger.info(f"Original image shape: {img.shape}, dtype: {img.dtype}")

        # Adjust target shape dynamically for large images
        if img.size * 4 > 1e8:  # Threshold: ~100 MB
            target_shape = tuple(max(s // 2, 64) for s in target_shape)
            logger.info(f"Adjusted target shape for memory optimization: {target_shape}")

        zoom_factors = [t / s for t, s in zip(target_shape, img.shape)]
        logger.info(f"Zoom factors: {zoom_factors}")

        img_resized = zoom(img, zoom_factors, order=3).astype(np.float32)
        transform = Normalize(mean=[0.5], std=[0.5])
        img_tensor = transform(torch.tensor(img_resized).unsqueeze(0).unsqueeze(0).float())
        logger.info(f"Image tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")

        return img_tensor
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def preprocess_and_predict(image_path, age, sex, model, device):
    try:
        # Preprocess the image
        img_tensor = preprocess_image(image_path).to(device)
        logger.info(f"Image tensor shape: {img_tensor.shape}")

        # Prepare auxiliary inputs
        aux_inputs = torch.tensor([age, sex], dtype=torch.float32).unsqueeze(0).to(device)
        logger.info(f"Auxiliary inputs: {aux_inputs}")

        # Forward pass through the model
        with torch.no_grad():
            logger.info("Starting model forward pass...")
            output = model(img_tensor, aux_inputs)
            logger.info(f"Raw model output: {output}")

            prediction = torch.argmax(output, dim=1).item()
            result = {'class': prediction, 'probabilities': output.cpu().numpy().tolist()}
            logger.info(f"Prediction result: {result}")
            return result
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise
    finally:
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()
