import torch
import numpy as np
import matplotlib.cm as cm


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, class_idx):
        output = self.model(input_image)
        self.model.zero_grad()

        loss = output[:, class_idx]
        loss.backward()

        weights = torch.mean(self.gradients, dim=(2, 3))
        cam = torch.sum(weights[:, :, None, None] * self.activations, dim=1)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        return cam



def show_cam_on_image(img, cam):
    # garantir tamanho correto
    cam = np.resize(cam, (img.shape[0], img.shape[1]))

    # aplicar colormap (JET)
    heatmap = cm.jet(cam)[..., :3]  # remove alpha channel

    heatmap = np.float32(heatmap)

    overlay = heatmap + np.float32(img)
    overlay = overlay / np.max(overlay)

    return overlay