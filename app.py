import gradio as gr
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image

# =========================
# CONFIGURACIÓN
# =========================

CONF_THRESHOLD = 0.25 #Confinza minima
NMS_THRESHOLD = 0.45  #Porcentaje Caja Superpuesta
IMG_SIZE = 640

CLASS_NAMES = [
    "Fuego",  
    "Humo"   
]

# =========================
# MODELO
# =========================

model = torch.jit.load("best.torchscript")
model.eval()

# =========================
# INFERENCIA
# =========================

def predict(image):

    img = np.array(image)

    original_h, original_w = img.shape[:2]

    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    tensor = (
        torch.tensor(
            resized,
            dtype=torch.float32
        )
        .permute(2, 0, 1)
        / 255.0
    ).unsqueeze(0)

    with torch.no_grad():
        preds = model(tensor)

    preds = preds[0].T

    boxes = []
    scores = []
    classes = []

    for det in preds:

        x = float(det[0])
        y = float(det[1])
        w = float(det[2])
        h = float(det[3])

        class_scores = det[4:]

        score, cls = torch.max(class_scores, dim=0)

        score = float(score)
        cls = int(cls)

        if score < CONF_THRESHOLD:
            continue

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        classes.append(cls)

    if len(boxes) == 0:
        return Image.fromarray(img)

    keep = torchvision.ops.nms(
        torch.tensor(boxes),
        torch.tensor(scores),
        NMS_THRESHOLD
    )

    for idx in keep:

        idx = int(idx)

        x1, y1, x2, y2 = boxes[idx]

        x1 = int(x1 * original_w / IMG_SIZE)
        y1 = int(y1 * original_h / IMG_SIZE)
        x2 = int(x2 * original_w / IMG_SIZE)
        y2 = int(y2 * original_h / IMG_SIZE)

        score = scores[idx]
        cls = classes[idx]

        label = f"{CLASS_NAMES[cls]} {score:.2f}"

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return Image.fromarray(img)

# =========================
# INTERFAZ
# =========================

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Detector de Objetos YOLOv8 TorchScript",
    description="Sube una imagen para realizar detección."
)

demo.launch()