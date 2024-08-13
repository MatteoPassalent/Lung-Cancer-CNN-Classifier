import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import preprocess_input

model = load_model('custom_CNN.keras')

layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

def predict_image(img):
    img = img.resize((256, 256))

    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    prediction = model(x, training=False)
    percent = prediction.numpy()[0][0]

    if percent < 0.5:
        result = "Positive for lung cancer"
        certainty = str(round((1 - percent) * 100, 2)) + "%"
    else:
        result = "Negative for lung cancer"
        certainty = str(round(percent * 100, 2)) + "%"

    activations = activation_model.predict(x)
    
    first_layer_activation = activations[0][0, :, :, :]
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('First Convolutional Layer Activations')
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.matshow(first_layer_activation[:, :, i], cmap='viridis')
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig('layer_visualization.png')
    plt.close()

    return result + ' with a certainty of ' + certainty, 'layer_visualization.png'

gr_interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=["text", gr.Image(type="filepath")],
    title="Lung Cancer Detection",
    description="Upload a CT scan image to predict whether it is positive or negative for lung cancer.",
    examples=["Malignant case (506).jpg", "Normal case (374).jpg"]
)
gr_interface.launch()
