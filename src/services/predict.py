import torch

from domain.models import ModelWrapper, get_output_targets
from domain.visualization import get_visualize_gradcam
from src import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(processor, model, image):
    print("Preprocessing image...")
    # Use the processor to prepare the image for the model
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Perform Inference
    print("Running inference...")
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(**inputs)
        logits = outputs.logits

    # Interpret the Results
    # Get the index of the highest logit score -> this is the predicted class ID
    predicted_class_idx = logits.argmax(-1).item()

    # Use the model's config to map the ID back to the label string ('ai' or 'hum')
    # predicted_label = model.config.id2label[predicted_class_idx]
    predicted_label = config.MODEL_ID_2_LABELS[predicted_class_idx]

    # Optional: Get probabilities using softmax
    probabilities = torch.softmax(logits, dim=-1)
    predicted_prob = probabilities[0, predicted_class_idx].item()

    return predicted_label, predicted_prob


def predict_with_gradcam(processor, model, image):
    input_tensor = processor(image, return_tensors="pt")["pixel_values"].to(
        model.device
    )

    # inference
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    with torch.no_grad():  # Disable gradient calculations for inference
        logits = wrapped_model.forward(input_tensor)

    # Interpret the Results
    # Get the index of the highest logit score -> this is the predicted class ID
    predicted_class_idx = logits.argmax(-1).item()
    # predicted_label = model.config.id2label[predicted_class_idx]
    predicted_label = config.MODEL_ID_2_LABELS[predicted_class_idx]

    probabilities = torch.softmax(logits, dim=-1)
    predicted_prob = probabilities[0, predicted_class_idx].item()

    # visualize
    target_layers = [wrapped_model.model.vision_model.encoder.layers[-1].layer_norm1]
    
    # Ensure we're using the correct target class for Grad-CAM
    targets = get_output_targets(predicted_class_idx)
    
    image_size = model.config.vision_config.image_size

    visualization = get_visualize_gradcam(
        wrapped_model, target_layers, targets, input_tensor, image, image_size
    )

    return predicted_label, predicted_prob, visualization
