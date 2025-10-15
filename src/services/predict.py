import torch

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
    predicted_label = model.config.id2label[predicted_class_idx]

    # Optional: Get probabilities using softmax
    probabilities = torch.softmax(logits, dim=-1)
    predicted_prob = probabilities[0, predicted_class_idx].item()

    return predicted_label, predicted_prob

