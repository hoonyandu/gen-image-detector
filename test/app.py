import gradio as gr
from PIL import Image, ImageFilter

def gaussian_blur(image):
    # numpy to pillow image
    image = Image.fromarray(image)
    # apply gaussian blur to pillow image
    blur = image.filter(ImageFilter.GaussianBlur(15))
    
    return blur

def user_greeting(name):
    return "안녕하세요! " + name + "님, 첫 번째 Gradio 애플리케이션에 오신 것을 환영합니다!😎"

# app = gr.Interface(fn=user_greeting, inputs="text", outputs="text")
app = gr.Interface(fn=gaussian_blur, inputs="image", outputs="image")
app.launch()