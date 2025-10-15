import gradio as gr

def user_greeting(name):
    return "안녕하세요! " + name + "님, 첫 번째 Gradio 애플리케이션에 오신 것을 환영합니다!😎"

app = gr.Interface(fn=user_greeting, inputs="text", outputs="text")
app.launch()