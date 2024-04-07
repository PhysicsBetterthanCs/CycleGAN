import gradio as gr
import torch

from models.Generator import Generator as gan


def model_initall():
    checkpoint = torch.load("../current.ckpt")
    model = gan()
    model.load_state_dict(checkpoint["g_ptm"])
    model.eval()
    return model


model = model_initall()

demo = gr.Interface(
    fn=model,
    inputs=gr.Image(type="numpy"),
    outputs=[
        "image"
    ]
)

demo.launch()