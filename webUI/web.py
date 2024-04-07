import gradio as gr
import torch

from models.Generator import Generator as gan


def model_initall():
    checkpoint = torch.load("../current.ckpt")
    model = gan()
    model = model.load_state_dict(checkpoint["g_ptm"])
    return model


model = model_initall()

demo = gr.Interface(
    fn=model,
    inputs="images",
    outputs="images"
)

demo.launch()