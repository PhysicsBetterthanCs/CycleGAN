import gradio as gr
import torch
from torchvision import transforms
from torchvision.utils import save_image

from models.model.generator import Generator as gan

# 模型加载权重
def model_initall():
    checkpoint = torch.load("../current.ckpt")
    model = gan()
    model.load_state_dict(checkpoint["g_ptm"])
    model.eval()
    return model

# 输出结果图像
def predict(inputs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = transforms.ToPILImage()(inputs)
    # 图片归一化处理
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    inputs = transform(inputs)
    generator = model_initall().to(device)
    outputs = generator(inputs.to(device))
    save_image(outputs, "monet_style.jpg", normalize=True)
    return "monet_style.jpg"


model = model_initall()

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[
        "image"
    ],
    title="CycleGAN",
    description="选择你的图片来转换成monet风格"
)

demo.launch()
