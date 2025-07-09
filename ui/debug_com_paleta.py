# Ficheiro: debug_com_paleta.py
# Versão FINALÍSSIMA com a correção do bug de estado atrasado usando JavaScript.

import gradio as gr
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import os

# --- 1. Definição da Paleta de Cores ---
COLOR_LIST = [
    # ... (a sua lista de cores completa, como antes) ...
    {"label": "building", "color": (219, 14, 154)},
    {"label": "pervious surface", "color": (147, 142, 123)},
    {"label": "impervious surface", "color": (248, 12, 0)},
    {"label": "bare soil", "color": (169, 113, 1)},
    {"label": "water", "color": (21, 83, 174)},
    {"label": "coniferous", "color": (25, 74, 38)},
    {"label": "deciduous", "color": (70, 228, 131)},
    {"label": "brushwood", "color": (243, 166, 13)},
    {"label": "vineyard", "color": (102, 0, 130)},
    {"label": "herbaceous vegetation", "color": (85, 255, 0)},
    {"label": "agricultural land", "color": (255, 243, 13)},
    {"label": "plowed land", "color": (228, 223, 124)},
    {"label": "swimming pool", "color": (61, 230, 235)},
    {"label": "snow", "color": (255, 255, 255)},
    {"label": "clear cut", "color": (138, 179, 160)},
    {"label": "mixed", "color": (107, 113, 79)},
    {"label": "pinus", "color": (95, 80, 231)},
]
DEFAULT_COLOR = COLOR_LIST[6]["color"]
# Converte as cores para o formato hexadecimal para o JavaScript
HEX_COLORS_WITH_LABELS = [(item["label"], f"#{item['color'][0]:02x}{item['color'][1]:02x}{item['color'][2]:02x}") for item in COLOR_LIST]


# --- 2. Carregamento do Modelo ---
print("Carregando modelos... Por favor, aguarde.")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16
caminho_do_modelo = "../finetuning/controlnet_training/model_pinus_finetuned"
controlnet = ControlNetModel.from_pretrained(caminho_do_modelo, torch_dtype=dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, torch_dtype=dtype
).to(device)
print("✅ Modelos carregados.")


# --- 3. Função de Geração de Imagem ---
def generate_map(image, prompt, steps, seed):
    if image is None:
        raise gr.Error("Por favor, desenhe ou envie uma máscara primeiro.")
    generated_image = pipe(
        prompt=prompt, num_inference_steps=int(steps),
        generator=torch.Generator(device="cuda").manual_seed(int(seed)),
        image=image,
    ).images[0]
    return generated_image

# --- 4. Construção da Interface com JavaScript ---
with gr.Blocks() as demo:
    gr.Markdown("# Site de Debug do Seg2Sat")
    gr.Markdown("Selecione uma classe na paleta para mudar a cor do pincel e desenhe a sua máscara.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Paleta e Canvas")
            color_selector = gr.Radio(
                choices=HEX_COLORS_WITH_LABELS, # Usamos a lista com cores hex
                label="Tipo de Pincel",
                value=HEX_COLORS_WITH_LABELS[6][1] # Pega o valor hex da cor padrão
            )
            input_image = gr.Image(
                label="Canvas",
                source="canvas",
                tool="color-sketch",
                type="pil",
                shape=(512, 512),
                elem_id="canvas_component" # <<< MUDANÇA 1: Damos um ID ao canvas
            )
            
        with gr.Column():
            gr.Markdown("### 2. Parâmetros e Resultado")
            prompt_input = gr.Textbox(label="Prompt", value="Aerial view of a forest with pinus trees in Paraná, Brazil. photorealistic, 4k, high detail.")
            steps_input = gr.Slider(label="Steps", minimum=10, maximum=50, step=1, value=30)
            seed_input = gr.Number(label="Seed", value=1024)
            generate_button = gr.Button("Gerar Imagem", variant="primary")
            output_image = gr.Image(label="Imagem Gerada", type="pil")

    # --- Lógica da Interface ---
    
    # MUDANÇA 2: A função agora é uma string de JavaScript
    js_update_function = """
    (selected_color_hex) => {
        // Encontra o componente do canvas pelo ID que definimos
        const canvas_comp = document.getElementById('canvas_component');
        // Encontra o input de cor específico dentro do componente
        const color_input = canvas_comp.querySelector('input[type="color"]');
        // Define o valor do input de cor para a nova cor
        color_input.value = selected_color_hex;
        // Dispara um evento de 'input' para notificar o canvas da mudança
        color_input.dispatchEvent(new Event('input'));
        // Retorna o valor para atualizar o próprio seletor de rádio (opcional, mas bom)
        return selected_color_hex;
    }
    """
    
    # MUDANÇA 3: Usamos a função JS no evento .select()
    color_selector.select(fn=None, _js=js_update_function, inputs=color_selector, outputs=color_selector)
    
    generate_button.click(
        fn=generate_map, 
        inputs=[input_image, prompt_input, steps_input, seed_input], 
        outputs=output_image
    )

# --- 5. Inicia a Aplicação ---
demo.launch()