# Fluxograma Analyzer - Notebook Final

# 1. Importações e Setup
import pytesseract
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import networkx as nx
import os
from fpdf import FPDF

# 2. OCR - Extração de Texto com Coordenadas
def extract_text_blocks(image_path):
    image = Image.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    draw = ImageDraw.Draw(image)
    blocks = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            blocks.append({'text': text, 'bbox': (x, y, w, h)})
            draw.rectangle([x, y, x + w, y + h], outline='red', width=2)
            draw.text((x, y - 10), text, fill='blue')
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title("Blocos de Texto Detectados")
    plt.show()
    return blocks

# 3. Análise com BLIP (modelo multimodal)
def analyze_with_blip(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# 4. Construção e Análise de Grafo
def draw_and_identify_bottlenecks(blocks):
    G = nx.DiGraph()
    for i, block in enumerate(blocks):
        G.add_node(i, label=block['text'])
        if i > 0:
            G.add_edge(i - 1, i)
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color='lightblue', font_size=10)
    bottlenecks = [n for n in G.nodes if G.in_degree(n) > 1]
    nx.draw_networkx_nodes(G, pos, nodelist=bottlenecks, node_color='red')
    plt.title("Grafo Gerado e Gargalos (vermelho)")
    plt.axis('off')
    plt.savefig("grafo_corrigido.png")
    plt.show()
    return G, bottlenecks

# 5. Sugestões de Correção para Gargalos
def sugerir_correcao(grafo, gargalos):
    sugestoes = []
    for g in gargalos:
        label = grafo.nodes[g].get('label', f'Nó {g}')
        sugestao = f"O bloco '{label}' é um gargalo. Sugestão: divida o fluxo de entrada ou revise a lógica de dependência."
        sugestoes.append(sugestao)
    return sugestoes

# 6. Gerar PDF da imagem corrigida
def gerar_pdf(imagem_path, sugestoes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Fluxograma Corrigido", ln=True, align='C')
    pdf.image(imagem_path, x=10, y=30, w=190)
    pdf.ln(120)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Sugestões de Correção:", ln=True)
    for s in sugestoes:
        pdf.multi_cell(0, 8, txt="- " + s)
    pdf.output("fluxograma_corrigido.pdf")

# 7. Execução Exemplo
image_path = "imagens/exemplo_fluxograma.png"
blocks = extract_text_blocks(image_path)
caption = analyze_with_blip(image_path)
print("Descrição:", caption)
G, bottlenecks = draw_and_identify_bottlenecks(blocks)
sugestoes = sugerir_correcao(G, bottlenecks)
gerar_pdf("grafo_corrigido.png", sugestoes)

# Instruções para VS Code e gitignore omitidas por brevidade...
