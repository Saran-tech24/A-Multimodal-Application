import whisper
import gradio as gr
from groq import Groq
from deep_translator import GoogleTranslator
from diffusers import StableDiffusionPipeline
import torch


# Replace with your actual API key
api_key ="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"
client = Groq(api_key=api_key)


model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id2 = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cpu")

prompt = """dreamlikeart, a grungy woman with rainbow hair, travelling between dimensions, dynamic pose, happy, soft eyes and narrow chin,
extreme bokeh, dainty figure, long hair straight down, torn kawaii shirt and baggy jeans
"""
image = pipe(prompt).images[0]

# Function to transcribe, translate, and analyze sentiment
def process_audio(audio_path, image_option):
    if audio_path is None:
        return "Please upload an audio file.", None, None, None

    # Step 1: Transcribe audio
    try:
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), file.read()),
                model="whisper-large-v3",
                language="ta",
                response_format="verbose_json",
            )
        tamil_text = transcription.text
    except Exception as e:
        return f"An error occurred during transcription: {str(e)}", None, None, None

    # Step 2: Translate Tamil to English
    try:
        translator = GoogleTranslator(source='ta', target='en')
        translation = translator.translate(tamil_text)
    except Exception as e:
        return tamil_text, f"An error occurred during translation: {str(e)}", None, None



    # Step 3: Generate image (if selected)
    image = None
    if image_option == "Generate Image":
        try:
            model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
            pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors=True)
            pipe = pipe.to("cpu")
            image = pipe(translation).images[0]
        except Exception as e:
            return tamil_text, translation, f"An error occurred during image generation: {str(e)}"

    return tamil_text, translation, image

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Base()) as iface:
    gr.Markdown("# Audio Transcription, Translation, and image Generate")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            image_option = gr.Dropdown(["Generate Image", "Skip Image"], label="Image Generation", value="Generate Image")
            submit_button = gr.Button("Process Audio")
        with gr.Column():
            tamil_text_output = gr.Textbox(label="Tamil Transcription")
            translation_output = gr.Textbox(label="English Translation")
            image_output = gr.Image(label="Generated Image")

    submit_button.click(
        fn=process_audio,
        inputs=[audio_input, image_option],
        outputs=[tamil_text_output, translation_output, image_output]
    )

# Launch the interface
iface.launch()
