import whisper
import gradio as gr
from groq import Groq
from deep_translator import GoogleTranslator
from diffusers import StableDiffusionPipeline
import os
import torch
import openai



api_key ="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"
client = Groq(api_key=api_key)

model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")



# Updated function for text generation using the new API structure
def generate_creative_text(prompt):
    chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content":prompt}
                ],
                model="llama-3.2-90b-text-preview"
            )
    chatbot_response = chat_completion.choices[0].message.content
    return chatbot_response

def process_audio(audio_path, image_option, creative_text_option):
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

    # Step 3: Generate creative text (if selected)
    creative_text = None
    if creative_text_option == "Generate Creative Text":
        creative_text = generate_creative_text(translation)


    # Step 4: Generate image (if selected)
    image = None
    if image_option == "Generate Image":
        try:
            model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
            pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors=True)
            pipe = pipe.to("cuda")
            image = pipe(translation).images[0]
        except Exception as e:
            return tamil_text, translation, creative_text, f"An error occurred during image generation: {str(e)}"

    return tamil_text, translation, creative_text, image

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Base()) as iface:
    gr.Markdown("# Audio Transcription, Translation, Image & Creative Text Generation")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            image_option = gr.Dropdown(["Generate Image", "Skip Image"], label="Image Generation", value="Generate Image")
            creative_text_option = gr.Dropdown(["Generate Creative Text", "Skip Creative Text"], label="Creative Text Generation", value="Generate Creative Text")
            submit_button = gr.Button("Process Audio")
        with gr.Column():
            tamil_text_output = gr.Textbox(label="Tamil Transcription")
            translation_output = gr.Textbox(label="English Translation")
            creative_text_output = gr.Textbox(label="Creative Text")
            image_output = gr.Image(label="Generated Image")
    submit_button.click(
        fn=process_audio,
        inputs=[audio_input, image_option, creative_text_option],
        outputs=[tamil_text_output, translation_output, creative_text_output, image_output]
    )

# Launch the interface
iface.launch()
