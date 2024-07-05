import streamlit as st
from transformers import pipeline
from PIL import Image
import librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


# Initialize Hugging Face pipelines
text_explainer = pipeline("text-generation", model="gpt2")
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
# If you have a speech-to-text model, you can initialize it similarly
# audio_classifier = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# Initialize model 
text_generator = pipeline("text-generation", model="gpt2")
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")


# Define the main function
def main():
    st.title("Multimodal AI Application")

    # Text input and processing
    st.header("Text Input")
    user_text = st.text_area("Enter some text")
    if user_text:
        explanation = text_generator(user_text, max_length=50, num_return_sequences=1)[0]['generated_text']
        st.write("Text explanation result:")
        st.write(explanation)


    # Upload and display an image
    st.header("Image Input")
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image_classification = image_classifier(image)
        st.write("Image classification result:")
        st.write(image_classification)

    # Upload and display audio
    st.header("Audio Input")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    if audio_file is not None:
     audio_data, sr = librosa.load(audio_file, sr=16000)
     input_values = tokenizer(audio_data, return_tensors="pt").input_values
     logits = model(input_values).logits
     predicted_ids = torch.argmax(logits, dim=-1)
     transcription = tokenizer.decode(predicted_ids[0])

     st.write("Transcription of audio:")
     st.write(transcription)



if __name__ == "__main__":
    main()