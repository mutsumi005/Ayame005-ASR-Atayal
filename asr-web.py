import torch
import torchaudio
import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration


model = WhisperForConditionalGeneration.from_pretrained("./results/checkpoint-1707")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
model.eval()


def trim_before_dollar_and_remove_all(text):
    dollar_pos = text.find('$')
    if dollar_pos != -1:
        text = text[:dollar_pos]
    return text.replace('$', '').strip()


def split_audio(audio_path, max_length_sec=30, sample_rate=16000):
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    max_samples = max_length_sec * sample_rate
    return [waveform[:, i:i + max_samples] for i in range(0, waveform.shape[1], max_samples)]


def transcribe_trimmed(audio_file):
    TOKENS_PER_SECOND = 3.5
    audio_chunks = split_audio(audio_file)

    full_pred = []

    for chunk in audio_chunks:
        duration_sec = chunk.shape[1] / 16000
        max_len = int(TOKENS_PER_SECOND * duration_sec) + 10

        input_features = processor.feature_extractor(
            chunk.numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(
                input_features,
                do_sample=False, num_beams=5, max_length=max_len + 10,
                repetition_penalty=1.2, early_stopping=True,
                temperature=0.1, forced_decoder_ids=forced_decoder_ids
            )
        pred = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        full_pred.append(pred)

    final_output = " ".join(full_pred).strip()
    trimmed_output = trim_before_dollar_and_remove_all(final_output)
    return trimmed_output


with gr.Blocks() as demo:
    gr.Markdown("## Whisper 泰雅語轉羅馬拼音\n上傳一個音檔，我們將模型轉錄後的羅馬拼音輸出")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="請上傳音檔")
        output_text = gr.Textbox(label="泰雅語轉羅馬拼音結果")

    with gr.Row():
        transcribe_btn = gr.Button("轉錄")
        clear_btn = gr.Button("清除")

    transcribe_btn.click(fn=transcribe_trimmed, inputs=audio_input, outputs=output_text)
    clear_btn.click(fn=lambda: "", inputs=[], outputs=output_text)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
