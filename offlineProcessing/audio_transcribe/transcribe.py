import utils
import whisperx
import json
import os
import glob


def transcribe_audio(args):
    args.VISUALISATION = os.path.join(args.OUTPUT_PATH, "visualisation")
    audio_transcibe_path = os.path.join(args.VISUALISATION, "audio_text.json")
    device = "cpu"
    audio_file = glob.glob(os.path.join(args.data, "*.wav"))
    if not audio_file:
        print("No .wav files found in the specified folder.")
        return
    batch_size = 64  # reduce if low on GPU mem
    compute_type = "float"  # change to "int8" if low on GPU mem (may reduce accuracy)
    use_keywords = args.keyword_transcribe_mode  # Set this flag to True for keyword-based transcription, False for full transcription

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file[0])
    result = model.transcribe(audio, batch_size=batch_size)

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    strat_word, end_word = args.audio_transcribe_keyword

    affordance = []

    if use_keywords:
        start = False
        sentence = {"text": "", "start": None, "end": None}
        for seg in result["segments"]:
            for word in seg["words"]:
                if word["word"].upper() == end_word.upper() or word["word"].upper() == end_word.upper() + ".":
                    start = False
                    sentence["end"] = word["end"]
                    affordance.append(sentence)
                    sentence = {"text": "", "start": None, "end": None}
                if start:
                    sentence["text"] += word["word"] + " "
                if word["word"].upper() == strat_word.upper():
                    start = True
                    sentence["start"] = word["start"]
    else:
        for seg in result["segments"]:
            sentence = {"text": seg["text"], "start": seg["start"], "end": seg["end"]}
            affordance.append(sentence)

    os.makedirs(os.path.dirname(audio_transcibe_path), exist_ok=True)
    # Save the output as a JSON file
    with open(audio_transcibe_path, "w") as outfile:
        json.dump(affordance, outfile, indent=4)

def main(args):
    transcribe_audio(args)

if __name__ == '__main__':
    main()