# Sentiment-Analysis-Using-Audio

Sentiment Analysis Using Audio is a machine learning project designed to predict emotional sentiment from audio inputs. It uses audio feature extraction and a pre-trained neural network to classify emotions into one of eight categories.

**Features**
- Predicts sentiment from an uploaded audio file or recorded audio.
- Recognizes eight emotional sentiments:
  - Neutral
  - Calm
  - Happy
  - Sad
  - Angry
  - Fearful
  - Disgust
  - Surprised
- User-friendly interface through command-line interaction.
- Integrated audio recording functionality.

**Dataset**
- The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
- Dataset Source: https://zenodo.org/records/1188976

**Requirements**
- Python 3.8 or 3.9
- Required Python libraries (Install via `pip`):
   - pip install -r requirements.txt.
   - For PyAudio installation:
      Use the `.whl` file provided or install via pip:
      ```bash
      pip install PyAudio-0.2.14-cp310-cp310-win_amd64.whl

**Command to run:**
- python prediction.py

**Output**

![Output](https://github.com/user-attachments/assets/b86e8e8c-5dc8-4075-8ecd-017e3c424131)

**Acknowledgements**
- TensorFlow for the deep learning framework.
- Librosa for audio processing.
- Speech Emotion Recognition researchers and contributors for inspiration and datasets.
