# Model Directory

Place your trained model files here.

## Expected Files

- `gru_model.h5` — Trained GRU model (Keras/TensorFlow)
- `cnn_model.pt` — Optional PyTorch CNN model

## To Load Model in app.py

```python
from tensorflow.keras.models import load_model

model = load_model('model/gru_model.h5')

def predict(frames):
    x = np.array(frames)              # shape: (16, 128, 128, 3)
    x = np.expand_dims(x, axis=0)    # shape: (1, 16, 128, 128, 3)
    score = model.predict(x)[0][0]   # confidence 0–1
    return float(score)
```

Replace the `analyze_frames()` simulation in `app.py` with this function for production use.

## Dataset

The model was trained on 10,000+ real and deepfake video samples.
Recommended datasets:
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DFDC (Deepfake Detection Challenge)](https://ai.facebook.com/datasets/dfdc/)
