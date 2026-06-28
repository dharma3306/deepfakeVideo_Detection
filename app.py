from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (128, 128))
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

    cap.release()
    return frames, total_frames, fps, duration


def analyze_frames(frames):
    """
    Heuristic deepfake analysis using real computer vision signals.

    Real videos typically have:
      - Higher temporal differences (natural motion between frames)
      - Higher Laplacian variance (natural texture and noise)
      - Higher pixel variance (complex natural scenes)

    Deepfake videos typically have:
      - Lower temporal differences (GAN frames are smoother)
      - Lower Laplacian variance (over-smoothed skin/faces)
      - Unusual RGB channel imbalance (GAN color artifacts)

    Replace the score block with model.predict() once you have a trained model.
    """
    if not frames:
        return None

    pixel_variances = []
    temporal_diffs = []
    noise_scores = []
    channel_diffs = []

    for i, frame in enumerate(frames):
        pixel_variances.append(float(np.var(frame)))

        if i > 0:
            diff = float(np.mean(np.abs(frames[i] - frames[i - 1])))
            temporal_diffs.append(diff)

        # Laplacian variance — measures sharpness/texture
        gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        noise_scores.append(laplacian_var)

        # RGB channel imbalance
        r_var = float(np.var(frame[:, :, 2]))
        g_var = float(np.var(frame[:, :, 1]))
        b_var = float(np.var(frame[:, :, 0]))
        channel_imbalance = float(np.std([r_var, g_var, b_var]))
        channel_diffs.append(channel_imbalance)

    avg_variance      = float(np.mean(pixel_variances))
    avg_temporal_diff = float(np.mean(temporal_diffs)) if temporal_diffs else 0.0
    avg_noise         = float(np.mean(noise_scores))
    avg_channel_diff  = float(np.mean(channel_diffs))

    # Low temporal diff → suspicious (deepfakes are temporally smoother)
    temporal_suspicion = float(np.clip(1.0 - (avg_temporal_diff / 0.05), 0.0, 1.0))

    # Low Laplacian → suspicious (over-smoothed)
    noise_suspicion = float(np.clip(1.0 - (avg_noise / 300.0), 0.0, 1.0))

    # Low pixel variance → suspicious (too uniform)
    variance_suspicion = float(np.clip(1.0 - (avg_variance / 0.04), 0.0, 1.0))

    # High channel imbalance → suspicious
    channel_suspicion = float(np.clip(avg_channel_diff / 0.01, 0.0, 1.0))

    score = (
        temporal_suspicion * 0.35 +
        noise_suspicion    * 0.30 +
        variance_suspicion * 0.20 +
        channel_suspicion  * 0.15
    )

    # Clamp to realistic range
    score = float(np.clip(score, 0.05, 0.95))

    return {
        'confidence':      round(score * 100, 2),
        'avg_pixel_variance': round(avg_variance, 6),
        'avg_temporal_diff':  round(avg_temporal_diff, 6),
        'avg_noise':          round(avg_noise, 2),
        'avg_channel_diff':   round(avg_channel_diff, 6),
        'frames_analyzed':    len(frames),
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file format. Use MP4, AVI, MOV, MKV, or WEBM.'}), 400

    filename = f"upload_{int(time.time())}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        start_time = time.time()
        frames, total_frames, fps, duration = extract_frames(filepath, num_frames=16)
        analysis = analyze_frames(frames)
        processing_time = round(time.time() - start_time, 2)

        if analysis is None:
            return jsonify({'error': 'Could not process video. Please try another file.'}), 500

        confidence = analysis['confidence']
        # Above 55% suspicion score = flagged as deepfake
        is_fake = confidence >= 55.0

        result = {
            'prediction': 'DEEPFAKE' if is_fake else 'AUTHENTIC',
            'confidence': confidence,
            'is_fake': is_fake,
            'video_info': {
                'total_frames': total_frames,
                'fps': round(fps, 2),
                'duration_seconds': round(duration, 2),
                'frames_analyzed': analysis['frames_analyzed'],
            },
            'analysis': {
                'pixel_variance':     analysis['avg_pixel_variance'],
                'temporal_inconsistency': analysis['avg_temporal_diff'],
                'noise_level':        analysis['avg_noise'],
                'channel_imbalance':  analysis['avg_channel_diff'],
            },
            'processing_time': processing_time,
        }

        os.remove(filepath)
        return jsonify(result)

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)