
# Crowd Counting AI

A sophisticated AI-powered crowd counting system that uses computer vision and deep learning to accurately estimate the number of people in images or video streams.

## 🚀 Features

- 🎯 Real-time crowd density estimation
- 🖼️ Support for both image and video input
- 🧠 High accuracy using state-of-the-art deep learning models
- 📊 Interactive visualization of crowd density maps
- 🖱️ Easy-to-use interface via Streamlit and API endpoints via FastAPI

## 🧰 Prerequisites

- Python 3.8+
- PyTorch
- OpenCV
- CUDA-capable GPU (recommended for optimal performance)



## 🧪 Usage

### Image-based crowd counting:
```bash
python count_crowd.py --input path/to/image.jpg
```

### Video-based crowd counting:
```bash
python count_crowd.py --input path/to/video.mp4 --mode video
```

### Streamlit UI (for testing only)
```bash
cd ui
streamlit run app.py
```

### FastAPI Backend
```bash
cd api
uvicorn main:app --reload
```

## 📁 Project Structure

```
crowd-counting-ai/
├── api/                      # FastAPI backend
│   └── main.py
├── core/                     # Core logic and model loading
│   └── crowd_counting_model.py
├── model/                    # YOLOv8 pretrained weights
│   └── yolov8n.pt
├── test/crowd-counting/      # Sample test videos
├── ui/                       # Streamlit UI for demo and interaction
│   └── app.py
├── requirements.txt          # Python dependencies
```

## 🧠 Model Architecture

The system uses YOLOv8 for object detection, optimized for real-time crowd counting tasks. It leverages convolutional neural networks (CNNs) to detect individuals and estimate density from video frames or images.

## 📈 Evaluation

You can evaluate the model using the provided test videos under `test/crowd-counting/` to see performance in various crowd conditions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for suggestions and bugs.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors who have helped with the development
- Special thanks to the research community for advancing crowd counting techniques


## 📬 Contact

For questions or support, feel free to open an issue on the GitHub repository.
