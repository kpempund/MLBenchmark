# ML Model Performance Comparison Report

This project benchmarks the performance of six deep learning models in a binary image classification task. The results highlight how different architectures perform across **accuracy**, **speed**, and **resource usage**, helping determine the best model for real-world deployment.

---

## 📊 Model Performance Comparison

The benchmarking results were obtained through a combination of PyTorch evaluation and system monitoring tools. Each model was tested on identical input data and metrics were recorded using both `time` and `psutil` for accuracy, inference speed, GPU utilization, and memory usage.

| Model           | Accuracy | Precision | Recall | F1-score | Inference Time (ms) | Params (M) | GPU Util (%) | VRAM Usage (MB) |
|----------------|----------|-----------|--------|----------|----------------------|------------|---------------|------------------|
| **ResNet50**        | **0.992**  | 0.992     | 0.992  | 0.992    | 3.08                 | 89.7       | 11.0%         | 212              |
| **EfficientNet-b0** | 0.988    | 0.988     | 0.988  | 0.988    | 1.65                 | 5.3        | 1.4%          | 292              |
| **MobileNet-v3**    | 0.971    | 0.972     | 0.971  | 0.971    | **1.38**             | 5.4        | 7.6%          | 189              |
| **ViT (B-16)**      | 0.980    | 0.980     | 0.980  | 0.980    | **26.08**            | 85.8       | **76.1%**     | 228              |
| **SqueezeNet**      | 0.955    | 0.955     | 0.955  | 0.955    | 1.80                 | **2.8**    | 12.5%         | 211              |
| **Inception-v3**    | 0.934    | 0.934     | 0.934  | 0.934    | 4.33                 | 27.2       | 50.05%        | **432**          |

---

### 🥇 Key Insights

- **🏆 ResNet50** achieves the highest overall accuracy, making it ideal when prediction quality is top priority.
- **⚡ MobileNet-v3** is the fastest and most lightweight model, perfect for edge deployment or mobile apps.
- **🧠 ViT** shows solid accuracy but is extremely resource-hungry, suggesting it’s better for high-performance systems.
- **💸 SqueezeNet** has the fewest parameters and low inference time but sacrifices some performance.
- **❌ Inception-v3** consumes the most VRAM and underperforms in accuracy, making it the least efficient overall.

---

## 🧪 How Benchmarking Was Performed

- Each model was initialized with `torchvision.models`, using pretrained weights.
- Images were passed through a shared transformation pipeline and evaluated using `torch.no_grad()`.
- Metrics collected include:
  - **Classification metrics**: via `sklearn.metrics.classification_report`
  - **Inference time**: using Python's `time` module
  - **Resource usage**: via `psutil` and `torch.cuda.memory_allocated()`

---

## 📁 Included in This Project

- `MLBenchmark.ipynb`: Model benchmarking notebook
- `app.py`: Streamlit web app for multi-image classification
- `requirements.txt`: Python dependency list
- `README.md`: Project instruction

## 📁 Result of This Project

- `model.pth`: Saved model weights for app use
- `benchmark_results.csv`: Result of model benchmarking in CSV format
- `enhanced_benchmark_results.png`: Visualized result of model benchmarking


---

## ▶️ Running the Benchmark and the Web App

Run the program first by

```bash
jupyter notebook MLBenchmark.ipynb
```

After the program finished, run the web app by

```bash
streamlit run app.py