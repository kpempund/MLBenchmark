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

## ▶️ Running the Benchmark

```bash
jupyter notebook MLBenchmark.ipynb
```

Once the notebook is running, you will be prompted to **select which models you want to benchmark** by entering numbers corresponding to each model:

```
1. EfficientNet  
2. ResNet50  
3. ViT  
4. SqueezeNet  
5. Inception-v3  
6. MobileNet-v3  
0. Exit  
```

Enter one model number at a time, pressing **Enter** after each selection.  
When you're done selecting, input **0** to start the benchmarking process.

The notebook will then:
- Run all selected models
- Measure performance metrics
- Display comparison graphs and resource usage summaries

---

## 💾 Exporting a Trained Model

After benchmarking, you can select **one of the evaluated models to export** for later use in the Streamlit app.

At the end of the notebook, you will be prompted to:

1. Choose a model from the list of benchmarked models by entering its number.
2. The selected model will be saved as `model.pth`, containing:
   - `model_type`: the architecture name (e.g., 'resnet50')
   - `num_classes`: number of output classes (e.g., 2)
   - `model_state_dict`: the trained model weights

This file can then be directly used by `app.py` for image classification in the web interface.

---

## 🖼️ Web App

After finished running the program, opne terminal and run:

```bash
streamlit run app.py
```

User can upload multiple images and get predictions and the confidence scores