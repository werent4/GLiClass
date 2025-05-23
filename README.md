# ⭐ GLiClass: Generalist and Lightweight Model for Sequence Classification

**GLiClass** is an efficient, zero-shot sequence classification model inspired by the [GLiNER](https://github.com/urchade/GLiNER/tree/main) framework. It achieves comparable performance to traditional cross-encoder models while being significantly more computationally efficient, offering classification results approximately **10 times faster** by performing classification in a single forward pass.

<p align="center">
    <a href="https://medium.com/@knowledgrator/pushing-zero-shot-classification-to-the-limit-696a2403032f">📄 Blog</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://discord.gg/dkyeAgs9DG">📢 Discord</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/spaces/knowledgator/GLiClass_SandBox">📺 Demo</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/models?sort=trending&search=gliclass">🤗 Available models</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://colab.research.google.com/github/Knowledgator/GLiClass/blob/main/finetuning.ipynb">
        <img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />
    </a>
</p>

### 🚀 Quick Start

Install GLiClass easily using pip:

```bash
pip install gliclass
```

#### Install from Source

Clone and install directly from GitHub:

```bash
git clone https://github.com/Knowledgator/GLiClass
cd GLiClass

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install .
```

Verify your installation:

```python
import gliclass
print(gliclass.__version__)
```

### 🧑‍💻 Usage Example

```python
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

model = GLiClassModel.from_pretrained("knowledgator/gliclass-small-v1.0")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-small-v1.0")

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type='multi-label', device='cuda:0'
)

text = "One day I will see the world!"
labels = ["travel", "dreams", "sport", "science", "politics"]
results = pipeline(text, labels, threshold=0.5)[0]

for result in results:
    print(f"{result['label']} => {result['score']:.3f}")
```

### 🌟 Retrieval-Augmented Classification (RAC)

With new models trained with retrieval-agumented classification, such as [this model](https://huggingface.co/knowledgator/gliclass-base-v2.0-rac-init) you can specify examples to improve classification accuracy:

```python
example = {
    "text": "A new machine learning platform automates complex data workflows but faces integration issues.",
    "all_labels": ["AI", "automation", "data_analysis", "usability", "integration"],
    "true_labels": ["AI", "integration", "automation"]
}

text = "The new AI-powered tool streamlines data analysis but has limited integration capabilities."
labels = ["AI", "automation", "data_analysis", "usability", "integration"]

results = pipeline(text, labels, threshold=0.1, rac_examples=[example])[0]

for predict in results:
    print(f"{predict['label']} => {predict['score']:.3f}")
```

### 🔊 Audio Classification
```python
glicalss_config = GLiClassModelConfig.from_pretrained("models_sampled/checkpoint-24/config.json")
model = GLiClassModel(glicalss_config)
tokenizer = AutoTokenizer.from_pretrained("models_sampled/checkpoint-24")
audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    glicalss_config.audio_model_config._name_or_path,
)
audio_path = "audio.wav"
labels = ["happy", "sad", "angry"]
pipline = ZeroShotClassificationPipeline(
    model, tokenizer,audio_tokenizer=audio_feature_extractor,
)
results = pipline(audio_path, labels, threshold=0)[0]
for predict in results:
    print(f"{predict['label']} => {predict['score']:.3f}")
```

### 🎯 Key Use Cases

- **Sentiment Analysis:** Rapidly classify texts as positive, negative, or neutral.
- **Document Classification:** Efficiently organize and categorize large document collections.
- **Search Results Re-ranking:** Improve relevance and precision by reranking search outputs.
- **News Categorization:** Automatically tag and organize news articles into predefined categories.
- **Fact Checking:** Quickly validate and categorize statements based on factual accuracy.

### 🛠️ How to Train GLiClass

Prepare your training data as follows:

```json
[
  {"text": "Sample text.", "all_labels": ["sports", "science", "business"], "true_labels": ["sports"]},
  ...
]
```

Optionally, specify confidence scores explicitly:

```json
[
  {"text": "Sample text.", "all_labels": ["sports", "science"], "true_labels": {"sports": 0.9}},
  ...
]
```

Please, refer to the `train.py` script to set up your training from scratch or fine-tune existing models.

### 🛠️ How to Train GLiClassAudio
```json
[
  {"audio_features_path": "path/to/already/processed/file", "all_labels": ["sports", "science", "business"], "true_labels": ["sports"]},
  ...
]
```

Please, refer to the `audiodataset_gliclass.py` and `train_gliclass_audio.py` scripts to set up your training from scratch or fine-tune existing models.
