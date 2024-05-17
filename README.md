# Path-RAG: Knowledge-Guided Key Region Retrieval for Open-ended Pathology Visual Question Answering

1. Clone this repository and navigate to path-rag folder

```Shell
git clone https://github.com/embedded-robotics/path-rag.git
cd path-rag
```

2. Install Package: Create conda environment

```Shell
conda create -n path-rag python=3.10 -y
conda activate path-rag
pip install --upgrade pip # enable PEP 660 support for LLaVA-Med
```

3. Install the dependent packages for LLaVA-Med and HistoCartography (compatible with both LLaVA-Med and HistoCartography)

```Shell
pip install -r requirements.txt
```

4. Download the PathVQA dataset from the following link

[PathVQA Dataset](https://github.com/UCSD-AI4H/PathVQA/blob/master/data/README.md)

5. Clone the HistoCartography tool and setup the model checkpoints in `histocartography/checkpoints`

```Shell
git clone https://github.com/BiomedSciAI/histocartography
```

6. Clone the LLaVA-Med repository

```Shell
git clone https://github.com/microsoft/LLaVA-Med
```

7. Download the LLaMA-7B model and weights from HuggingFace

```Shell
python llama_7B_model_weights.py # LLaMA-7B weights/model stored into $HF_HOME (By Default $HF_HOME = ~/.cache/huggingface)
```

8. Download LLaVA-Med delta weights `llava_med_in_text_60k_ckpt2_delta` and `pvqa-9epoch_delta` from `https://github.com/microsoft/LLaVA-Med#model-download`. Put them inside a folder named `model_delta_weights`

9. Apply the LLaVA-Med delta weights to base LLaMA-7B to come up with the final weights for LLaVA-Med

```Shell
cd LLaVA-Med
```

#### LLaVA-Med pre-trained on general biomedicine data

```Shell
!python3 -m llava.model.apply_delta \
    --base ~/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16 \
    --target ../final_models/llava_med \
    --delta ../model_delta_weights/llava_med_in_text_60k_ckpt2_delta
```

#### LLaVA-Med fine-tuned on PathVQA

```Shell
!python -m llava.model.apply_delta \
    --base ~/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16 \
    --target ../final_models/llava_med_pvqa \
    --delta ../model_delta_weights/pvqa-9epoch_delta
```

```Shell
cd ..
```

10. Generate the top patches for open-ended PathVQA images using HistoCartography

```Shell
python generate_histo_patches.py
```

11. Generate the files for query to be asked for LLaVA-Med for both the images and patches

```Shell
python generate_llava_med_query.py
```

12. Now we need to generate the answer for all the query files using raw model (`./final_models/llava_med`) and fine-tuned model (`./final_models/llava_med_pvqa`)

```Shell
cd LLaVA-Med
```

#### Raw Model
```Shell
python llava/eval/model_vqa.py --model-name ../final_models/llava_med \
    --question-file ../files/query/image_direct.jsonl \
    --image-folder ../pvqa/images/test \
    --answers-file ../files/answer/raw/answer_image_direct.jsonl
```

```Shell
python llava/eval/model_vqa.py --model-name ../final_models/llava_med \
    --question-file ../files/query/patch_direct.jsonl \
    --image-folder ../pvqa/images/test \
    --answers-file ../files/answer/raw/answer_patch_direct.jsonl
```

```Shell
python llava/eval/model_vqa.py --model-name ../final_models/llava_med \
    --question-file ../files/query/image_description.jsonl \
    --image-folder ../pvqa/images/test \
    --answers-file ../files/answer/raw/answer_image_description.jsonl
```

```Shell
python llava/eval/model_vqa.py --model-name ../final_models/llava_med \
    --question-file ../files/query/patch_description.jsonl \
    --image-folder ../pvqa/images/test \
    --answers-file ../files/answer/raw/answer_patch_description.jsonl
```

#### Fine-Tuned Model
```Shell
python llava/eval/model_vqa.py --model-name ../final_models/llava_med_pvqa \
    --question-file ../files/query/image_direct.jsonl \
    --image-folder ../pvqa/images/test \
    --answers-file ../files/answer/fine-tuned/answer_image_direct.jsonl
```

```Shell
python llava/eval/model_vqa.py --model-name ../final_models/llava_med_pvqa \
    --question-file ../files/query/patch_direct.jsonl \
    --image-folder ../pvqa/images/test \
    --answers-file ../files/answer/fine-tuned/answer_patch_direct.jsonl
```

```Shell
python llava/eval/model_vqa.py --model-name ../final_models/llava_med_pvqa \
    --question-file ../files/query/image_description.jsonl \
    --image-folder ../pvqa/images/test \
    --answers-file ../files/answer/fine-tuned/answer_image_description.jsonl
```

```Shell
python llava/eval/model_vqa.py --model-name ../final_models/llava_med_pvqa \
    --question-file ../files/query/patch_description.jsonl \
    --image-folder ../pvqa/images/test \
    --answers-file ../files/answer/fine-tuned/answer_patch_description.jsonl
```

13. Evaluate the results for different use-cases using `recall_calculation.py`

(i) Path-RAG w/o GPT: Combine the answer of image + all patches to be the final predicted answer
(ii) Path-RAG (description): Combine the description of image + all patches. Then involve GPT-4 for reasoning to ge the final predicted answer (See Supplementary Section for Prompts)
(iii) Path-RAG (answer): Combine the answer of image + all patches. Then involve GPT-4 for reasoning to ge the final predicted answer (See Supplementary Section for Prompts)
