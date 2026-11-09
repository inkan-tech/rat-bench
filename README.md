# RAT-Bench 
<p align="center">
<img src="Rat_Bench.png" width="350" align="center">
</p>

Benchmark for evaluating PII removal and text anonymization tools for the paper [RAT-Bench](https://arxiv.org/pdf/2602.12806v1), with a focus on **re-identification risk**. 

Our [benchmark dataset](https://huggingface.co/datasets/imperial-cpg/rat-bench) is openly accessible. We also maintain an [extendable leaderboard](https://huggingface.co/spaces/imperial-cpg/rat-bench) of text anonymization tools.

# ⚙️ Installation

This project uses the [`uv`](https://github.com/astral-sh/uv) package manager.

### 1. Set up the virtual environment

Run the following inside the project root:

```bash
uv sync
```

This will create a virtual environment inside a `.venv` folder. Run all next commands within this virtual environment.

Make sure to then activate the environment, by running `source .venv/bin/activate`.

To install the [correctmatch](https://github.com/imperial-aisp/pycorrectmatch/tree/master) package from Rocher et al. that is used to compute re-identification risk, run the following:

```bash
python install_correctmatch.py
```

You'll need to install the `spacy` transformers by running

```
uv pip install pip
python -m spacy download en_core_web_sm
```

If you would need anonymizers also for other languages, repeat the same process to download `es_core_news_lg`(Spanish) and `zh_core_web_lg`(Chinese).

### Apple Silicon (MLX)

vLLM needs CUDA, so `uv sync` fails on macOS. On Apple Silicon, the Llama anonymizers (`llama_*`) run on [MLX](https://github.com/ml-explore/mlx-lm) instead. Install every dependency except vLLM and the CUDA packages, then add `mlx-lm`:

```bash
uv export --frozen --no-hashes --no-emit-project \
  | grep -vE "^(vllm|nvidia-cudnn-frontend|nvidia-cutlass-dsl|flashinfer-python|cupy-cuda12x|cuda-[a-z-]+)==" > requirements-mac.txt
uv venv && uv pip install -r requirements-mac.txt mlx-lm
```

The backend is chosen automatically (`mlx` on Apple Silicon, `vllm` elsewhere); force it with `RATBENCH_LLM_BACKEND=mlx|vllm`. The default checkpoint is `mlx-community/Meta-Llama-{llama_version}-bf16` (bf16, as vLLM uses on recent GPUs, ~16 GB). Set `RATBENCH_MLX_MODEL` to use another one, e.g. a `-4bit` variant. MLX runs prompts one after the other, so it is slower than vLLM batching.

Copy ``pii_benchmark/credentials_example.py`` → ``pii_benchmark/credentials.py``.

Add your API keys inside ``credentials.py``.

(Note: ``credentials.py`` is already in ``.gitignore`` and won’t be committed.)

To run our pipeline for the Textwash anonymizer, download their models from [here](https://drive.google.com/file/d/1YBccngYE3lvod87TI6UIhBzrN7nY9vHS/view). Create a directory named ``data_textwash`` and place all the files in there (make sure the files are directly in `data_textwash`: the path to the models should be `data_textwash/en`, not `data_textwash/models/en`).

# ⚙️ Data

We have included the processed datasets needed to generate the benchmark and run our experiments in this repo:

`data/300_profiles.csv` contains the profiles, including direct and indirect identifiers, used to generate benchmark entries.
`data/population_sample.csv` contains a 100-row example of [US census data](https://www.census.gov/programs-surveys/acs/microdata/access/2010.html) which is used to compute re-identification rate. The full dataset needed for computation, preprocessed to account for weighting of each row, can be found [here](https://drive.google.com/drive/folders/1k4ZP1E4r4YCQ9ihuelOi_D-C6onBF_jp). **Please download this dataset and place it in ``data/pums/pums_pwgtp_sample_.csv`` in order to get accurate risk scores.**

# 🚀 Adding and Evaluating a New Anonymizer

All anonymizers are built on the abstract **`Anonymizer`** class.  

## 🛠️ Step 1: Add Your Anonymizer

1. Create a new file in **`pii_benchmark/anonymizers/`** (e.g., `my_anonymizer.py`).  
2. Inside, define a class that:  
   - Inherits from **`Anonymizer`**  
   - Implements the **`anonymize`** method.  
3. Open **`pii_benchmark/anonymizers/get_anonymizers.py`** and register your anonymizer by adding it to the `cases` dictionary.  

✅ Example structure:  

```python
from pii_benchmark.anonymizers.base import Anonymizer

class MyAnonymizer(Anonymizer):
    def anonymize(self, text: str) -> str:
        # Your anonymization logic here
        return text
```

Open `get_anonymizers.py` and register your anonymizer by adding it to the cases dictionary.

## ✨ Step 2: Run Anonymization

To download the benchmark tests from hugging face and anonymize with your method, run:

```bash
sh scripts/anonymize/medical/all_levels_hf.sh
sh scripts/anonymize/chatbot/all_levels_hf.sh
```

To anonymize with your method on a locally stored version of the benchmark tests, run:

```bash
sh scripts/anonymize/medical/all_levels.sh
sh scripts/anonymize/chatbot/all_levels.sh
```
⚠️ Important: Update the ``anon_methods`` parameter in each script to match your anonymizer’s name.

⏱️ Note: If you want to see the runtime performance of your anonymizer per profile, set the ``timing_flag`` parameter in each script to 1.

## 🔍 Step 3: Evaluate Re-identification Risk
After anonymization, compute the re-identification risk with:
```bash
sh scripts/attack/medical/all_levels.sh
sh scripts/attack/chatbot/all_levels.sh
```


## 📊 (Optional) Step 4: Compute utility scores of anonymized data
Compute the utility scores of each anonymizer with:
```bash
sh scripts/utility/medical/all_levels.sh
sh scripts/utility/chatbot/all_levels.sh
```

## Generating the data from scratch

We have included 100 benchmark entries per level in the `benchmark` folder in this repo. To run a new generation, run:
```bash
sh scripts/generate/level_{LEVEL}.sh
```
for the desired level of difficulty. You can also instead obtain our generated benchmark dataset [here](https://huggingface.co/datasets/imperial-cpg/text_anonymization_benchmark).

Anonymization scripts are included in `scripts/anonymize`, and attacker scripts are included in `scripts/attack`.

A full run of the pipeline for one difficulty level, from generation to re-identification rate computation, can be achieved by running the following:

1. `sh scripts/generate/level_1.sh`
2. `sh scripts/anonymize/level_1.sh`
3. `sh scripts/attack/level_1.sh`

# References

If you found our work useful for your research, kindly cite our paper: 

```
@article{krvco2026rat,
  title={RAT-Bench: A Comprehensive Benchmark for Text Anonymization},
  author={Kr{\v{c}}o, Nata{\v{s}}a and Yao, Zexi and Meeus, Matthieu and de Montjoye, Yves-Alexandre},
  journal={arXiv preprint arXiv:2602.12806},
  year={2026}
}
```
