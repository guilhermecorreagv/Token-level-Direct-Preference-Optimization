# Granular Annotation Experiment

This repository is a fork from [Token-level Direct Preference Optimization](https://github.com/Vance0124/Token-level-Direct-Preference-Optimization). Please also check their paper [here](https://arxiv.org/pdf/2404.11999.pdf).

We extend their work by adding training using token-level masks. In order to run our pipeline, you'll need the following:

1. A dataset contained in a JSON as a list with the following fields:
   1. **prompt**: Text containing the prompt to be given to the assistant.
   2. **correct_response**: Text containing the preferred response to the given prompt.
   3. **incorrect_response**: Text containing the unpreferred response to the given prompt.
   4. **masked_region**: A tuple containing (start_index, end_index, response_flag). The response flag should be 1 if the highlighted region is in the correct response and -1 otherwise.

Similarly to TDPO our pipeline has two stages:

1. Run supervised fine-tuning (SFT) on the dataset(s) of interest. Generally, $(x, y_w)$ from the preference dataset is directly used as the supervised fine-tuning target. You can also set the granular option in SFT trainign although we did not notice any significant benefits from it in the SFT phase.

2. Run preference learning on the model from step 1, using preference data (ideally from the same distribution as the SFT examples). The dataset is generally composed of $\mathcal{D} = \{(x, y_w, y_l)_i\}_{i=1}^N$, where $x$ represents the prompt, $y_w$ and $y_l$ denote the preferred and dispreferred completion.

The files in this repo are:

- `train.py`: the main entry point for training (either SFT or TDPO preference-based training)
- `trainers.py`: the trainer classes (e.g., implementing the loop of learning as well as multi-GPU logic)
- `utils.py`: some convenience functions used by multiple other files
- `preference_datasets.py`: dataset processing logic for both SFT and TDPO preference-based training; **this is where you'll need to make some additions to train on your own data**

The code here supports any causal HuggingFace model- look at our examples in `config/model` to add your own. Adding your own datasets is also easy. See [the README section](https://github.com/huggingface/peft) on adding datasets.

## Example

Let's work through a complete example training in our created dataset.

### Step 1: Set up environment

    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt

### Step 2: Run SFT

    python -u train.py model=llama8b-fine-tuned datasets=[ours] loss=sft granular=false custom_train_path=/content/granular_annotation_dataset.json custom_eval_path=/content/granular_annotation_evaluation_dataset.json  exp_name=experiment_name gradient_accumulation_steps=2 eval_every=2000 batch_size=16 eval_batch_size=8 n_epochs=10 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16

### Step 3: Run DPO/TDPO

```
python -u train.py model=llama8b-fine-tuned datasets=[ours] loss=dpo/tdpo loss.alpha=0.5 loss.beta=0.1 exp_name=experiment_name gradient_accumulation_steps=2 batch_size=4 eval_batch_size=2 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=1000 model.archive=/path/to/trained/model/in/step1/policy.pt
```

Keep in mind that when using granular annotations the learning rate (lr) should be increased. In our experiments we increased by a factor of 100x. For more experimental details and information, please refer to our paper.
