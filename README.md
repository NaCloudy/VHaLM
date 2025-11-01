# VHaLM: From Visual Moments to Haiku

# Visual-Haiku Language Model for Multi-Image Poetic Generation

> Project for UW-Madison 2025 Fall STAT 453, group 3.

### File dir structure (up-to-date)

```
./
  data/
    test_data/ 			# Test data for midway code
    	model1/			# For stage 1 model(finetuning)
            annotations/
                train.jsonl
            images/
                test/   # e.g. 0000006_c.jpg
                train/
                val/
  models/				# Paras of trained models
  src/					# Source code
    model1/				# For stage 1 model(finetuning)
        outputs/
            checkpoints/
            logs/
            samples/
        dataset.py
        eval.py
        model.py
        train.py
  results/				# For model eval results (data & img)
```

### Prompt

> output 1 description for 3 images (model 1 test data)

```
Below are three image about similar scenes. Write ONE concise(< 50 tokens), objective sentence that summarizes what all three images depict together. Avoid repetition and avoid listing each image separately.
```
