# EACS
Title: An Extractive-and-Abstractive Framework for Source Code Summarization

## Requirements
The dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```

## Original Dataset
The CodeSearchNet original dataset can be downloaded from the github repo: [https://github.com/github/CodeSearchNet](https://github.com/github/CodeSearchNet), and the cleaned dataset (CodeXGLUE) can be downloaded from the [https://drive.google.com/open?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h](https://drive.google.com/open?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h)

The JCSD and PCSD dataset can be downloaded from the github repo: [https://github.com/xing-hu/TL-CodeSum](https://github.com/xing-hu/TL-CodeSum) and [https://github.com/wanyao1992/code_summarization_public](https://github.com/wanyao1992/code_summarization_public)

## Quick Start
### Extractor
Run the `Extractor_ classifier/make_label/make_dataset_label.py` to generate the classification labels for classifier training.

For example:
```bash
cd Extractor_ classifier/make_label/
```
```bash
python make_dataset_label.py
```

And run the `Extractor_ classifier/train.py` to train the classifier model.
For example:
```bash
cd Extractor_ classifier/
```
```bash
python train.py {language}
```

The classifier model will be saved in the `Extractor_ classifier/model/`, and run the `Extractor_ classifier/classifier.py` to generate the important sentences predicted value.
```bash
cd Extractor_ classifier/
```
```bash
python classifier.py {language}
```

The {language} can be selected in `java, python, go, php, ruby, javascript, JCSD, PCSD`

the output samples are as follows:
```
- output samples:
{
    "idx":0, 
    "code_tokens":["private", "int", "currentDepth", ...], 
    "docstring_tokens": ["returns", "a", "0", ...],  
    ...,  
    "ex_labels": [1, 0, 1, ...], 
    "cleaned_seqs_pred": [1, 0, 1, ...] 
}
...
```

### EACS + CodeBert
To train the EACS CodeBert model:
```bash
cd EACS_codeBert/
```
```bash
python train.py {language}
```

To test and output the EACS CodeBert results:
```bash
python test.py {language}
```

The {language} can be selected in `java, python, go, php, ruby, javascript, JCSD, PCSD`

### EACS + CodeT5

To train the EACS CodeBert model and it also will outputs the results: 
```bash
cd EACS_codeT5/
```
```bash
python run_gen.py {language}
```

The {language} can be selected in `java, python, go, php, ruby, javascript, JCSD, PCSD`

### Evaluation
After trainning the EACS + CodeBert and EACS + CodeT5 models, run the evaluation code to output Bleu, Meteor and Rouge-L:

(*Switch into python 2.7*)
```bash
cd Evaluation/
```
```bash
python evaluate.py
```
