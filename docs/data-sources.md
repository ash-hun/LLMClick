# Training data sources

Jeff's weights were trained on the sources below. **We release the model weights and code, not the training data.**
Each source keeps its own licence; some are share-alike (CC BY-SA). Licences are shown where our code records them;
for the others, see the source. Every training question is checked against the evaluation sets (the benchmark panel
and JevBench), and near-duplicates are removed.

## Public datasets converted into decisions

| Dataset | Source | Licence |
|---|---|---|
| boolq | [google/boolq](https://huggingface.co/datasets/google/boolq) | see source |
| squad2 | [rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2) | see source |
| paws | [google-research-datasets/paws](https://huggingface.co/datasets/google-research-datasets/paws) | see source |
| civil_comments | [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | see source |
| aegis2 | [nvidia/Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | see source |
| pubmedqa | [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) | see source |
| vitaminc | https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip | see source |
| multinli | https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip | see source |
| massive | https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz | see source |
| snli | [stanfordnlp/snli](https://huggingface.co/datasets/stanfordnlp/snli) | CC BY-SA-4.0 |
| commonsense_qa | [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa) | MIT |
| openbookqa | [allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa) | not stated by the source |
| arc_challenge | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | CC BY-SA-4.0 |
| arc_easy | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | CC BY-SA-4.0 |
| social_iqa | [allenai/social_i_qa](https://huggingface.co/datasets/allenai/social_i_qa) | CC BY-4.0 |
| cosmos_qa | [allenai/cosmos_qa](https://huggingface.co/datasets/allenai/cosmos_qa) | CC BY-4.0 |
| quartz | [allenai/quartz](https://huggingface.co/datasets/allenai/quartz) | CC BY-4.0 |
| qasc | [allenai/qasc](https://huggingface.co/datasets/allenai/qasc) | CC BY-4.0 |
| truthful_qa | [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa) | APACHE-2.0 |
| twitter_financial | [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | MIT |
| liar2 | [chengxuphd/liar2](https://huggingface.co/datasets/chengxuphd/liar2) | APACHE-2.0 |
| halueval_qa | [pminervini/HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) | APACHE-2.0 |
| halueval_dialogue | [pminervini/HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) | APACHE-2.0 |
| halueval_summarization | [pminervini/HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) | APACHE-2.0 |
| wikibio | [potsawee/wiki_bio_gpt3_hallucination](https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination) | CC BY-SA-3.0 |
| ragtruth_train | [wandb/RAGTruth-processed](https://huggingface.co/datasets/wandb/RAGTruth-processed) | MIT |
| winogrande_train | [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande) | CC BY |

## Long documents

Long real documents with human labels, as decision rows: ContractNLI (non-disclosure agreements, CC BY 4.0), ConditionalQA (UK government guidance pages; release under BSD-2, pages under the Open Government Licence) and CUAD (commercial contracts annotated by lawyers for clause types, CC BY 4.0).

## Voice commands

Built in code from Amazon MASSIVE (English voice-assistant commands) and the CMU Pronouncing Dictionary, by swapping
words for others that sound the same, to imitate speech-recognition errors.

## Built in code

- 10,000 probability questions (dice, cards, raffles and so on) with exact probabilities as targets.
- Object-tracking and date-arithmetic questions, generated with exact answers.

## Synthetic questions

About 31,000 questions written and checked by an open-weight teacher model (Qwen3.8-Flash-Next) served locally. No
closed-model output is in the training data; a closed model was used only to spot-check the quality of a sample.
