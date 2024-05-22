from datasets import load_dataset, DatasetDict
from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM, 
                          AutoTokenizer, pipeline, get_scheduler,
                          DataCollatorForSeq2Seq, MBartTokenizer, 
                          MBartTokenizerFast, default_data_collator)
import evaluate
import torch
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
import math
import collections
import numpy as np
import random
import json
import os
valkey="test"#"validation"
Dualevaluation=False



#https://huggingface.co/facebook/wmt21-dense-24-wide-en-x
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-en-x")
# tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-en-x")

# inputs = tokenizer("To translate into a target language, the target language id is forced as the first generated token. To force the target language id as the first generated token, pass the forced_bos_token_id parameter to the generate method.", return_tensors="pt")

# # translate English to Chinese
# generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("zh")) #max_new_tokens
# result=tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# print(result)

# model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
# translator = pipeline("translation", model=model_checkpoint)
# print(translator("Default to expanded threads"))

# from transformers import AutoTokenizer

# model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
# from transformers import AutoModelForSeq2SeqLM
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

def modelparameters(model, unfreezename=""):
    if unfreezename:
        for name, param in model.named_parameters():
            if name.startswith(unfreezename): # choose whatever you like here
                param.requires_grad = True
            else:
                param.requires_grad = False
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

def loadmodel(model_checkpoint, task="Seq2SeqLM", mycache_dir="", pretrained="", hpc=True, unfreezename=""):
    if hpc==True:
        localpath=os.path.join(mycache_dir, model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(localpath, local_files_only=True)
        if task=="Seq2SeqLM":
            model = AutoModelForSeq2SeqLM.from_pretrained(localpath, local_files_only=True)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(localpath, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)#, cache_dir=mycache_dir)
        if task=="Seq2SeqLM":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)#"distilroberta-base")
    starting_epoch = 0
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        print("Pretrained epoch:", checkpoint['epoch'])
        starting_epoch = checkpoint['epoch'] +1
        model.load_state_dict(checkpoint['model_state_dict'])
    embedding_size = model.get_input_embeddings().weight.shape[0]
    print("Embeeding size:", embedding_size) #65001
    print("Tokenizer length:", len(tokenizer)) #65001
    # if len(tokenizer) > embedding_size:
    #     model.resize_token_embeddings(len(tokenizer))

    model_num_parameters = model.num_parameters() / 1_000_000
    print(f"'>>> Model number of parameters: {round(model_num_parameters)}M'")
    #print(f"'>>> BERT number of parameters: 110M'")
    modelparameters(model, unfreezename)
    return model, tokenizer, starting_epoch

# max_length = 128
# def preprocess_function(examples):
#     inputs = [ex[source_lang] for ex in examples["translation"]] #1000
#     targets = [ex[target_lang] for ex in examples["translation"]] #1000
#     model_inputs = globaltokenizer(
#         inputs, text_target=targets, max_length=max_length, truncation=True
#     )
#     return model_inputs

def loaddata(args, USE_HPC):
    if args.data_type == "huggingface":
        if USE_HPC:
            if args.data_name=='kde4':
                #raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
                datasetpath=os.path.join(mycache_dir, args.data_name, "en-fr-lang1\=en\,lang2\=fr", "0.0.0", "/243129fb2398d5b0b4f7f6831ab27ad84774b7ce374cf10f60f6e1ff331648ac") #"/data/cmpe249-fa23/Huggingfacecache/imdb/plain_text/1.0.0/d613c88cf8fa3bab83b4ded3713f1f74830d1100e171db75bbddb80b3345c9c0"
                #raw_datasets = load_dataset(args.data_name, cache_dir=mycache_dir) #eli5
                datasetpath=os.path.join(mycache_dir, args.data_name)
                trainarrowpath=os.path.join(mycache_dir, args.data_name, args.data_name+'-train.arrow')
                #valarrowpath=os.path.join(mycache_dir, datasetpath, args.data_name+'-validation.arrow')
                #testarrowpath=os.path.join(mycache_dir, datasetpath, args.data_name+'-test.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
            elif args.data_name=='opus100':
                datasetpath=os.path.join(mycache_dir, args.data_name, 'enzh')
                trainarrowpath=os.path.join(datasetpath, args.data_name+'-train.arrow')
                valarrowpath=os.path.join(datasetpath, args.data_name+'-validation.arrow')
                testarrowpath=os.path.join(datasetpath, args.data_name+'-test.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
                #raw_datasets = load_dataset("opus100", language_pair="en-zh")
            elif args.data_name == 'wmt19':
                datasetpath=os.path.join(mycache_dir, args.data_name, 'zh-en-24b9c423f6ba2174/0.0.0/29e210fae5690e843cae5dc43b53db36c4e02f927db50cd5235a22ab42dde90a')
                trainarrowpath=os.path.join(datasetpath, args.data_name+'-train*.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
            else:#wmt19
                raw_datasets = load_dataset(args.data_name, language_pair=(args.target_lang,args.source_lang))
        else:
            if args.data_name=='kde4':
                raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
            elif args.data_name=='opus100':
                raw_datasets = load_dataset("opus100", language_pair="en-zh")
            elif args.data_name=='opus_books':
                raw_datasets = load_dataset("opus_books", "en-fr")
            else: #wmt19
                #raw_datasets = load_dataset(args.data_name, args.dataconfig) #dataconfig="train_asks[:5000]"
                raw_datasets = load_dataset(args.data_name, language_pair=(args.target_lang,args.source_lang))
        #Download to home/.cache/huggingface/dataset
        
        print("All keys in raw datasets:", raw_datasets['train'][0].keys()) #obly one ['translation'] key
        split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
        # rename the "test" key to "validation" 
        #split_datasets["validation"] = split_datasets.pop("test")
        #one element
        sampletext = split_datasets["train"][1]["translation"]
        raw_datasets = split_datasets
        if args.subset>0:
            if args.subset<1:
                trainlen=int(len(raw_datasets["train"])*args.subset)
                testlen=int(len(raw_datasets[valkey])*args.subset)
            else:
                trainlen = int(min(args.subset, len(raw_datasets["train"])))
                testlen = int(trainlen/10)
            print("trainlen:", trainlen)
            raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select([i for i in list(range(trainlen))])
            raw_datasets[valkey] = raw_datasets[valkey].shuffle(seed=42).select([i for i in list(range(testlen))])
    
        #limit the evaluation set size
        maxtestlen = 5000
        if len(raw_datasets[valkey])>maxtestlen:
            raw_datasets[valkey] = raw_datasets[valkey].shuffle(seed=42).select([i for i in list(range(maxtestlen))])

    return raw_datasets

def get_myoptimizer(model, learning_rate=2e-5, weight_decay=0.0):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

def postprocess(predictions, labels, ignore_pad_token_for_loss=True):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels

import sacrebleu
class myEvaluator:
    def __init__(self, metricname, useHFevaluator=False, language="en", dualevaluator=False):
        self.useHFevaluator = useHFevaluator
        self.dualevaluator = dualevaluator
        self.language = language
        if useHFevaluator==True or dualevaluator==True:
            self.HFmetric = evaluate.load(metricname) #"sacrebleu" pip install sacrebleu
        else:
            self.HFmetric = None
        self.preds = []
        self.refs = []

    
    def compute(self, predictions=None, references=None):
        if predictions is not None and references is not None:
            if self.useHFevaluator==True:
                results = self.HFmetric.compute(predictions=predictions, references=references)
                #keys: ['score', 'counts', 'totals', 'precisions', 'bp', 'sys_len', 'ref_len']
            else:
                bleu = sacrebleu.corpus_bleu(predictions, references)
                results = {'score':bleu.score, 'counts':bleu.counts, 'totals': bleu.totals,
                        'precisions': bleu.precisions, 'bp': bleu.bp, 
                        'sys_len': bleu.sys_len, 'ref_len': bleu.ref_len
                        }
        else: #evaluate the whole dataset
            if self.useHFevaluator==True or self.dualevaluator==True:
                results = self.HFmetric.compute()
                print("HF evaluator:", results["score"])
            else:
                #self.refs should be list of list strings
                #Tokenization method to use for BLEU. If not provided, defaults to `zh` for Chinese, `ja-mecab` for Japanese, `ko-mecab` for Korean and `13a` (mteval) otherwise
                if self.language=="zh":
                    bleu = sacrebleu.corpus_bleu(self.preds, [self.refs], tokenize="zh")
                else:
                    bleu = sacrebleu.corpus_bleu(self.preds, [self.refs], tokenize="none")
                results = {'score':bleu.score, 'counts':bleu.counts, 'totals': bleu.totals,
                        'precisions': bleu.precisions, 'bp': bleu.bp, 
                        'sys_len': bleu.sys_len, 'ref_len': bleu.ref_len
                        }
                print("Local evaluator:", results["score"])
        return results
    
    def add_batch(self, predictions, references):
        if self.useHFevaluator==True or self.dualevaluator==True:
            self.HFmetric.add_batch(predictions=predictions, references=references)
        else:
            #self.preds.append(predictions)
            #self.refs.append(references)
            self.preds.extend(predictions)
            #references: list of list
            for ref in references:
                self.refs.append(ref[0])
            #print(len(self.refs))

def evaluate_dataset(model, tokenizer, eval_dataloader, use_accelerator, accelerator, device, max_target_length, num_beams, metric):
    # Evaluation
    totallen = len(eval_dataloader)
    print("Total evaluation length:", totallen)
    #evalprogress_bar = tqdm(range(num_training_steps))
    model.eval()
    gen_kwargs = {
        "max_length": max_target_length,
        "num_beams": num_beams,
    }
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            if not use_accelerator:
                batch = {k: v.to(device) for k, v in batch.items()}
                generated_tokens = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
            else:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
            #evalprogress_bar.update(1)
        labels = batch["labels"]
        if use_accelerator:
            # Necessary to pad predictions and labels for being gathered
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(generated_tokens)
            labels_gathered = accelerator.gather(labels)

            decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered, ignore_pad_token_for_loss)
        else:
            decoded_preds, decoded_labels = postprocess(generated_tokens, labels, ignore_pad_token_for_loss)
        
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        #evalmetric.add_batch(predictions=decoded_preds, references=decoded_labels)
    
    results = metric.compute()
    #evalresults = evalmetric.compute()
    #print(f"BLEU score: {results['score']:.2f}")
    #print(evalresults['score'])
    return results


import shutil
def savemodels(model, optimizer, epoch, trainoutput):
    modelfilepath=os.path.join(trainoutput, 'savedmodel.pth')
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, modelfilepath)
    modelfilepathwithepoch=os.path.join(trainoutput, 'epoch'+str(epoch)+'_savedmodel.pth')
    shutil.copy(modelfilepath, modelfilepathwithepoch)
    #Huggingface format:
    model.save_pretrained(trainoutput)

#data_name="imdb", dataconfig="", model_checkpoint="distilbert-base-uncased"
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--data_type', type=str, default="huggingface",
                    help='data type name: huggingface, custom')
    parser.add_argument('--data_name', type=str, default="opus_books",
                    help='data name: opus_books, kde4, opus100')
    parser.add_argument('--dataconfig', type=str, default='',
                    help='train_asks[:5000]')
    parser.add_argument('--subset', type=float, default=0,
                    help='0 means all dataset')
    parser.add_argument('--data_path', type=str, default="/data/cmpe249-fa23/Huggingfacecache",
                    help='path to get data ') #r"E:\Dataset\NLPdataset\aclImdb"
    parser.add_argument('--model_checkpoint', type=str, default="t5-base",
                    help='Model checkpoint name from HF, t5-small, t5-base, Helsinki-NLP/opus-mt-en-zh, Helsinki-NLP/opus-mt-en-fr, t5-small, facebook/wmt21-dense-24-wide-en-x')
    parser.add_argument('--task', type=str, default="Seq2SeqLM",
                    help='NLP tasks: Seq2SeqLM')
    parser.add_argument('--evaluate', type=str, default="localevaluate",
                    help='perform evaluation via HFevaluate, localevaluate')
    parser.add_argument("--source_lang", type=str, default="en", help="Source language id for translation.")
    parser.add_argument("--target_lang", type=str, default="fr", help="Target language id for translation.")
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument('--pretrained', type=str, default="",
                    help='Pretrained model path')
    parser.add_argument('--unfreezename', type=str, default="",
                    help='Unfreezename in models')
    parser.add_argument('--outputdir', type=str, default="./output",
                    help='output path')
    parser.add_argument('--traintag', type=str, default="1124",
                    help='Name the current training')
    parser.add_argument('--training', type=bool, default=True,
                    help='Perform training')
    parser.add_argument('--usehpc', type=bool, default=False,
                    help='Use HPC')
    parser.add_argument('--useHFaccelerator', type=bool, default=False,
                    help='Use Huggingface accelerator')
    parser.add_argument('--gpuid', default=0, type=int, help='GPU id')
    parser.add_argument('--total_epochs', default=16, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learningrate', default=2e-5, type=float, help='Learning rate')
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass, ref: https://kozodoi.me/blog/20210219/gradient-accumulation.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=True,
        help=(
            "Whether to pad all samples to model maximum sentence length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        ),
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128, #1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded "
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    args = parser.parse_args()

    global task
    task = args.task
    print(' '.join(f'{k}={v}' for k, v in vars(args).items())) #get the arguments as a dict by calling vars(args)

    use_accelerator = args.useHFaccelerator
    model_checkpoint = args.model_checkpoint
    use_fp16 = True
    
    USE_HPC=args.usehpc
    if USE_HPC:
        #https://huggingface.co/docs/transformers/installation#offline-mode
        #HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        mycache_dir=args.data_path #"/data/cmpe249-fa23/Huggingfacecache"
        os.environ['TRANSFORMERS_CACHE'] = mycache_dir
        os.environ['HF_HOME'] = mycache_dir
        os.environ['HF_DATASETS_CACHE'] = mycache_dir
        os.environ['HF_EVALUATE_OFFLINE'] = "1"
        os.environ['HF_DATASETS_OFFLINE'] = "1"
        os.environ['TRANSFORMERS_OFFLINE'] = "1"
        os.environ['http_proxy'] = "http://172.16.1.2:3128"
        os.environ['HTTP_PROXY'] = "http://172.16.1.2:3128"
        os.environ['https_proxy'] = "https://172.16.1.2:3128"
        os.environ['HTTPS_PROXY'] = "https://172.16.1.2:3128"
        trainoutput="/data/cmpe249-fa23/trainoutput/huggingface"
        taskname=args.traintag #"eli5asksciencemodeling"
    else:
        trainoutput=args.outputdir #"./output"
        taskname=args.traintag #taskname="eli5asksciencemodeling"
        mycache_dir="./data/"
    trainoutput=os.path.join(trainoutput, model_checkpoint, args.data_name+'_'+args.traintag)
    os.makedirs(trainoutput, exist_ok=True)
    print("Trainoutput folder:", trainoutput)

    model, tokenizer, starting_epoch = loadmodel(model_checkpoint, task=task, mycache_dir=mycache_dir, pretrained=args.pretrained, hpc=USE_HPC, unfreezename=args.unfreezename)
    #tokenizer.model_max_len=512
    print(tokenizer.pad_token)
    print(tokenizer.eos_token)
    #tokenizer.pad_token = tokenizer.eos_token
    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if args.source_lang is not None:
            tokenizer.src_lang = args.source_lang
        if args.target_lang is not None:
            tokenizer.tgt_lang = args.target_lang
    # Get the language codes for input/target.
    source_lang = args.source_lang.split("_")[0]
    target_lang = args.target_lang.split("_")[0]
    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        assert (
            args.target_lang is not None and args.source_lang is not None
        ), "mBart requires --target_lang and --source_lang"
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(args.target_lang)
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    raw_datasets = loaddata(args, USE_HPC)
    column_names = raw_datasets["train"].column_names
    print("column_names:", column_names) #['translation']
    padding = "max_length" if args.pad_to_max_length else False
    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    ignore_pad_token_for_loss = True

    def translationpreprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = raw_datasets.map(
            translationpreprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=raw_datasets["train"].column_names,
        )#The default batch size is 1000, but you can adjust it with the batch_size argument
    tokenized_datasets.set_format("torch")
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets[valkey]
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if use_fp16 else None,
        )

    #To test this on a few samples
    batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
    print(batch.keys()) #['input_ids', 'attention_mask', 'labels']
    #batch["labels"] #our labels have been padded to the maximum length of the batch, using -100:
    #batch["decoder_input_ids"] #shifted versions of the labels

    predictions = [
        "This plugin lets you translate web pages between several languages automatically."
    ]
    references = [
        [
            "This plugin allows you to automatically translate web pages between several languages."
        ]
    ]
    if args.evaluate=="HFevaluate":
        #metric = evaluate.load("sacrebleu") #pip install sacrebleu
        #results = metric.compute(predictions=predictions, references=references)
        #print("Test evaluation via HFevaluate:", round(results["score"], 1))
        metric = myEvaluator(metricname="sacrebleu", useHFevaluator=True, language=target_lang, dualevaluator=Dualevaluation)
    else:
        metric = myEvaluator(metricname="sacrebleu", useHFevaluator=False, language=target_lang, dualevaluator=Dualevaluation)
    results = metric.compute(predictions=predictions, references=references)
    print("Test evaluation:", round(results["score"], 1))

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.batch_size
    )

    #optimizer = AdamW(model.parameters(), lr=2e-5)
    optimizer = get_myoptimizer(model, learning_rate=args.learningrate)

    num_train_epochs = args.total_epochs
    #num_update_steps_per_epoch = len(train_dataloader)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    completed_steps = starting_epoch * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    if use_accelerator:
        #accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=2)
        accelerator = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
        device = accelerator.device
        print("Using HF Accelerator and device:", device)
    else:
        accelerator = None
        if torch.cuda.is_available():
            device = torch.device('cuda:'+str(args.gpuid))  # CUDA GPU 0
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model.to(device)
        print("Using device:", device)
    
    

    evaluate_dataset(model, tokenizer, eval_dataloader, use_accelerator, accelerator, device, max_target_length, args.num_beams, metric)

    if args.training == True:
        print("Start training, total steps:", num_training_steps)
        progress_bar = tqdm(range(num_training_steps))
        model.train()
        for epoch in range(starting_epoch, num_train_epochs):
            # Training
            for step, batch in enumerate(train_dataloader):
                #batch = {k: v.to(device) for k, v in batch.items()}
                if not use_accelerator:
                    batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                if not use_accelerator:
                    loss.backward()
                else:
                    accelerator.backward(loss)

                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

            # Evaluation
            results = evaluate_dataset(model, tokenizer, eval_dataloader, use_accelerator, accelerator, device, max_target_length, args.num_beams, metric)
            print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")
            #print(evalresults['score'])
            # Save the results
            with open(os.path.join(trainoutput, "eval_results.json"), "w") as f:
                json.dump({"eval_bleu": results["score"]}, f)

            if use_accelerator:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                #unwrapped_model.save_pretrained(trainoutput, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(trainoutput)
                savemodels(model, optimizer, epoch, trainoutput)
            else:
                #model.save_pretrained(trainoutput)
                #torch.save(model.state_dict(), os.path.join(trainoutput, 'savedmodel.pth'))
                savemodels(model, optimizer, epoch, trainoutput)
                tokenizer.save_pretrained(trainoutput)

    del model, optimizer, lr_scheduler
    if use_accelerator:
        accelerator.free_memory()

