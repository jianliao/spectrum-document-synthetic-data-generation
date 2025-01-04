from datasets import concatenate_datasets, load_dataset
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer, SentenceTransformer
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.losses import MultipleNegativesRankingLoss

def main():
    # Load dataset from HuggingFace
    dataset = load_dataset("JianLiao/spectrum-design-docs", split="train")

    # Add an id column to the dataset
    dataset = dataset.add_column("id", range(len(dataset)))

    # Split dataset into a 10% test set
    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset  = train_test_split["test"]

    # Combine test and train set into one
    corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

    # Convert the datasets to dictionaries
    corpus = dict(zip(corpus_dataset["id"], corpus_dataset["positive"])) # c_id => positive

    queries = dict(zip(test_dataset["id"], test_dataset["anchor"])) # q_id => anchor

    # Create a mapping of relevant document (1 in our case) for each query
    relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
    for q_id in queries:
        relevant_docs[q_id] = [q_id]

    # InformationRetrievalEvaluator is a evaluator designed for measuring RAG performance
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="sds",
        truncate_dim=1024,
        score_functions={"cosine": cos_sim},
    )

    # HuggingFace base model ID: https://huggingface.co/BAAI/bge-large-en-v1.5
    model_id = "BAAI/bge-large-en-v1.5"

    model = SentenceTransformer(model_id)

    train_loss = MultipleNegativesRankingLoss(model)

    # define training arguments
    train_args = SentenceTransformerTrainingArguments(
        output_dir="bge-base-sds-ft",               # output directory and huggingface model ID
        num_train_epochs=30,                        # number of epochs
        per_device_train_batch_size=22,             # train batch size
        gradient_accumulation_steps=16,             # the number of forward and backward passes (steps) performed before updating the model's weights during training
        per_device_eval_batch_size=16,              # evaluation batch size
        warmup_ratio=0.1,                           # warmup ratio
        learning_rate=2e-5,                         # learning rate, 2e-5 is a good value
        lr_scheduler_type="cosine",                 # use constant learning rate scheduler
        optim="adamw_torch_fused",                  # use fused adamw optimizer
        tf32=True,                                  # use tf32 precision
        bf16=True,                                  # use bf16 precision
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        eval_strategy="epoch",                      # evaluate after each epoch
        save_strategy="epoch",                      # save after each epoch
        logging_steps=10,                           # log every 10 steps
        save_total_limit=3,                         # save only the last 3 models
        load_best_model_at_end=True,                # load the best model when training ends
        metric_for_best_model="eval_sds_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score for the 1024 dimension
        dataloader_drop_last=True,                  # When using DistributedDataParallel (DDP), it is recommended to set `dataloader_drop_last=True` to avoid hanging issues with an uneven last batch.       
        prompts= {
            "anchor": "Represent this sentence for searching relevant passages: " # The key is "anchor", because this prompt/instruction should only apply to the anchor for retrieval.
        }
    )

    trainer = SentenceTransformerTrainer(
        model=model, # base model
        args=train_args,  # training arguments
        train_dataset=train_dataset.select_columns(
            ["anchor", "positive"]
        ),  # training dataset
        loss=train_loss,
        evaluator=evaluator,
    )

    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()

    # save the best model
    trainer.save_model()
    

if __name__ == "__main__":
    main()