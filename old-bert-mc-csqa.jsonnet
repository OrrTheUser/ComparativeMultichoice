local train_size = 9741;
local batch_size = 6;
local grad_accumulate = 1;
local num_epochs = 3;
local lr = 0.00001;
local warmup = 0.1;
local bert_model = "bert-base-uncased";

{
  "dataset_reader": {
    "type": "bert_mc_qa",
    "sample": -1,
    "random_seed" : "1",
    "pretrained_model": bert_model,
    "max_pieces": 128,
    "token_indexers": {
        "bert": {
              "type": "bert-pretrained",
              "pretrained_model": "bert-base-uncased",
              "do_lowercase": true,
              "use_starting_offsets": true
          }
    }
  },
  "train_data_path": "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl",
  "validation_data_path": "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl",
  "evaluate_on_test": false,
  "model": {
    "type": "bert_mc_qa",
    "pretrained_model": bert_model
  },
  "iterator": {
    "type": "basic",
    "batch_size": batch_size
  },
  "trainer": {
    "optimizer": {
        "type": "bert_adam",
        "lr": lr,
        "warmup": warmup,
        "t_total": train_size / batch_size * num_epochs
    },
    "validation_metric": "+EM",
    "num_serialized_models_to_keep": 1,
    "should_log_learning_rate": true,
    //"gradient_accumulation_steps": grad_accumulate,
    "num_epochs": num_epochs,
    "cuda_device": -1
  }
}
