## Generate unit test using open-source LLMs

### Set up
```bash
cd code-llm-evaluator
pip install .

cd ..
pip install -r requirements.txt
```

### Generate
```bash
python3 generate.py \
--model #Specifies the path or name of the LLM to use
--instruct_model #Flag to indicate that the model is instruction-tuned
--arg_context #Flag to use argument context in the input prompt
--file_context #Flag to use file context in the input prompt
--split #Indicates the dataset split (rust, go, julia)
--lang #The language name (e.g., Rust, Go, Julia)
--task_name #The name of the task
--max_tokens #Maximum number of tokens to generate per sample.
--batch_size #Number of examples to generate in parallel.
--save_dir #Output directory to save generated outputs.
--num_return_sequences #Number of code sequences to return per prompt.
--repetition_penalty #A penalty to discourage the model from repeating tokens.
--top_p #Top-p (nucleus) sampling threshold. Here, 1 means no filtering.
--top_k #Top-k sampling cutoff. -1 disables top-k.
--temperature 0	#Controls randomness in sampling. 0 means greedy decoding.
```

An example of how to run the script with multiple models and languages is provided in the `run.sh` script.

### Output file
The output is formatted as JSON with the following structure:
```json
{
    "prompt": "The input prompt for the LLM",
    "response": "The generated output, which includes the unit test"
}
```