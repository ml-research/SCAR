poetry run python -m lm_eval \
    --model_args "pretrained=RuHae/SCAR,dtype=float16,trust_remote_code=True,mod_features=0,mod_scaling=0.0" \
    --limit 1000 \
    --tasks "mmlu,hellaswag,openbookqa,arc_easy,boolq,lambada_openai,triviaqa,winogrande" \
    --batch_size auto \
    --output_path "./$1-a0_0.json" \
    --trust_remote_code

poetry run python -m lm_eval \
    --model_args "pretrained=RuHae/SCAR,dtype=float16,trust_remote_code=True,mod_features=0,mod_scaling=1.0" \
    --limit 1000 \
    --tasks "mmlu,hellaswag,openbookqa,arc_easy,boolq,lambada_openai,triviaqa,winogrande" \
    --batch_size auto \
    --output_path "./$1-a1_0.json" \
    --trust_remote_code

poetry run python -m lm_eval \
    --model_args "pretrained=RuHae/SCAR,dtype=float16,trust_remote_code=True,mod_features=0,mod_scaling=-1.0" \
    --limit 1000 \
    --tasks "mmlu,hellaswag,openbookqa,arc_easy,boolq,lambada_openai,triviaqa,winogrande" \
    --batch_size auto \
    --output_path "./$1-a-1_0.json" \
    --trust_remote_code

poetry run python -m lm_eval \
    --model_args "pretrained=RuHae/SCAR,dtype=float16,trust_remote_code=True,mod_features=0,mod_scaling=100.0" \
    --limit 1000 \
    --tasks "mmlu,hellaswag,openbookqa,arc_easy,boolq,lambada_openai,triviaqa,winogrande" \
    --batch_size auto \
    --output_path "./$1-a100_0.json" \
    --trust_remote_code

poetry run python -m lm_eval \
    --model_args "pretrained=RuHae/SCAR,dtype=float16,trust_remote_code=True,mod_features=0,mod_scaling=-100.0" \
    --limit 1000 \
    --tasks "mmlu,hellaswag,openbookqa,arc_easy,boolq,lambada_openai,triviaqa,winogrande" \
    --batch_size auto \
    --output_path "./$1-a-100_0.json" \
    --trust_remote_code

poetry run python -m lm_eval \
    --model_args "pretrained=RuHae/SCAR,dtype=float16,trust_remote_code=True,mod_features=0,mod_scaling=50.0" \
    --limit 1000 \
    --tasks "mmlu,hellaswag,openbookqa,arc_easy,boolq,lambada_openai,triviaqa,winogrande" \
    --batch_size auto \
    --output_path "./$1-a50_0.json" \
    --trust_remote_code

poetry run python -m lm_eval \
    --model_args "pretrained=RuHae/SCAR,dtype=float16,trust_remote_code=True,mod_features=0,mod_scaling=-50.0" \
    --limit 1000 \
    --tasks "mmlu,hellaswag,openbookqa,arc_easy,boolq,lambada_openai,triviaqa,winogrande" \
    --batch_size auto \
    --output_path "./$1-a-50_0.json" \
    --trust_remote_code