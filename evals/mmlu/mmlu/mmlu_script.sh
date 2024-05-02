model=$1
exp=$2
results="results/"
tokenizer_path=""
num_experts=1
num_select=1

case ${model} in
    "gpt")
        echo "Model is GPT"
        tokenizer_path="tiktoken"
        case ${exp} in
            "std")
                model_path="model_weights"/${model}/${exp}/"out-wikitexts-gpt-wikitexts_gpt_exp_num_1_topk_exp_1_num_iter_2373_lr_0.0006_wd_0.1_seed_2_tm_1713627144.8572702_0/ckpt.pt"
                num_experts=1
                save_dir=$results"gpt/std"
                ;;
            "exp_4_top_2")
                model_path="model_weights"/${model}/${exp}/"out-wikitexts-gpt-wikitexts_exp_num_4_topk_exp_2_num_iter_5616_lr_0.0006_wd_0.1_seed_2_ts_1713626623.2495303_0/ckpt.pt"
                num_experts=4
                save_dir=$results"gpt/exp_4_top_2"
                ;;
            "exp_6_top_3")
                model_path="model_weights"/${model}/${exp}/"todo"
                num_experts=6
                save_dir=$results"gpt/exp_6_top_3"
                ;;
            *)
                echo "No compatible expert setup"
                ;;
        esac
        ;;
        
    "byt5")
        echo "Model is BYT5"
        tokenizer_path="google/byt5-base"
        case ${exp} in
            "std")
                model_path="model_weights"/${model}/${exp}/"out-wikitexts-gpt-wikitexts_exp_num_1_topk_exp_1_lr_0.0006_wd_0.1_num_iter_1600_seed_2_ts_1712870865.8761034_0/ckpt.pt"
                num_experts=1
                save_dir=$results"byt5/std"
                ;;
            "exp_4_top_2")
                model_path="model_weights"/${model}/${exp}/"out-wikitexts-gpt-wikitexts_exp_num_4_topk_exp_2_lr_0.0006_wd_0.1_num_iter_4885_seed_2_tm_1713456988.1967895_0/ckpt.pt"
                num_experts=4
                num_select=2
                save_dir=$results"byt5/exp_4_top_2"
                ;;
            "exp_6_top_3")
                model_path="model_weights"/${model}/${exp}/""
                num_experts=6
                save_dir=$results"byt5/exp_6_top_3"
                ;;
            *)
                echo "No compatible expert setup"
                ;;
        esac
        ;;
    *)
        echo "Model is unknown"
        ;;
esac

echo "Model path: ${model_path}"
echo "Tokenizer path: $tokenizer_path"
echo "Number of experts: $num_experts"
echo "save_dir: $save_dir"

script="evals/mmlu/mmlu/mmlu_eval.py"
data_dir="evals/mmlu/mmlu/data"

test -f $model_path && echo 'Model exists' || echo 'Model does not exist'

# Run the evaluation script
python $script --model_path $model_path --tokenizer_path $tokenizer_path --num_experts $num_experts --num_selects $num_select --save_dir $save_dir --data_dir $data_dir

