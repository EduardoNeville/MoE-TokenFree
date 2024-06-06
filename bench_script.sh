model=$1
exp=$2
eval_name="$3"
echo "$eval_name"
if [[ $eval_name = "mmlu" ]]; then
    results="results/mmlu/"
    data_dir="evals/mmlu/mmlu/data"
elif  [[ $eval_name = "arc_easy" ]]; then
    results="results/arc_easy/"
    data_dir="evals/arc_easy/data"
elif  [[ $eval_name = "coqa" ]]; then
    results="results/coqa/"
    data_dir="evals/coqa/data"
else
    echo "No evaluation provided \n"
fi
tokenizer_path=""
num_experts=1
num_select=1

case ${model} in
    "gpt")
        echo "Model is GPT"
        tokenizer_path="tiktoken"
        case ${exp} in
            "std")
                model_path="openweb-tiktoken-exp1-top1/ckpt.pt"
                num_experts=1
                num_select=1
                save_dir=$results"gpt/std"
                ;;
            "exp_4_top_2")
                model_path="openweb-tiktoken-exp4-top2/ckpt.pt"
                num_experts=4
                num_select=2
                save_dir=$results"gpt/exp_4_top_2"
                ;;
            "exp_4_top_1")
                model_path="openweb-tiktoken-exp4-top1/ckpt.pt"
                num_experts=4
                num_select=1
                save_dir=$results"gpt/exp_4_top_1"
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
                model_path="openweb-byt5-exp1-top1/ckpt.pt"
                num_experts=1
                num_select=1
                save_dir=$results"byt5/std"
                ;;
            "exp_4_top_2")
                model_path="openweb-byt5-exp4-top2/ckpt.pt"
                num_experts=4
                num_select=2
                save_dir=$results"byt5/exp_4_top_2"
                ;;
            "exp_4_top_1")
                model_path="openweb-byt5-exp4-top1/ckpt.pt"
                num_experts=4
                num_select=1
                save_dir=$results"byt5/exp_4_top_1"
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
#data_dir="evals/arc_easy/data"

#test -f $model_path && echo 'Model exists' || echo 'Model does not exist'

# Run the evaluation script
python $script --model_path $model_path --tokenizer_path $tokenizer_path --num_experts $num_experts --num_selects $num_select --save_dir $save_dir --data_dir $data_dir

