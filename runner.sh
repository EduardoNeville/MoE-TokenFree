echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "BYT5 RUNS"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"

echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "Running MMLU: BYT5 STANDARD"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
./evals/mmlu/mmlu/mmlu_script.sh byt5 std arc_easy
echo "----------- DONE -----------"

echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "Running MMLU: BYT5 EXP_4_TOP_2"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
./evals/mmlu/mmlu/mmlu_script.sh byt5 exp_4_top_2 arc_easy
echo "----------- DONE -----------"

echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "Running MMLU: BYT5 EXP_4_TOP_1"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
./evals/mmlu/mmlu/mmlu_script.sh byt5 exp_4_top_1 arc_easy
echo "----------- DONE -----------"

echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "GPT RUNS"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"

echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "Running MMLU: GPT STANDARD"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
./evals/mmlu/mmlu/mmlu_script.sh gpt std arc_easy
echo "----------- DONE -----------"

echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "Running MMLU: GPT EXP_4_TOP_2"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
./evals/mmlu/mmlu/mmlu_script.sh gpt exp_4_top_2 arc_easy
echo "----------- DONE -----------"

echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "Running MMLU: GPT EXP_4_TOP_1"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
echo "::::::::::::::::::::::::::::::::::::::::::::::::"
./evals/mmlu/mmlu/mmlu_script.sh byt5 exp_4_top_1 arc_easy
echo "----------- DONE -----------"
