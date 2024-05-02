#/usr/bin/bash
# This script is used to tokenize the datasets

echo "Pulling changes from git... \n"
git pull
echo "Updating required packages... \n"
pip install -r requirements.txt


idx=$1
case $idx in
    0)
        # Run wikitext data generation
        python data/wikitexts/generate_data.py
    ;;
    1)
        # Run wikitext data generation
        python data/multilingual_wiki/generate_data.py --dir data/multilingual_wiki/ 
    ;;
    2)
        # Run wikitext data generation
        python data/multilingual_wiki/generate_data_GPT2.py
    ;;
    *)
        echo "Invalid option"
    ;;
esac
#
#echo "Data generation completed... \n"
