This is the repository for the scripts used for the TurEngMix corpus building and experimentation.

Scripts found here include scripts for collecting and filtering Turkish-English code-mixed posts from Eksi Sozluk, tokenizing posts, running LID and NER tests, and generating synthetic code-mixed data and annotating with LLM-as-Judge.

**/Dataset_Creation**: This folder includes the **eksi_sozluk_scraping.py** and the **tokenizer.py** scripts. **eksi_sozluk_scraping.py** contains the full script for scraping 200 pages of posts from Eksi Sozluk using a given topic list, filtering them with langdetect, and filtering them with two separate GPT4o prompts, then saving them all to a csv. **tokenizer.py** tokenizes a csv of given posts collected using the scraping script.

**/LID_and_NER**: This folder includes the **/LID** folder and the **/NER** folders. The **LID** folder contains the **gpt_lid.py** and **qwen_lid.py** scripts. Each script takes the csv creates by the tokenizer script and runs the model on each token, creating an output csv with the initial csv and a column of LID predictions. API keys need to be added or loaded in from the environment. The **NER** folder contains the **gpt_ner.py** and **qwen_ner.py** scripts. Each script takes the csv creates by the tokenizer script and runs the model on each token, creating an output csv with the initial csv and a column of NER predictions. 

Both folders also contain the **\prompts** folder with zero-shot and three-shot prompt text. 

This folder also includes the **encoder_finetuning** folder, which contains **encoder_LID** and **encoder_NER**, with scripts for splitting the dataset, training the models, and testing model capabilities for encoder models.


