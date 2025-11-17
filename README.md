# CulShield

the following is the main functions in different python files

## crawler.py
crawl candidate paragraphes (including cultural taboos) given the url 

## data_construct.py
extract taboos from text and generate questions basea on these taboos
1. write_jailbreak_queries_yiwei
generate candidate jailbreak queries step by step

2. judge_jailbreak_qs_yiwei
judge whether the candidate query is a jailbreak query

3. gen_nonsent_qs
generate queries to evaluate whether the target llm has the knowledge

4. judge_oversent_qs
judge whether the generated queries is based on the give taboo

5. filter_again
judge whether the standard answer of the query is correct

6. gen_oversent_qs
generate queries to evaluate whether the target llm is overly sensitivity to some cultural taboos

## data_construct4sft.py
generate data for supervised fine-tune
1. generate_ft_data
generate refusal output for jailbreak questions

2. know_and_defend
classify all taboos to four categories (e.g., t_f, t_t, f_t, f_f)

3. read_score4target_country
generate extra data for supervised fine-tune according to the world value survey

## evaluation.py
1. eval_explicit
get answers from different llms given explicit questions, and compare their answers with the groundtruth answers

2. infer_implicit
get answers from different llms given jailbreak questions

3. eval_implicit
employ gpt-4o to judge whether the answer outputed by llms are jailbroken

## statisitic.py
1. statistic4explicit
get accuarcy (acc), oversensitivity rate (osr) for explicit questions

2. statistic4implicit
get jailbreak success rate (jsr) for jailbreak questions

## analysis.py
1. draw_pearson_pic
calculate pearson correlation coefficients between jsr and cultural distance for different llms, and draw the picture

2. draw_jsr_pic
show jsr of different countries in the world map
