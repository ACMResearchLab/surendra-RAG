#https://huggingface.co/SEBIS/code_trans_t5_base_source_code_summarization_python
from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline

#TODO: this doesnt seem to work
pipeline = SummarizationPipeline(
    model=AutoModelWithLMHead.from_pretrained("SEBIS/code_trans_t5_base_source_code_summarization_python"),
    tokenizer=AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_base_source_code_summarization_python", skip_special_tokens=True),
    device=0
)

tokenized_code = '''with open ( CODE_STRING , CODE_STRING ) as in_file : buf = in_file . readlines ( )  with open ( CODE_STRING , CODE_STRING ) as out_file : for line in buf :          if line ==   " ; Include this text   " :              line = line +   " Include below  "          out_file . write ( line ) '''
pipeline([tokenized_code])
