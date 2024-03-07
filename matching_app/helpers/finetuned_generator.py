# %%
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from detoxify import Detoxify
import torch

# %%
fine_tuned_model = GPT2LMHeadModel.from_pretrained('../fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('../fine_tuned_model')
toxic_model = Detoxify("original")

# %%  
def generate_text(prompt_text):
    '''
    Generates text from the user's input using the fine-tuned model, ensuring that the user's input is not harmful using Detoxify.
    Input: The text of the prompt (str)
    Output: The generated output if input & output passes the check or a message telling the user their input/output was harmful. 
    '''

    ## Do not generate text if input bio fails toxicity test
    toxicity_rubric_input = toxic_model.predict(prompt_text)
    if toxicity_rubric_input['severe_toxicity'] > 0.1 or toxicity_rubric_input['threat'] > 0.01:
        return "Your generated match cannot be shown due to harmful material in your bio. Please modify and try again."
    
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    output = fine_tuned_model.generate(input_ids, 
                                        attention_mask=attention_mask,
                                        pad_token_id=tokenizer.eos_token_id,
                                        min_length=200, 
                                        max_length=500, 
                                        num_return_sequences=1, 
                                        temperature=0.9,  
                                        do_sample=True)  
    decoded_str = tokenizer.decode(output[0], skip_special_tokens=True)
    
    toxicity_rubric_generated = toxic_model.predict(decoded_str)
    
    ## Do not return generated text if it fails toxicity test
    if toxicity_rubric_generated['severe_toxicity'] > 0.1 or toxicity_rubric_generated['threat'] > 0.01:
        return "Your generated match cannot be shown. Please try again."
    
    if prompt_text in decoded_str:
        decoded_str = decoded_str.replace(prompt_text, '').strip()

    ## Return generated bio if generated bio passes toxicity tests
    return decoded_str
