import torch
import pandas as pd
import numpy as np
import gradio as gr
import textwrap

# All the embeddings are belong to "handbook-of-international-law.pdf"

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the SentenceTransformer model
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device=device)

# Convert texts and embedding df to a list of dicts
text_chunks_and_embeddings_df = pd.read_csv("text_chunks_and_embeddings.csv")
# Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
text_chunks_and_embeddings_df["embedding"] = text_chunks_and_embeddings_df["embedding"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" "))
# Convert DataFrame to list of dictionaries
pages_and_chunks = text_chunks_and_embeddings_df.to_dict(orient='records')
# Convert embeddings to torch tensor and send to device (Note: NumPy arrays are float64, torch tensors are float32 by default)
embeddings = torch.tensor(np.array(text_chunks_and_embeddings_df["embedding"].tolist()),
                          dtype=torch.float32).to(device)

from sentence_transformers import util

def retrieve(query: str,
             embeddings: torch.tensor,
             embedding_model: SentenceTransformer = embedding_model,
             n_results: int = 5):
    # 1. Embed the query to the same numerical space as the text examples
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    # 2. Get similarity scores with the dot product
    dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]

    # 3. Get the top-k results (we'll keep this to 5)
    scores, indices = torch.topk(dot_scores, k=n_results)
    return scores, indices


def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available


# Get GPU memory
if device == "cuda":
    use_quantization = True
    gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (2 ** 30))
    if gpu_memory_gb > 19.0:
        use_quantization = False
    else:
        use_quantization = True

# Create quantization config for smaller model loading
# For models that require 4-bit quantization
from transformers import BitsAndBytesConfig

quantization_fig = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_use_double_quant=True,
                                      bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.float16)

# Flash Attention 2 for faster inference, default to "sdpa" ("scaled do product attention")
if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
    attn_implementation = "flash_attn_2"
else:
    attn_implementation = "sdpa"

# Pick a model we'd like to use
# Note: The model I'm using required login to Hugging Face CLI to be able to access otherwise a HTTPS error will be thrown
# You can also substitute with a model that doesn't require login like: "microsoft/Phi-3-mini-4k-instruct"
# For more details, read the README.md file

model_id = "google/gemma-2b-it"
print(f"[INFO] Using model: {model_id}")

# Instantiate tokenizer (tokenizer turns text into numbers ready for the model)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id, trust_remote_code=True)

# Instantiate the model
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True,
                                                 low_cpu_mem_usage=False,
                                                 quantization_config=quantization_fig if use_quantization else None)
llm_model = llm_model.to(device)


# Define a prompt formatter
def prompt_formatter(query: str,
                     context_items: list[dict]) -> str:
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Customizable prompt for the user, should change according to the PDF content
    base_prompt = """
Based on the following legal documents and case studies, please answer the query.
Don't include your thought process, just provide the answer in a clear and comprehensive way.
Strive to emulate the following examples for the ideal answer format:
Example 1:
Query: What are the elements of a contract?
Answer: A valid contract entails several elements: offer, acceptance, consideration, capacity, and legality. An offer signifies a willingness to enter into an agreement, outlining the proposed terms. Acceptance represents a clear and unequivocal agreement to the offer's terms. Consideration refers to the exchange of something of value between the parties, which can be a good or service. Capacity ensures both parties possess the legal authority to form a contract. Legality implies the contract's purpose adheres to the law.
Example 2:
Query: Describe the concept of negligence in tort law.
Answer: In tort law, negligence signifies a failure to exercise reasonable care, resulting in harm to another person or their property. It encompasses four key elements: duty of care, breach of duty, causation, and damages. The duty of care mandates that individuals act with a degree of caution to avoid foreseeable risks to others. A breach of duty occurs when someone fails to uphold this standard of care. Causation establishes a link between the breach of duty and the resulting harm. Damages represent the losses or injuries suffered by the plaintiff due to the defendant's negligence.
Example 3:
Query: Explain the Miranda rights in the United States.
Answer: The Miranda rights, established in the landmark Miranda v. Arizona case, safeguard individuals suspected of criminal activity during custodial interrogation. These rights encompass the right to remain silent, the right to an attorney, and the right to have an attorney present during questioning. If a suspect is not informed of these rights, any statements they make during questioning may be inadmissible in court.
Now replace the placeholders with specific legal documents, case studies, or relevant legal topics, and ask your question.
**{context}** (Replace with specific legal documents, case studies, or relevant legal topics)
**Relevant passages:** <extract relevant passages from the legal context here>
**User query:** {query}
"""

    # Update base prompt with context items and query   
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
         "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt


# Define the function for the Gradio interface
# To enable the context items to be returned, set 'only_answer' to False
def ask(query, temperature=0.7, max_new_tokens=512, only_answer=True):
    # Get just the scores and indices of top related results
    scores, indices = retrieve(query=query,
                               embeddings=embeddings)

    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]

    # Add score to context items
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu()  # return score back to CPU

    # Format the prompt with context items
    prompt = prompt_formatter(query=query,
                              context_items=context_items)

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate an output of tokens
    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)

    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    # Replace special tokens and unnecessary help message (different from models)
    output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace(
        "Sure, here is the answer to the user query:\n\n", "")
    if only_answer:
        return output_text

    return output_text, context_items


# Define the Gradio interface
interface = gr.Interface(
    fn=ask,
    inputs=[
        gr.Textbox(show_label=False, placeholder="Enter your query here..."),
    ],
    outputs="text",
)


def main():
    interface.launch()


if __name__ == "__main__":
    main()
