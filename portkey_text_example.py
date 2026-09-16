# Install the Portkey AI Gateway package with pip
#   pip -i portkey-ai
# 
# For more information on the SDK see https://portkey.ai/docs/api-reference/sdk/python
# 
from portkey_ai import Portkey
import os

# Before executing this code, define the API Key within an environment variable in your OS
# Linux BASH example: export AI_SANDBOX_KEY=<key provided to you>

# Import API key from OS environment variables
AI_SANDBOX_KEY = os.getenv("AI_SANDBOX_KEY")

# Check if API key is set
if not AI_SANDBOX_KEY:
    raise ValueError("AI_SANDBOX_KEY environment variable is not set. Please set it before running this script.")

client = Portkey(api_key=AI_SANDBOX_KEY)

# Set the model deployment name that the prompt should be sent to
# Models that support image analysis are indicated by image_analysis: True
available_models = [
                    # OpenAI models are made available via OpenAI.  
                    # For a list of all available OpenAI models see https://platform.openai.com/docs/models
                   {
                        "model_name": "gpt-5-mini",
                        "supports_image_analysis": True
                    },
                    {
                        "model_name": "gpt-4o-mini",
                        "supports_image_analysis": True
                    },
                    # Gemini models are made available via Google Cloud.  
                    # For a list of all available Gemini models see https://ai.google.dev/gemini-api/docs/models
                    {
                        "model_name": "gemini-3.1-pro-preview",
                        "supports_image_analysis": True
                    },
                    # The following specific models are made available via Microsoft Azure
                    {
                        "model_name": "claude-opus-4-7",
                        "supports_image_analysis": True
                    },
                    {
                        "model_name": "claude-sonnet-4-6",
                        "supports_image_analysis": True
                    },
                    {
                        "model_name": "claude-haiku-4-5",
                        "supports_image_analysis": True
                    },
                    {
                        "model_name": "Llama-3.3-70B-Instruct",
                        "supports_image_analysis": False
                    },
                    {
                        "model_name": "Llama-4-Scout-17B-16E-Instruct",
                        "supports_image_analysis": True
                    },
                    {
                        "model_name": "mistral-small-2503",
                        "supports_image_analysis": True
                    },
                    {
                        "model_name": "mistral-medium-2505",
                        "supports_image_analysis": True
                    }
                ]

# This function will submit a simple text prompt to the chosen model
def text_prompt_example(model_to_be_used):
    # Establish a connection to your Azure OpenAI instance
    
    try:
        response = client.chat.completions.create(
        model=model_to_be_used, 
        # Prompt parameters may also be defined, depending on model capabilities
        #temperature=0.5, # temperature = how creative/random the model is in generating response - 0 to 1 with 1 being most creative
        #max_tokens=1000, # max_tokens = token limit on context to send to the model
        #top_p=0.5, # top_p = diversity of generated text by the model considering probability attached to token - 0 to 1 - ex. top_p of 0.1 = only tokens within the top 10% probability are considered
        messages=[
        {"role": "system", "content": "You are a helpful junior physics professor."}, # describes model identity and purpose
        {"role": "user", "content": "Please explain quantum mechanics in 100 words or less."}, # user prompt
               ]
        )
        print("\n"+response.choices[0].message.content)

    except Exception as e:
        print(e.message)


#
# Execute the example functions
if __name__ == "__main__":

    for model in available_models:
        # Execute the image prompt example
        print("\nModel: " + model["model_name"])
        text_prompt_example(model["model_name"])

