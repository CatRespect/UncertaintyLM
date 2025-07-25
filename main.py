# --- IMPORT SCRAPER ---
from huggingface_scraper import get_top_models, get_fun_messages

get_top_models()
fun_messages = get_fun_messages()

import os
import warnings
import random
import time
warnings.filterwarnings("ignore")
# os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# os.environ["TRANSFORMERS_NO_PROGRESS_BARS"] = "1"
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from transformers import logging
logging.set_verbosity_error()
from dotenv import load_dotenv, dotenv_values
load_dotenv() 

from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_polygraph.utils.model import WhiteboxModel
model_path = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = WhiteboxModel(base_model, tokenizer, model_path=model_path)

# model = BlackboxModel.from_openai(
#     openai_api_key=os.getenv("API_KEY"),
#     model_path='gpt-4o',
#     supports_logprobs=True  # Enable for deployments 
# )
# import google.generativeai as genai
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# model = genai.GenerativeModel("gemini-2.5-flash")
# print(genai.list_models())
print("BEGINNING")

from lm_polygraph.estimators import *
ue_method = MeanTokenEntropy()
from lm_polygraph.utils.manager import estimate_uncertainty

def pretty_print_ue(ue):
    print("\n======= РЕЗУЛЬТАТ =======")
    print(f"Вопрос: {ue.input_text}")
    print(f"Ответ: {ue.generation_text}")
    print(f"\033[1mНеопределённость: {ue.uncertainty:.4f}\033[0m")
    print(f"Модель: {ue.model_path}")
    print(f"Оценщик: {ue.estimator}")
    print("========================\n")

print("\nДобро пожаловать в интерактивный режим! (Ctrl+C для выхода)\n")

while True:
    try:
        input_text = input("\033[1mВведите ваш вопрос: \033[0m")
        if not input_text.strip():
            print("Пустой ввод. Попробуйте ещё раз!\n")
            continue
        time.sleep(random.uniform(0.7, 1.5))
        print(random.choice(fun_messages))
        ue = estimate_uncertainty(model, ue_method, input_text=input_text)
        pretty_print_ue(ue)
    except KeyboardInterrupt:
        print("\n\nДо новых встреч!\n")
        break
    except Exception as e:
        print(f"Ошибка: {e}\n")