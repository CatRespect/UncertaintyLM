import os
import logging
import random
import time
import requests
from bs4 import BeautifulSoup
from huggingface_scraper import get_top_models, get_fun_messages
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import MeanTokenEntropy
from lm_polygraph.utils.manager import estimate_uncertainty
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# Load environment
load_dotenv()
hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO)

# Globals
MODEL_LIMIT = 10

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message with inline buttons to choose a model."""
    # Fetch trending models
    models = get_top_models(limit=MODEL_LIMIT)
    keyboard = []
    for model_name in models:
        keyboard.append([InlineKeyboardButton(model_name, callback_data=f"select:{model_name}")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        '🤖Here is the list of the most trending models on huggingface.\n Please choose a model for your questions:', reply_markup=reply_markup
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data.startswith('select:'):
        model_name = data.split(':', 1)[1]
        context.user_data['model_name'] = model_name
        # Load model
        try:
            base = AutoModelForCausalLM.from_pretrained(model_name, device_map='cpu', trust_remote_code=True)
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tok.pad_token = tok.eos_token
            base.config.pad_token_id = base.config.eos_token_id
        except Exception as e:
            await query.edit_message_text(f"❌ This model seems unsupported: {e}")
            return
        context.user_data['model'] = WhiteboxModel(base, tok, model_path=model_name)
        context.user_data['ue_method'] = MeanTokenEntropy()
        await query.edit_message_text(text=f"✅ Model set to: {model_name}\nNow send me a question.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    model = context.user_data.get('model')
    ue_method = context.user_data.get('ue_method')
    if not model:
        await update.message.reply_text("⚠️ No model selected. Use /start to choose a model first.")
        return
    # Fun message
    fun_messages = get_fun_messages()
    await update.message.reply_text(random.choice(fun_messages))
    time.sleep(random.uniform(0.7, 1.5))
    # Estimate uncertainty and reply
    try:
        ue = estimate_uncertainty(model, ue_method, input_text=user_text)
        reply = (f"💬 *Question:* {ue.input_text}\n"
                 f"*Answer:* {ue.generation_text}\n"
                 f"*Uncertainty:* `{ue.uncertainty:.4f}`\n"
                 f"*Model:* {ue.model_path}\n"
                 f"*Estimator:* {ue.estimator}")
        await update.message.reply_markdown(reply)
    except Exception as e:
        await update.message.reply_text(f"❌ Error during inference: {e}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Use /start to choose a model and ask questions inline.")

if __name__ == '__main__':
    token = os.getenv('TG_BOT_TOKEN')
    if not token:
        logging.error('TG_BOT_TOKEN not set in environment')
        exit(1)
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logging.info('Bot started')
    app.run_polling()
