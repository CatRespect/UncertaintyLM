import asyncio
import os
import logging
import random
import time

import lm_polygraph.estimators
import requests
from bs4 import BeautifulSoup
from huggingface_scraper import get_top_models, get_fun_messages
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import *
from lm_polygraph.utils.manager import estimate_uncertainty
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# Load environment variables
dotenv_loaded = load_dotenv()
hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO)

# --- Uncertainty Estimation Methods ---
# Each tuple: (class_name, display_label with year), sorted from most recent to oldest
ESTIMATOR_METHODS = [
    # White-box, Information-based (2023)
    ('MonteCarloSequenceEntropy',         'MCSeqEntropy (2023)'),

    # White-box, Density-based (2023)
    ('RelativeMahalanobisDistanceSeq',    'RelMahalanobis (2023)'),

    # Black-box, Meaning Diversity (2023)
    ('DegMat',                            'DegMat (2023)'),
    ('Eccentricity',                      'Eccentricity (2023)'),
    ('EigValLaplacian',                   'EigValLaplacian (2023)'),

    # White-box, Reflexive (2022)
    ('ConditionalPointwiseMutualInformation',  'CondPMI (2022)'),
    ('MeanConditionalPointwiseMutualInformation', 'MeanCondPMI (2022)'),
    ('PTrue',                             'p(True) (2022)'),
    ('PTrueSampling',                     'p(True)Sampling (2022)'),

    # White-box, Information-based (2020)
    ('Perplexity',                        'Perplexity (2020)'),
    ('MeanTokenEntropy',                  'MeanTokenEntropy (2020)'),
    ('MaxTokenEntropy',                   'MaxTokenEntropy (2020)'),
    ('LexicalSimilarity',                 'LexSim (2020)'),

    # White-box, Information-based (2019)
    ('PointwiseMutualInformation',        'PMI (2019)'),
    ('MeanPointwiseMutualInformation',    'MeanPMI (2019)'),

    # White-box, Information-based (2018)
    ('MaximumSequenceProbability',        'MaxSeqProb (2018)'),
]

# Globals
MODEL_LIMIT = 10

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Initiate model selection with inline buttons."""
    models = get_top_models(limit=MODEL_LIMIT)
    keyboard = [[InlineKeyboardButton(name, callback_data=f"model:{name}")] for name in models]
    markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        '🤖 Select a model:', reply_markup=markup
    )

async def send_estimator_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Prompt user to choose an uncertainty estimator, showing publication years."""
    keyboard = []
    for class_name, label in ESTIMATOR_METHODS:
        keyboard.append([InlineKeyboardButton(label, callback_data=f"estimator:{class_name}")])
    markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.message.reply_text(
        '✏️ Choose an uncertainty estimator:', reply_markup=markup
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data.startswith('model:'):
        model_name = data.split(':', 1)[1]
        context.user_data['model_name'] = model_name
        try:
            base = AutoModelForCausalLM.from_pretrained(model_name, device_map='cpu', trust_remote_code=True)
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tok.pad_token = tok.eos_token
            base.config.pad_token_id = base.config.eos_token_id
        except Exception as e:
            await query.edit_message_text(f"❌ Unsupported model: {e}")
            return
        context.user_data['model'] = WhiteboxModel(base, tok, model_path=model_name)
        await query.edit_message_text(f"✅ Model set to: {model_name}")
        await send_estimator_menu(update, context)

    elif data.startswith('estimator:'):
        estimator_name = data.split(':', 1)[1]
        try:
            estimator_cls = globals()[estimator_name]
            context.user_data['ue_method'] = estimator_cls()
        except Exception as e:
            await query.edit_message_text(f"❌ Error loading estimator {estimator_name}: {e}")
            return
        await query.edit_message_text(
            f"✅ Estimator set to: {estimator_name}\nYou can now ask your question."
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    model = context.user_data.get('model')
    ue_method = context.user_data.get('ue_method')
    if not model or not ue_method:
        await update.message.reply_text("⚠️ Please select both model and estimator using /start first.")
        return
    await update.message.reply_text(random.choice(get_fun_messages()))
    # time.sleep(random.uniform(0.7, 1.5))
    try:
        # msg = await update.message.reply_text("⏳ Generating answer… 0%")
        # def progress_cb(done: int, total: int):
        #     pct = int(done / total * 100)
        #     asyncio.create_task(msg.edit_text(f"⏳ Generating answer… {pct}%"))
        ue = estimate_uncertainty(
            model, ue_method, input_text=user_text, max_new_tokens=256)
        reply = (f"💬 *Question:* {ue.input_text}\n"
                 f"*Answer:* {ue.generation_text}\n"
                 f"*Uncertainty:* `{ue.uncertainty:.4f}`\n"
                 f"*Model:* {ue.model_path}\n"
                 f"*Estimator:* {ue.estimator}")
        await update.message.reply_markdown(reply)
    except Exception as e:
        await update.message.reply_text(f"❌ Error during inference: {e}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Use /start to choose model and estimator, then ask your question."
    )

if __name__ == '__main__':
    token = os.getenv('TG_BOT_TOKEN')
    if not token:
        logging.error('TG_BOT_TOKEN not set')
        exit(1)
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logging.info('Bot starting')
    app.run_polling()
