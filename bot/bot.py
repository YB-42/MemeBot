import asyncio
import logging
import sys
import os

print("Импорт начался")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("✅ sys.path настроен")

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command

try:
    from config_reader import config

    print("✅ config импортирован:", config.bot_token.get_secret_value())
except Exception as e:
    print("❌ Ошибка при импорте config:", e)

from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from PIL import Image

try:
    from models.inference_model import run_inference

    print("✅ inference импортирован:")
except Exception as e:
    print("❌ Ошибка при импорте inferenc'a:", e)

print("✅ Все импорты прошли")

logging.basicConfig(level=logging.INFO)
print("🟢 Логгер настроен")

bot = Bot(token=config.bot_token.get_secret_value())
dp = Dispatcher()

print("🚀 Bot и Dispatcher созданы")

user_model = {}


def model_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="EfficientNet"),
                KeyboardButton(text="ViT"),
                KeyboardButton(text="ResNet"),
            ],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def next_action_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Вернуться к моделям", callback_data="back_to_models"
                ),
                InlineKeyboardButton(text="Очистить чат", callback_data="clear_chat"),
            ]
        ]
    )


@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    logging.info("Команда /start получена от пользователя: %s", message.from_user.id)
    await message.answer(
        "Привет 👋\n\nЯ Бот и могу помочь определить, мем из твоего паблика или нет"
    )
    await message.answer(
        "Выбери модель, с которой будем работать:", reply_markup=model_keyboard()
    )


@dp.message(F.text.in_(["EfficientNet", "ViT", "ResNet"]))
async def model_chosen(message: types.Message):
    user_model[message.from_user.id] = message.text
    await message.answer("Супер, теперь скидывай фото с мемом")


@dp.message(lambda message: message.photo)
async def handle_photo(message: types.Message):
    user_id = message.from_user.id

    if user_id not in user_model:
        await message.answer("Пожалуйста, сначала выбери модель с помощью /start")
        return

    logging.info(f"Получено фото от пользователя: {user_id}")

    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)

        image = await bot.download_file(file.file_path)
        image_filt = Image.open(image).convert("RGB")
    except Exception as e:
        await message.answer(f"❌ Ошибка при загрузке изображения: {e}")
        return

    model_name = user_model[user_id]
    processing_msg = await message.answer("🔍 Обрабатываю изображение...")

    try:
        result = run_inference(model_name, image_filt)
        await bot.delete_message(
            chat_id=message.chat.id, message_id=processing_msg.message_id
        )

        await message.answer(
            f"📊 Результат от {model_name}: `{result:.4f}`",
            parse_mode="Markdown",
            reply_markup=next_action_keyboard(),
        )

        user_model.pop(user_id, None)

    except Exception as e:
        await bot.delete_message(
            chat_id=message.chat.id, message_id=processing_msg.message_id
        )
        await message.answer(f"❌ Ошибка при обработке: {e}")


@dp.callback_query(lambda c: c.data in ["back_to_models", "clear_chat"])
async def process_callback(callback_query: types.CallbackQuery):
    user_id = callback_query.from_user.id
    data = callback_query.data

    if data == "back_to_models":
        await callback_query.message.answer(
            "Выбери модель, с которой будем работать:", reply_markup=model_keyboard()
        )
        await callback_query.answer()

    elif data == "clear_chat":
        chat_id = callback_query.message.chat.id
        history = await bot.get_chat_history(chat_id, limit=100)
        for msg in history:
            try:
                await bot.delete_message(chat_id=chat_id, message_id=msg.message_id)
            except Exception:
                pass
        await callback_query.answer("Чат очищен")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.info("Запуск бота...")
    asyncio.run(main())
