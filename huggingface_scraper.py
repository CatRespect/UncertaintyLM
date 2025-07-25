import requests
from bs4 import BeautifulSoup


import requests
from bs4 import BeautifulSoup

def get_top_models(limit=21):
    url = "https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:0,max:32B&sort=trending"
    print("🤗 Top HuggingFace Models(<32B params) (Trending - Scraped):")
    modelList = []
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Look for model links under <a href="/model-name"> inside model cards
        model_links = soup.select("a[href^='/']")

        seen = set()
        count = 0
        for link in model_links:
            href = link.get("href")
            if href and href.count("/") == 2 and "/datasets/" not in href and "/spaces/" not in href:
                model_name = href.strip("/")
                if model_name not in seen:
                    if model_name=="search/full-text?type=model": continue
                    print(f"{count + 1}. {model_name}")
                    modelList.append(model_name)
                    seen.add(model_name)
                    count += 1
            if count >= limit:
                break

    except Exception as e:
        print(f"⚠️ Ошибка при парсинге Hugging Face: {e}")
    return modelList

def get_fun_messages():
    return [
        "Электроны бегут по проводам...",
        "Голубь уже летит к вам с ответом!",
        "Квантовые биты спорят между собой...",
        "Сервер греется, как чайник!",
        "Мозговой штурм искусственного интеллекта...",
        "Пакетики данных прыгают по интернету...",
        "Модель вспоминает всё, что знает...",
        "Вычисляем вероятность истины...",
        "Генерируем ответ с помощью магии!",
        "Проверяем, не галлюцинирует ли модель...",
        "🤖 Мозги модели греются от вашего вопроса...",
        "🎩 Достаём кролика из нейросетевой шляпы...",
        "💾 Подключаемся к серверу за пределами Млечного Пути...",
        "🌐 AI ищет ответ в тёмных уголках интернета...",
        "Эм, кажется, я не уверен в ответе...",
        "Хмм, может быть, это не совсем точно...",
        "Похоже, я немного сомневаюсь...",
        "Возможно, это не стопроцентно...",
        "Мой алгоритм подсказывает, но с сомнением..."
    ]
