import json
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# Загрузка модели
model = load_model('model.h5')

# Загрузка данных рецептов из JSON-файла
with open('recipes.json', 'r', encoding='utf-8') as file:
    recipes = json.load(file)

# Функция для обработки ингредиентов
def process_ingredients(ingredients):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(ingredients)
    sequences = tokenizer.texts_to_sequences(ingredients)

    # Получение максимальной длины последовательности
    max_length = max([len(seq) for seq in sequences])

    # Приведение всех последовательностей к одинаковой длине
    padded_sequences = np.array([seq + [0] * (max_length - len(seq)) for seq in sequences])

    return padded_sequences

# Функция для получения предсказания от модели
def get_prediction(inputs):
    predictions = model.predict(inputs)
    predicted_class_idx = np.argsort(predictions[0])[::-1]
    predicted_recipes = [recipe_names[idx] for idx in predicted_class_idx]
    predicted_probabilities = [predictions[0][idx] for idx in predicted_class_idx]
    return predicted_recipes, predicted_probabilities

# Основная функция
def main():
    print("Введите ингредиенты через запятую:")
    ingredients = input().split(",")
    ingredients = [item.strip() for item in ingredients]

    # Проверка, что данные рецептов загружены корректно
    print(f"Загружено {len(recipes)} рецептов")
    for recipe in recipes[:5]:
        print(recipe['title'])

    inputs = process_ingredients([" ".join(ingredients)])
    predicted_recipes, predicted_probabilities = get_prediction(inputs)

    print("Предсказанные рецепты:")
    for recipe, probability in zip(predicted_recipes, predicted_probabilities):
        print(f"{recipe} ({probability*100:.2f}%)")

if __name__ == "__main__":
    # Список названий рецептов, соответствующих выходным классам модели
    recipe_names = [recipe['title'] for recipe in recipes]
    main()
