# mythical_speedrunners_rag.py
import requests
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import sys

class MythicalSpeedrunnersRAG:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
    def load_tz_from_github(self):
        """Загрузка ТЗ с GitHub"""
        url = "https://raw.githubusercontent.com/CasualCalf/First-task/main/MythicalSpeedrunners.md"
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Ошибка загрузки ТЗ: {response.status_code}")
    
    def chunk_text(self, text):
        """Разбивка текста на смысловые блоки"""
        # Очищаем текст от лишних символов
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Разделяем по заголовкам Markdown
        sections = re.split(r'\n## ', text)
        
        chunks = []
        metadata = []
        
        for section in sections:
            if not section.strip():
                continue
                
            # Извлекаем заголовок и содержание
            lines = section.split('\n')
            title = lines[0].strip().replace('#', '').strip()
            content = '\n'.join(lines[1:]).strip()
            
            if not content:
                continue
                
            # Разбиваем содержание на абзацы
            paragraphs = [p.strip() for p in re.split(r'\n\n+', content) if p.strip()]
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 30:
                    chunks.append(paragraph)
                    metadata.append({
                        'section': title,
                        'paragraph_index': i,
                        'full_content': f"{title}\n\n{paragraph}"
                    })
        
        return chunks, metadata
    
    def create_vector_index(self, chunks):
        """Создание векторного индекса"""
        print("Векторизация текста...")
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        return index, embeddings
    
    def setup(self):
        """Полная настройка системы"""
        print("🦄 Загрузка ТЗ Mythical Speedrunners с GitHub...")
        tz_text = self.load_tz_from_github()
        
        print("📝 Разбивка на смысловые блоки...")
        self.chunks, self.chunk_metadata = self.chunk_text(tz_text)
        
        self.index, self.embeddings = self.create_vector_index(self.chunks)
        
        print(f"✅ Создано {len(self.chunks)} блоков для поиска")
        
    def save_vectors(self, filename='mythical_speedrunners_vectors.pkl'):
        """Сохранение векторов в файл"""
        vector_data = {
            'embeddings': self.embeddings,
            'chunks': self.chunks,
            'metadata': self.chunk_metadata,
            'index': faiss.serialize_index(self.index)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(vector_data, f)
        
        print(f"💾 Векторы сохранены в {filename}")
        return filename
    
    def load_vectors(self, filename='mythical_speedrunners_vectors.pkl'):
        """Загрузка векторов из файла"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Файл {filename} не найден. Сначала запустите setup().")
            
        with open(filename, 'rb') as f:
            vector_data = pickle.load(f)
        
        self.embeddings = vector_data['embeddings']
        self.chunks = vector_data['chunks']
        self.chunk_metadata = vector_data['metadata']
        self.index = faiss.deserialize_index(vector_data['index'])
        
        print(f"📂 Векторы загружены из {filename}")
        return True
    
    def search(self, query, top_k=3):
        """Поиск релевантных блоков"""
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'content': self.chunks[idx],
                    'metadata': self.chunk_metadata[idx],
                    'score': distances[0][i]
                })
        
        return results
    
    def generate_response(self, query, context):
        """Генерация ответа на основе найденного контекста"""
        response = f"""🤖 ОТВЕТ НА ВОПРОС: "{query}"

📚 На основе технического задания Mythical Speedrunners:

{context}

💡 Эта информация взята из соответствующих разделов ТЗ игры."""
        
        return response
    
    def ask_question(self, query, top_k=3):
        """Полный цикл: поиск + генерация ответа"""
        search_results = self.search(query, top_k)
        
        if not search_results:
            return {
                'answer': "❌ По вашему запросу не найдено релевантной информации в ТЗ.",
                'sources': []
            }
        
        # Формирование контекста
        context = "\n\n".join([
            f"► Раздел: {result['metadata']['section']}\n{result['content']}"
            for result in search_results
        ])
        
        # Генерация ответа
        response = self.generate_response(query, context)
        
        return {
            'answer': response,
            'sources': search_results
        }

def initialize_system():
    """Инициализация системы с созданием векторной базы"""
    rag = MythicalSpeedrunnersRAG()
    rag.setup()
    rag.save_vectors()
    return rag

def load_existing_system():
    """Загрузка существующей системы"""
    rag = MythicalSpeedrunnersRAG()
    rag.load_vectors()
    return rag

def run_demo():
    """Демонстрация работы системы"""
    print("🎯 ДЕМОНСТРАЦИЯ RAG СИСТЕМЫ")
    print("=" * 60)
    
    try:
        rag = load_existing_system()
        print("✅ Система загружена!")
    except FileNotFoundError:
        print("📥 Создание новой векторной базы...")
        rag = initialize_system()
    
    # Тестовые вопросы
    questions = [
        "Какая система прогрессии в игре?",
        "Какие персонажи доступны и их способности?",
        "Как работает монетизация игры?",
        "Что такое ядра хаоса и как их использовать?",
        "Какие технические требования к игре?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. ❓ ВОПРОС: {question}")
        result = rag.ask_question(question, top_k=2)
        
        # Показываем начало ответа
        answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
        print(f"📝 ОТВЕТ: {answer_preview}")
        
        print("📚 ИСТОЧНИКИ:")
        for j, source in enumerate(result['sources'], 1):
            print(f"   {j}. {source['metadata']['section']} (сходство: {source['score']:.4f})")
        
        print("-" * 60)

def interactive_mode():
    """Интерактивный режим вопрос-ответ"""
    print("🎮 RAG Система для Mythical Speedrunners")
    print("=" * 50)
    
    try:
        rag = load_existing_system()
        print("✅ Система загружена и готова к работе!")
    except FileNotFoundError:
        print("📥 Создание новой векторной базы...")
        rag = initialize_system()
    
    print("\n💡 Примеры вопросов:")
    print("  - Какая система прогрессии в игре?")
    print("  - Какие персонажи доступны?") 
    print("  - Как работает монетизация?")
    print("  - Что такое ядра хаоса?")
    print("  - Какие технические требования?")
    print("\nВведите 'quit' для выхода\n")
    
    while True:
        print("-" * 50)
        question = input("🎯 Ваш вопрос: ").strip()
        
        if question.lower() in ['quit', 'exit', 'выход', '']:
            print("👋 До свидания!")
            break
            
        print("🔍 Поиск в ТЗ...")
        result = rag.ask_question(question)
        
        print(f"\n{result['answer']}")
        
        if result['sources']:
            print(f"\n📖 Использованные разделы ТЗ:")
            for i, source in enumerate(result['sources'], 1):
                print(f"   {i}. {source['metadata']['section']} (сходство: {source['score']:.4f})")

def setup_only():
    """Только создание векторной базы"""
    print("🦄 Создание векторной базы для Mythical Speedrunners...")
    rag = initialize_system()
    print("✅ Векторная база успешно создана!")

def show_help():
    """Показать справку"""
    print("""
🎮 RAG система для Mythical Speedrunners

Использование:
  python mythical_speedrunners_rag.py [команда]

Команды:
  demo      - запустить демонстрацию (по умолчанию)
  interactive - интерактивный режим вопрос-ответ
  setup     - только создать векторную базу
  help      - показать эту справку

Примеры:
  python mythical_speedrunners_rag.py
  python mythical_speedrunners_rag.py interactive
  python mythical_speedrunners_rag.py setup
""")

def main():
    """Основная функция"""
    if len(sys.argv) == 1:
        # Если аргументов нет - запускаем демо
        run_demo()
    else:
        command = sys.argv[1].lower()
        
        if command == 'demo':
            run_demo()
        elif command == 'interactive':
            interactive_mode()
        elif command == 'setup':
            setup_only()
        elif command in ['help', '--help', '-h']:
            show_help()
        else:
            print(f"❌ Неизвестная команда: {command}")
            show_help()

if __name__ == "__main__":
    main()
