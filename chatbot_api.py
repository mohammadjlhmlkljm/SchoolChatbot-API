import os
import json
import logging
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader # لقراءة PDF

# قم بتفعيل تسجيل الأخطاء لرؤية مشاكل API أو الملفات
logging.basicConfig(level=logging.INFO)

# تحميل المتغيرات من ملف .env
load_dotenv()

app = Flask(__name__)
# 💡 الحصول على المفتاح من متغيرات البيئة
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 💡 مسار مجلد المعرفة (يفترض أنه في نفس مسار ملف Python)
KNOWLEDGE_PATH = os.path.join(os.getcwd(), "Knowledge") 

# دالة لتحديد اللغة (بسيطة)
def is_arabic(text):
    return any('\u0600' <= char <= '\u06FF' for char in text)

# دالة لاستخلاص السياق من الملفات (TXT/CSV/PDF)
def find_relevant_context(question):
    context = []
    
    # 🛑 تأكد من وجود مجلد Knowledge
    if not os.path.exists(KNOWLEDGE_PATH):
        logging.error(f"Knowledge path not found: {KNOWLEDGE_PATH}")
        return ""

    try:
        for filename in os.listdir(KNOWLEDGE_PATH):
            filepath = os.path.join(KNOWLEDGE_PATH, filename)
            extension = os.path.splitext(filename)[1].lower()
            content = ""

            # === 1. قراءة الملفات النصية العادية ===
            if extension in ['.txt', '.csv']:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

            # === 2. قراءة ملفات PDF ===
            elif extension == '.pdf':
                reader = PdfReader(filepath)
                text_content = []
                for page in reader.pages:
                    text_content.append(page.extract_text())
                content = "\n".join(text_content)
            
            # منطق البحث البسيط (هل السؤال يحتوي على كلمة مفتاحية؟)
            if content and any(word.lower() in content.lower() for word in question.split()):
                 context.append(f"--- محتوى من ملف: {filename} ---\n{content}")
                 
        return "\n\n".join(context)
    
    except Exception as e:
        # تسجيل أي ملف يسبب مشكلة وعدم انهيار الـ API
        logging.error(f"Error reading knowledge files: {e}")
        return ""

# دالة بناء الـ System Prompt بناءً على الدور والسياق
def build_system_prompt(user_role, context, question):
    base_school_name = "مدرسة الأمير زيد بن الحسين المهنية"
    language = "بالعربية" if is_arabic(question) else "باللغة الإنجليزية"
    
    base_prompt = f"أنت مساعد ذكي لـ {base_school_name}. الإجابة يجب أن تكون {language}."

    if context:
        # إضافة تعليمات RAG (استخدم السياق فقط)
        base_prompt += f"\nملاحظة: اعتمد في إجابتك على السياق المرفق فقط. إذا لم تجد الإجابة في السياق، أجب بأنك لا تعلم.\n\nالسياق:\n{context}"

    if user_role == "Teacher":
        return f"أنت مساعد متخصص للمعلمين في {base_school_name}. أجب على الأسئلة الإدارية والتعليمية للمدرسة. {base_prompt}"
    elif user_role == "Student":
        return f"أنت مرشد أكاديمي للطلاب في {base_school_name}. كن ودوداً وموضحاً. {base_prompt}"
    else: # Visitor/Parent/General User
        return f"أنت مساعد عام للزوار وأولياء الأمور. أجب بأدب ووضوح عن أسئلة القبول والتسجيل والأخبار العامة للمدرسة. {base_prompt}"

# 💡 نقطة النهاية (Endpoint) لـ API
@app.route('/api/chatbot/ask_python', methods=['POST'])
def ask_chatbot():
    try:
        data = request.get_json()
        question = data.get('question', '')
        # استلام الدور من الباك إند C#
        user_role = data.get('user_role', 'Visitor/Parent') 

        if not question:
            return jsonify({"message": "Question is required."}), 400

        # 1. استخلاص السياق
        context = find_relevant_context(question)
        
        # 2. بناء الـ Prompt
        system_prompt = build_system_prompt(user_role, context, question)

        # 3. الاتصال بـ OpenAI (gpt-3.5-turbo)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )

        bot_reply = response.choices[0].message.content
        return jsonify({"message": bot_reply}), 200

    except Exception as e:
        logging.error(f"An error occurred in OpenAI API call: {e}")
        return jsonify({"message": "عذراً، حدث خطأ داخلي أثناء معالجة الطلب في خادم البوت."}), 500

#if __name__ == '__main__':
#    app.run(host='0.0.0.0', port=5001, debug=True)