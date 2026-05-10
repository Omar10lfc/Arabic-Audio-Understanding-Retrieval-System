"""
generate_test_audio.py — Synthesize a ~3-minute Arabic AI lecture for demo testing.

The text mixes Arabic with English technical terms the way a real Arab CS
lecturer would code-switch (AI, Machine Learning, Transformer, GPU, BERT, etc.).
This stresses Whisper on the Arabic/Latin boundary, which is the more realistic
test case for the pipeline.

Install once:
    pip install edge-tts

Run:
    python generate_test_audio.py

Output:
    arabic_lecture_sample.mp3   (~3 min, MSA, neural male voice)
"""

import asyncio
import edge_tts


# ~500 words of MSA with natural English code-switching on technical terms.
LECTURE = """
السلام عليكم ورحمة الله وبركاته. أهلًا بكم في محاضرة اليوم. سنتحدث عن مجال
الذكاء الاصطناعي، أو ما يُعرف بـ Artificial Intelligence، وكيف يغير حياتنا
اليومية بشكل جذري.

الـ AI ينقسم إلى عدة فروع رئيسية. أولها Machine Learning، أو تعلم الآلة،
وهو الفرع الذي يهتم بتطوير خوارزميات قادرة على التعلم من البيانات دون
الحاجة إلى برمجة صريحة. ثم لدينا Deep Learning، أو التعلم العميق، الذي
يعتمد على Neural Networks ذات الطبقات المتعددة، ويتطلب عادةً قوة حسابية
عالية يوفرها الـ GPU.

في الـ Deep Learning، نستخدم خوارزمية شهيرة تسمى Backpropagation لتدريب
الشبكة. تبدأ الشبكة بأوزان عشوائية، ثم نمرر البيانات عبرها، ونحسب الـ Loss،
وبعد ذلك نُحدّث الأوزان باستخدام Gradient Descent. هذه العملية تتكرر آلاف
المرات حتى يصل النموذج إلى أداء جيد.

من أهم التطورات الحديثة في هذا المجال ظهور معمارية الـ Transformer عام
ألفين وسبعة عشر، والتي قدمتها شركة Google في ورقة بحثية شهيرة بعنوان
Attention Is All You Need. هذه المعمارية أحدثت ثورة في معالجة اللغة
الطبيعية، أو الـ Natural Language Processing.

من تطبيقات الـ Transformer نماذج مثل BERT و GPT و T5. وفي مجال اللغة
العربية، لدينا نماذج متخصصة مثل AraBERT و AraBART، وهي مدربة على مليارات
الكلمات العربية. وأيضًا في مجال التعرف على الكلام، نموذج Whisper من شركة
OpenAI يُعتبر من أقوى النماذج المتاحة، ويدعم اللغة العربية بكفاءة عالية.

دعونا نتحدث عن خط الأنابيب، أو الـ Pipeline، الذي بُني عليه هذا المشروع.
نبدأ بمرحلة الـ Speech to Text، حيث نستخدم Whisper لتحويل الصوت إلى نص.
ثم تأتي مرحلة الـ Embedding، حيث نستخدم نموذج CAMeL-BERT لتمثيل النص
كمتجهات عددية. بعد ذلك نبني فهرس بحث باستخدام مكتبة FAISS، وهي مكتبة
طورتها شركة Meta لإجراء البحث الدلالي بسرعة عالية.

أخيرًا، في مرحلة التلخيص، نستخدم نموذج AraBART الذي قمنا بعمل Fine-Tuning
له على مجموعة بيانات XL-Sum العربية. النتائج كانت ممتازة، حيث حصلنا على
أداء يضاهي أحدث النماذج المنشورة في الأبحاث العلمية.

لكن مع كل هذه الإمكانيات، هناك تحديات حقيقية. أولها قضية الـ Hallucination،
حيث قد يولد النموذج معلومات غير صحيحة بثقة عالية. ثانيًا، قضية الـ Bias،
أو التحيز، الذي قد ينتقل من بيانات التدريب إلى النموذج. وثالثًا، تكلفة
الـ Compute العالية، فتدريب نموذج لغوي كبير قد يحتاج إلى مئات الـ GPUs
ويستغرق أسابيع.

في الختام، الـ AI ثورة حقيقية ستعيد تشكيل العالم في السنوات القادمة. مهم
جدًا أن نواكب هذه التطورات، وأن نتعلم استخدام أدوات مثل PyTorch و
TensorFlow و Hugging Face لبناء حلول عربية تخدم مجتمعاتنا.
شكرًا لكم على حسن الاستماع.
"""

# MSA male voice. The voice will pronounce English terms with an Arabic
# accent, which is exactly how real Arab lecturers code-switch.
# Alternatives:
#   ar-EG-ShakirNeural  (Egyptian male, warmer)
#   ar-SY-LaithNeural   (Levantine male)
VOICE = "ar-SA-HamedNeural"
RATE  = "-5%"                 # slightly slower → easier ASR target
OUT   = "arabic_lecture_sample.mp3"


async def main():
    print(f"Synthesizing {len(LECTURE.split())} words with {VOICE} ...")
    communicate = edge_tts.Communicate(LECTURE.strip(), VOICE, rate=RATE)
    await communicate.save(OUT)
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
