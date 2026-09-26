"""Gemini Multilingual Conversational Weather Guide for WeatherGPT.

Provides natural, friendly weather guidance for all people (daily routines, clothing,
travel plans, outdoor activities, umbrella alerts, and 7-day future predictions)
supporting ALL Indian Languages:
- Hindi (हिन्दी)
- Tamil (தமிழ்)
- Telugu (తెలుగు)
- Kannada (ಕನ್ನಡ)
- Malayalam (മലയാളം)
- Bengali (বাংলা)
- Marathi (मराठी)
- Gujarati (ગુજરાતી)
- Punjabi (ਪੰਜਾਬੀ)
- Odia (ଓଡ଼ିଆ)
- Assamese (অসমীয়া)
- Urdu (اردو)
- English (India / Global)
"""

import base64
import os
import io
import wave
import struct
import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from app.config import settings
from app.services.tts_service import tts_service

LANGUAGE_CONFIG = {
    "en": {"name": "English", "native": "English (India)", "voice": "Puck", "speech_locale": "en-IN"},
    "hi": {"name": "Hindi", "native": "हिन्दी (Hindi)", "voice": "Puck", "speech_locale": "hi-IN"},
    "ta": {"name": "Tamil", "native": "தமிழ் (Tamil)", "voice": "Aoede", "speech_locale": "ta-IN"},
    "te": {"name": "Telugu", "native": "తెలుగు (Telugu)", "voice": "Aoede", "speech_locale": "te-IN"},
    "kn": {"name": "Kannada", "native": "ಕನ್ನಡ (Kannada)", "voice": "Aoede", "speech_locale": "kn-IN"},
    "ml": {"name": "Malayalam", "native": "മലയാളം (Malayalam)", "voice": "Aoede", "speech_locale": "ml-IN"},
    "bn": {"name": "Bengali", "native": "বাংলা (Bengali)", "voice": "Aoede", "speech_locale": "bn-IN"},
    "mr": {"name": "Marathi", "native": "मराठी (Marathi)", "voice": "Puck", "speech_locale": "mr-IN"},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી (Gujarati)", "voice": "Aoede", "speech_locale": "gu-IN"},
    "pa": {"name": "Punjabi", "native": "ਪੰਜਾਬੀ (Punjabi)", "voice": "Puck", "speech_locale": "pa-IN"},
    "or": {"name": "Odia", "native": "ଓଡ଼ିଆ (Odia)", "voice": "Aoede", "speech_locale": "or-IN"},
    "as": {"name": "Assamese", "native": "অসমীয়া (Assamese)", "voice": "Aoede", "speech_locale": "as-IN"},
    "ur": {"name": "Urdu", "native": "اردو (Urdu)", "voice": "Puck", "speech_locale": "ur-IN"},
}

class LLMService:
    def __init__(self):
        self.default_api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self._cached_clients: Dict[str, Any] = {}

    def _get_client(self, api_key_override: Optional[str] = None):
        """Get or initialize Google GenAI Client with given or default API key."""
        active_key = api_key_override or self.default_api_key or os.getenv("GEMINI_API_KEY", "")
        if not active_key:
            return None

        if active_key in self._cached_clients:
            return self._cached_clients[active_key]

        try:
            from google import genai
            client = genai.Client(api_key=active_key)
            self._cached_clients[active_key] = client
            return client
        except Exception as e:
            print(f"[LLMService] Failed to initialize Gemini Client: {e}")
            return None

    async def chat_guide(
        self,
        user_message: str,
        weather_metrics: Dict[str, Any],
        daily_forecast: List[Dict[str, Any]],
        location_name: str = "your area",
        language_code: str = "en",
        chat_history: Optional[List[Dict[str, str]]] = None,
        api_key_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Conversational weather guide for all people across all Indian languages.
        
        Responds to questions about rain, clothing, travel, outdoor plans,
        and weekly outlook in an articulate, clear, and friendly spoken tone.
        """
        lang_info = LANGUAGE_CONFIG.get(language_code, LANGUAGE_CONFIG["en"])
        lang_name = lang_info["name"]
        native_name = lang_info["native"]

        client = self._get_client(api_key_override)
        if client:
            try:
                return await self._call_gemini_chat(
                    client=client,
                    user_message=user_message,
                    weather_metrics=weather_metrics,
                    daily_forecast=daily_forecast,
                    location_name=location_name,
                    lang_info=lang_info,
                    lang_code=language_code,
                    chat_history=chat_history or [],
                )
            except Exception as e:
                print(f"[LLMService] Gemini generation error: {e}. Using intelligent fallback.")

        # Fallback to intelligent rule-based responses if API key is not present or offline
        return await self._generate_fallback_response(
            user_message=user_message,
            weather_metrics=weather_metrics,
            daily_forecast=daily_forecast,
            location_name=location_name,
            lang_code=language_code,
        )

    async def _call_gemini_chat(
        self,
        client: Any,
        user_message: str,
        weather_metrics: Dict[str, Any],
        daily_forecast: List[Dict[str, Any]],
        location_name: str,
        lang_info: Dict[str, str],
        lang_code: str,
        chat_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        from google.genai import types

        lang_name = lang_info["name"]
        native_name = lang_info["native"]

        # Calculate live time-of-day context for deeply relatable, timely answers
        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            time_of_day = "Morning"
        elif 12 <= hour < 16:
            time_of_day = "Afternoon"
        elif 16 <= hour < 20:
            time_of_day = "Evening"
        else:
            time_of_day = "Night"
        current_time_str = now.strftime("%I:%M %p")
        current_date_str = now.strftime("%A, %d %B %Y")

        # Prepare 7-day compact summary for context
        forecast_summary_lines = []
        for d in daily_forecast[:7]:
            forecast_summary_lines.append(
                f"- {d.get('day_name')} ({d.get('date')}): {d.get('weather_desc')}, "
                f"Max: {d.get('temp_max')}°C, Min: {d.get('temp_min')}°C, "
                f"Rain chance: {d.get('precipitation_probability')}%, Wind: {d.get('wind_speed_max')} km/h"
            )
        forecast_context = "\n".join(forecast_summary_lines)

        # 1. Format multi-turn conversation history for context continuity
        history_text = ""
        if chat_history:
            history_lines = []
            for turn in chat_history[-6:]:
                role = "User" if turn.get("role") == "user" else "WeatherGPT"
                content = (turn.get("content") or "").strip()
                if content:
                    history_lines.append(f"{role}: {content}")
            if history_lines:
                history_text = "Recent Conversation Context:\n" + "\n".join(history_lines) + "\n\n"

        system_instruction = (
            f"You are WeatherGPT, a friendly, ultra-concise AI Weather Companion talking with people in {location_name}. "
            f"You speak out loud in {lang_name} ({native_name}).\n\n"
            f"STRICT SPOKEN CONVERSATION RULES:\n"
            f"1. SHORT & CRISP: Answer in strictly 1 to 2 short sentences (maximum 20 to 30 words total). Never write lengthy paragraphs or elaborate essays.\n"
            f"2. DIRECT VERDICT FIRST: Directly give the clear verdict first (e.g., 'Yes, you can dry clothes today' or 'Carry an umbrella, rain is likely' or 'Great day for outdoor sports').\n"
            f"3. 1 LIVE WEATHER FACT: Back your answer with just one live metric fact (e.g., 'Rain probability is only 15%' or 'Temperature is 32°C with high humidity').\n"
            f"4. TIME-OF-DAY RELEVANCE: It is currently {time_of_day} ({current_time_str}). Make it timely for this moment.\n"
            f"5. 100% PURE NATIVE SCRIPT: Write ENTIRELY in {lang_name} ({native_name}) script (except if English is selected). Never mix English script.\n"
            f"6. CLEAN SPOKEN DICTION: Do NOT include asterisks (*), hashtags (#), bullet points, markdown bolding, or emojis so the voice synthesizer speaks aloud seamlessly."
        )

        prompt = (
            f"Current Time: {time_of_day} ({current_time_str})\n"
            f"Live Weather in {location_name}: {weather_metrics.get('temperature_2m')}°C ({weather_metrics.get('weather_desc')}), "
            f"Rain Chance: {weather_metrics.get('daily_rain_probability')}%, Wind: {weather_metrics.get('wind_speed_10m')} km/h, "
            f"Humidity: {weather_metrics.get('relative_humidity_2m')}%\n\n"
            f"{history_text}"
            f"User's Question: '{user_message}'\n\n"
            f"Give a SHORT, direct 1-2 sentence response (under 25 words) in {lang_name} ({native_name}) for voice readout:"
        )

        text_output = ""

        config_text = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.65,
        )

        # Determine candidate models dynamically and prioritize active 2026 models
        models_to_try = []
        if self.model_name:
            models_to_try.append(self.model_name)
        if settings.GEMINI_MODEL and settings.GEMINI_MODEL not in models_to_try:
            models_to_try.append(settings.GEMINI_MODEL)

        # Dynamically query active models from Google for this client
        try:
            for m in client.models.list():
                clean_id = (getattr(m, 'name', '') or '').replace('models/', '')
                actions = getattr(m, 'supported_actions', []) or []
                if not actions or 'generateContent' in actions:
                    if clean_id and clean_id not in models_to_try:
                        models_to_try.append(clean_id)
        except Exception as e_list:
            print(f"[LLMService] Note: client.models.list notice: {e_list}")

        # Current 2026 fallback candidates if dynamic list is unavailable
        fallback_candidates = [
            "gemini-2.5-flash",
            "gemini-3.6-flash",
            "gemini-2.5-pro",
            "gemini-3.5-flash",
            "gemini-3.7-flash",
        ]
        for fc in fallback_candidates:
            if fc not in models_to_try:
                models_to_try.append(fc)

        # Prioritize flash models, deprioritize retired 2.0-flash-lite
        def _model_priority(name: str):
            n = name.lower()
            if "lite" in n and "2.0" in n:
                return 99  # Deprecated
            if "flash" in n and "lite" not in n:
                return 0
            if "flash" in n:
                return 1
            if "pro" in n:
                return 2
            return 3

        models_to_try.sort(key=_model_priority)

        for model_candidate in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_candidate,
                    contents=prompt,
                    config=config_text,
                )
                text_output = (response.text or "").strip()
                if text_output:
                    self.model_name = model_candidate  # Cache working model for future calls
                    settings.GEMINI_MODEL = model_candidate
                    break
            except Exception as e:
                err_str = str(e).lower()
                print(f"[LLMService] Gemini notice with '{model_candidate}': {e}")
                if "api_key_invalid" in err_str or "invalid api key" in err_str or "api_key not valid" in err_str:
                    print(f"[LLMService] Invalid Gemini API key. Halting candidate loop.")
                    break
                continue

        if not text_output:
            return await self._generate_fallback_response(
                user_message, weather_metrics, daily_forecast, location_name, lang_code
            )

        # Clean text from markdown symbols and emojis for clean voice synthesis
        clean_text = self._clean_text_for_speech(text_output)

        # Synthesize real native spoken audio in the respected Indian language
        audio_base64, audio_mime = await tts_service.synthesize_speech(clean_text, lang_code)

        if not audio_base64:
            audio_base64, audio_mime = self._generate_chime_audio()

        return {
            "response_text": clean_text,
            "language": lang_code,
            "language_name": lang_name,
            "speech_locale": lang_info.get("speech_locale", "en-IN"),
            "audio_base64": audio_base64,
            "audio_mime_type": audio_mime,
            "source": f"gemini_{self.model_name.replace('-', '_')}",
        }

    def _clean_text_for_speech(self, text: str) -> str:
        """Strip markdown markers, asterisks, hashtags, bullets, and emojis for smooth speech synthesis."""
        clean = re.sub(r'[*#_`~>]', '', text)
        clean = re.sub(r'\[.*?\]\(.*?\)', '', clean)
        clean = re.sub(r'[\U00010000-\U0010ffff]', '', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    def _detect_intent(self, user_message: str) -> str:
        """Detect user's specific conversational activity or question intent across all Indian languages."""
        q = user_message.lower().strip()

        # 1. Gratitude / Farewell
        if any(k in q for k in [
            "thank", "thx", "dhanyawad", "shukriya", "nandri", "bye", "goodbye",
            "धन्यवाद", "शुक्रिया", "நன்றி", "ధన్యవాదాలు", "ಧನ್ಯವಾದ",
            "നന്ദി", "ধন্যবাদ", "आभार", "આભાર", "ਧੰਨਵਾਦ", "ଧନ୍ୟବାଦ", "ধন্যবাদ"
        ]):
            return "thanks"

        # 2. Greetings
        if any(k in q for k in [
            "hi", "hello", "hey", "good morning", "good evening", "good afternoon", "how are you",
            "नमस्ते", "नमस्कार", "हैलो", "सुप्रभात", "शुभ संध्या", "कैसे हो", "हाल चाल",
            "வணக்கம்", "காலை வணக்கம்", "மாலை வணக்கம்", "எப்படி இருக்கீங்க", "ஹலோ",
            "నమస్కారం", "హలో", "శుభోదయం", "ఎలా ఉన్నారు",
            "ನಮಸ್ಕಾರ", "ಶುಭೋದಯ", "ಹೇಗಿದ್ದೀರಿ", "ಹಲೋ",
            "നമസ്കാരം", "സുപ്രഭാതം", "ഹലോ", "എങ്ങനെയുണ്ട്",
            "নমস্কার", "হ্যালো", "সুপ্রভাত", "শুভ সকাল", "কেমন আছেন",
            "नमस्कार", "शुभ प्रभात", "कसे आहात", "हॅलो",
            "નમસ્તે", "નમસ્કાર", "સુપ્રભાત", "કેમ છો", "હેલો",
            "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ", "ਨਮਸਤੇ", "ਕਿਵੇਂ ਹੋ", "ਹੈਲੋ",
            "ନମସ୍କାର", "ଶୁଭ ସକାଳ", "କେମିତି ଅଛନ୍ତି", "ହେଲୋ",
            "নমস্কাৰ", "শুভ প্ৰভাত", "কেনেকুৱা আছে",
            "سلام", "آداب", "صبح بخیر", "کیسے ہیں", "ہیلو"
        ]):
            return "greeting"

        # 3. Laundry / Drying Clothes
        if (("dry" in q or "drying" in q or "hang" in q) and ("cloth" in q or "laundry" in q or "outside" in q)) or \
           ("wash" in q and "cloth" in q) or any(k in q for k in [
            "dry cloth", "drying", "laundry", "wash cloth", "hang cloth", "dry outside", "clothes dry",
            "कपड़े सुखा", "कपड़े धो", "धोना", "सुखाना", "कपड़े बाहर",
            "துணி காய", "துவைக்க", "துணி", "உலர்த்த",
            "బట్టలు ఆరబెట్ట", "బట్టలు ఉతక", "బట్టలు",
            "ಬಟ್ಟೆ ಒಣಗಿಸ", "ಬಟ್ಟೆ ತೊಳೆಯ", "ಬಟ್ಟೆಗಳು", "ಬಟ್ಟೆ",
            "തുണി ഉണക്കാൻ", "തുണി കഴുകാൻ", "തുണികൾ", "തുണി",
            "জামা কাপড় শুকানো", "কাপড় কাঁচা", "শুকোতে", "কাপড়",
            "कपडे वाळवणे", "कपडे धुणे", "वाळत",
            "કપડાં સૂકવ", "કપડાં ધોવા", "સૂકવણી", "સૂકવ", "સુકવ",
            "ਕੱਪੜੇ ਸੁਕਾਉਣੇ", "ਕੱਪੜੇ ਧੋਣੇ",
            "ଲୁଗା ଶୁଖାଇବା", "ଲୁଗା ଧୋଇବା",
            "কাপোৰ শুকুওৱা", "কাপোৰ ধোৱা",
            "کپڑے سکھانا", "کپڑے دھونا"
        ]):
            return "laundry"

        # 4. Vehicle / Car / Bike Wash
        if (("wash" in q or "clean" in q) and any(v in q for v in ["car", "bike", "vehicle", "scooter", "auto"])) or any(k in q for k in [
            "wash car", "wash bike", "vehicle", "cleaning car", "clean bike", "wash vehicle",
            "गाड़ी धो", "कार धो", "बाइक धो",
            "கார் கழுவ", "வண்டி கழுவ", "வாகனம்",
            "కారు కడగ", "బండి కడగ", "వాహనం",
            "ಕಾರು ತೊಳೆಯ", "ಬೈಕ್ ತೊಳೆಯ", "ವಾಹನ",
            "കാർ കഴുകാൻ", "വണ്ടി കഴുകാൻ",
            "গাড়ি ধোয়া", "বাইক ধোয়া",
            "गाडी धुणे", "कार धुणे", "गाडी धु",
            "ગાડી ધોવી", "કાર ધોવી", "ગાડી ધો",
            "ਗੱਡੀ ਧੋਣੀ", "ਕਾਰ ਧੋਣੀ", "ਗੱਡੀ ਧੋ",
            "ଗାଡ଼ି ଧୋଇବା", "ଗାଡ଼ି ଧୋ",
            "গাড়ী ধোৱা", "গাড়ী ধো",
            "گاڑی دھونا", "کار دھونا"
        ]):
            return "car_wash"

        # 5. Night / Sleep / Cold Tonight
        if any(k in q for k in [
            "night", "tonight", "sleep", "cold tonight", "blanket", "ac needed", "fan", "chilly night",
            "रात", "ठंड", "सर्दी", "सोने", "कंबल", "एसी",
            "இரவு", "குளிர்", "இரவில்", "தூங்க",
            "రాత్రి", "చలి", "పడుకో",
            "ರಾತ್ರಿ", "ಚಳಿ", "ನಿದ್ರೆ",
            "രാത്രി", "തണുപ്പ്", "ഉറങ്ങാൻ",
            "রাত", "ঠান্ডা", "ঘুম",
            "रात्र", "थंडी", "झोप",
            "રાત", "ઠંડી", "ઊંઘ",
            "ਰਾਤ", "ਠੰਢ", "ਨੀਂਦ",
            "ରାତି", "ଥଣ୍ଡା", "ଶୋଇବା",
            "ৰাতি", "ঠাণ্ডা", "টোপনি",
            "رات", "سردی", "ٹھنڈ", "نیند"
        ]):
            return "night_sleep"

        # 6. Humidity / Sweat / Sticky
        if any(k in q for k in [
            "humid", "humidity", "sweat", "sticky", "muggy", "moisture",
            "उमस", "पसीना", "चिपचिपा",
            "ஈரப்பதம்", "வியர்வை", "புழுக்கம்",
            "ఉక్కపోత", "చెమట", "తేమ",
            "ಆರ್ದ್ರತೆ", "ಬೆವರು", "ಸೆಕೆ",
            "ഈർപ്പം", "വിയർപ്പ്", "ചൂട്",
            "আর্দ্রতা", "ঘাম", "ভ্যাপসা",
            "दमट", "घाम", "उकाडा",
            "ભેજ", "પરસેવો", "બફારો",
            "ਨਮੀ", "ਮੁੜ੍ਹਕਾ",
            "ଆର୍ଦ୍ରତା", "ଝାଳ",
            "আৰ্দ্ৰতা", "ঘাম",
            "نمی", "پسینہ", "حبس"
        ]):
            return "humidity_sweat"

        # 7. Wind / Breeze / Gale / Storm
        if any(k in q for k in [
            "wind", "windy", "breeze", "storm", "cyclone", "gust", "blow",
            "हवा", "आंधी", "तूफान", "झोंका",
            "காற்று", "புயல்", "தென்றல்",
            "గాలి", "తుఫాను", "గాలులు",
            "ಗಾಳಿ", "ಬಿರುಗಾಳಿ", "ಚಂಡಮಾರುತ",
            "കാറ്റ്", "ചുഴലിക്കാറ്റ്",
            "বাতাস", "ঝড়", "ঘূর্ণিঝড়",
            "वारा", "वादळ", "झुळूक",
            "પવન", "વાવાઝોડું",
            "ਹਵਾ", "ਤੂਫ਼ਾਨ",
            "ପବନ", "ଝଡ଼",
            "বতাহ", "ধুমুহা",
            "ہوا", "طوفان", "آندھی"
        ]):
            return "wind_storm"

        # 8. Rain / Umbrella / Precipitation
        if any(k in q for k in [
            "rain", "umbrella", "wet", "shower", "pour", "drizzle",
            "बारिश", "छाता", "बरसात", "बूंदाबांदी",
            "மழை", "குடை", "தூறல்",
            "వర్షం", "గొడుగు", "చినుకులు",
            "ಮಳೆ", "ಛತ್ರಿ", "ಹನಿಮಳೆ",
            "മഴ", "കുട", "തുള്ളിക്കൊരുക്കം",
            "বৃষ্টি", "ছাতা", "বৃষ্টিপাত",
            "पाऊस", "छत्री", "शिडकावा",
            "વરસાદ", "છત્રી", "છાંટણા",
            "ਮੀਂਹ", "ਛਤਰੀ", "ਬਰਸਾਤ",
            "ବର୍ଷା", "ଛତା",
            "বৰষুণ", "ছাতি",
            "بارش", "چھتری", "برسات"
        ]):
            return "rain_umbrella"

        # 9. Outdoor Activities / Sports / Running / Cricket
        if any(k in q for k in [
            "run", "jog", "walk", "sport", "cricket", "play", "outdoor", "cycling", "exercise",
            "खेल", "दौड़", "टहलना", "क्रिकेट", "कसरत",
            "விளையாட்டு", "நடைப்பயிற்சி", "ஓட்டம்", "கிரிக்கெட்", "உடற்பயிற்சி",
            "ఆట", "నడక", "పరుగు", "క్రికెట్", "వ్యాయామం",
            "ಆಟ", "ನಡಿಗೆ", "ಓಟ", "ಕ್ರಿಕೆಟ್", "ವ್ಯಾಯಾಮ",
            "കളി", "നടത്തം", "ഓട്ടം", "ക്രിക്കറ്റ്", "വ്യായാമം",
            "খেলা", "হাঁটা", "দৌড়ানো", "ক্রিকেট", "ব্যায়াম",
            "खेळ", "फिरणे", "धावणे", "क्रिकेट", "व्यायाम",
            "રમત", "ચાલવું", "દોડવું", "ક્રિકેટ", "કસરત",
            "ਖੇਡ", "ਸੈਰ", "ਦੌੜ", "ਕ੍ਰਿਕਟ",
            "ଖେଳ", "ଚାଲିବା", "ଦୌଡ଼ିବା", "କ୍ରିକେଟ",
            "খেল", "খোজ কঢ়া", "দৌৰ", "ক্ৰিকেট",
            "کھیل", "سیر", "دوڑ", "کرکٹ", "ورزش"
        ]):
            return "walk_jog_sports"

        # 10. Heat / Sun / Attire / Clothing
        if any(k in q for k in [
            "wear", "cloth", "dress", "jacket", "hot", "heat", "sunny", "sun", "warm",
            "कपड़े", "गर्मी", "धूप", "पहनें",
            "ஆடை", "உடை", "வெயில்", "சூடு", "அணிய",
            "దుస్తులు", "ఎండ", "వేడి", "ధరించ",
            "ಬಟ್ಟೆ", "ಬಿಸಿಲು", "ಶಾಖ", "ಧರಿಸಲು",
            "വസ്ത്രം", "ചൂട്", "വെയിൽ", "ധരിക്കാൻ",
            "পোশাক", "গরম", "রোদ", "পরব",
            "कपडे", "उष्णता", "ऊन", "घालावे",
            "કપડાં", "ગરમી", "તડકો", "પહેરવા",
            "ਕੱਪੜੇ", "ਗਰਮੀ", "ਧੁੱਪ", "ਪਾਉਣੇ",
            "ପୋଷାକ", "ଖରା", "ଗରମ", "ପିନ୍ଧିବା",
            "কাপোৰ", "গৰম", "ৰ'দ",
            "لباس", "گرمی", "دھوپ", "پہننا"
        ]):
            return "heat_clothing"

        # 11. Tomorrow / Weekend / Future
        if any(k in q for k in [
            "tomorrow", "weekend", "saturday", "sunday", "future", "week", "next day", "upcoming",
            "कल", "सप्ताहांत", "रविवार", "शनिवार", "आने वाले", "अगले",
            "நாளை", "வார இறுதி", "ஞாயிறு", "சனிக்கிழமை", "அடுத்த",
            "రేపు", "వారం", "ఆదివారం", "శనివారం", "రాబోయే",
            "ನಾಳೆ", "ವಾರಾಂತ್ಯ", "ಭಾನುವಾರ", "ಮುಂದಿನ",
            "നാളെ", "വാരാന്ത്യം", "ഞായറാഴ്ച", "അടുത്ത",
            "কাল", "আগামীকাল", "সপ্তাহান্ত", "রবিবার", "পরবর্তী",
            "उद्या", "आठवडा", "रविवार", "पुढील",
            "આવતીકાલે", "વીકેન્ડ", "રવિવાર", "આગામી",
            "ਭਲਕੇ", "ਕੱਲ੍ਹ", "ਹਫ਼ਤਾਵਾਰੀ", "ਐਤਵਾਰ",
            "ଆସନ୍ତାକାଲି", "ସପ୍ତାହ", "ରବିବାର",
            "কাইলৈ", "সপ্তাহ", "দেওবাৰ",
            "کل", "ہفتہ", "اتوار", "آئندہ"
        ]):
            return "tomorrow_future"

        return "general"

    async def _generate_fallback_response(
        self,
        user_message: str,
        weather_metrics: Dict[str, Any],
        daily_forecast: List[Dict[str, Any]],
        location_name: str,
        lang_code: str,
    ) -> Dict[str, Any]:
        """Highly relatable, natural spoken responses across all 13 Indian languages when running offline."""
        temp = round(float(weather_metrics.get("temperature_2m", 28.0)), 1)
        feels_like = round(float(weather_metrics.get("apparent_temperature", temp)), 1)
        rain_prob = int(weather_metrics.get("daily_rain_probability", 20))
        precip = float(weather_metrics.get("precipitation", 0.0))
        wind = round(float(weather_metrics.get("wind_speed_10m", 12.0)), 1)
        rh = int(weather_metrics.get("relative_humidity_2m", 65))
        desc = weather_metrics.get("weather_desc", "Clear")
        min_temp = round(float(weather_metrics.get("daily_min_temp", temp - 4)), 1)
        max_temp = round(float(weather_metrics.get("daily_max_temp", temp + 3)), 1)

        intent = self._detect_intent(user_message)

        responses = {
            "en": {
                "greeting": f"Hello! The weather in {location_name} is currently {desc} at {temp}°C. How can I help guide your day?",
                "thanks": f"You are very welcome! If you have any more questions about the weather, feel free to ask. Have a fantastic day!",
                "laundry": f"Yes, it is a great day to dry clothes outside! With rain chance at only {rain_prob}% and pleasant warmth, your laundry will dry quickly." if (rain_prob < 40 and precip == 0) else f"I recommend drying clothes indoors today, as there is a {rain_prob}% chance of rain in {location_name}.",
                "car_wash": f"Yes, it is a perfect day to wash your car! The weather will stay dry with only {rain_prob}% rain chance." if rain_prob < 40 else f"You might want to hold off on washing your vehicle today, as rain is likely ({rain_prob}% chance) and could muddy it up again.",
                "night_sleep": f"Tonight in {location_name}, temperatures will dip to a comfortable {min_temp}°C with {rh}% humidity. A restful night is expected.",
                "humidity_sweat": f"Relative humidity is {rh}%, so it may feel somewhat sticky or muggy. Stay hydrated and in well-ventilated areas.",
                "wind_storm": f"Wind speed is currently {wind} km/h with pleasant breezes in {location_name}." if wind < 25 else f"It is quite windy in {location_name} with gusts around {wind} km/h. Take care if outdoors.",
                "rain_umbrella": f"Yes, rain is likely in {location_name} today with a {rain_prob}% chance. Please carry an umbrella when stepping out." if (rain_prob >= 40 or precip > 0.1) else f"Rain is unlikely today in {location_name} with only a {rain_prob}% chance. You can comfortably step out without an umbrella.",
                "walk_jog_sports": f"The weather is very pleasant with gentle breezes of {wind} km/h, ideal for walking, running, or outdoor sports." if (rain_prob < 40 and wind < 25) else f"Outdoor sports or running are not recommended right now due to rain risk of {rain_prob}% and gusty conditions.",
                "heat_clothing": f"With temperatures around {temp}°C (feels like {feels_like}°C), wear light, breathable cotton clothes and stay well hydrated today." if temp >= 31.0 else f"Weather is comfortable today at {temp}°C. Normal everyday clothing is perfect.",
                "tomorrow_future": f"Over the coming days in {location_name}, temperatures will range between {min_temp}°C and {max_temp}°C with rain chance around {rain_prob}%.",
                "general": f"The weather in {location_name} is currently {desc} at {temp}°C with {rh}% humidity. Have a wonderful day!"
            },
            "ta": {
                "greeting": f"வணக்கம்! {location_name} பகுதியில் இன்று வானிலை {desc} மற்றும் {temp} டிகிரி செல்சியஸாக உள்ளது. உங்கள் நாளுக்கான வானிலை வழிகாட்டலில் என்ன உதவ வேண்டும்?",
                "thanks": f"மிக்க மகிழ்ச்சி! வானிலை குறித்து எந்த உதவி தேவைப்பட்டாலும் எப்போது வேண்டுமானாலும் கேளுங்கள். உங்கள் நாள் இனிதாக அமையட்டும்!",
                "laundry": f"தாராளமாக துணிகளை வெளியில் காய வைக்கலாம்! இன்று {location_name} பகுதியில் மழை வாய்ப்பு வெறும் {rain_prob}% மட்டுமே உள்ளதால் துணிகள் விரைவாக காய்ந்துவிடும்." if (rain_prob < 40 and precip == 0) else f"இன்று {location_name} பகுதியில் {rain_prob}% மழை பெய்ய வாய்ப்புள்ளதால் துணிகளை வெளியில் காய வைப்பது நல்லதல்ல. வீட்டிற்குள்ளேயே உலர்த்துவது நல்லது.",
                "car_wash": f"தாராளமாக வண்டி கழுவலாம்! இன்று வானிலை தெளிவாகவும் மழை வாய்ப்பு குறைவாகவும் உள்ளதால் கார் நீண்ட நேரம் பளபளப்பாக இருக்கும்." if rain_prob < 40 else f"இன்று {rain_prob}% மழை வாய்ப்பு உள்ளதால் கார் அல்லது வண்டி கழுவுவதை தவிர்க்கவும். மழை நீரால் மீண்டும் அழுக்காகிவிடும்.",
                "night_sleep": f"இன்று இரவில் {location_name} பகுதியில் குறைந்தபட்ச வெப்பநிலை {min_temp} டிகிரி செல்சியஸ் வரை குறையும். இரவு இதமாகவும் அமைதியாகவும் இருக்கும்.",
                "humidity_sweat": f"தற்போது காற்றில் ஈரப்பதம் {rh}% ஆக உள்ளதால் சற்று புழுக்கமாக அல்லது வியர்வையாக உணரலாம். போதுமான தண்ணீர் குடித்து காற்றோட்டமாக இருங்கள்.",
                "wind_storm": f"தற்போது காற்றின் வேகம் மணிக்கு {wind} கிலோமீட்டராக உள்ளது. மென்மையான இதமான காற்று வீசுகிறது." if wind < 25 else f"காற்றின் வேகம் மணிக்கு {wind} கி.மீ ஆக வீசுவதால் வெளியில் கவனமாக இருக்கவும்.",
                "rain_umbrella": f"ஆம், இன்று {location_name} பகுதியில் {rain_prob}% மழை பெய்ய வாய்ப்புள்ளது. வெளியே செல்லும்போது குடை எடுத்துச் செல்லவும்." if (rain_prob >= 40 or precip > 0.1) else f"இன்று {location_name} பகுதியில் மழைக்கு வாய்ப்பு குறைவு, வெறும் {rain_prob}% மட்டுமே. நீங்கள் குடை இல்லாமல் தைரியமாக செல்லலாம்.",
                "walk_good": f"வெளியில் நடைப்பயிற்சி, ஓட்டம் அல்லது விளையாட்டுகளுக்கு வானிலை மிகவும் இதமாக உள்ளது. காற்றின் வேகம் மணிக்கு {wind} கி.மீ ஆக உள்ளது.",
                "walk_jog_sports": f"வெளியில் நடைப்பயிற்சி, ஓட்டம் அல்லது விளையாட்டுகளுக்கு வானிலை மிகவும் இதமாக உள்ளது. காற்றின் வேகம் மணிக்கு {wind} கி.மீ ஆக உள்ளது." if (rain_prob < 40 and wind < 25) else f"மழை வாய்ப்பு {rain_prob}% மற்றும் காற்று வீசுவதால் இப்போது வெளிப்புற விளையாட்டுகளை தவிர்ப்பது நல்லது.",
                "heat_clothing": f"வெப்பநிலை {temp} டிகிரி செல்சியஸாக உள்ளது (உணரப்படுவது {feels_like}°C). மெல்லிய பருத்தி ஆடைகளை அணிந்து போதுமான தண்ணீர் குடிக்கவும்." if temp >= 31.0 else f"வானிலை {temp} டிகிரி செல்சியஸுடன் இதமாக உள்ளது. சாதாரண வசதியான ஆடைகள் போதுமானது.",
                "tomorrow_future": f"அடுத்த சில நாட்களுக்கு {location_name} பகுதியில் வெப்பநிலை {min_temp} முதல் {max_temp} டிகிரி வரை நிலவும். மழை வாய்ப்பு {rain_prob}%.",
                "general": f"{location_name} பகுதியில் தற்போது வானிலை {desc}, வெப்பநிலை {temp} டிகிரி செல்சியஸ் மற்றும் ஈரப்பதம் {rh}% ஆக உள்ளது. உங்கள் நாள் சிறக்க வாழ்த்துகள்!"
            },
            "hi": {
                "greeting": f"नमस्ते! {location_name} में वर्तमान तापमान {temp} डिग्री सेल्सियस और मौसम {desc} है। आज आपकी क्या मदद करूँ?",
                "thanks": f"बहुत-बहुत धन्यवाद! मौसम से जुड़ी कोई भी जानकारी चाहिए हो तो कभी भी पूछें। आपका दिन मंगलमय हो!",
                "laundry": f"हाँ, आप आराम से कपड़े बाहर सुखा सकते हैं! आज बारिश की संभावना केवल {rain_prob}% है और अच्छी धूप से कपड़े जल्दी सूखेंगे।" if (rain_prob < 40 and precip == 0) else f"आज कपड़े बाहर न सुखाएं तो बेहतर होगा, क्योंकि {rain_prob}% बारिश की संभावना है। कपड़े भीग सकते हैं।",
                "car_wash": f"हाँ, आज गाड़ी धोने के लिए बहुत अच्छा दिन है! मौसम साफ रहेगा और धूप खिली रहेगी।" if rain_prob < 40 else f"आज गाड़ी धोने से बचें, क्योंकि {rain_prob}% बारिश की संभावना होने से गाड़ी फिर से गंदी हो सकती है।",
                "night_sleep": f"आज रात {location_name} में न्यूनतम तापमान {min_temp} डिग्री सेल्सियस तक गिरेगा। रात आरामदायक रहेगी।",
                "humidity_sweat": f"हवा में नमी {rh}% है, जिससे थोड़ी उमस या पसीना महसूस हो सकता है। पंखे या खुली हवा में रहें।",
                "wind_storm": f"हवा की रफ्तार {wind} किलोमीटर प्रति घंटा है। हल्की ठंडी हवा चल रही है।" if wind < 25 else f"हवा की रफ्तार {wind} किमी/घंटा है, बाहर तेज हवा चल रही है।",
                "rain_umbrella": f"हाँ, आज {location_name} में {rain_prob}% बारिश की संभावना है। बाहर निकलते समय छाता जरूर साथ रखें।" if (rain_prob >= 40 or precip > 0.1) else f"आज बारिश की संभावना केवल {rain_prob}% है। आप बिना छाते के आराम से बाहर जा सकते हैं।",
                "walk_jog_sports": f"मौसम बहुत सुहावना है और हल्की हवाएं चल रही हैं। बाहर टहलने, दौड़ने या क्रिकेट खेलने के लिए यह समय बहुत उत्तम है।" if (rain_prob < 40 and wind < 25) else f"बारिश की संभावना {rain_prob}% होने के कारण अभी बाहर खेलने या दौड़ने से बचना बेहतर रहेगा।",
                "heat_clothing": f"तापमान {temp} डिग्री सेल्सियस (महसूस {feels_like}°C) है। हल्के सूती कपड़े पहनें और खूब पानी पिएं।" if temp >= 31.0 else f"मौसम {temp} डिग्री के साथ बहुत आरामदायक है। सामान्य रोजमर्रा के कपड़े उपयुक्त हैं।",
                "tomorrow_future": f"कल और आने वाले दिनों में तापमान {min_temp} से {max_temp} डिग्री सेल्सियस के बीच रहेगा और बारिश की संभावना {rain_prob}% है।",
                "general": f"{location_name} में वर्तमान तापमान {temp} डिग्री सेल्सियस है और मौसम {desc} बना हुआ है। आपका दिन शुभ हो!"
            },
            "te": {
                "greeting": f"నమస్కారం! {location_name}లో ప్రస్తుత ఉష్ణోగ్రత {temp} డిగ్రీలు మరియు వాతావరణం {desc}గా ఉంది. ఈ రోజు మీకు ఏ విధంగా సహాయపడగలను?",
                "thanks": f"చాలా ధన్యవాదాలు! వాతావరణానికి సంబంధించిన ఎలాంటి సందేహాలున్నా ఎప్పుడైనా అడగవచ్చు. మీ రోజు శుభంగా సాగాలి!",
                "laundry": f"అవును, బట్టలు నిరభ్యంతరంగా ఆరబెట్టుకోవచ్చు! వర్షం పడే అవకాశం కేవలం {rain_prob}% మాత్రమే ఉన్నందున బట్టలు త్వరగా ఆరిపోతాయి." if (rain_prob < 40 and precip == 0) else f"ఈ రోజు {rain_prob}% వర్ష సూచన ఉన్నందున బట్టలను బయట ఆరబెట్టకపోవడమే మంచిది. ఇంట్లోనే ఆరబెట్టండి.",
                "car_wash": f"ఖచ్చితంగా కారు లేదా బైక్ కడుక్కోవచ్చు! వాతావరణం పొడిగా ఉండి వాహనం ఎక్కువ సమయం శుభ్రంగా ఉంటుంది." if rain_prob < 40 else f"ఈ రోజు వాహనం కడగకపోవడమే మంచిది, ఎందుకంటే {rain_prob}% వర్షం పడి మళ్లీ మురికి అయ్యే అవకాశం ఉంది.",
                "night_sleep": f"ఈ రాత్రి {location_name}లో కనిష్ట ఉష్ణోగ్రత {min_temp} డిగ్రీలకు చేరుకుంటుంది. రాత్రి వేళ వాతావరణం చల్లగా ఉంటుంది.",
                "humidity_sweat": f"గాలిలో తేమ {rh}% గా ఉంది, అందువల్ల కొద్దిగా ఉక్కపోతగా ఉండవచ్చు. చల్లని ప్రదేశంలో విశ్రాంతి తీసుకోండి.",
                "wind_storm": f"ప్రస్తుతం గాలి వేగం గంటకు {wind} కిలోమీటర్లుగా నమోదైంది. ఆహ్లాదకరమైన గాలులు వీస్తున్నాయి.",
                "rain_umbrella": f"అవును, ఈ రోజు {location_name}లో {rain_prob}% వర్షం పడే అవకాశం ఉంది. బయటకు వెళ్లేటప్పుడు గొడుగు తీసుకెళ్లండి." if (rain_prob >= 40 or precip > 0.1) else f"ఈ రోజు వర్ష సూచన చాలా తక్కువ, కేవలం {rain_prob}% మాత్రమే. గొడుగు అవసరం లేదు.",
                "walk_jog_sports": f"నడకకు, పరుగుకు లేదా బహిరంగ ఆటలకు వాతావరణం చాలా అనుకూలంగా ఉంది. గాలి వేగం గంటకు {wind} కి.మీ." if (rain_prob < 40 and wind < 25) else f"వర్షం మరియు గాలుల వల్ల ప్రస్తుతం బహిరంగ ఆటలు లేదా నడకను వాయిదా వేయడం మంచిది.",
                "heat_clothing": f"ఉష్ణోగ్రత {temp} డిగ్రీలుగా ఉంది. తేలికపాటి కాటన్ దుస్తులు ధరించి ఎక్కువ నీరు త్రాగండి." if temp >= 31.0 else f"వాతావరణం {temp} డిగ్రీలతో ఆహ్లాదకరంగా ఉంది. సాధారణ సౌకర్యవంతమైన దుస్తులు సరిపోతాయి.",
                "tomorrow_future": f"రాబోయే రోజుల్లో ఉష్ణోగ్రతలు {min_temp} నుండి {max_temp} డిగ్రీల మధ్య ఉంటాయి. వర్ష సూచన {rain_prob}%.",
                "general": f"{location_name}లో ప్రస్తుత ఉష్ణోగ్రత {temp} డిగ్రీలు, తేమ {rh}%. మీ రోజు ఆనందంగా గడవాలని కోరుకుంటున్నాము!"
            },
            "kn": {
                "greeting": f"ನಮಸ್ಕಾರ! {location_name} ನಲ್ಲಿ ಪ್ರಸ್ತುತ ತಾಪಮಾನ {temp} ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್ ಮತ್ತು ಹವಾಮಾನ {desc} ಆಗಿದೆ. ಇಂದು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?",
                "thanks": f"ತುಂಬಾ ಧನ್ಯವಾದಗಳು! ಹವಾಮಾನದ ಕುರಿತು ಯಾವುದೇ ಮಾಹಿತಿ ಬೇಕಿದ್ದರೂ ಕೇಳಬಹುದು. ನಿಮ್ಮ ದಿನ ಶುಭವಾಗಿರಲಿ!",
                "laundry": f"ಹೌದು, ಬಟ್ಟೆಗಳನ್ನು ಹೊರಗೆ ಒಣಗಿಸಲು ಉತ್ತಮ ದಿನ! ಮಳೆ ಬರುವ ಸಾಧ್ಯತೆ ಕೇವಲ {rain_prob}% ಮಾತ್ರ ಇದ್ದು, ಬೇಗನೆ ಒಣಗುತ್ತವೆ." if (rain_prob < 40 and precip == 0) else f"ಇಂದು {rain_prob}% ಮಳೆಯಾಗುವ ಸಾಧ್ಯತೆ ಇರುವುದರಿಂದ ಬಟ್ಟೆಗಳನ್ನು ಹೊರಗೆ ಒಣಗಿಸುವುದು ಬೇಡ.",
                "car_wash": f"ಖಂಡಿತವಾಗಿಯೂ ವಾಹನ ತೊಳೆಯಬಹುದು! ಹವಾಮಾನವು ಸ್ವಚ್ಛವಾಗಿದ್ದು ಗಾಡಿ ಹೊಳೆಯುತ್ತದೆ." if rain_prob < 40 else f"ಇಂದು ಗಾಡಿ ತೊಳೆಯುವುದನ್ನು ಮುಂದೂಡಿ, ಮಳೆಯಿಂದಾಗಿ ವಾಹನ ಮತ್ತೆ ಕೊಳೆಯಾಗಬಹುದು.",
                "night_sleep": f"ಇಂದು ರಾತ್ರಿ {location_name} ನಲ್ಲಿ ಕನಿಷ್ಠ ತಾಪಮಾನ {min_temp} ಡಿಗ್ರಿವರೆಗೆ ಇಳಿಯುತ್ತದೆ. ನಿದ್ರೆಗೆ ಹಿತಕರವಾಗಿರುತ್ತದೆ.",
                "humidity_sweat": f"ಆರ್ದ್ರತೆ {rh}% ಇದ್ದು ಸ್ವಲ್ಪ ಸೆಕೆ ಅಥವಾ ಬೆವರು ಅನಿಸಬಹುದು. ಗಾಳಿಯಾಡುವ ಜಾಗದಲ್ಲಿರಿ.",
                "wind_storm": f"ಗಾಳಿಯ ವೇಗ ಗಂಟೆಗೆ {wind} ಕಿ.ಮೀ ಆಗಿದೆ. ತಂಪಾದ ಹಿತಕರ ಗಾಳಿ ಬೀಸುತ್ತಿದೆ.",
                "rain_umbrella": f"ಹೌದು, ಇಂದು {location_name} ನಲ್ಲಿ {rain_prob}% ಮಳೆಯಾಗುವ ಸಾಧ್ಯತೆಯಿದೆ. ಹೊರಹೋಗುವಾಗ ಛತ್ರಿ ಇಟ್ಟುಕೊಳ್ಳಿ." if (rain_prob >= 40 or precip > 0.1) else f"ಇಂದು ಮಳೆಯ ಸಾಧ್ಯತೆ ಕೇವಲ {rain_prob}% ಮಾತ್ರ. ಛತ್ರಿ ಅಗತ್ಯವಿಲ್ಲದೆ ಆರಾಮವಾಗಿ ಹೋಗಬಹುದು.",
                "walk_jog_sports": f"ಹೊರಾಂಗಣ ನಡಿಗೆ, ಓಟ ಅಥವಾ ಆಟಗಳಿಗೆ ಹವಾಮಾನವು ಅತ್ಯಂತ ಆಹ್ಲಾದಕರವಾಗಿದೆ. ಗಾಳಿಯ ವೇಗ {wind} ಕಿ.ಮೀ." if (rain_prob < 40 and wind < 25) else f"ಮಳೆಯ ಸಂಭವ ಇರುವುದರಿಂದ ಹೊರಾಂಗಣ ಕ್ರೀಡೆಗಳನ್ನು ಸದ್ಯಕ್ಕೆ ಮುಂದೂಡುವುದು ಸೂಕ್ತ.",
                "heat_clothing": f"ತಾಪಮಾನ {temp} ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್ ಇದೆ. ಹತ್ತಿ ಬಟ್ಟೆಗಳನ್ನು ಧರಿಸಿ ಮತ್ತು ಸಾಕಷ್ಟು ನೀರು ಕುಡಿಯಿರಿ." if temp >= 31.0 else f"ಹವಾಮಾನವು {temp} ಡಿಗ್ರಿಗಳೊಂದಿಗೆ ಹಿತಕರವಾಗಿದೆ. ಸಾಮಾನ್ಯ ಉಡುಪುಗಳು ಸೂಕ್ತವಾಗಿವೆ.",
                "tomorrow_future": f"ಮುಂದಿನ ದಿನಗಳಲ್ಲಿ ತಾಪಮಾನ {min_temp} ರಿಂದ {max_temp} ಡಿಗ್ರಿಗಳಷ್ಟಿರುತ್ತದೆ. ಮಳೆ ಸಾಧ್ಯತೆ {rain_prob}%.",
                "general": f"{location_name} ನಲ್ಲಿ ಪ್ರಸ್ತುತ ತಾಪಮಾನ {temp} ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್ ಆಗಿದೆ. ನಿಮಗೆ ಶುಭ ದಿನ!"
            },
            "ml": {
                "greeting": f"നമസ്കാരം! {location_name} ൽ ഇപ്പോൾ താപനില {temp} ഡിഗ്രിയും കാലാവസ്ഥ {desc} യുമാണ്. ഇന്ന് ഞാൻ എങ്ങനെ സഹായിക്കണം?",
                "thanks": f"വളരെ നന്ദി! കാലാവസ്ഥയെക്കുറിച്ചുള്ള ഏത് ചോദ്യങ്ങൾക്കും എപ്പോഴും ഇവിടെയുണ്ട്. നല്ലൊരു ദിവസം ആശംസിക്കുന്നു!",
                "laundry": f"തീർച്ചയായും തുണികൾ പുറത്ത് ഉണക്കാൻ ഇടാം! മഴ സാധ്യത {rain_prob}% മാത്രമായതിനാൽ തുണികൾ വേഗം ഉണങ്ങും." if (rain_prob < 40 and precip == 0) else f"ഇന്ന് {rain_prob}% മഴ സാധ്യത ഉള്ളതിനാൽ തുണികൾ പുറത്ത് ഇടുന്നത് ഒഴിവാക്കുക. അകത്ത് ഉണക്കാൻ ശ്രമിക്കുക.",
                "car_wash": f"തീർച്ചയായും വാഹനം കഴുകാം! തെളിഞ്ഞ കാലാവസ്ഥയായതിനാൽ വാഹനം വൃത്തിയായിരിക്കും." if rain_prob < 40 else f"ഇന്ന് വാഹനം കഴുകേണ്ടതില്ല, കാരണം മഴ പെയ്ത് വീണ്ടും ചെളി പുരളാൻ സാധ്യതയുണ്ട്.",
                "night_sleep": f"ഇന്ന് രാത്രി താപനില {min_temp} ഡിഗ്രി വരെ താഴാൻ സാധ്യതയുണ്ട്. സുഖകരമായ ഉറക്കം ലഭിക്കും.",
                "humidity_sweat": f"ഈർപ്പം {rh}% ഉള്ളതിനാൽ ചെറിയ ചൂടും വിയർപ്പും അനുഭവപ്പെടാം. ധാരാളം വെള്ളം കുടിക്കുക.",
                "wind_storm": f"കാറ്റിന്റെ വേഗത മണിക്കൂറിൽ {wind} കിലോമീറ്ററാണ്. നല്ല ഇളം കാറ്റ് വീശുന്നുണ്ട്.",
                "rain_umbrella": f"അതെ, ഇന്ന് {location_name} ൽ {rain_prob}% മഴയ്ക്ക് സാധ്യതയുണ്ട്. പുറത്തിറങ്ങുമ്പോൾ കുട കരുതുക." if (rain_prob >= 40 or precip > 0.1) else f"ഇന്ന് മഴയ്ക്ക് സാധ്യത {rain_prob}% മാത്രമാണ്. കുടയുടെ ആവശ്യമില്ല.",
                "walk_jog_sports": f"നടത്തത്തിനും വ്യായാമത്തിനും കളികൾക്കും ഏറ്റവും അനുയോജ്യമായ സുഖകരമായ കാലാവസ്ഥയാണ്." if (rain_prob < 40 and wind < 25) else f"മഴ സാധ്യതയുള്ളതിനാൽ ഇപ്പോൾ പുറത്തുള്ള വ്യായാമവും കളികളും ഒഴിവാക്കുന്നതാണ് ഉചിതം.",
                "heat_clothing": f"താപനില {temp} ഡിഗ്രി ആയതിനാൽ കോട്ടൺ വസ്ത്രങ്ങൾ ധരിക്കുകയും ധാരാളം വെള്ളം കുടിക്കുകയും ചെയ്യുക." if temp >= 31.0 else f"കാലാവസ്ഥ {temp} ഡിഗ്രിയോടെ സുഖകരമാണ്. സാധാരണ വസ്ത്രങ്ങൾ അനുയോജ്യമാണ്.",
                "tomorrow_future": f"അടുത്ത ദിവസങ്ങളിൽ താപനില {min_temp} മുതൽ {max_temp} ഡിഗ്രി വരെയായിരിക്കും. മഴ സാധ്യത {rain_prob}%.",
                "general": f"{location_name} ൽ നിലവിൽ താപനില {temp} ഡിഗ്രി സെൽഷ്യസ് ആണ്. സന്തോഷകരമായ ദിനം ആശംസിക്കുന്നു!"
            },
            "bn": {
                "greeting": f"নমস্কার! {location_name}-এ বর্তমান তাপমাত্রা {temp} ডিগ্রি সেলসিয়াস এবং আবহাওয়া {desc}। আজ আপনাকে কীভাবে সাহায্য করতে পারি?",
                "thanks": f"অনেক ধন্যবাদ! আবহাওয়া সম্পর্কিত যে কোনো তথ্যের জন্য সবসময় পাশে আছি। দিনটি ভালো কাটুক!",
                "laundry": f"হ্যাঁ, নির্দ্বিধায় জামাকাপড় বাইরে শুকোতে দিতে পারেন! বৃষ্টির সম্ভাবনা মাত্র {rain_prob}%, কাপড় তাড়াতাড়ি শুকিয়ে যাবে।" if (rain_prob < 40 and precip == 0) else f"আজ জামাকাপড় বাইরে না মেলানোই ভালো, কারণ {rain_prob}% বৃষ্টির সম্ভাবনা রয়েছে।",
                "car_wash": f"হ্যাঁ, গাড়ি বা বাইক ধোয়ার জন্য আজ চমৎকার দিন! রোদ ঝলমলে আবহাওয়া থাকবে।" if rain_prob < 40 else f"আজ গাড়ি ধোয়া স্থগিত রাখুন, কারণ বৃষ্টির কারণে গাড়ি আবার কাদায় নোংরা হতে পারে।",
                "night_sleep": f"আজ রাতে {location_name}-এ সর্বনিম্ন তাপমাত্রা {min_temp} ডিগ্রি পর্যন্ত নামবে। রাত আরামদায়ক হবে।",
                "humidity_sweat": f"বাতাসে আর্দ্রতা {rh}% থাকায় একটু ভ্যাপসা গরম বা ঘাম হতে পারে। পর্যাপ্ত জল খান।",
                "wind_storm": f"বাতাসের গতিবেগ ঘণ্টায় {wind} কিলোমিটার। সুন্দর মৃদু হাওয়া বইছে।",
                "rain_umbrella": f"হ্যাঁ, আজ {location_name}-এ {rain_prob}% বৃষ্টির সম্ভাবনা রয়েছে। বের হওয়ার সময় ছাতা সঙ্গে রাখুন।" if (rain_prob >= 40 or precip > 0.1) else f"আজ বৃষ্টির সম্ভাবনা মাত্র {rain_prob}%। ছাতা ছাড়াই নিশ্চিন্তে বাইরে বের হতে পারেন।",
                "walk_jog_sports": f"আবহাওয়া খুবই মনোরম। বাইরে হাঁটা, দৌড়ানো বা ক্রিকেট খেলার জন্য চমৎকার সময়।" if (rain_prob < 40 and wind < 25) else f"বৃষ্টির সম্ভাবনার কারণে এখন বাইরের খেলাধুলা বা দৌড়াদৌড়ি এড়িয়ে চলাই ভালো।",
                "heat_clothing": f"তাপমাত্রা {temp} ডিগ্রি সেলসিয়াস। হালকা সুতির পোশাক পরুন এবং প্রচুর জল পান করুন।" if temp >= 31.0 else f"আবহাওয়া {temp} ডিগ্রির সাথে বেশ আরামদায়ক। স্বাভাবিক পোশাকই উপযুক্ত।",
                "tomorrow_future": f"আগামী দিনগুলিতে তাপমাত্রা {min_temp} থেকে {max_temp} ডিগ্রির মধ্যে থাকবে এবং বৃষ্টির সম্ভাবনা {rain_prob}%.",
                "general": f"{location_name}-এ বর্তমান তাপমাত্রা {temp} ডিগ্রি সেলসিয়াস। আপনার দিনটি শুভ হোক!"
            },
            "mr": {
                "greeting": f"नमस्कार! {location_name} मध्ये सध्या तापमान {temp} अंश सेल्सिअस असून हवामान {desc} आहे. आज मी आपली काय मदत करू शकेन?",
                "thanks": f"मनःपूर्वक धन्यवाद! हवामानाबद्दल कोणतीही माहिती हवी असल्यास नक्की विचारा. आपला दिवस आनंदात जावो!",
                "laundry": f"होय, कपडे बाहेर वाळत घालण्यासाठी छान दिवस आहे! पावसाची शक्यता फक्त {rain_prob}% असल्याने कपडे लवकर वाळतील." if (rain_prob < 40 and precip == 0) else f"आज कपडे बाहेर वाळत घालू नका, कारण {rain_prob}% पावसाची शक्यता आहे. कपडे ओले होऊ शकतात.",
                "car_wash": f"नक्कीच, गाडी धुण्यासाठी आज उत्तम दिवस आहे! हवामान स्वच्छ राहील." if rain_prob < 40 else f"आज गाडी धुणे टाळा, कारण पावसामुळे गाडी पुन्हा खराब होऊ शकते.",
                "night_sleep": f"आज रात्री {location_name} मध्ये तापमान {min_temp} अंशांपर्यंत खाली येईल. रात्रीचे वातावरण सुखद राहील.",
                "humidity_sweat": f"हवेत दमटपणा {rh}% असल्याने थोडा उकाडा जाणवू शकतो. मोकळ्या हवेत राहा.",
                "wind_storm": f"सध्या वाऱ्याचा वेग ताशी {wind} किमी आहे. छान मंद वारे वाहत आहेत.",
                "rain_umbrella": f"होय, आज {location_name} मध्ये {rain_prob}% पावसाची शक्यता आहे. बाहेर जाताना छत्री नक्की सोबत ठेवा." if (rain_prob >= 40 or precip > 0.1) else f"आज पावसाची शक्यता फक्त {rain_prob}% आहे. आपण छत्रीशिवाय आरामात बाहेर जाऊ शकता.",
                "walk_jog_sports": f"फिरण्यासाठी, धावण्यासाठी किंवा खेळांसाठी वातावरण अतिशय आल्हाददायक आहे. वाऱ्याचा वेग ताशी {wind} किमी आहे." if (rain_prob < 40 and wind < 25) else f"पावसामुळे मैदानी खेळ किंवा फिरणे सध्या टाळणे अधिक योग्य राहील.",
                "heat_clothing": f"तापमान {temp} अंश आहे. हलके सुती कपडे वापरा आणि भरपूर पाणी प्या." if temp >= 31.0 else f"हवामान {temp} अंशांसह अतिशय सुखद आहे. नेहमीचे कपडे उत्तम आहेत.",
                "tomorrow_future": f"येत्या दिवसांत तापमान {min_temp} ते {max_temp} अंशांदरम्यान राहील आणि पावसाची शक्यता {rain_prob}% आहे.",
                "general": f"{location_name} मध्ये सध्या तापमान {temp} अंश सेल्सिअस आहे. आपला दिवस उत्तम जावो!"
            },
            "gu": {
                "greeting": f"નમસ્તે! {location_name} માં હાલનું તાપમાન {temp} ડિગ્રી સેલ્સિયસ અને વાતાવરણ {desc} છે. આજે હું આપને શું મદદ કરી શકું?",
                "thanks": f"ખૂબ ખૂબ આભાર! હવામાન સંબંધી કોઈપણ માહિતી માટે સદાય હાજર છું. આપનો દિવસ ખુશનુમા રહે!",
                "laundry": f"હા, કપડાં બહાર સૂકવવા માટે સરસ દિવસ છે! વરસાદની શક્યતા માત્ર {rain_prob}% હોવાથી કપડાં જલ્દી સૂકાઈ જશે." if (rain_prob < 40 and precip == 0) else f"આજે કપડાં બહાર ન સૂકવવા હિતાવહ છે, કારણ કે {rain_prob}% વરસાદની શક્યતા છે.",
                "car_wash": f"ચોક્કસ, આજે ગાડી ધોવા માટે સારો સમય છે! હવામાન સાફ રહેશે." if rain_prob < 40 else f"આજે ગાડી ધોવાનું ટાળો, કારણ કે વરસાદના કારણે ગાડી ફરીથી ગંદી થઈ શકે છે.",
                "night_sleep": f"આજે રાત્રે {location_name} માં લઘુત્તમ તાપમાન {min_temp} ડિગ્રી સુધી જશે. રાત શાંત અને આરામદાયક રહેશે.",
                "humidity_sweat": f"હવામાં ભેજ {rh}% છે, જેથી થોડો બફારો કે પરસેવો થઈ શકે છે.",
                "wind_storm": f"પવનની ગતિ કલાકના {wind} કિમી છે. સરસ ઠંડો પવન ફૂંકાઈ રહ્યો છે.",
                "rain_umbrella": f"હા, આજે {location_name} માં {rain_prob}% વરસાદની શક્યતા છે. બહાર નીકળતી વખતે છત્રી જરૂર સાથે રાખશો." if (rain_prob >= 40 or precip > 0.1) else f"આજે વરસાદની શક્યતા માત્ર {rain_prob}% છે. છત્રી વિના નિશ્ચિંત થઈને જઈ શકો છો.",
                "walk_jog_sports": f"બહાર ચાલવા, દોડવા કે રમતગમત માટે હવામાન ખૂબ જ અનુકૂળ અને ખુશનુમા છે." if (rain_prob < 40 and wind < 25) else f"વરસાદની શક્યતાને લીધે અત્યારે આઉટડોર પ્રવૃત્તિઓ ટાળવી યોગ્ય રહેશે.",
                "heat_clothing": f"તાપમાન {temp} ડિગ્રી છે. સુતરાઉ કપડાં પહેરો અને પૂરતું પાણી પીવો." if temp >= 31.0 else f"હવામાન {temp} ડિગ્રી સાથે ઘણું આરામદાયક છે. સામાન્ય વસ્ત્રો યોગ્ય રહેશે.",
                "tomorrow_future": f"આગામી દિવસોમાં તાપમાન {min_temp} થી {max_temp} ડિગ્રી રહેશે અને વરસાદની શક્યતા {rain_prob}% છે.",
                "general": f"{location_name} માં હાલ તાપમાન {temp} ડિગ્રી સેલ્સિયસ છે. આપનો દિવસ શુભ રહે!"
            },
            "pa": {
                "greeting": f"ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ ਜੀ! {location_name} ਵਿੱਚ ਤਾਪਮਾਨ {temp} ਡਿਗਰੀ ਸੈਲਸੀਅਸ ਅਤੇ ਮੌਸਮ {desc} ਹੈ। ਅੱਜ ਮੈਂ ਤੁਹਾਡੀ ਕੀ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
                "thanks": f"ਬਹੁਤ ਧੰਨਵਾਦ ਜੀ! ਮੌਸਮ ਬਾਰੇ ਕੋਈ ਵੀ ਜਾਣਕਾਰੀ ਚਾਹੀਦੀ ਹੋਵੇ ਤਾਂ ਜ਼ਰੂਰ ਪੁੱਛੋ। ਤੁਹਾਡਾ ਦਿਨ ਚੰਗਾ ਲੰਘੇ!",
                "laundry": f"ਹਾਂ ਜੀ, ਕੱਪੜੇ ਬਾਹਰ ਸੁਕਾਉਣ ਲਈ ਬਹੁਤ ਵਧੀਆ ਦਿਨ ਹੈ! ਮੀਂਹ ਦੀ ਸੰਭਾਵਨਾ ਸਿਰਫ਼ {rain_prob}% ਹੈ ਅਤੇ ਕੱਪੜੇ ਜਲਦੀ ਸੁੱਕਣਗੇ।" if (rain_prob < 40 and precip == 0) else f"ਅੱਜ ਕੱਪੜੇ ਬਾਹਰ ਨਾ ਸੁਕਾਓ ਕਿਉਂਕਿ {rain_prob}% ਮੀਂਹ ਪੈਣ ਦਾ ਖ਼ਤਰਾ ਹੈ।",
                "car_wash": f"ਬਿਲਕੁਲ, ਗੱਡੀ ਧੋਣ ਲਈ ਅੱਜ ਬਹੁਤ ਵਧੀਆ ਦਿਨ ਹੈ! ਮੌਸਮ ਸਾਫ਼ ਰਹੇਗਾ।" if rain_prob < 40 else f"ਅੱਜ ਗੱਡੀ ਨਾ ਧੋਵੋ ਕਿਉਂਕਿ ਮੀਂਹ ਕਾਰਨ ਗੱਡੀ ਮੁੜ ਗੰਦੀ ਹੋ ਸਕਦੀ ਹੈ।",
                "night_sleep": f"ਅੱਜ ਰਾਤ {location_name} ਵਿੱਚ ਤਾਪਮਾਨ {min_temp} ਡਿਗਰੀ ਤੱਕ ਹੇਠਾਂ ਆਵੇਗਾ। ਰਾਤ ਸੁਹਾਵਣੀ ਰਹੇਗੀ।",
                "humidity_sweat": f"ਹਵਾ ਵਿੱਚ ਨਮੀ {rh}% ਹੈ, ਜਿਸ ਕਾਰਨ ਥੋੜ੍ਹੀ ਗਰਮੀ ਮਹਿਸੂਸ ਹੋ ਸਕਦੀ ਹੈ।",
                "wind_storm": f"ਹਵਾ ਦੀ ਗਤੀ {wind} ਕਿਲੋਮੀਟਰ ਪ੍ਰਤੀ ਘੰਟਾ ਹੈ। ਠੰਢੀ ਹਵਾ ਚੱਲ ਰਹੀ ਹੈ।",
                "rain_umbrella": f"ਹਾਂ ਜੀ, ਅੱਜ {location_name} ਵਿੱਚ {rain_prob}% ਮੀਂਹ ਪੈਣ ਦੀ ਸੰਭਾਵਨਾ ਹੈ। ਬਾਹਰ ਜਾਣ ਵੇਲੇ ਛਤਰੀ ਜ਼ਰੂਰ ਨਾਲ ਰੱਖੋ।" if (rain_prob >= 40 or precip > 0.1) else f"ਅੱਜ ਮੀਂਹ ਦੀ ਸੰਭਾਵਨਾ ਸਿਰਫ਼ {rain_prob}% ਹੈ। ਛਤਰੀ ਦੀ ਕੋਈ ਲੋੜ ਨਹੀਂ ਹੈ।",
                "walk_jog_sports": f"ਸੈਰ ਕਰਨ ਜਾਂ ਖੇਡਣ ਲਈ ਮੌਸਮ ਬਹੁਤ ਸੁਹਾਵਣਾ ਹੈ। ਹਵਾ ਦੀ ਰਫ਼ਤਾਰ {wind} ਕਿਲੋਮੀਟਰ ਪ੍ਰਤੀ ਘੰਟਾ ਹੈ।" if (rain_prob < 40 and wind < 25) else f"ਮੀਂਹ ਦੀ ਸੰਭਾਵਨਾ ਹੋਣ ਕਰਕੇ ਬਾਹਰੀ ਖੇਡਾਂ ਕੁਝ ਸਮੇਂ ਲਈ ਟਾਲਣਾ ਬਿਹਤਰ ਰਹੇਗਾ।",
                "heat_clothing": f"ਤਾਪਮਾਨ {temp} ਡਿਗਰੀ ਹੈ। ਹਲਕੇ ਸੂਤੀ ਕੱਪੜੇ ਪਾਓ ਅਤੇ ਖੂਬ ਪਾਣੀ ਪੀਓ।" if temp >= 31.0 else f"ਮੌਸਮ {temp} ਡਿਗਰੀ ਨਾਲ ਬਹੁਤ ਆਰਾਮਦਾਇਕ ਹੈ। ਆਮ ਕੱਪੜੇ ਢੁੱਕਵੇਂ ਹਨ।",
                "tomorrow_future": f"ਆਉਣ ਵਾਲੇ ਦਿਨਾਂ ਵਿੱਚ ਤਾਪਮਾਨ {min_temp} ਤੋਂ {max_temp} ਡਿਗਰੀ ਵਿਚਕਾਰ ਰਹੇਗਾ ਅਤੇ ਮੀਂਹ ਦੀ ਸੰਭਾਵਨਾ {rain_prob}% ਹੈ।",
                "general": f"{location_name} ਵਿੱਚ ਮੌਜੂਦਾ ਤਾਪਮਾਨ {temp} ਡਿਗਰੀ ਸੈਲਸੀਅਸ ਹੈ। ਤੁਹਾਡਾ ਦਿਨ ਸ਼ੁਭ ਰਹੇ!"
            },
            "or": {
                "greeting": f"ନମସ୍କାର! {location_name} ରେ ବର୍ତ୍ତମାନ ତାପମାତ୍ରା {temp} ଡିଗ୍ରୀ ଏବଂ ପାଣିପାଗ {desc} ଅଛି। ଆଜି ମୁଁ ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରିପାରିବି?",
                "thanks": f"ବହୁତ ଧନ୍ୟବାଦ! ପାଣିପାଗ ବିଷୟରେ କିଛି ବି ଜାଣିବାକୁ ଥିଲେ ନିଶ୍ଚୟ ପଚାରନ୍ତୁ। ଦିନଟି ଆନନ୍ଦରେ କଟୁ!",
                "laundry": f"ହଁ, ଲୁଗା ବାହାରେ ଶୁଖାଇବା ପାଇଁ ଆଜି ଭଲ ଦିନ! ବର୍ଷା ସମ୍ଭାବନା ମାତ୍ର {rain_prob}% ଥିବାରୁ ଲୁଗା ଶୀଘ୍ର ଶୁଖିଯିବ।" if (rain_prob < 40 and precip == 0) else f"ଆଜି ଲୁଗା ବାହାରେ ଶୁଖାନ୍ତୁ ନାହିଁ, କାରଣ {rain_prob}% ବର୍ଷା ହେବାର ସମ୍ଭାବନା ଅଛି।",
                "car_wash": f"ହଁ, ଆଜି ଗାଡ଼ି ଧୋଇବା ପାଇଁ ଅନୁକୂଳ ପାଣିପାଗ ଅଛି! ଆକାଶ ନିର୍ମଳ ରହିବ।" if rain_prob < 40 else f"ଆଜି ଗାଡ଼ି ଧୋଇବା ସ୍ଥଗିତ ରଖନ୍ତୁ, ବର୍ଷା ପାଣିରେ ଗାଡ଼ି ପୁଣି ମଇଳା ହୋଇପାରେ।",
                "night_sleep": f"ଆଜି ରାତିରେ ସର୍ବନିମ୍ନ ତାପମାତ୍ରା {min_temp} ଡିଗ୍ରୀ ପର୍ଯ୍ୟନ୍ତ କମିବ। ରାତି ଆରାମଦାୟକ ହେବ।",
                "humidity_sweat": f"ବାୟୁରେ ଆର୍ଦ୍ରତା {rh}% ଥିବାରୁ ଝାଳ ବୋହିପାରେ। ଶାନ୍ତ ଓ ଥଣ୍ଡା ସ୍ଥାନରେ ରୁହନ୍ତୁ।",
                "wind_storm": f"ପବନର ବେଗ ଘଣ୍ଟା ପ୍ରତି {wind} କି.ମି. ଅଛି। ମନୋରମ ପବନ ବହୁଛି।",
                "rain_umbrella": f"ହଁ, ଆଜି {location_name} ରେ {rain_prob}% ବର୍ଷା ହେବାର ସମ୍ଭାବନା ଅଛି। ବାହାରକୁ ଗଲେ ଛତା ସାଙ୍ଗରେ ନିଅନ୍ତୁ।" if (rain_prob >= 40 or precip > 0.1) else f"ଆଜି ବର୍ଷା ସମ୍ଭାବନା ମାତ୍ର {rain_prob}%। ଛତା ବିନା ନିଶ୍ଚିନ୍ତରେ ଯାଇପାରିବେ।",
                "walk_jog_sports": f"ବାହାରେ ବୁଲିବା, ଦୌଡ଼ିବା କିମ୍ବା ଖେଳିବା ପାଇଁ ପାଣିପାଗ ଖୁବ ମନୋରମ ଅଛି।" if (rain_prob < 40 and wind < 25) else f"ବର୍ଷା ସମ୍ଭାବନା ଥିବାରୁ ବାହାର ଖେଳ କିଛି ସମୟ ପାଇଁ ବନ୍ଦ ରଖିବା ଭଲ।",
                "heat_clothing": f"ତାପମାତ୍ରା {temp} ଡିଗ୍ରୀ ଅଛି। ହାଲୁକା ସୂତା ପୋଷାକ ପିନ୍ଧନ୍ତୁ ଏବଂ ପର୍ଯ୍ୟାପ୍ତ ପାଣି ପିଅନ୍ତୁ।" if temp >= 31.0 else f"ପାଣିପାଗ {temp} ଡିଗ୍ରୀ ସହିତ ବେଶ୍ ଆରାମଦାୟକ ଅଛି। ସାଧାରଣ ପୋଷାକ ଉପଯୁକ୍ତ।",
                "tomorrow_future": f"ଆଗାମୀ ଦିନରେ ତାପମାତ୍ରା {min_temp} ରୁ {max_temp} ଡିଗ୍ରୀ ମଧ୍ୟରେ ରହିବ ଏବଂ ବର୍ଷା ସମ୍ଭାବନା {rain_prob}%.",
                "general": f"{location_name} ରେ ବର୍ତ୍ତମାନ ତାପମାତ୍ରା {temp} ଡିଗ୍ରୀ ସେଲସିୟସ ଅଛି। ଆପଣଙ୍କ ଦିନ ଶୁଭ ହେଉ!"
            },
            "as": {
                "greeting": f"নমস্কাৰ! {location_name}ত বৰ্তমান উত্তাপ {temp} ডিগ্ৰী আৰু বতৰ {desc} হৈ আছে। আজি আপোনাক কেনেকৈ সহায় কৰিব পাৰোঁ?",
                "thanks": f"অশেষ ধন্যবাদ! বতৰ সম্পৰ্কে যিকোনো কথা জানিবলৈ সদায় সুধিব পাৰে। আপোনাৰ দিনটো আনন্দৰে পাৰ হওক!",
                "laundry": f"হয়, কাপোৰ বাহিৰত শুকুৱাবলৈ আজি বৰ ভাল দিন! বৰষুণৰ সম্ভাৱনা মাত্ৰ {rain_prob}% আৰু সোনকালে কাপোৰ শুকাব।" if (rain_prob < 40 and precip == 0) else f"আজি কাপোৰ বাহিৰত নিদিয়াই ভাল, কাৰণ {rain_prob}% বৰষুণৰ সম্ভাৱনা আছে।",
                "car_wash": f"নিশ্চয়, আজি গাড়ী ধুব পাৰে! বতৰ পৰিষ্কাৰ থাকিব।" if rain_prob < 40 else f"আজি গাড়ী ধোৱা স্থগিত ৰাখক, কিয়নো বৰষুণৰ ফলত গাড়ী আকৌ লেতেৰা হ'ব পাৰে।",
                "night_sleep": f"আজি ৰাতি নিম্নতম উত্তাপ {min_temp} ডিগ্ৰীলৈ নামিব। ৰাতিটো শান্ত আৰু আৰামদায়ক হ'ব।",
                "humidity_sweat": f"বায়ুত আৰ্দ্ৰতা {rh}% থকাত সামান্য ঘাম ওলাব পাৰে। পানী বেছিকৈ খাওক।",
                "wind_storm": f"বতাহৰ গতিবেগ প্ৰতি ঘণ্টাত {wind} কিমি। শীতল মৃদু বতাহ বলিছে।",
                "rain_umbrella": f"হয়, আজি {location_name}ত {rain_prob}% বৰষুণৰ সম্ভাৱনা আছে। বাহিৰ ওলালে লগত ছাতি ৰাখিব।" if (rain_prob >= 40 or precip > 0.1) else f"আজি বৰষুণৰ সম্ভাৱনা মাত্ৰ {rain_prob}%। ছাতিৰ কোনো প্ৰয়োজন নাই।",
                "walk_jog_sports": f"বাহিৰত খোজ কঢ়া বা খেল-ধেমালিৰ বাবে বতৰ অতি অনুকূল আৰু আনন্দদায়ক।" if (rain_prob < 40 and wind < 25) else f"বৰষুণৰ সম্ভাৱনা থকা বাবে এতিয়া বাহিৰৰ খেল-ধেমালি পৰিহাৰ কৰা ভাল।",
                "heat_clothing": f"উত্তাপ {temp} ডিগ্ৰী চেলচিয়াছ। পাতল কপাহী কাপোৰ পৰিধান কৰক আৰু প্ৰচুৰ পানী খাওক।" if temp >= 31.0 else f"বতৰ {temp} ডিগ্ৰীৰ সৈতে অতি মনোৰম। স্বাভাৱিক কাপোৰ উপযোগী।",
                "tomorrow_future": f"অহা দিনবোৰত উত্তাপ {min_temp} ৰ পৰা {max_temp} ডিগ্ৰীৰ মাজত থাকিব আৰু বৰষুণৰ সম্ভাৱনা {rain_prob}%.",
                "general": f"{location_name}ত বৰ্তমান উত্তাপ {temp} ডিগ্ৰী চেলচিয়াছ। আপোনাৰ দিনটো শুভ হওক!"
            },
            "ur": {
                "greeting": f"سلام! {location_name} میں اس وقت درجہ حرارت {temp} ڈگری اور موسم {desc} ہے۔ آج آپ کی کیا مدد کر سکتا ہوں?",
                "thanks": f"بہت بہت شکریہ! موسم کے متعلق کوئی بھی سوال ہو تو بلا جھجھک پوچھیں۔ آپ کا دن اچھا گزرے!",
                "laundry": f"جی ہاں، کپڑے باہر سکھانے کے لیے بہترین دن ہے! بارش کا امکان صرف {rain_prob}% ہے اور دھوپ سے کپڑے جلد سوکھ جائیں گے۔" if (rain_prob < 40 and precip == 0) else f"آج کپڑے باہر نہ سکھائیں کیونکہ {rain_prob}% بارش کا امکان ہے۔ کپڑے بھیگ سکتے ہیں۔",
                "car_wash": f"جی ہاں، گاڑی دھونے کے لیے موسم بہت اچھا اور صاف رہے گا۔" if rain_prob < 40 else f"آج گاڑی دھونے سے گریز کریں کیونکہ بارش سے گاڑی دوبارہ گندی ہو سکتی ہے۔",
                "night_sleep": f"آج رات کم سے کم درجہ حرارت {min_temp} ڈگری تک رہے گا۔ رات پرسکون رہے گی۔",
                "humidity_sweat": f"ہوا میں نمی {rh}% ہے، جس سے تھوڑی حبس یا پسینہ آ سکتا ہے۔",
                "wind_storm": f"ہوا کی رفتار {wind} کلومیٹر فی گھنٹہ ہے۔ خوشگوار ہوا چل رہی ہے۔",
                "rain_umbrella": f"جی ہاں، آج {location_name} میں {rain_prob}% بارش کا امکان ہے۔ باہر جاتے ہوئے چھتری ساتھ رکھیں۔" if (rain_prob >= 40 or precip > 0.1) else f"آج بارش کا امکان صرف {rain_prob}% ہے۔ چھتری کی ضرورت نہیں ہے۔",
                "walk_jog_sports": f"سیر و تفریح، دوڑ یا کھیلوں کے لیے موسم بے حد خوشگوار ہے۔ ہوا کی رفتار {wind} کلومیٹر ہے۔" if (rain_prob < 40 and wind < 25) else f"بارش کے امکان کی وجہ سے آؤٹ ڈور کھیلیں فی الحال موخر کرنا بہتر ہے۔",
                "heat_clothing": f"درجہ حرارت {temp} ڈگری ہے۔ ہلکے سوتی کپڑے پہنیں اور خوب پانی پیئیں۔" if temp >= 31.0 else f"موسم {temp} ڈگری کے ساتھ پرسکون ہے۔ عام لباس بہترین ہے۔",
                "tomorrow_future": f"آنے والے دنوں میں درجہ حرارت {min_temp} سے {max_temp} ڈگری کے درمیان رہے گا اور بارش کا امکان {rain_prob}% ہے۔",
                "general": f"{location_name} میں اس وقت درجہ حرارت {temp} ڈگری سینٹی گریڈ ہے۔ آپ کا دن خوشگوار گزرے!"
            },
        }

        lang_dict = responses.get(lang_code, responses["en"])
        lang_info = LANGUAGE_CONFIG.get(lang_code, LANGUAGE_CONFIG["en"])

        text = lang_dict.get(intent, lang_dict["general"])
        clean_text = self._clean_text_for_speech(text)

        # Synthesize real native spoken audio in the respected Indian language
        audio_base64, audio_mime = await tts_service.synthesize_speech(clean_text, lang_code)

        if not audio_base64:
            audio_base64, audio_mime = self._generate_chime_audio()

        return {
            "response_text": clean_text,
            "language": lang_code,
            "language_name": lang_info["name"],
            "speech_locale": lang_info.get("speech_locale", "en-IN"),
            "audio_base64": audio_base64,
            "audio_mime_type": audio_mime,
            "source": "smart_weather_fallback",
        }

    @staticmethod
    def _generate_chime_audio() -> Tuple[str, str]:
        """Generate a clean audio chime for voice start/end alert."""
        sample_rate = 16000
        duration_s = 0.35
        n_samples = int(sample_rate * duration_s)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            frames = bytearray()
            import math
            for i in range(n_samples):
                t = float(i) / sample_rate
                freq = 659.25 if t < 0.15 else 880.0  # E5 -> A5
                envelope = max(0.0, 1.0 - (t / duration_s))
                val = int(9000.0 * envelope * math.sin(2.0 * math.pi * freq * t))
                val = max(-32767, min(32767, val))
                frames.extend(struct.pack("<h", val))
            wav_file.writeframes(frames)

        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return b64, "audio/wav"

llm_service = LLMService()
