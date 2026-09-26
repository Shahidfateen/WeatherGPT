"""Weather Service for Open-Meteo REST API Ingestion, 7-Day Forecast, Reverse Geocoding, and Offline Bundles."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import httpx
from app.config import settings
from app.services.cache_service import cache_service

WMO_WEATHER_CODES = {
    0: {"desc": "Clear Sky", "icon": "fa-sun", "category": "clear"},
    1: {"desc": "Mainly Clear", "icon": "fa-sun", "category": "clear"},
    2: {"desc": "Partly Cloudy", "icon": "fa-cloud-sun", "category": "cloudy"},
    3: {"desc": "Overcast", "icon": "fa-cloud", "category": "cloudy"},
    45: {"desc": "Foggy", "icon": "fa-smog", "category": "fog"},
    48: {"desc": "Depositing Rime Fog", "icon": "fa-smog", "category": "fog"},
    51: {"desc": "Light Drizzle", "icon": "fa-cloud-rain", "category": "rain"},
    53: {"desc": "Moderate Drizzle", "icon": "fa-cloud-rain", "category": "rain"},
    55: {"desc": "Dense Drizzle", "icon": "fa-cloud-rain", "category": "rain"},
    56: {"desc": "Light Freezing Drizzle", "icon": "fa-snowflake", "category": "snow"},
    57: {"desc": "Dense Freezing Drizzle", "icon": "fa-snowflake", "category": "snow"},
    61: {"desc": "Slight Rain", "icon": "fa-cloud-rain", "category": "rain"},
    63: {"desc": "Moderate Rain", "icon": "fa-cloud-showers-heavy", "category": "rain"},
    65: {"desc": "Heavy Rain", "icon": "fa-cloud-showers-heavy", "category": "rain"},
    66: {"desc": "Light Freezing Rain", "icon": "fa-snowflake", "category": "snow"},
    67: {"desc": "Heavy Freezing Rain", "icon": "fa-snowflake", "category": "snow"},
    71: {"desc": "Slight Snowfall", "icon": "fa-snowflake", "category": "snow"},
    73: {"desc": "Moderate Snowfall", "icon": "fa-snowflake", "category": "snow"},
    75: {"desc": "Heavy Snowfall", "icon": "fa-snowflake", "category": "snow"},
    77: {"desc": "Snow Grains", "icon": "fa-snowflake", "category": "snow"},
    80: {"desc": "Slight Rain Showers", "icon": "fa-cloud-rain", "category": "rain"},
    81: {"desc": "Moderate Rain Showers", "icon": "fa-cloud-rain", "category": "rain"},
    82: {"desc": "Violent Rain Showers", "icon": "fa-cloud-showers-heavy", "category": "rain"},
    85: {"desc": "Slight Snow Showers", "icon": "fa-snowflake", "category": "snow"},
    86: {"desc": "Heavy Snow Showers", "icon": "fa-snowflake", "category": "snow"},
    95: {"desc": "Thunderstorm", "icon": "fa-bolt", "category": "thunder"},
    96: {"desc": "Thunderstorm with Slight Hail", "icon": "fa-cloud-bolt", "category": "thunder"},
    99: {"desc": "Thunderstorm with Heavy Hail", "icon": "fa-cloud-bolt", "category": "thunder"},
}


PLACES_I18N = {
    # Countries
    "India": {
        "ta": "இந்தியா", "hi": "भारत", "te": "భారతదేశం", "kn": "ಭಾರತ", "ml": "ഇന്ത്യ",
        "bn": "ভারত", "mr": "भारत", "gu": "ભારત", "pa": "ਭਾਰਤ", "or": "ଭାରତ",
        "as": "ভাৰত", "ur": "ہندوستان", "en": "India"
    },
    # Indian States & UTs
    "Tamil Nadu": {
        "ta": "தமிழ்நாடு", "hi": "तमिलनाडु", "te": "తమిళనాడు", "kn": "ತಮಿಳುನಾಡು", "ml": "തമിഴ്നാട്",
        "bn": "তামিলনাড়ু", "mr": "तमिळनाडू", "gu": "તમિલનાડુ", "pa": "ਤਾਮਿਲਨਾਡੂ", "or": "ତାମିଲନାଡୁ",
        "as": "তামিলনাডু", "ur": "تمل ناڈو", "en": "Tamil Nadu"
    },
    "Karnataka": {
        "ta": "கர்நாடகா", "hi": "कर्नाटक", "te": "కర్ణాటక", "kn": "ಕರ್ನಾಟಕ", "ml": "കർണാടക",
        "bn": "কর্ণাটক", "mr": "कर्नाटक", "gu": "કર્ણાટક", "pa": "ਕਰਨਾਟਕ", "or": "କର୍ଣ୍ଣାଟକ",
        "as": "কৰ্ণাটক", "ur": "کرناٹک", "en": "Karnataka"
    },
    "Kerala": {
        "ta": "கேரளா", "hi": "केरल", "te": "కేరళ", "kn": "ಕೇರಳ", "ml": "കേരളം",
        "bn": "কেরল", "mr": "केरळ", "gu": "કેરળ", "pa": "ਕੇਰਲ", "or": "କେରଳ",
        "as": "কেৰালা", "ur": "کیرالہ", "en": "Kerala"
    },
    "Andhra Pradesh": {
        "ta": "ஆந்திரப் பிரதேசம்", "hi": "आंध्र प्रदेश", "te": "ఆంధ్రప్రదేశ్", "kn": "ಆಂಧ್ರಪ್ರದೇಶ", "ml": "ആന്ധ്രാപ്രദേശ്",
        "bn": "অন্ধ্রপ্রদেশ", "mr": "आंध्र प्रदेश", "gu": "આંધ્રપ્રદેશ", "pa": "ਆਂਧਰਾ ਪ੍ਰਦੇਸ਼", "or": "ଆନ୍ଧ୍ର ପ୍ରଦେଶ",
        "as": "অন্ধ্ৰ প্ৰদেশ", "ur": "آندھرا پردیش", "en": "Andhra Pradesh"
    },
    "Telangana": {
        "ta": "தெலுங்கானா", "hi": "तेलंगाना", "te": "తెలంగాణ", "kn": "ತೆಲಂಗಾಣ", "ml": "തെലങ്കാന",
        "bn": "তেলেঙ্গানা", "mr": "तेलंगणा", "gu": "તેલંગાણા", "pa": "ਤੇਲੰਗਾਨਾ", "or": "ତେଲେଙ୍ଗାନା",
        "as": "তেলেংগানা", "ur": "تلنگانہ", "en": "Telangana"
    },
    "Maharashtra": {
        "ta": "மகாராஷ்டிரா", "hi": "महाराष्ट्र", "te": "మహారాష్ట్ర", "kn": "ಮಹಾರಾಷ್ಟ್ರ", "ml": "മഹാരാഷ്ട്ര",
        "bn": "মহারাষ্ট্র", "mr": "महाराष्ट्र", "gu": "મહારાષ્ટ્ર", "pa": "ਮਹਾਰਾਸ਼ਟਰ", "or": "ମହାରାଷ୍ଟ୍ର",
        "as": "মহাৰাষ্ট্ৰ", "ur": "مہاراشٹر", "en": "Maharashtra"
    },
    "Gujarat": {
        "ta": "குஜராத்", "hi": "गुजरात", "te": "గుజరాత్", "kn": "ಗುಜರಾತ್", "ml": "ഗുജറാത്ത്",
        "bn": "গুজরাট", "mr": "गुजरात", "gu": "ગુજરાત", "pa": "ਗੁਜਰਾਤ", "or": "ଗୁଜରାଟ",
        "as": "গুজৰাট", "ur": "گجرات", "en": "Gujarat"
    },
    "West Bengal": {
        "ta": "மேற்கு வங்காளம்", "hi": "पश्चिम बंगाल", "te": "పశ్చిమ బెంగాల్", "kn": "ಪಶ್ಚಿಮ ಬಂಗಾಳ", "ml": "പശ്ചിമ ബംഗാൾ",
        "bn": "পশ্চিমবঙ্গ", "mr": "पश्चिम बंगाल", "gu": "પશ્ચિમ બંગાળ", "pa": "ਪੱਛਮੀ ਬੰਗਾਲ", "or": "ପଶ୍ଚିମ ବଙ୍ଗ",
        "as": "পশ্চিম বংগ", "ur": "مغربی بنگال", "en": "West Bengal"
    },
    "Punjab": {
        "ta": "பஞ்சாப்", "hi": "पंजाब", "te": "పంజాబ్", "kn": "ಪಂಜಾಬ್", "ml": "പഞ്ചാബ്",
        "bn": "পাঞ্জাব", "mr": "पंजाब", "gu": "પંજાબ", "pa": "ਪੰਜਾਬ", "or": "ପଞ୍ଜାବ",
        "as": "পাঞ্জাৱ", "ur": "پنجاب", "en": "Punjab"
    },
    "Odisha": {
        "ta": "ஒடிசா", "hi": "ओडिशा", "te": "ఒడిశా", "kn": "ಒಡಿಶಾ", "ml": "ഒഡീഷ",
        "bn": "ওড়িশা", "mr": "ओडिशा", "gu": "ઓડિશા", "pa": "ਓਡੀਸ਼ਾ", "or": "ଓଡ଼ିଶା",
        "as": "ওড়িশা", "ur": "اوڈیشہ", "en": "Odisha"
    },
    "Assam": {
        "ta": "அசாம்", "hi": "असम", "te": "అస్సాం", "kn": "ಅಸ್ಸಾಂ", "ml": "അസം",
        "bn": "আসাম", "mr": "आसाम", "gu": "આસામ", "pa": "ਅਸਾਮ", "or": "ଆସାମ",
        "as": "অসম", "ur": "آسام", "en": "Assam"
    },
    "Delhi": {
        "ta": "தில்லி", "hi": "दिल्ली", "te": "ఢిల్లీ", "kn": "ದೆಹಲಿ", "ml": "ദില്ലി",
        "bn": "দিল্লি", "mr": "दिल्ली", "gu": "દિલ્હી", "pa": "ਦਿੱਲੀ", "or": "ଦିଲ୍ଲୀ",
        "as": "দিল্লী", "ur": "دہلی", "en": "Delhi"
    },
    "Puducherry": {
        "ta": "புதுச்சேரி", "hi": "पुदुचेरी", "te": "పుదుచ్చేరి", "kn": "ಪುದುಚೇರಿ", "ml": "പുതുച്ചേരി",
        "bn": "পুদুচেরি", "mr": "पुदुचेरी", "gu": "પુડુચેરી", "pa": "ਪੁਡੂਚੇਰੀ", "or": "ପୁଡୁଚେରୀ",
        "as": "পুডুচেৰী", "ur": "پڈوچیری", "en": "Puducherry"
    },
    # Cities, Towns & Localities
    "Oragadam": {
        "ta": "ஒரகடம்", "hi": "ओरगदम", "te": "ఒరగడం", "kn": "ಒರಗಡಂ", "ml": "ഒറഗടം",
        "bn": "ওরাগাদাম", "mr": "ओरागडम", "gu": "ઓરાગડમ", "pa": "ਓਰਾਗਦਮ", "or": "ଓରାଗାଦାମ",
        "as": "ওৰাগাদাম", "ur": "اوراگدم", "en": "Oragadam"
    },
    "Sriperumbudur": {
        "ta": "ஸ்ரீபெரும்புதூர்", "hi": "श्रीपेरंबदूर", "te": "శ్రీపెరంబుదూర్", "kn": "ಶ್ರೀಪೆರಂಬುದೂರ್", "ml": "ശ്രീപെരുമ്പുദൂർ",
        "bn": "শ্রীপেরুম্বুদুর", "mr": "श्रीपेरुंबुदूर", "gu": "શ્રીપેરમ્બુદૂર", "pa": "ਸ਼੍ਰੀਪੇਰੰਬੁਦੂਰ", "or": "ଶ୍ରୀପେରୁମ୍ବୁଦୁର",
        "as": "শ্ৰীপেৰুম্বুদ্বুৰ", "ur": "سری پیرومبدور", "en": "Sriperumbudur"
    },
    "Kanchipuram": {
        "ta": "காஞ்சிபுரம்", "hi": "कांचीपुरम", "te": "కాంచీపురం", "kn": "ಕಾಂಚೀಪುರಂ", "ml": "കാഞ്ചീപുരം",
        "bn": "কাঞ্চীপুরম", "mr": "कांचीपुरम", "gu": "કાંચીપુરમ", "pa": "ਕਾਂਚੀਪੁਰਮ", "or": "କାଞ୍ଚିପୁରମ୍",
        "as": "কাঞ্চীপুৰম", "ur": "کانچی پورم", "en": "Kanchipuram"
    },
    "Chengalpattu": {
        "ta": "செங்கல்பட்டு", "hi": "चेंगलपट्टू", "te": "చెంగల్పట్టు", "kn": "ಚೆಂಗಲ್ಪಟ್ಟು", "ml": "ചെങ്കൽപട്ട്",
        "bn": "চেঙ্গালপট্টু", "mr": "चेंगलपट्टू", "gu": "ચેંગલપટ્ટુ", "pa": "ਚੇਂਗਲਪੱਟੂ", "or": "ଚେଙ୍ଗଲପଟ୍ଟୁ",
        "as": "চেংগলপট্টু", "ur": "چنگل پٹو", "en": "Chengalpattu"
    },
    "Tambaram": {
        "ta": "தாம்பரம்", "hi": "तांबरम", "te": "తాంబరం", "kn": "ತಾಂಬರಂ", "ml": "താംബരം",
        "bn": "তাম্বারাম", "mr": "तांबरम", "gu": "તાંબરમ", "pa": "ਤਾਂਬਰਮ", "or": "ତାମ୍ବରମ",
        "as": "তাম্বাৰাম", "ur": "تامبرم", "en": "Tambaram"
    },
    "Guindy": {
        "ta": "கிண்டி", "hi": "गिंडी", "te": "గిండి", "kn": "ಗಿಂಡಿ", "ml": "ഗിണ്ടി",
        "bn": "গিন্ডি", "mr": "गिंडी", "gu": "ગિન્ડી", "pa": "ਗਿੰਡੀ", "or": "ଗିଣ୍ଡି",
        "as": "গিণ্ডি", "ur": "گنڈی", "en": "Guindy"
    },
    "Chennai": {
        "ta": "சென்னை", "hi": "चेन्नई", "te": "చెన్నై", "kn": "ಚೆನ್ನೈ", "ml": "ചെന്നൈ",
        "bn": "চেন্নাই", "mr": "चेन्नई", "gu": "ચેન્નઈ", "pa": "ਚੇਨਈ", "or": "ଚେନ୍ନାଇ",
        "as": "চেন্নাই", "ur": "چنئی", "en": "Chennai"
    },
    "Bengaluru": {
        "ta": "பெங்களூரு", "hi": "बेंगलुरु", "te": "బెంగళూరు", "kn": "ಬೆಂಗಳೂರು", "ml": "ബെംഗളൂരു",
        "bn": "বেঙ্গালুরু", "mr": "बंगळुरू", "gu": "બેંગલુરુ", "pa": "ਬੈਂਗਲੁਰੂ", "or": "ବେଙ୍ଗାଲୁରୁ",
        "as": "বেংগালুৰু", "ur": "بنگلور", "en": "Bengaluru"
    },
    "Hyderabad": {
        "ta": "ஹைதராபாத்", "hi": "हैदराबाद", "te": "హైదరాబాద్", "kn": "ಹೈದರಾಬಾದ್", "ml": "ഹൈദരാബാദ്",
        "bn": "হায়দ্রাবাদ", "mr": "हैदराबाद", "gu": "હૈદરાબાદ", "pa": "ਹੈਦਰਾਬਾਦ", "or": "ହାଇଦ୍ରାବାଦ",
        "as": "হায়দৰাবাদ", "ur": "حیدرآباد", "en": "Hyderabad"
    },
    "Mumbai": {
        "ta": "மும்பை", "hi": "मुंबई", "te": "ముంబై", "kn": "ಮುಂಬೈ", "ml": "മുംബൈ",
        "bn": "মুম্বাই", "mr": "मुंबई", "gu": "મુંબઈ", "pa": "ਮੁੰਬਈ", "or": "ମୁମ୍ବାଇ",
        "as": "মুম্বাই", "ur": "ممبئی", "en": "Mumbai"
    },
    "Kolkata": {
        "ta": "கொல்கத்தா", "hi": "कोलकाता", "te": "కోల్‌కతా", "kn": "ಕೋಲ್ಕತ್ತಾ", "ml": "കൊൽക്കത്ത",
        "bn": "কলকাতা", "mr": "कोलकाता", "gu": "કોલકાતા", "pa": "ਕੋਲਕਾਤਾ", "or": "କୋଲକାତା",
        "as": "কলকাতা", "ur": "کولکتہ", "en": "Kolkata"
    },
    "Thanjavur": {
        "ta": "தஞ்சாவூர்", "hi": "तंजावुर", "te": "తంజావూరు", "kn": "ತಂಜಾವೂರು", "ml": "തഞ്ചാവൂർ",
        "bn": "তাঞ্জাভুর", "mr": "तंजावर", "gu": "તંજાવુર", "pa": "ਤੰਜਾਵੁਰ", "or": "ତାଞ୍ଜାଭୁର",
        "as": "তাঞ্জাভুৰ", "ur": "تنچاور", "en": "Thanjavur"
    },
    "Coimbatore": {
        "ta": "கோயம்புத்தூர்", "hi": "कोयंबटूर", "te": "కోయంబత్తూర్", "kn": "ಕೊಯಮತ್ತೂರು", "ml": "കോയമ്പത്തൂർ",
        "bn": "কোয়েম্বাটুর", "mr": "कोइम्बतूर", "gu": "કોયમ્બતૂર", "pa": "ਕੋਇੰਬਟੂਰ", "or": "କୋଏମ୍ବାଟୁର",
        "as": "কোয়েম্বাটোৰ", "ur": "کوئمبٹور", "en": "Coimbatore"
    },
    "Madurai": {
        "ta": "மதுரை", "hi": "मदुरै", "te": "మధురై", "kn": "ಮಧುರೈ", "ml": "മധുര",
        "bn": "মাদুরাই", "mr": "मदुराई", "gu": "મદુરાઇ", "pa": "ਮਦੁਰਾਈ", "or": "ମଦୁରାଇ",
        "as": "মাদুৰাই", "ur": "مدورائی", "en": "Madurai"
    },
    "Tiruchirappalli": {
        "ta": "திருச்சிராப்பள்ளி", "hi": "तिरुचिरापल्ली", "te": "తిరుచిరాపల్లి", "kn": "ತಿರುಚಿರಾಪಳ್ಳಿ", "ml": "തിരുച്ചിറപ്പള്ളി",
        "bn": "তিরুচিরাপল্লী", "mr": "तिरुचिरापल्ली", "gu": "તિરુચિરાપલ્લી", "pa": "ਤਿਰੂਚਿਰਾਪੱਲੀ", "or": "ତିରୁଚିରାପଲ୍ଲୀ",
        "as": "তিৰুচিৰাপল্লী", "ur": "تروچیراپلی", "en": "Tiruchirappalli"
    },
    "Salem": {
        "ta": "சேலம்", "hi": "सलेम", "te": "సేలం", "kn": "ಸೇಲಂ", "ml": "സേലം",
        "bn": "সালেম", "mr": "सेलम", "gu": "સેલમ", "pa": "ਸਲੇਮ", "or": "ସାଲେମ୍",
        "as": "চালেম", "ur": "سیلم", "en": "Salem"
    },
    "Tirunelveli": {
        "ta": "திருநெல்வேலி", "hi": "तिरुनेलवेली", "te": "తిరునెల్వేలి", "kn": "ತಿರುನೆಲ್ವೇಲಿ", "ml": "തിരുനെൽവേലി",
        "bn": "তিরুনেলভেলি", "mr": "तिरुनेलवेली", "gu": "તિરુનેલવેલી", "pa": "ਤਿਰੂਨੇਲਵੇਲੀ", "or": "ତିରୁନେଲଭେଲି",
        "as": "তিৰুনেলভেলি", "ur": "ترونلویلی", "en": "Tirunelveli"
    },
    "Vellore": {
        "ta": "வேலூர்", "hi": "वेल्लोर", "te": "వెల్లూరు", "kn": "ವೆಲ್ಲೂರು", "ml": "വെല്ലൂർ",
        "bn": "ভেলোর", "mr": "वेल्लोर", "gu": "વેલ્લોર", "pa": "ਵੇਲੋਰ", "or": "ଭେଲୋର୍",
        "as": "ভেলোৰ", "ur": "ویلور", "en": "Vellore"
    },
    "Erode": {
        "ta": "ஈரோடு", "hi": "ईरोड", "te": "ఈరోడ్", "kn": "ಈರೋಡ್", "ml": "ഈറോഡ്",
        "bn": "ইরোড", "mr": "इरोड", "gu": "ઈરોડ", "pa": "ਈਰੋਡ", "or": "ଇରୋଡ୍",
        "as": "ইৰোড", "ur": "ایروڈ", "en": "Erode"
    },
    "Your Location": {
        "ta": "உங்கள் இருப்பிடம்", "hi": "आपका स्थान", "te": "మీ స్థానం", "kn": "ನಿಮ್ಮ ಸ್ಥಳ", "ml": "നിങ്ങളുടെ സ്ഥലം",
        "bn": "আপনার অবস্থান", "mr": "आपले स्थान", "gu": "તમારું સ્થાન", "pa": "ਤੁਹਾਡਾ ਸਥਾਨ", "or": "ଆପଣଙ୍କ ସ୍ଥାନ",
        "as": "আপোনাৰ স্থান", "ur": "آپ کا مقام", "en": "Your Location"
    }
}

class WeatherService:
    def __init__(self):
        self.mock_file = settings.DATA_DIR / "weather_mock.json"

    async def get_forecast(
        self,
        lat: float,
        lon: float,
        force_refresh: bool = False,
        api_key: Optional[str] = None
    ) -> Tuple[Dict[str, Any], str]:
        """Fetch real-time metrics and 7-day forecast from Open-Meteo REST API.
        
        Supports optional Open-Meteo API key, falling back seamlessly to free public endpoint,
        SQLite offline cache, or local mock data.
        """
        # 1. Check SQLite Cache if not forcing refresh
        if not force_refresh:
            cached = cache_service.get_cached_forecast(lat, lon)
            if cached:
                return cached, "sqlite_cache"

        # 2. Query Open-Meteo REST API
        active_key = api_key or settings.OPEN_METEO_API_KEY
        endpoint = settings.get_open_meteo_endpoint(active_key)

        params: Dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,"
                "precipitation,rain,weather_code,cloud_cover,surface_pressure,"
                "wind_speed_10m,wind_direction_10m"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "apparent_temperature_max,apparent_temperature_min,sunrise,sunset,"
                "uv_index_max,precipitation_sum,precipitation_probability_max,wind_speed_10m_max"
            ),
            "timezone": "auto",
            "forecast_days": 7,
        }

        # If Open-Meteo API Key is provided, pass it in query parameters
        if active_key:
            params["apikey"] = active_key

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get(endpoint, params=params)
                if response.status_code == 200:
                    payload = response.json()
                    # Cache in SQLite
                    cache_service.save_forecast(lat, lon, payload)
                    return payload, "open_meteo"
                elif response.status_code in (400, 401, 403) and active_key:
                    # If invalid or non-commercial API key provided, fallback smoothly to free public endpoint
                    print(f"[WeatherService] Open-Meteo API key returned {response.status_code}. Retrying seamlessly with public endpoint...")
                    params.pop("apikey", None)
                    free_resp = await client.get(settings.OPEN_METEO_URL, params=params)
                    if free_resp.status_code == 200:
                        payload = free_resp.json()
                        cache_service.save_forecast(lat, lon, payload)
                        return payload, "open_meteo_public"
        except Exception as e:
            print(f"[WeatherService] Open-Meteo query failed: {e}. Falling back to cache/mock.")

        # 3. Fallback to any cached data in SQLite (even if expired)
        stale_cached = cache_service.get_any_cached_forecast(lat, lon)
        if stale_cached:
            return stale_cached, "offline_cache_expired"

        # 4. Fallback to local mock payload
        fallback_data = self._load_mock_data()
        return fallback_data, "mock_fallback"

    def localize_place(self, name: str, state: str, country: str, language: str) -> str:
        """Translate city/town, state, and country into native script of the selected language."""
        if not language or language == "en":
            parts = [p for p in [name, state, country] if p]
            return ", ".join(parts)

        # 1. Translate city
        city_trans = name
        for k, v in PLACES_I18N.items():
            if k.lower() == name.lower() or k.lower() in name.lower():
                city_trans = v.get(language, name)
                break

        # 2. Translate state
        state_trans = state
        for k, v in PLACES_I18N.items():
            if k.lower() == state.lower() or k.lower() in state.lower():
                state_trans = v.get(language, state)
                break

        # 3. Translate country
        country_trans = country
        for k, v in PLACES_I18N.items():
            if k.lower() == country.lower() or k.lower() in country.lower():
                country_trans = v.get(language, country)
                break

        parts = [p for p in [city_trans, state_trans, country_trans] if p]
        return ", ".join(parts)

    async def reverse_geocode(self, lat: float, lon: float, language: str = "en") -> Dict[str, Any]:
        """Convert latitude and longitude coordinates into a human-readable city and location name in requested language."""
        try:
            headers = {
                "User-Agent": "WeatherGPT/1.0 (weathergpt@local.dev)",
                "Accept-Language": f"{language},en;q=0.7",
            }
            url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language={language}"
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    address = data.get("address", {})
                    city = (
                        address.get("city")
                        or address.get("town")
                        or address.get("village")
                        or address.get("suburb")
                        or address.get("county")
                        or data.get("name")
                        or "Your Location"
                    )
                    state = address.get("state", "")
                    country = address.get("country", "")

                    # Localize place, state, country
                    display = self.localize_place(city, state, country, language)
                    raw_display = f"{city}, {state}, {country}".strip(", ")

                    return {
                        "name": city,
                        "state": state,
                        "country": country,
                        "display_name": display,
                        "raw_display_name": raw_display,
                        "latitude": lat,
                        "longitude": lon,
                        "language": language,
                    }
        except Exception as e:
            print(f"[WeatherService] Reverse geocode error: {e}")

        default_loc = self.localize_place("Your Location", "", "India", language)
        return {
            "name": f"Location ({lat:.2f}°, {lon:.2f}°)",
            "state": "",
            "country": "",
            "display_name": default_loc,
            "raw_display_name": f"Location ({lat:.2f}°, {lon:.2f}°)",
            "latitude": lat,
            "longitude": lon,
            "language": language,
        }

    async def search_locations(self, query: str, count: int = 6) -> List[Dict[str, Any]]:
        """Search cities/locations by name using Open-Meteo Geocoding API."""
        if not query or len(query.strip()) < 2:
            return []

        try:
            params = {
                "name": query.strip(),
                "count": count,
                "language": "en",
                "format": "json"
            }
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(settings.OPEN_METEO_GEOCODING_URL, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    results = []
                    for item in data.get("results", []):
                        city = item.get("name", "")
                        admin1 = item.get("admin1", "")
                        country = item.get("country", "")
                        label_parts = [city]
                        if admin1:
                            label_parts.append(admin1)
                        if country:
                            label_parts.append(country)
                        results.append({
                            "name": city,
                            "admin1": admin1,
                            "country": country,
                            "display_name": ", ".join(label_parts),
                            "latitude": item.get("latitude"),
                            "longitude": item.get("longitude"),
                            "timezone": item.get("timezone", "auto")
                        })
                    return results
        except Exception as e:
            print(f"[WeatherService] Location search error: {e}")

        return []

    def _load_mock_data(self) -> Dict[str, Any]:
        """Load local fallback weather_mock.json."""
        if self.mock_file.exists():
            try:
                with open(self.mock_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WeatherService] Error reading mock file: {e}")

        today = datetime.now().strftime("%Y-%m-%d")
        return {
            "latitude": settings.DEFAULT_LAT,
            "longitude": settings.DEFAULT_LON,
            "current": {
                "temperature_2m": 29.5,
                "apparent_temperature": 32.0,
                "relative_humidity_2m": 72.0,
                "surface_pressure": 1010.0,
                "precipitation": 0.0,
                "rain": 0.0,
                "wind_speed_10m": 12.0,
                "wind_direction_10m": 140,
                "cloud_cover": 35,
                "weather_code": 2,
                "is_day": 1
            },
            "daily": {
                "time": [today],
                "weather_code": [2],
                "temperature_2m_max": [32.5],
                "temperature_2m_min": [24.0],
                "apparent_temperature_max": [35.0],
                "apparent_temperature_min": [25.0],
                "sunrise": [f"{today}T06:05"],
                "sunset": [f"{today}T18:15"],
                "uv_index_max": [7.5],
                "precipitation_sum": [0.0],
                "precipitation_probability_max": [25],
                "wind_speed_10m_max": [16.0]
            }
        }

    @staticmethod
    def extract_current_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract atmospheric features from Open-Meteo payload for UI and AI guide."""
        current = payload.get("current", {})
        daily = payload.get("daily", {})

        temp = float(current.get("temperature_2m", 28.0))
        feels_like = float(current.get("apparent_temperature", temp))
        rh = float(current.get("relative_humidity_2m", 65.0))
        sp = float(current.get("surface_pressure", 1010.0))
        precip = float(current.get("precipitation", 0.0))
        rain = float(current.get("rain", precip))
        wind = float(current.get("wind_speed_10m", 10.0))
        wind_dir = float(current.get("wind_direction_10m", 0.0))
        cloud = float(current.get("cloud_cover", 20.0))
        is_day = int(current.get("is_day", 1))
        wcode = int(current.get("weather_code", 0))

        wmo_info = WMO_WEATHER_CODES.get(wcode, {"desc": "Clear", "icon": "fa-sun", "category": "clear"})
        weather_desc = wmo_info["desc"]
        weather_icon = wmo_info["icon"]
        if is_day == 0 and weather_icon == "fa-sun":
            weather_icon = "fa-moon"

        # Daily extremes for today
        max_temps = daily.get("temperature_2m_max", [temp])
        min_temps = daily.get("temperature_2m_min", [temp - 5.0])
        precip_prob = daily.get("precipitation_probability_max", [20])
        uv_max = daily.get("uv_index_max", [6.0])

        return {
            "temperature_2m": round(temp, 1),
            "apparent_temperature": round(feels_like, 1),
            "relative_humidity_2m": round(rh, 1),
            "surface_pressure": round(sp, 1),
            "precipitation": round(precip, 1),
            "rain": round(rain, 1),
            "wind_speed_10m": round(wind, 1),
            "wind_direction_10m": round(wind_dir, 0),
            "cloud_cover": round(cloud, 0),
            "weather_code": wcode,
            "weather_desc": weather_desc,
            "weather_icon": weather_icon,
            "weather_category": wmo_info.get("category", "clear"),
            "is_day": is_day,
            "daily_max_temp": round(float(max_temps[0]), 1) if max_temps else round(temp + 2, 1),
            "daily_min_temp": round(float(min_temps[0]), 1) if min_temps else round(temp - 4, 1),
            "daily_rain_probability": int(precip_prob[0]) if precip_prob else 20,
            "uv_index": round(float(uv_max[0]), 1) if uv_max else 5.0,
        }

    @staticmethod
    def extract_daily_forecast(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract structured 7-day future weather prediction array."""
        daily = payload.get("daily", {})
        times = daily.get("time", [])
        if not times:
            return []

        wcodes = daily.get("weather_code", [0] * len(times))
        max_temps = daily.get("temperature_2m_max", [28.0] * len(times))
        min_temps = daily.get("temperature_2m_min", [22.0] * len(times))
        precip_sums = daily.get("precipitation_sum", [0.0] * len(times))
        precip_probs = daily.get("precipitation_probability_max", [0] * len(times))
        wind_maxs = daily.get("wind_speed_10m_max", [10.0] * len(times))
        uv_maxs = daily.get("uv_index_max", [5.0] * len(times))

        days_list = []
        for i, dt_str in enumerate(times[:7]):
            try:
                dt = datetime.strptime(dt_str, "%Y-%m-%d")
                if i == 0:
                    day_name = "Today"
                elif i == 1:
                    day_name = "Tomorrow"
                else:
                    day_name = dt.strftime("%A")  # e.g. Friday
                formatted_date = dt.strftime("%b %d")  # e.g. Sep 24
            except Exception:
                day_name = f"Day {i+1}"
                formatted_date = dt_str

            code = int(wcodes[i]) if i < len(wcodes) else 0
            wmo_info = WMO_WEATHER_CODES.get(code, {"desc": "Clear", "icon": "fa-sun", "category": "clear"})

            days_list.append({
                "date": dt_str,
                "formatted_date": formatted_date,
                "day_name": day_name,
                "weather_code": code,
                "weather_desc": wmo_info["desc"],
                "weather_icon": wmo_info["icon"],
                "weather_category": wmo_info.get("category", "clear"),
                "temp_max": round(float(max_temps[i]), 1) if i < len(max_temps) else 30.0,
                "temp_min": round(float(min_temps[i]), 1) if i < len(min_temps) else 22.0,
                "precipitation_sum": round(float(precip_sums[i]), 1) if i < len(precip_sums) else 0.0,
                "precipitation_probability": int(precip_probs[i]) if i < len(precip_probs) else 0,
                "wind_speed_max": round(float(wind_maxs[i]), 1) if i < len(wind_maxs) else 10.0,
                "uv_index_max": round(float(uv_maxs[i]), 1) if i < len(uv_maxs) else 5.0,
            })

        return days_list

    def generate_offline_bundle(
        self,
        lat: float,
        lon: float,
        location_name: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Bundle all 7-day weather information, insights, and offline instructions for local device saving."""
        current_metrics = self.extract_current_metrics(payload)
        daily_forecast = self.extract_daily_forecast(payload)

        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # General weather summary for everyday users
        max_rain_prob = max([d["precipitation_probability"] for d in daily_forecast]) if daily_forecast else 0
        rainy_days = [d["day_name"] for d in daily_forecast if d["precipitation_probability"] >= 50]
        hot_days = [d["day_name"] for d in daily_forecast if d["temp_max"] >= 35.0]

        summary_points = []
        if current_metrics["precipitation"] > 0 or current_metrics["daily_rain_probability"] > 50:
            summary_points.append("Rain expected today - carry an umbrella when heading out.")
        else:
            summary_points.append("Fair weather today with comfortable conditions.")

        if rainy_days:
            summary_points.append(f"Rain likely on: {', '.join(rainy_days)}.")
        else:
            summary_points.append("Dry conditions predicted across the upcoming 7 days.")

        if hot_days:
            summary_points.append(f"High temperatures expected on {', '.join(hot_days)}; stay hydrated.")

        return {
            "version": "1.0",
            "app": "WeatherGPT",
            "downloaded_at": timestamp_str,
            "location": {
                "latitude": lat,
                "longitude": lon,
                "name": location_name,
            },
            "current_weather": current_metrics,
            "seven_day_forecast": daily_forecast,
            "offline_insights": summary_points,
            "data_source": "Open-Meteo REST API",
            "note": "This 7-day weather dataset is saved offline on your device and can be viewed anytime without internet."
        }

weather_service = WeatherService()
