import os

# Define the glossary data with exactly 40 terms mapped across Hindi, Kannada, Telugu, Tamil, Marathi
glossary_data = [
    # --- Crops (15 Terms) ---
    {
        "english": "Wheat",
        "hindi": "गेहूं (Gehu)",
        "kannada": "ಗೋಧಿ (Godhi)",
        "telugu": "గోధుమ (Godhuma)",
        "tamil": "கோதுமை (Godhumai)",
        "marathi": "गहू (Gahu)",
        "category": "Crop"
    },
    {
        "english": "Rice / Paddy",
        "hindi": "धान (Dhan)",
        "kannada": "ಭತ್ತ (Bhatta)",
        "telugu": "వరి (Vari)",
        "tamil": "நெல் (Nel)",
        "marathi": "भात (Bhat) / धान (Dhan)",
        "category": "Crop"
    },
    {
        "english": "Cotton",
        "hindi": "कपास (Kapas)",
        "kannada": "ಹತ್ತಿ (Hatti)",
        "telugu": "పత్తి (Patti)",
        "tamil": "பருத்தி (Paruthi)",
        "marathi": "कापूस (Kapus)",
        "category": "Crop"
    },
    {
        "english": "Soybean",
        "hindi": "सोयाबीन (Soyabean)",
        "kannada": "ಸೋಯಾಬೀನ್ (Soyabean)",
        "telugu": "సోయాబీన్ (Soyabean)",
        "tamil": "சோயாபீன் (Soyabean)",
        "marathi": "सोयाबीन (Soyabean)",
        "category": "Crop"
    },
    {
        "english": "Tomato",
        "hindi": "टमाटर (Tamatar)",
        "kannada": "ಟೊಮೆಟೊ (Tomato)",
        "telugu": "టమోటా (Tomato)",
        "tamil": "தக்காளி (Thakkali)",
        "marathi": "टोमॅटो (Tomato)",
        "category": "Crop"
    },
    {
        "english": "Onion",
        "hindi": "प्याज (Pyaaz)",
        "kannada": "ಈರುಳ್ಳಿ (Eerulli)",
        "telugu": "ఉల్లిపాయ (Ullipaya)",
        "tamil": "வெங்காயம் (Vengayam)",
        "marathi": "कांदा (Kanda)",
        "category": "Crop"
    },
    {
        "english": "Potato",
        "hindi": "आलू (Aloo)",
        "kannada": "ಆಲೂಗಡ್ಡೆ (Aloogadde)",
        "telugu": "బంగాళాదుంప (Bangaladumpa)",
        "tamil": "உருளைக்கிழங்கு (Urulaikilangu)",
        "marathi": "बटाटा (Batata)",
        "category": "Crop"
    },
    {
        "english": "Chickpea",
        "hindi": "चना (Chana)",
        "kannada": "ಕಡಲೆ (Kadale)",
        "telugu": "శనగలు (Senagalu)",
        "tamil": "கொண்டைக்கடலை (Kondaikadalai)",
        "marathi": "हरभरा (Harbhara)",
        "category": "Crop"
    },
    {
        "english": "Chilli",
        "hindi": "मिर्च (Mirch)",
        "kannada": "ಮೆಣಸಿನಕಾಯಿ (Menasinakai)",
        "telugu": "మిరపకాయ (Mirapakaya)",
        "tamil": "மிளகாய் (Milagai)",
        "marathi": "मिरची (Mirchi)",
        "category": "Crop"
    },
    {
        "english": "Sugarcane",
        "hindi": "गन्ना (Ganna)",
        "kannada": "ಕಬ್ಬು (Kabbu)",
        "telugu": "చెరకు (Cheraku)",
        "tamil": "கரும்பு (Karumbu)",
        "marathi": "ऊस (Uus)",
        "category": "Crop"
    },
    {
        "english": "Pigeon Pea",
        "hindi": "अरहर (Arhar)",
        "kannada": "ತೊಗರಿ (Togari)",
        "telugu": "కంది (Kandi)",
        "tamil": "துவரை (Thuvarai)",
        "marathi": "तूर (Tur)",
        "category": "Crop"
    },
    {
        "english": "Groundnut",
        "hindi": "मूंगफली (Mungfali)",
        "kannada": "ಕಡಲೇಕಾಯಿ (Kadalekayi)",
        "telugu": "పల్లీలు (Pallilu) / వేరుశనగ (Verusanaga)",
        "tamil": "நிலக்கடலை (Nilakkadalai)",
        "marathi": "भुईमूग (Bhuimug)",
        "category": "Crop"
    },
    {
        "english": "Maize",
        "hindi": "मक्का (Makka)",
        "kannada": "ಮೆಕ್ಕೆಜೋಳ (Mekkejola)",
        "telugu": "مొక్కజొన్న (Mokkajonna)",
        "tamil": "மக்காச்சோளம் (Makkacholam)",
        "marathi": "मका (Maka)",
        "category": "Crop"
    },
    {
        "english": "Brinjal",
        "hindi": "बैंगन (Baigan)",
        "kannada": "ಬದನೆಕಾಯಿ (Badanekayi)",
        "telugu": "వంకాయ (Vankaya)",
        "tamil": "கத்தரிக்காய் (Kathirikai)",
        "marathi": "वांगे (Wange)",
        "category": "Crop"
    },
    {
        "english": "Coconut",
        "hindi": "नारियल (Nariyal)",
        "kannada": "ತೆಂಗಿನಕಾಯಿ (Tenginakayi)",
        "telugu": "కొబ్బరికాయ (Kobbarikaya)",
        "tamil": "தேங்காய் (Thengai)",
        "marathi": "नारळ (Naral)",
        "category": "Crop"
    },

    # --- Diseases & Pests (10 Terms) ---
    {
        "english": "Blight",
        "hindi": "झुलसा (Jhulsa)",
        "kannada": "ಬೆಂಕಿ ರೋಗ (Benki roga)",
        "telugu": "ఆకు ఎండు తెగులు (Aaku endu thegulu)",
        "tamil": "கருகல் நோய் (Karugal noi)",
        "marathi": "करपा (Karpa)",
        "category": "Disease/Pest"
    },
    {
        "english": "Leaf Spot",
        "hindi": "पत्ती धब्बा (Patti dhabba)",
        "kannada": "ಚುಕ್ಕೆ ರೋಗ (Chukke roga)",
        "telugu": "ఆకు మచ్చ తెగులు (Aaku macha thegulu)",
        "tamil": "இலைப்புள்ளி நோய் (Ilai pulli noi)",
        "marathi": "पानांवरील ठिपके (Paanavaril thipke) / करपा (Karpa)",
        "category": "Disease/Pest"
    },
    {
        "english": "Rust",
        "hindi": "गेरूई (Gerui) / तांबेरा (Tambera)",
        "kannada": "ತುಕ್ಕು ರೋಗ (Tukku roga)",
        "telugu": "తుప్పు తెగులు (Tuppu thegulu)",
        "tamil": "துரு நோய் (Thuru noi)",
        "marathi": "तांबेरा (Tambera)",
        "category": "Disease/Pest"
    },
    {
        "english": "Wilt",
        "hindi": "उकठा (Uktha)",
        "kannada": "ಬಾಡಲು ರೋಗ (Badalu roga) / ಒಣಗು ರೋಗ (Onagu roga)",
        "telugu": "వడలు తెగులు (Vadalu thegulu)",
        "tamil": "வாடல் நோய் (Vadal noi)",
        "marathi": "मर (Mar)",
        "category": "Disease/Pest"
    },
    {
        "english": "Powdery Mildew",
        "hindi": "भभूत रोग (Bhabhoot rog) / भस्म रोग (Bhasma rog)",
        "kannada": "ಬೂದಿ ರೋಗ (Boodi roga)",
        "telugu": "బూడిద తెగులు (Budida thegulu)",
        "tamil": "சாம்பல் நோய் (Sambal noi)",
        "marathi": "भुरी (Bhuri)",
        "category": "Disease/Pest"
    },
    {
        "english": "Stem Borer",
        "hindi": "तना छेदक (Tana chhedak)",
        "kannada": "ಕಾಂಡ ಕೊರಕ ಹುಳು (Kanda koraka hula)",
        "telugu": "కాండం తొలిచే పురుగు (Kanda tholuchu purugu)",
        "tamil": "தண்டு துளைப்பான் (Thandu thulaipan)",
        "marathi": "खोडकिडा (Khodkida)",
        "category": "Disease/Pest"
    },
    {
        "english": "Caterpillar",
        "hindi": "सुंडी (Sundi) / इल्ली (Illi)",
        "kannada": "ಕಂಬಳಿ ಹುಳು (Kambali hula)",
        "telugu": "పచ్చ పురుగు (Pacha purugu) / గొంగళి పురుగు (Gongali purugu)",
        "tamil": "புழு (Puzhu)",
        "marathi": "अळी (Aali)",
        "category": "Disease/Pest"
    },
    {
        "english": "Thrips",
        "hindi": "थ्रिप्स (Thrips)",
        "kannada": "ನುಸಿ ಹುಳು (Nusi hula) / ಸೂಜಿ ಹುಳು (Suzi hula)",
        "telugu": "తామర పురుగులు (Thamara purugulu) / నల్లి (Nalli)",
        "tamil": "இலைப்பேன் (Ilaippen)",
        "marathi": "थ्रिप्स (Thrips)",
        "category": "Disease/Pest"
    },
    {
        "english": "Whitefly",
        "hindi": "सफेद मक्खी (Safed makkhi)",
        "kannada": "ಬಿಳಿ ನೊಣ (Bili nona)",
        "telugu": "తెల్ల దోమ (Tella doma)",
        "tamil": "வெள்ளை ஈ (Vellai ee)",
        "marathi": "पांढरी माशी (Pandhri mashi)",
        "category": "Disease/Pest"
    },
    {
        "english": "Bollworm",
        "hindi": "डोडे की सुंडी (Dode ki sundi)",
        "kannada": "ಕಾಯಿ ಕೊರಕ ಹುಳು (Kayi koraka hula)",
        "telugu": "గోల పురుగు (Gola purugu)",
        "tamil": "பஞ்சுப் புழு (Panju puzhu) / காய் புழு (Kayi puzhu)",
        "marathi": "बोंड अळी (Bonda aali)",
        "category": "Disease/Pest"
    },

    # --- Weather & Farm Actions (10 Terms) ---
    {
        "english": "Rain",
        "hindi": "बारिश (Barish) / वर्षा (Varsha)",
        "kannada": "ಮಳೆ (Male)",
        "telugu": "వర్షం (Varsham)",
        "tamil": "மழை (Mazhai)",
        "marathi": "पाऊस (Paus)",
        "category": "Weather/Action"
    },
    {
        "english": "Sowing",
        "hindi": "बुवाई (Buwai)",
        "kannada": "ಬಿತ್ತನೆ (Bittane)",
        "telugu": "విత్తడం (Vittadam)",
        "tamil": "விதைப்பு (Vithaipu)",
        "marathi": "पेरणी (Perni)",
        "category": "Weather/Action"
    },
    {
        "english": "Harvesting",
        "hindi": "कटाई (Katai)",
        "kannada": "ಕಟಾವು (Katavu) / ಕೊಯ್ಲು (Koylu)",
        "telugu": "కోత (Kotha)",
        "tamil": "அறுவடை (Aruvadai)",
        "marathi": "कापणी (Kapni) / काढणी (Kadhni)",
        "category": "Weather/Action"
    },
    {
        "english": "Fertilizer",
        "hindi": "खाद (Khad)",
        "kannada": "ಗೊಬ್ಬರ (Gobbara)",
        "telugu": "ఎరువు (Eruvu)",
        "tamil": "உரம் (Uram)",
        "marathi": "खत (Khat)",
        "category": "Weather/Action"
    },
    {
        "english": "Irrigation",
        "hindi": "सिंचाई (Sinchai) / पानी लगाना (Paani lagana)",
        "kannada": "ನೀರಾವರಿ (Neeravari)",
        "telugu": "నీటి పారుదల (Neeti parudala) / నీళ్లు పెట్టడం (Neellu pettadam)",
        "tamil": "நீர்ப்பாசனம் (Neerpasanam)",
        "marathi": "पाणी देणे (Paani dene) / सिंचन (Sinchan)",
        "category": "Weather/Action"
    },
    {
        "english": "Seed",
        "hindi": "बीज (Beej)",
        "kannada": "ಬೀಜ (Beeja)",
        "telugu": "విత్తనం (Vittanam)",
        "tamil": "விதை (Vidhai)",
        "marathi": "बी (Bi) / बियाणे (Biyane)",
        "category": "Weather/Action"
    },
    {
        "english": "Soil",
        "hindi": "मिट्टी (Mitti)",
        "kannada": "ಮಣ್ಣು (Mannu)",
        "telugu": "మట్టి (Matti) / నేల (Nela)",
        "tamil": "மண் (Man)",
        "marathi": "माती (Mati)",
        "category": "Weather/Action"
    },
    {
        "english": "Pesticide",
        "hindi": "कीटनाशक (Keetnashak) / दवा (Dawa)",
        "kannada": "ಕೀಟನಾಶಕ (Keetanashaka)",
        "telugu": "పురుగుల మందు (Purugula mandu)",
        "tamil": "பூச்சிக்கொல்லி (Poochikkolli)",
        "marathi": "कीटकनाशक (Keetaknashak)",
        "category": "Weather/Action"
    },
    {
        "english": "Disease",
        "hindi": "रोग (Rog) / बीमारी (Beemari)",
        "kannada": "ರೋಗ (Roga)",
        "telugu": "తెగులు (Thegulu)",
        "tamil": "நோய் (Noi)",
        "marathi": "रोग (Rog)",
        "category": "Disease/Pest"
    },
    {
        "english": "Drought",
        "hindi": "सूखा (Sukha)",
        "kannada": "ಬರ (Bara)",
        "telugu": "కరువు (Karuvu)",
        "tamil": "வறட்சி (Varatchi)",
        "marathi": "दुष्काळ (Dushkal)",
        "category": "Weather/Action"
    },

    # --- Market / Mandi (5 Terms) ---
    {
        "english": "Mandi / Market",
        "hindi": "मंडी (Mandi) / बाजार (Bazar)",
        "kannada": "ಮಾರುಕಟ್ಟೆ (Marukatte) / ಮಂಡಿ (Mandi)",
        "telugu": "మార్కెట్ (Market) / మండి (Mandi)",
        "tamil": "மண்டி (Mandi) / சந்தை (Santhai)",
        "marathi": "मंडी (Mandi) / बाजार (Bazar)",
        "category": "Market"
    },
    {
        "english": "Price",
        "hindi": "दाम (Daam) / कीमत (Keemat)",
        "kannada": "ಬೆಲೆ (Bele)",
        "telugu": "ధర (Dhara)",
        "tamil": "விலை (Vilai)",
        "marathi": "दर (Dar) / भाव (Bhav)",
        "category": "Market"
    },
    {
        "english": "Rate",
        "hindi": "भाव (Bhav)",
        "kannada": "ದರ (Dhara)",
        "telugu": "రేటు (Rate)",
        "tamil": "விலை (Vilai)",
        "marathi": "भाव (Bhav)",
        "category": "Market"
    },
    {
        "english": "MSP (Minimum Support Price)",
        "hindi": "न्यूनतम समर्थन मूल्य (Nyunatam samarthan mulya)",
        "kannada": "ಕನಿಷ್ಠ ಬೆಂಬಲ ಬೆಲೆ (Kanishtha bembala bele)",
        "telugu": "కనీస మద్దతు ధర (Kaneesa maddathu dhara)",
        "tamil": "குறைந்தபட்ச ஆதரவு விலை (Kurainthabatcha aatharavu vilai)",
        "marathi": "हमीभाव (Hamibhav)",
        "category": "Market"
    },
    {
        "english": "Yield",
        "hindi": "पैदावार (Paidawar) / उपज (Upaj)",
        "kannada": "ಇಳುವರಿ (Iluvari)",
        "telugu": "దిగుబడి (Digubadi)",
        "tamil": "மகசூல் (Mahasool)",
        "marathi": "उत्पादन (Utpadan) / पीक (Pik)",
        "category": "Market"
    }
]

# Validation
print(f"Total entries: {len(glossary_data)}")
assert len(glossary_data) >= 30, "Glossary must have at least 30 terms"

for entry in glossary_data:
    for lang in ['hindi', 'kannada', 'telugu', 'tamil', 'marathi']:
        assert entry[lang].strip(), f"Missing translation for {entry['english']} in {lang}"
    assert entry['category'] in ['Crop', 'Disease/Pest', 'Weather/Action', 'Market'], f"Invalid category {entry['category']}"

print("All entries validated successfully!")

# Write to agricultural_glossary.md
glossary_path = r"d:\Projects\FarmGenius\agricultural_glossary.md"

markdown_content = """# FarmGenius — Regional Agricultural Glossary

This glossary compiles key agricultural terms translated and localized across five major Indian languages: **Hindi, Kannada, Telugu, Tamil, and Marathi**. It is based on authentic farmer dialect vocabulary and queries (from `FARMER_CORPUS.md`), mapping standard English concepts to the actual regional terminology used by farmers in the field.

This resource is designed to support multi-lingual NLP model training, localized chatbot response generation, and voice translation pipelines in the FarmGenius project.

---

## Glossary Summary
- **Total Terms**: {total_terms}
- **Categories**: Crops, Diseases & Pests, Weather & Farm Actions, Market & Mandi
- **Target Languages**: Hindi, Kannada, Telugu, Tamil, Marathi

---

## Glossary Table

| English Term | Hindi | Kannada | Telugu | Tamil | Marathi | Category |
|---|---|---|---|---|---|---|
""".format(total_terms=len(glossary_data))

for entry in glossary_data:
    markdown_content += f"| {entry['english']} | {entry['hindi']} | {entry['kannada']} | {entry['telugu']} | {entry['tamil']} | {entry['marathi']} | {entry['category']} |\n"

with open(glossary_path, "w", encoding="utf-8") as f:
    f.write(markdown_content)

print(f"Glossary successfully written to {glossary_path}")
