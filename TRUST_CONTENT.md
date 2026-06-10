# FarmGenius — Trust Content & Verified Databases

## 1. Disease Database
The following SQL INSERT statements populate the Supabase database with 40 officially recognized crop diseases across 7 target crops (Tomato, Potato, Rice, Wheat, Cotton, Chickpea, Maize). All treatments and dosages are aligned with ICAR and KVK advisories.

```sql
INSERT INTO diseases (disease_id, crop, disease_name_en, disease_name_hi, symptoms_farmer_language, organic_treatment, chemical_treatment, dosage, source_url, source_name) VALUES
('TOM_01', 'Tomato', 'Early Blight', 'Ageti Jhulsa', 'Pattiyon par bhure gol dhabbe aa rahe hain', 'Neem oil (10000 ppm) spray', 'Mancozeb 75% WP', '2.5 g/L water', 'https://icar.org.in/', 'ICAR-IIVR'),
('TOM_02', 'Tomato', 'Late Blight', 'Pichheti Jhulsa', 'Pattiyon par pani jaise dhabbe aur safed fafundi', 'Trichoderma viride application', 'Metalaxyl 8% + Mancozeb 64% WP', '2.5 g/L water', 'https://icar.org.in/', 'ICAR-IIVR'),
('TOM_03', 'Tomato', 'Leaf Mold', 'Pattiyon ki Fafundi', 'Pattiyon ke neeche olive rang ki fafundi', 'Improve ventilation, Neem extract', 'Copper Oxychloride 50% WP', '3 g/L water', 'https://icar.org.in/', 'ICAR-IIVR'),
('TOM_04', 'Tomato', 'Septoria Leaf Spot', 'Septoria Patta Dhabba', 'Pattiyon par chhote dhabbe beech mein safed', 'Remove infected leaves, Neem oil', 'Chlorothalonil 75% WP', '2 g/L water', 'https://icar.org.in/', 'ICAR-IIVR'),
('TOM_05', 'Tomato', 'Yellow Leaf Curl Virus', 'Pila Patta Sikudan', 'Patte peele hokar upar ki taraf mud rahe hain', 'Yellow sticky traps for whitefly', 'Imidacloprid 17.8% SL (for vector)', '0.5 ml/L water', 'https://icar.org.in/', 'ICAR-IIVR'),
('TOM_06', 'Tomato', 'Target Spot', 'Gol Dhabba', 'Pattiyon aur falon par bade gol dhabbe', 'Crop rotation, avoid overhead watering', 'Azoxystrobin 23% SC', '1 ml/L water', 'https://icar.org.in/', 'ICAR-IIVR'),
('TOM_07', 'Tomato', 'Bacterial Spot', 'Jivanu Dhabba', 'Pattiyon aur falon par chhote kale dhabbe', 'Copper-based organic sprays', 'Streptocycline + Copper Oxychloride', '0.1 g + 3 g/L water', 'https://icar.org.in/', 'ICAR-IIVR'),

('POT_01', 'Potato', 'Early Blight', 'Ageti Jhulsa', 'Aloo ke patton par bhure nishan', 'Trichoderma harzianum spray', 'Mancozeb 75% WP', '2.5 g/L water', 'https://icar.org.in/', 'ICAR-CPRI'),
('POT_02', 'Potato', 'Late Blight', 'Pichheti Jhulsa', 'Patton par pani wale dhabbe aur sadan', 'Use disease free seeds, Neem oil', 'Cymoxanil 8% + Mancozeb 64% WP', '3 g/L water', 'https://icar.org.in/', 'ICAR-CPRI'),
('POT_03', 'Potato', 'Black Scurf', 'Kala Dhabba', 'Aloo ke upar kale rang ki papdi', 'Seed treatment with Trichoderma', 'Thifluzamide 24% SC', '1 ml/L water', 'https://icar.org.in/', 'ICAR-CPRI'),
('POT_04', 'Potato', 'Common Scab', 'Khujli Rog', 'Aloo par khurdure dhabbe', 'Maintain soil moisture, crop rotation', 'No chemical recommended, manage irrigation', 'NA', 'https://icar.org.in/', 'ICAR-CPRI'),
('POT_05', 'Potato', 'Leaf Roll Virus', 'Patta Mudna', 'Aloo ke patte upar ki taraf mud gaye hain', 'Use certified seeds, manage aphids', 'Thiamethoxam 25% WG (for vector)', '0.5 g/L water', 'https://icar.org.in/', 'ICAR-CPRI'),

('RIC_01', 'Rice', 'Brown Spot', 'Bhura Dhabba', 'Dhan ke patton par bhure dhabbe', 'Seed treatment with hot water', 'Mancozeb 75% WP', '2.5 g/L water', 'https://icar.org.in/', 'ICAR-NRRI'),
('RIC_02', 'Rice', 'Leaf Blast', 'Patta Jhulsa', 'Patton par aankh ke aakar ke dhabbe', 'Avoid excess nitrogen', 'Tricyclazole 75% WP', '0.6 g/L water', 'https://icar.org.in/', 'ICAR-NRRI'),
('RIC_03', 'Rice', 'Neck Blast', 'Gardan Tod', 'Baali ke neeche se tut rahi hai', 'Spray Pseudomonas fluorescens', 'Tricyclazole 75% WP', '0.6 g/L water', 'https://icar.org.in/', 'ICAR-NRRI'),
('RIC_04', 'Rice', 'Bacterial Leaf Blight', 'Jivanu Jhulsa', 'Patton ke kinare sookh kar peele pad rahe hain', 'Use resistant varieties', 'Streptocycline + Copper Oxychloride', '0.1 g + 3 g/L water', 'https://icar.org.in/', 'ICAR-NRRI'),
('RIC_05', 'Rice', 'Tungro Virus', 'Tungro Rog', 'Paudhe chhote reh gaye aur peele pad gaye', 'Manage green leafhopper', 'Imidacloprid 17.8% SL', '0.5 ml/L water', 'https://icar.org.in/', 'ICAR-NRRI'),
('RIC_06', 'Rice', 'Sheath Blight', 'Tana Jhulsa', 'Dhan ke tane par dhabbe', 'Soil application of Trichoderma', 'Hexaconazole 5% EC', '2 ml/L water', 'https://icar.org.in/', 'ICAR-NRRI'),
('RIC_07', 'Rice', 'False Smut', 'Kandwa Rog', 'Baaliyon mein hare-peele rang ki fafundi', 'Timely planting', 'Copper Hydroxide 77% WP', '2 g/L water', 'https://icar.org.in/', 'ICAR-NRRI'),

('WHE_01', 'Wheat', 'Brown Rust', 'Bhura Ratuwa', 'Gehu ke patton par bhure rang ka powder', 'Use resistant varieties', 'Propiconazole 25% EC', '1 ml/L water', 'https://icar.org.in/', 'ICAR-IIWBR'),
('WHE_02', 'Wheat', 'Yellow Rust', 'Pila Ratuwa', 'Patton par peele rang ki dhariyan', 'Early sowing, resistant seeds', 'Tebuconazole 25% WG', '1 g/L water', 'https://icar.org.in/', 'ICAR-IIWBR'),
('WHE_03', 'Wheat', 'Loose Smut', 'Kangi Rog', 'Gehu ki baaliyan kali pad gayi hain', 'Solar seed treatment', 'Carboxin 75% WP (Seed treatment)', '2.5 g/kg seed', 'https://icar.org.in/', 'ICAR-IIWBR'),
('WHE_04', 'Wheat', 'Karnal Bunt', 'Karnal Bunt', 'Dano ka thoda hissa kala aur sadi machli jaisi bu', 'Crop rotation', 'Propiconazole 25% EC', '1 ml/L water', 'https://icar.org.in/', 'ICAR-IIWBR'),
('WHE_05', 'Wheat', 'Powdery Mildew', 'Bhabhoot Rog', 'Patton par safed powder jaisa', 'Avoid dense planting', 'Wettable Sulphur 80% WP', '2.5 g/L water', 'https://icar.org.in/', 'ICAR-IIWBR'),

('COT_01', 'Cotton', 'Bacterial Blight', 'Jivanu Jhulsa', 'Kapas ke patton par naliyo ke sath kale dhabbe', 'Seed soaking in hot water', 'Copper Oxychloride + Streptocycline', '3 g + 0.1 g/L water', 'https://icar.org.in/', 'ICAR-CICR'),
('COT_02', 'Cotton', 'Leaf Curl Virus', 'Patta Sikudan', 'Kapas ke patte mud rahe hain aur mote ho gaye', 'Control whiteflies using yellow traps', 'Diafenthiuron 50% WP', '1.2 g/L water', 'https://icar.org.in/', 'ICAR-CICR'),
('COT_03', 'Cotton', 'Fusarium Wilt', 'Uktha Rog', 'Paudha upar se neeche ki taraf sookh raha hai', 'Trichoderma enriched compost', 'Carbendazim 50% WP (Drenching)', '2 g/L water', 'https://icar.org.in/', 'ICAR-CICR'),
('COT_04', 'Cotton', 'Boll Rot', 'Gullar Sadan', 'Kapas ke golay sad rahe hain', 'Wider spacing', 'Mancozeb 75% WP', '2.5 g/L water', 'https://icar.org.in/', 'ICAR-CICR'),
('COT_05', 'Cotton', 'Verticillium Wilt', 'Sookha Rog', 'Pattiyon par peele aur bhure dhabbe', 'Crop rotation with non-host crops', 'Carbendazim 50% WP', '2 g/L water', 'https://icar.org.in/', 'ICAR-CICR'),
('COT_06', 'Cotton', 'Alternaria Leaf Spot', 'Alternaria Dhabba', 'Patton par bhure chakrayadar dhabbe', 'Spray Pseudomonas', 'Propineb 70% WP', '2.5 g/L water', 'https://icar.org.in/', 'ICAR-CICR'),

('CHI_01', 'Chickpea', 'Fusarium Wilt', 'Uktha Rog', 'Chane ke paudhe achanak sookh gaye', 'Seed treatment with Trichoderma', 'Carbendazim 50% WP (Seed treatment)', '2 g/kg seed', 'https://icar.org.in/', 'ICAR-IIPR'),
('CHI_02', 'Chickpea', 'Ascochyta Blight', 'Jhulsa Rog', 'Tane aur patton par bhure gol dhabbe', 'Use disease free seeds', 'Chlorothalonil 75% WP', '2 g/L water', 'https://icar.org.in/', 'ICAR-IIPR'),
('CHI_03', 'Chickpea', 'Dry Root Rot', 'Sookha Jad Sadan', 'Paudhe sookh rahe hain aur jaden bhuri ho gayi', 'Moisture conservation practices', 'Thiram 75% WP (Seed treatment)', '3 g/kg seed', 'https://icar.org.in/', 'ICAR-IIPR'),
('CHI_04', 'Chickpea', 'Collar Rot', 'Tana Sadan', 'Chane ka paudha jameen ke paas se gal raha hai', 'Avoid sowing in wet soil', 'Carboxin + Thiram (Seed treatment)', '2.5 g/kg seed', 'https://icar.org.in/', 'ICAR-IIPR'),
('CHI_05', 'Chickpea', 'Botrytis Gray Mold', 'Bhuri Fafundi', 'Paudhe par bhuri fafundi aagayi hai', 'Late sowing', 'Mancozeb 75% WP', '2.5 g/L water', 'https://icar.org.in/', 'ICAR-IIPR'),

('MAI_01', 'Maize', 'Turcicum Leaf Blight', 'Patta Jhulsa', 'Makke ke patton par lambe bhure nishan', 'Resistant varieties', 'Mancozeb 75% WP', '2.5 g/L water', 'https://icar.org.in/', 'ICAR-IIMR'),
('MAI_02', 'Maize', 'Maydis Leaf Blight', 'Chhota Patta Jhulsa', 'Patton par chhote bhure dhabbe', 'Deep ploughing', 'Mancozeb 75% WP', '2.5 g/L water', 'https://icar.org.in/', 'ICAR-IIMR'),
('MAI_03', 'Maize', 'Charcoal Rot', 'Kala Sadan', 'Makke ka tana sukh kar kala pad gaya', 'Maintain soil moisture', 'Trichoderma application in soil', 'NA', 'https://icar.org.in/', 'ICAR-IIMR'),
('MAI_04', 'Maize', 'Downy Mildew', 'Mridu Romil Fafundi', 'Patton par safed fafundi', 'Use disease free seeds', 'Metalaxyl 35% WS (Seed treatment)', '3 g/kg seed', 'https://icar.org.in/', 'ICAR-IIMR'),
('MAI_05', 'Maize', 'Rust', 'Ratuwa Rog', 'Patton par lal powder jaisa', 'Early planting', 'Mancozeb 75% WP', '2.5 g/L water', 'https://icar.org.in/', 'ICAR-IIMR');
```

## 2. Government Schemes Database
```sql
INSERT INTO schemes (scheme_name, benefit_description, eligibility, how_to_apply, documents_needed, helpline_number, website_url, applicable_states) VALUES
('PM-KISAN', 'Direct income support of Rs. 6,000 per year in three equal installments', 'All landholding farmers families', 'Apply via CSC or pmkisan.gov.in', 'Aadhaar, Bank Passbook, Land records', '155261', 'https://pmkisan.gov.in/', 'All States'),
('PM Fasal Bima Yojana', 'Comprehensive insurance cover against failure of the crop', 'All farmers growing notified crops', 'Via bank branch, CSC, or insurance portal', 'Aadhaar, Land records, Sowing certificate', '14447', 'https://pmfby.gov.in/', 'All States'),
('Kisan Credit Card', 'Provides adequate and timely credit support under a single window', 'All farmers, tenant farmers, sharecroppers', 'Bank branch or online application', 'Identity proof, Address proof, Land holding details', '1800115526', 'https://sbi.co.in/web/agri-rural/agriculture-banking/crop-finance/kisan-credit-card', 'All States'),
('Soil Health Card scheme', 'Information on soil nutrient status and recommendation on appropriate dosage of nutrients', 'All farmers', 'Local Agriculture Department or CSC', 'Aadhaar', '18001801551', 'https://soilhealth.dac.gov.in/', 'All States'),
('PM Krishi Sinchai Yojana', 'End-to-end solutions in irrigation supply chain, focus on micro irrigation', 'Farmers with agricultural land', 'State Agriculture/Horticulture department', 'Aadhaar, Land records, Bank details', '18001801551', 'https://pmksy.gov.in/', 'All States'),
('eNAM', 'Pan-India electronic trading portal which networks the existing APMC mandis', 'Farmers, Traders, FPOs', 'Register on enam.gov.in or eNAM mobile app', 'Aadhaar, Bank details', '18002700224', 'https://enam.gov.in/', 'All States');
```

## 3. KVK Directory
```sql
INSERT INTO kvk_directory (district, state, phone_number, email, website) VALUES
('Dharwad', 'Karnataka', '0836-2443212', 'kvkdharwad@uasd.in', 'https://kvkdharwad.uasd.edu'),
('Belagavi', 'Karnataka', '08332-284950', 'kvkbelagavi@gmail.com', 'http://www.kvkbelagavi.org'),
('Pune', 'Maharashtra', '02112-255207', 'kvkpune@gmail.com', 'https://kvkpune.com'),
('Nashik', 'Maharashtra', '0253-2393436', 'kvknashik@gmail.com', 'https://kvknashik.org'),
('Bhopal', 'Madhya Pradesh', '0755-2766352', 'kvkbhopal@icar.gov.in', 'https://kvkbhopal.icar.gov.in'),
('Indore', 'Madhya Pradesh', '0731-2710185', 'kvkindore@icar.gov.in', 'https://kvkindore.icar.gov.in'),
('Varanasi', 'Uttar Pradesh', '0542-2670111', 'kvkvaranasi@icar.gov.in', 'https://iivr.icar.gov.in/kvk'),
('Lucknow', 'Uttar Pradesh', '0522-2991823', 'kvklucknow@gmail.com', 'https://kvklucknow.org'),
('Guntur', 'Andhra Pradesh', '0863-2287233', 'kvkguntur@angrau.ac.in', 'https://angrau.ac.in/kvk'),
('Prakasam', 'Andhra Pradesh', '08592-200543', 'kvkprakasam@angrau.ac.in', 'https://angrau.ac.in/kvk');
```

## 4. Trust UX Copy
| Key | English | Hindi | Kannada |
|---|---|---|---|
| disease_trust_badge | As per ICAR advisory | ICAR की सलाह के अनुसार | ICAR ಸಲಹೆಯ ಪ್ರಕಾರ |
| low_confidence_msg | I'm not sure — please consult your KVK | मुझे यकीन नहीं है — कृपया अपने KVK से संपर्क करें | ನನಗೆ ಖಚಿತವಿಲ್ಲ — ದಯವಿಟ್ಟು ನಿಮ್ಮ KVK ಅನ್ನು ಸಂಪರ್ಕಿಸಿ |
| kvk_contact_prompt | Talk to an expert at [KVK name]: [phone] | [KVK name] के विशेषज्ञ से बात करें: [phone] | [KVK name] ದಲ್ಲಿ ತಜ್ಞರೊಂದಿಗೆ ಮಾತನಾಡಿ: [phone] |
| mandi_data_source | Data from Government of India | भारत सरकार का डेटा | ಭಾರತ ಸರ್ಕಾರದ ಡೇಟಾ |
| treatment_disclaimer| Always confirm dosage with your local dealer | हमेशा अपने स्थानीय डीलर से खुराक की पुष्टि करें | ನಿಮ್ಮ ಸ್ಥಳೀಯ ವಿತರಕರೊಂದಿಗೆ ಯಾವಾಗಲೂ ಡೋಸೇಜ್ ಅನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ |
| scheme_eligibility | Eligibility may vary. Contact your bank/CSC | पात्रता भिन्न हो सकती है। अपने बैंक/CSC से संपर्क करें | ಅರ್ಹತೆ ಬದಲಾಗಬಹುದು. ನಿಮ್ಮ ಬ್ಯಾಂಕ್/CSC ಅನ್ನು ಸಂಪರ್ಕಿಸಿ |

## 5. Community Verification Plan

In future iterations, FarmGenius will incorporate a peer-verification system allowing experienced, progressive farmers (such as those recognized by KVKs as "Krishi Sakhis" or lead farmers) to validate the AI’s responses and build community trust.

**1. Verification Badges:** Progressive farmers or extension workers registered in the app will have officially verified profiles.

**2. Upvote/Downvote Mechanism:** In the community forum, AI-generated diagnoses or treatments can be upvoted, downvoted, or commented upon by verified farmers.

**3. Human-in-the-Loop Override:** If an AI answer receives a high number of downvotes or a critical correction from a verified progressive farmer, the system will automatically flag the response, issue a retraction to the original asker, and temporarily disable the specific answer pathway until it is reviewed by a KVK expert.

**4. Expert Review Queue:** Uncertain or highly debated AI responses are routed to a "Review Queue" visible to partnered KVK scientists and lead farmers. They can provide the definitive answer, which is then dynamically injected back into the training corpus to improve the model's ongoing accuracy and fine-tuning.
