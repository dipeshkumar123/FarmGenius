import os

import psycopg2
from psycopg2 import OperationalError
from dotenv import load_dotenv


load_dotenv()


def _get_db_url() -> str:
    candidates = [
        os.getenv("DATABASE_URL"),
        os.getenv("SUPABASE_DB_URL"),
        os.getenv("SUPABASE_POOLER_URL"),
    ]

    for candidate in candidates:
        if candidate:
            return candidate

    raise RuntimeError(
        "No database connection string found. Set DATABASE_URL or SUPABASE_POOLER_URL in .env."
    )


db_url = _get_db_url()

def run_setup():
    try:
        print("Connecting to Supabase...")
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cursor = conn.cursor()

        print("Creating uuid-ossp extension...")
        cursor.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')

        create_scripts = """
-- Farmers profile data
CREATE TABLE IF NOT EXISTS farmers (
    id UUID PRIMARY KEY,
    phone VARCHAR(20) UNIQUE NOT NULL,
    district VARCHAR(100),
    crops_grown TEXT[],
    language_pref VARCHAR(10) DEFAULT 'hi'
);

-- Audit log of LLM chat queries
CREATE TABLE IF NOT EXISTS queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    farmer_id UUID REFERENCES farmers(id),
    query_text TEXT NOT NULL,
    language VARCHAR(10),
    response TEXT NOT NULL,
    category VARCHAR(50),
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Farmer feedback on LLM answers
CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID REFERENCES queries(id),
    was_helpful BOOLEAN NOT NULL,
    follow_up_action TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Mandi prices fallback cache
CREATE TABLE IF NOT EXISTS prices_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    commodity VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    min_price NUMERIC,
    max_price NUMERIC,
    modal_price NUMERIC,
    date DATE NOT NULL,
    UNIQUE(commodity, district, date)
);

-- Static disease reference dictionary
CREATE TABLE IF NOT EXISTS diseases (
    disease_id VARCHAR(20) PRIMARY KEY,
    crop VARCHAR(100) NOT NULL,
    disease_name_en VARCHAR(100) NOT NULL,
    disease_name_hi VARCHAR(100) NOT NULL,
    symptoms_farmer_language TEXT NOT NULL,
    organic_treatment TEXT,
    chemical_treatment TEXT,
    dosage TEXT,
    source_url TEXT,
    source_name VARCHAR(100)
);

-- Government schemes
CREATE TABLE IF NOT EXISTS schemes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scheme_name VARCHAR(200) NOT NULL,
    benefit_description TEXT,
    eligibility TEXT,
    how_to_apply TEXT,
    documents_needed TEXT,
    helpline_number VARCHAR(20),
    website_url TEXT,
    applicable_states VARCHAR(100)
);

-- KVK Directory
CREATE TABLE IF NOT EXISTS kvk_directory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    district VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    phone_number VARCHAR(20),
    email VARCHAR(100),
    website TEXT
);
        """

        print("Executing CREATE TABLE scripts...")
        cursor.execute(create_scripts)
        
        insert_scripts = """
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
('MAI_05', 'Maize', 'Rust', 'Ratuwa Rog', 'Patton par lal powder jaisa', 'Early planting', 'Mancozeb 75% WP', '2.5 g/L water', 'https://icar.org.in/', 'ICAR-IIMR')
ON CONFLICT (disease_id) DO NOTHING;

INSERT INTO schemes (scheme_name, benefit_description, eligibility, how_to_apply, documents_needed, helpline_number, website_url, applicable_states) VALUES
('PM-KISAN', 'Direct income support of Rs. 6,000 per year in three equal installments', 'All landholding farmers families', 'Apply via CSC or pmkisan.gov.in', 'Aadhaar, Bank Passbook, Land records', '155261', 'https://pmkisan.gov.in/', 'All States'),
('PM Fasal Bima Yojana', 'Comprehensive insurance cover against failure of the crop', 'All farmers growing notified crops', 'Via bank branch, CSC, or insurance portal', 'Aadhaar, Land records, Sowing certificate', '14447', 'https://pmfby.gov.in/', 'All States'),
('Kisan Credit Card', 'Provides adequate and timely credit support under a single window', 'All farmers, tenant farmers, sharecroppers', 'Bank branch or online application', 'Identity proof, Address proof, Land holding details', '1800115526', 'https://sbi.co.in/web/agri-rural/agriculture-banking/crop-finance/kisan-credit-card', 'All States'),
('Soil Health Card scheme', 'Information on soil nutrient status and recommendation on appropriate dosage of nutrients', 'All farmers', 'Local Agriculture Department or CSC', 'Aadhaar', '18001801551', 'https://soilhealth.dac.gov.in/', 'All States'),
('PM Krishi Sinchai Yojana', 'End-to-end solutions in irrigation supply chain, focus on micro irrigation', 'Farmers with agricultural land', 'State Agriculture/Horticulture department', 'Aadhaar, Land records, Bank details', '18001801551', 'https://pmksy.gov.in/', 'All States'),
('eNAM', 'Pan-India electronic trading portal which networks the existing APMC mandis', 'Farmers, Traders, FPOs', 'Register on enam.gov.in or eNAM mobile app', 'Aadhaar, Bank details', '18002700224', 'https://enam.gov.in/', 'All States');

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
        """
        
        print("Executing INSERT statements...")
        cursor.execute("TRUNCATE TABLE schemes, kvk_directory RESTART IDENTITY CASCADE;")
        cursor.execute(insert_scripts)
        
        print("Database seeding completed successfully.")
        
    except OperationalError as e:
        message = str(e)
        if "could not translate host name" in message:
            print(
                "Error: your DATABASE_URL points at a Supabase host that this machine cannot resolve or reach. "
                "Use the exact Postgres connection string from Supabase Dashboard -> Settings -> Database -> Connection string, "
                "or switch to the pooler host if your network does not support the direct database endpoint."
            )
        else:
            print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            cursor.close()
            conn.close()

if __name__ == '__main__':
    run_setup()
