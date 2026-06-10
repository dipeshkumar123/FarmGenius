#!/usr/bin/env python
"""
Script to create training datasets for FarmGenius chatbot models.
Downloads and generates FAQ, crop recommendation, and other training data.
"""
import csv
import os
import random
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── 1. FAQ Dataset ──────────────────────────────────────────────────────────────
FAQ_DATA = [
    # Crop rotation & management
    ("What is crop rotation?", "Crop rotation is the practice of growing different types of crops in the same area across seasons. It helps maintain soil fertility, reduce pest pressure, and improve crop yields."),
    ("Why is crop rotation important?", "Crop rotation is important because it prevents soil nutrient depletion, breaks pest and disease cycles, improves soil structure, and can increase overall farm productivity."),
    ("What are the benefits of intercropping?", "Intercropping benefits include better use of resources like sunlight, water, and nutrients; natural pest control; reduced risk of total crop failure; and improved soil health."),
    ("How do I practice crop rotation?", "Practice crop rotation by alternating between different plant families each season. For example, follow legumes (which fix nitrogen) with heavy feeders like corn, then root crops like potatoes."),
    ("What is monoculture farming?", "Monoculture farming is growing a single crop species on the same land year after year. While efficient for large-scale production, it depletes specific nutrients and increases pest vulnerability."),
    ("What is companion planting?", "Companion planting is growing different plant species near each other for mutual benefit. For example, planting marigolds alongside tomatoes helps repel pests, while beans fix nitrogen that benefits neighbouring corn."),
    ("What is cover cropping?", "Cover cropping involves planting specific crops primarily to protect and enrich the soil rather than for harvest. Cover crops prevent erosion, suppress weeds, improve soil fertility, and enhance water retention."),
    ("How often should I rotate crops?", "Ideally, rotate crops every season or at least annually. A 3-4 year rotation cycle is recommended, especially to break disease and pest cycles effectively."),

    # Soil health
    ("How do I test my soil?", "You can test soil by collecting samples from different areas of your field at 6-8 inch depth, mixing them, and sending to a soil testing laboratory. Home testing kits are also available for basic pH and nutrient tests."),
    ("What is soil pH and why does it matter?", "Soil pH measures how acidic or alkaline your soil is on a scale of 0-14. Most crops grow best in slightly acidic to neutral soil (pH 6.0-7.0). pH affects nutrient availability to plants."),
    ("How can I improve soil fertility?", "Improve soil fertility by adding organic matter (compost, manure), using cover crops, practicing crop rotation, applying appropriate fertilizers based on soil tests, and minimizing tillage."),
    ("What is composting?", "Composting is the natural process of decomposing organic matter like kitchen scraps, leaves, and manure into nutrient-rich humus. It improves soil structure, water retention, and provides slow-release nutrients."),
    ("How do I make compost?", "Make compost by layering green materials (grass, food scraps) with brown materials (dry leaves, straw) in a 1:3 ratio. Keep the pile moist, turn it regularly, and it should be ready in 2-6 months."),
    ("What is the role of nitrogen in soil?", "Nitrogen is essential for plant leaf and stem growth. It is a key component of chlorophyll and amino acids. Deficiency causes yellowing of older leaves and stunted growth."),
    ("What is the role of phosphorus in soil?", "Phosphorus promotes root development, flowering, and fruit production. It is essential for energy transfer in plants. Deficiency causes purple discoloration of leaves and poor root growth."),
    ("What is the role of potassium in soil?", "Potassium strengthens plant stems, improves drought resistance, and enhances disease resistance. It regulates water use and enzyme activation. Deficiency causes brown leaf edges."),
    ("What is organic matter in soil?", "Organic matter consists of decomposed plant and animal residues in soil. It improves soil structure, water-holding capacity, nutrient retention, and supports beneficial microorganisms."),
    ("How do I prevent soil erosion?", "Prevent soil erosion by maintaining ground cover, using contour farming, building terraces on slopes, planting windbreaks, practicing conservation tillage, and using mulch."),
    ("What is mulching?", "Mulching is covering the soil surface with organic or inorganic materials to conserve moisture, suppress weeds, regulate soil temperature, and add nutrients as organic mulch decomposes."),
    ("What causes soil salinity?", "Soil salinity is caused by excessive irrigation without proper drainage, use of saline water, high evaporation rates, and naturally occurring salt deposits. It reduces crop productivity."),

    # Pest management
    ("How do I control pests organically?", "Control pests organically using integrated pest management (IPM): encourage beneficial insects, use neem oil or diatomaceous earth, practice crop rotation, use companion planting, and introduce biological controls like ladybugs."),
    ("What is integrated pest management?", "Integrated Pest Management (IPM) is a sustainable approach combining biological, cultural, physical, and chemical methods to manage pests while minimizing economic, health, and environmental risks."),
    ("How do I identify common crop pests?", "Identify pests by examining leaf damage patterns, checking for holes, discoloration, wilting, or presence of insects. Look under leaves, at stem bases, and in the soil. Use magnifying glasses for small pests."),
    ("What are biological pest control methods?", "Biological pest control uses natural predators, parasites, or pathogens to manage pests. Examples include releasing ladybugs for aphid control, using Bt bacteria for caterpillars, and encouraging birds."),
    ("How do I control aphids?", "Control aphids by spraying with water to dislodge them, applying neem oil or insecticidal soap, introducing ladybugs or lacewings, planting repellent herbs like basil, and removing heavily infested parts."),
    ("What is neem oil used for?", "Neem oil is a natural pesticide derived from the neem tree. It controls over 200 pest species including aphids, whiteflies, and mites while being safe for beneficial insects when used correctly."),
    ("How do I prevent pest infestations?", "Prevent infestations by maintaining plant health, removing crop debris, using resistant varieties, rotating crops, monitoring regularly, and maintaining beneficial insect habitats."),
    ("What are common tomato pests?", "Common tomato pests include tomato hornworm, aphids, whiteflies, spider mites, flea beetles, and cutworms. Regular monitoring and IPM practices help control them effectively."),
    ("How do I control weeds?", "Control weeds through mulching, hand weeding, hoeing, cover cropping, crop rotation, proper spacing, flame weeding, and targeted herbicide application as a last resort."),

    # Disease management
    ("How do I prevent plant diseases?", "Prevent diseases by using disease-resistant varieties, practicing crop rotation, ensuring proper spacing for air circulation, avoiding overhead watering, removing infected plants, and maintaining soil health."),
    ("What causes leaf blight?", "Leaf blight is caused by fungal or bacterial pathogens that thrive in warm, humid conditions. It appears as brown or yellow spots that enlarge and kill leaf tissue, reducing photosynthesis and yield."),
    ("How do I treat fungal diseases in crops?", "Treat fungal diseases by removing infected plant parts, improving air circulation, applying fungicides (copper-based for organic farming), avoiding wet foliage, and using resistant varieties."),
    ("What is mosaic virus?", "Mosaic virus causes patchy light and dark green patterns on leaves, stunted growth, and reduced yields. It spreads through infected seeds, tools, and insect vectors like aphids. There is no cure; prevention is key."),
    ("How do I identify bacterial blight?", "Bacterial blight appears as water-soaked spots on leaves that turn brown with yellow halos. It spreads in warm, wet conditions. Affected leaves eventually die and fall off."),
    ("What is powdery mildew?", "Powdery mildew is a fungal disease appearing as white powdery spots on leaves and stems. It thrives in warm, dry conditions with cool nights. Treat with sulfur sprays, neem oil, or baking soda solutions."),
    ("How do I control root rot?", "Control root rot by improving soil drainage, avoiding overwatering, using raised beds, applying fungicides, removing infected plants, and using disease-free planting material."),
    ("What is anthracnose?", "Anthracnose is a fungal disease causing dark, sunken lesions on leaves, stems, flowers, and fruits. It spreads in warm, humid weather through rain splashes and infected seeds."),
    ("How do I manage crop diseases without chemicals?", "Manage diseases without chemicals by using resistant varieties, practicing good sanitation, ensuring proper spacing, applying compost tea, using biological controls, and maintaining balanced soil nutrition."),

    # Irrigation
    ("What is drip irrigation?", "Drip irrigation delivers water directly to plant roots through a network of tubes, emitters, and drippers. It conserves water (up to 60% compared to flood irrigation), reduces weed growth, and minimizes disease."),
    ("How much water do crops need?", "Water needs vary by crop, climate, and growth stage. Generally, most crops need 1-2 inches of water per week. Sandy soils need more frequent watering, while clay soils retain moisture longer."),
    ("What is the best time to irrigate?", "The best time to irrigate is early morning (before 10 AM) when evaporation is lowest and plants can absorb water before afternoon heat. Avoid evening watering as it promotes fungal diseases."),
    ("What is flood irrigation?", "Flood irrigation involves flooding the entire field with water. It is simple and low-cost but wastes water through evaporation and runoff. It works best for rice paddies and level fields."),
    ("How do I save water in farming?", "Save water by using drip irrigation, mulching, planting drought-resistant varieties, rainwater harvesting, scheduling irrigation based on soil moisture, and reducing evaporation with ground cover."),
    ("What is furrow irrigation?", "Furrow irrigation channels water through small parallel channels (furrows) along crop rows. It is more efficient than flood irrigation and suitable for row crops like corn, cotton, and vegetables."),
    ("How does irrigation scheduling work?", "Irrigation scheduling determines when and how much water to apply based on soil moisture sensors, weather data, crop growth stage, and evapotranspiration rates to optimize water use."),

    # Fertilizers
    ("What is NPK fertilizer?", "NPK fertilizer contains three primary nutrients: Nitrogen (N) for leaf growth, Phosphorus (P) for root and flower development, and Potassium (K) for overall plant health. The numbers represent the percentage of each."),
    ("When should I apply fertilizer?", "Apply fertilizer based on crop growth stage: at planting for root establishment, during vegetative growth for leaf development, and before flowering for fruit production. Always base on soil test results."),
    ("What is organic fertilizer?", "Organic fertilizer comes from natural sources like compost, manure, bone meal, and fish emulsion. It releases nutrients slowly, improves soil structure, and supports microbial life."),
    ("How do I calculate fertilizer requirements?", "Calculate fertilizer needs based on soil test results, crop nutrient requirements, and expected yield. The formula: Amount = (Nutrient needed × 100) / Fertilizer nutrient percentage."),
    ("What is vermicompost?", "Vermicompost is organic fertilizer produced by earthworms decomposing organic waste. It is rich in nutrients, beneficial microorganisms, and growth hormones. It improves soil structure and water retention."),
    ("What is the difference between organic and chemical fertilizers?", "Organic fertilizers release nutrients slowly, improve soil health long-term, and are environmentally friendly. Chemical fertilizers provide immediate nutrients but can degrade soil structure and pollute water over time."),
    ("What is biofertilizer?", "Biofertilizer contains living microorganisms that enhance soil fertility by fixing nitrogen, solubilizing phosphorus, or decomposing organic matter. Examples include Rhizobium, Azotobacter, and mycorrhizal fungi."),

    # Specific crops
    ("How do I grow tomatoes?", "Grow tomatoes in well-drained soil with pH 6.0-6.8, full sunlight (6-8 hours), regular watering at the base, and support with stakes or cages. Plant after the last frost, 2-3 feet apart."),
    ("How do I grow maize/corn?", "Grow maize in well-drained fertile soil with pH 5.8-7.0. Plant seeds 1-2 inches deep, 8-12 inches apart in blocks for pollination. Needs full sun, regular watering, and nitrogen-rich fertilizer."),
    ("How do I grow cassava?", "Grow cassava by planting stem cuttings (20-30 cm) at an angle in well-drained soil. Space 1m apart. It thrives in tropical climates, tolerates drought, and is harvested 8-12 months after planting."),
    ("How do I grow cashew trees?", "Grow cashew trees from seeds or grafted seedlings in well-drained sandy soil with pH 5-6.5. Plant 8-10m apart. They need tropical climate, minimal water once established, and protection from frost."),
    ("How do I grow rice?", "Grow rice in flooded paddy fields with clay soil that retains water. Transplant seedlings 20-25 days old, 20cm apart. Rice needs warm temperatures (20-35°C), consistent water, and nitrogen fertilizer."),
    ("How do I grow wheat?", "Grow wheat in cool climates in well-drained loamy soil with pH 6.0-7.0. Sow seeds in autumn or spring, 1-2 inches deep. Needs moderate water and nitrogen fertilizer at tillering stage."),
    ("How do I grow potatoes?", "Grow potatoes in loose, well-drained acidic soil (pH 5.0-6.0). Plant seed potatoes 4 inches deep, 12 inches apart. Hill soil around stems as they grow. Harvest when foliage dies back."),
    ("What crops grow best in sandy soil?", "Sandy soil suits crops like carrots, radishes, potatoes, peanuts, watermelon, and cashews. These crops prefer good drainage and don't need heavy clay. Amend sandy soil with organic matter for better results."),
    ("What crops grow best in clay soil?", "Clay soil suits rice, wheat, cabbage, broccoli, lettuce, and beans. These crops tolerate heavier moisture. Improve clay soil by adding organic matter, gypsum, and avoiding compaction."),
    ("What crops grow in acidic soil?", "Acidic soil (pH <6) suits blueberries, potatoes, sweet potatoes, tea, and most tropical fruits. Add lime to raise pH if needed for other crops."),

    # Weather and climate
    ("How does weather affect crop growth?", "Weather affects crops through temperature (growth rate), rainfall (water supply), sunlight (photosynthesis), humidity (disease risk), and wind (pollination and damage). Extreme weather can devastate harvests."),
    ("What is the best season to plant crops?", "The best planting season depends on the crop and region. Generally, cool-season crops (wheat, peas) are planted in autumn/winter, while warm-season crops (corn, tomatoes) are planted in spring/summer."),
    ("How do I protect crops from frost?", "Protect from frost by covering plants with row covers or mulch, watering soil before frost (wet soil retains heat), using cold frames, planting near buildings, and choosing frost-resistant varieties."),
    ("How does drought affect crops?", "Drought reduces water availability causing wilting, stunted growth, poor pollination, reduced yields, and plant death. Manage with drip irrigation, mulching, drought-resistant varieties, and water conservation."),
    ("How do I prepare for the monsoon season?", "Prepare for monsoon by improving drainage, building raised beds, repairing bunds, storing harvested crops properly, stocking seeds for post-monsoon planting, and securing farm structures."),
    ("What is climate-smart agriculture?", "Climate-smart agriculture practices sustainably increase productivity, adapt to climate change, and reduce greenhouse gases. Includes conservation tillage, agroforestry, and efficient water management."),

    # Harvesting and storage
    ("When should I harvest my crops?", "Harvest timing depends on the crop. Look for signs of maturity: color change (tomatoes turn red), firmness (melons sound hollow), dryness (grains have specific moisture content). Test a few before full harvest."),
    ("How do I store grains properly?", "Store grains in clean, dry, airtight containers at moisture content below 13%. Use proper silos or sealed bags. Protect from pests with neem leaves or diatomaceous earth. Monitor for moisture and insects regularly."),
    ("What is post-harvest loss?", "Post-harvest loss is the reduction in quantity or quality of crops after harvest due to pests, diseases, poor storage, transportation damage, and spoilage. It can account for 20-40% of production in developing countries."),
    ("How do I reduce post-harvest losses?", "Reduce losses by harvesting at proper maturity, handling carefully, drying to correct moisture, using improved storage facilities, controlling pests, and minimizing time between harvest and storage/sale."),
    ("How do I dry crops for storage?", "Dry crops using sun drying on tarpaulins or drying platforms, mechanical dryers for large quantities, or solar dryers. Ensure even drying and protect from rain. Target moisture content varies by crop."),

    # Market and economics
    ("How do I find market prices for my crops?", "Find market prices through agricultural market boards, government portals like data.gov.in, local mandi (market) visits, agricultural apps on your smartphone, and through farmer cooperatives."),
    ("What is the minimum support price?", "Minimum Support Price (MSP) is a guaranteed price set by the government to protect farmers from market fluctuations. It covers major crops like rice, wheat, pulses, and oilseeds."),
    ("How do I sell my crops at the best price?", "Get better prices by grading and sorting produce, understanding market demand, selling through farmer producer organizations, exploring direct-to-consumer models, and timing sales based on market trends."),
    ("What are commodity prices influenced by?", "Commodity prices are influenced by supply and demand, weather conditions, government policies, international trade, storage costs, transportation, currency exchange rates, and speculative trading."),
    ("What is contract farming?", "Contract farming is an agreement between farmers and buyers where the buyer guarantees purchase at a pre-agreed price and quality standard. It provides price certainty but may limit flexibility."),

    # Technology in farming
    ("What is precision agriculture?", "Precision agriculture uses technology like GPS, sensors, drones, and data analytics to optimize farming operations. It enables variable-rate application of inputs, reducing waste and improving yields."),
    ("How can drones help in farming?", "Drones help in farming by surveying crops, detecting pest infestations early, monitoring crop health through NDVI imaging, spraying pesticides precisely, and mapping field boundaries."),
    ("What is vertical farming?", "Vertical farming grows crops in vertically stacked layers in controlled environments. It uses less water, no pesticides, and can operate year-round. Best suited for leafy greens and herbs."),
    ("How can I use mobile apps for farming?", "Use farming apps for weather forecasts, market prices, pest identification, crop planning, soil testing interpretation, agricultural advisory services, and connecting with buyers and suppliers."),

    # Sustainable farming
    ("What is sustainable agriculture?", "Sustainable agriculture meets current food needs without compromising future generations' ability to meet theirs. It balances economic viability, environmental health, and social equity through responsible practices."),
    ("What is organic farming?", "Organic farming avoids synthetic chemicals and GMOs, relying on natural processes, organic inputs, and biological controls. It promotes biodiversity, soil health, and produces chemical-free food."),
    ("What is agroforestry?", "Agroforestry combines trees with crops on the same land. Trees provide shade, windbreaks, nitrogen fixation, and additional income from fruits/timber while crops grow between them."),
    ("What is conservation tillage?", "Conservation tillage minimizes soil disturbance during planting, leaving at least 30% of crop residue on the surface. It reduces erosion, improves soil moisture, and lowers fuel and labor costs."),
    ("How do I start organic farming?", "Start organic farming by stopping chemical inputs, building soil health with compost and cover crops, implementing biological pest control, getting certification, and connecting with organic markets. The transition takes 2-3 years."),
    ("What is regenerative agriculture?", "Regenerative agriculture goes beyond sustainable farming by actively rebuilding degraded soil, increasing biodiversity, improving the water cycle, and sequestering carbon. Practices include no-till, cover cropping, and diverse rotations."),

    # Animal husbandry
    ("How do I integrate livestock with crop farming?", "Integrate livestock by using crop residues as animal feed, applying animal manure as fertilizer, using animals for draft power, practicing rotational grazing on crop fields after harvest, and growing fodder crops."),
    ("What is integrated farming?", "Integrated farming combines crop production, livestock, aquaculture, and agroforestry on the same farm. Each component supports others, reducing waste, diversifying income, and improving sustainability."),

    # Seeds
    ("How do I select good seeds?", "Select seeds based on variety suitability for your climate, disease resistance, yield potential, and seed quality. Use certified seeds from reputable sources. Test germination rate before planting."),
    ("What are hybrid seeds?", "Hybrid seeds are produced by crossing two different parent varieties to combine desirable traits. They offer higher yields and disease resistance but must be purchased new each season as saved seeds don't breed true."),
    ("What is seed treatment?", "Seed treatment applies fungicides, insecticides, or biological agents to seeds before planting to protect against soil-borne diseases and early-stage pests, improving germination and establishment."),
    ("How do I store seeds?", "Store seeds in cool (below 15°C), dry (below 40% humidity), dark conditions in airtight containers. Label with variety and date. Most vegetable seeds remain viable for 2-5 years under proper storage."),

    # General farming
    ("What is the Green Revolution?", "The Green Revolution (1960s-70s) introduced high-yielding crop varieties, chemical fertilizers, and irrigation to dramatically increase food production, particularly in Asia. It saved millions from famine but raised environmental concerns."),
    ("What is subsistence farming?", "Subsistence farming produces just enough food for the farmer's family with little surplus for sale. It relies on traditional methods, limited inputs, and is common in developing regions."),
    ("What is commercial farming?", "Commercial farming produces crops primarily for sale and profit. It typically involves larger acreage, mechanization, modern inputs, and market-oriented production strategies."),
    ("How do I get a loan for farming?", "Get farming loans through agricultural banks (like NABARD), cooperative societies, commercial banks with agriculture schemes, microfinance institutions, and government subsidy programs. Prepare a farm plan and required documents."),
    ("What government schemes are available for farmers?", "Government schemes include PM-KISAN (income support), crop insurance (PMFBY), soil health cards, irrigation subsidies, agricultural marketing reforms, and interest subvention on crop loans."),
    ("What is greenhouse farming?", "Greenhouse farming grows crops in enclosed structures with controlled temperature, humidity, and light. It extends growing seasons, protects from weather extremes, and enables year-round production of high-value crops."),
    ("What is hydroponics?", "Hydroponics grows plants without soil using mineral nutrient solutions in water. Benefits include faster growth, higher yields, less water use, and year-round production. Common for lettuce, herbs, and tomatoes."),
    ("What is aquaponics?", "Aquaponics combines fish farming (aquaculture) with soilless plant growing (hydroponics). Fish waste provides nutrients for plants, and plants filter water for fish, creating a symbiotic system."),
]


def create_faq_dataset():
    """Create the FAQ CSV dataset."""
    faq_path = os.path.join(DATASET_DIR, "faq_dataset.csv")
    with open(faq_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answers"])
        writer.writeheader()
        for q, a in FAQ_DATA:
            writer.writerow({"question": q, "answers": a})
    logger.info(f"Created FAQ dataset with {len(FAQ_DATA)} entries at {faq_path}")
    return faq_path


# ── 2. Crop Recommendation Dataset ─────────────────────────────────────────────
# Features: N, P, K, ph, EC, S, Cu, Fe, Mn, Zn, B, label
# Based on typical soil nutrient ranges for Indian crop recommendations

CROP_PROFILES = {
    "Rice":       {"N": (60, 120), "P": (20, 60), "K": (20, 50), "ph": (5.5, 7.0), "EC": (0.1, 0.5), "S": (10, 25), "Cu": (0.2, 1.5), "Fe": (5, 30), "Mn": (1, 10), "Zn": (0.5, 3), "B": (0.2, 1.0)},
    "Wheat":      {"N": (80, 140), "P": (30, 70), "K": (25, 55), "ph": (6.0, 7.5), "EC": (0.1, 0.4), "S": (12, 30), "Cu": (0.3, 2.0), "Fe": (8, 35), "Mn": (2, 12), "Zn": (0.6, 4), "B": (0.3, 1.2)},
    "Maize":      {"N": (100, 180), "P": (30, 80), "K": (30, 60), "ph": (5.5, 7.5), "EC": (0.1, 0.5), "S": (10, 28), "Cu": (0.2, 2.0), "Fe": (5, 30), "Mn": (1, 10), "Zn": (0.5, 3.5), "B": (0.2, 1.0)},
    "Cotton":     {"N": (40, 100), "P": (20, 50), "K": (30, 70), "ph": (6.0, 8.0), "EC": (0.1, 0.6), "S": (15, 35), "Cu": (0.3, 2.0), "Fe": (5, 25), "Mn": (2, 8), "Zn": (0.4, 3), "B": (0.3, 1.5)},
    "Sugarcane":  {"N": (120, 200), "P": (25, 60), "K": (40, 80), "ph": (6.0, 7.5), "EC": (0.2, 0.6), "S": (15, 30), "Cu": (0.3, 1.8), "Fe": (8, 35), "Mn": (2, 10), "Zn": (0.6, 3), "B": (0.3, 1.2)},
    "Soybean":    {"N": (20, 60),  "P": (30, 70), "K": (30, 60), "ph": (6.0, 7.0), "EC": (0.1, 0.4), "S": (10, 25), "Cu": (0.2, 1.5), "Fe": (5, 25), "Mn": (1, 8), "Zn": (0.5, 3), "B": (0.2, 1.0)},
    "Groundnut":  {"N": (10, 40),  "P": (20, 60), "K": (20, 50), "ph": (5.5, 7.0), "EC": (0.1, 0.3), "S": (15, 35), "Cu": (0.2, 1.5), "Fe": (5, 20), "Mn": (1, 8), "Zn": (0.4, 2.5), "B": (0.3, 1.5)},
    "Potato":     {"N": (80, 150), "P": (40, 90), "K": (60, 120), "ph": (5.0, 6.5), "EC": (0.1, 0.5), "S": (12, 28), "Cu": (0.2, 1.5), "Fe": (5, 25), "Mn": (1, 8), "Zn": (0.4, 3), "B": (0.2, 1.0)},
    "Tomato":     {"N": (80, 150), "P": (40, 80), "K": (50, 100), "ph": (6.0, 6.8), "EC": (0.2, 0.5), "S": (12, 28), "Cu": (0.3, 2.0), "Fe": (5, 30), "Mn": (1, 10), "Zn": (0.5, 3), "B": (0.3, 1.2)},
    "Onion":      {"N": (60, 120), "P": (30, 70), "K": (40, 80), "ph": (6.0, 7.0), "EC": (0.1, 0.4), "S": (20, 40), "Cu": (0.2, 1.5), "Fe": (5, 25), "Mn": (1, 8), "Zn": (0.4, 2.5), "B": (0.2, 1.0)},
    "Chickpea":   {"N": (15, 40),  "P": (20, 50), "K": (20, 45), "ph": (6.0, 7.5), "EC": (0.1, 0.3), "S": (10, 25), "Cu": (0.2, 1.5), "Fe": (5, 20), "Mn": (1, 8), "Zn": (0.4, 2.5), "B": (0.2, 1.0)},
    "Mustard":    {"N": (60, 100), "P": (20, 50), "K": (20, 45), "ph": (6.0, 7.5), "EC": (0.1, 0.4), "S": (20, 40), "Cu": (0.2, 1.5), "Fe": (5, 25), "Mn": (1, 8), "Zn": (0.4, 2.5), "B": (0.3, 1.5)},
    "Lentil":     {"N": (15, 35),  "P": (20, 50), "K": (15, 40), "ph": (6.0, 7.5), "EC": (0.1, 0.3), "S": (10, 25), "Cu": (0.2, 1.0), "Fe": (4, 20), "Mn": (1, 6), "Zn": (0.3, 2.0), "B": (0.2, 0.8)},
    "Cassava":    {"N": (30, 80),  "P": (15, 40), "K": (40, 90), "ph": (5.5, 7.0), "EC": (0.1, 0.4), "S": (8, 20),  "Cu": (0.1, 1.0), "Fe": (3, 20), "Mn": (1, 8), "Zn": (0.3, 2.0), "B": (0.1, 0.8)},
    "Cashew":     {"N": (20, 60),  "P": (10, 30), "K": (20, 50), "ph": (5.0, 6.5), "EC": (0.1, 0.3), "S": (8, 20),  "Cu": (0.1, 1.0), "Fe": (5, 25), "Mn": (1, 8), "Zn": (0.3, 2.0), "B": (0.1, 0.8)},
}
SAMPLES_PER_CROP = 80


def create_crop_dataset():
    """Create synthetic crop recommendation dataset."""
    random.seed(42)
    crop_path = os.path.join(DATASET_DIR, "crop_data.csv")
    fieldnames = ["N", "P", "K", "ph", "EC", "S", "Cu", "Fe", "Mn", "Zn", "B", "label"]
    with open(crop_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for crop_name, ranges in CROP_PROFILES.items():
            for _ in range(SAMPLES_PER_CROP):
                row = {feat: round(random.uniform(*rng), 2) for feat, rng in ranges.items()}
                # Add some noise for realism
                for feat in row:
                    if feat == "ph":
                        row[feat] = round(row[feat] + random.gauss(0, 0.15), 2)
                    else:
                        noise = row[feat] * random.gauss(0, 0.05)
                        row[feat] = round(max(0, row[feat] + noise), 2)
                row["label"] = crop_name
                writer.writerow(row)
    total = SAMPLES_PER_CROP * len(CROP_PROFILES)
    logger.info(f"Created crop recommendation dataset with {total} samples ({len(CROP_PROFILES)} crops) at {crop_path}")
    return crop_path


# ── 3. Weather dataset (stub for processor) ────────────────────────────────────

def create_weather_dataset():
    """Create a minimal weather dataset."""
    random.seed(42)
    weather_path = os.path.join(DATASET_DIR, "weather_data.csv")
    cities = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore", "Hyderabad", "Pune", "Jaipur", "Lucknow", "Ahmedabad"]
    fieldnames = ["city", "date", "temperature_c", "humidity", "rainfall_mm", "wind_speed_kmh", "condition"]
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Thunderstorm", "Foggy", "Hazy"]
    with open(weather_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for city in cities:
            base_temp = random.uniform(20, 35)
            for day in range(1, 31):
                temp = round(base_temp + random.gauss(0, 3), 1)
                hum = random.randint(30, 95)
                rain = round(max(0, random.gauss(2, 5)), 1)
                wind = round(random.uniform(5, 30), 1)
                cond = random.choice(conditions)
                writer.writerow({
                    "city": city,
                    "date": f"2026-01-{day:02d}",
                    "temperature_c": temp,
                    "humidity": hum,
                    "rainfall_mm": rain,
                    "wind_speed_kmh": wind,
                    "condition": cond,
                })
    logger.info(f"Created weather dataset at {weather_path}")
    return weather_path


# ── 4. Price dataset (stub for processor) ──────────────────────────────────────

def create_price_dataset():
    """Create a minimal commodity price dataset matching PriceProcessor expectations."""
    random.seed(42)
    price_path = os.path.join(DATASET_DIR, "commodity_prices.csv")
    fieldnames = ["commodity_name", "date", "state", "district", "market", "min_price", "max_price", "modal_price"]
    commodities = {
        "Rice": (1800, 2200), "Wheat": (1600, 2100), "Maize": (1200, 1800),
        "Cotton": (4000, 6000), "Sugarcane": (250, 350), "Soybean": (3200, 4500),
        "Groundnut": (3500, 5000), "Potato": (800, 1500), "Tomato": (600, 2000),
        "Onion": (800, 2500),
    }
    markets = [
        ("Delhi", "Delhi", "Azadpur"),
        ("Maharashtra", "Mumbai", "Vashi"),
        ("West Bengal", "Kolkata", "Koley Market"),
        ("Tamil Nadu", "Chennai", "Koyambedu"),
        ("Karnataka", "Bangalore", "Yeshwanthpur"),
    ]
    with open(price_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for comm, (lo, hi) in commodities.items():
            for state, district, market in markets:
                for day in range(1, 31):
                    modal = round(random.uniform(lo, hi), 2)
                    mn = round(modal * random.uniform(0.85, 0.95), 2)
                    mx = round(modal * random.uniform(1.05, 1.15), 2)
                    writer.writerow({
                        "commodity_name": comm,
                        "date": f"2026-01-{day:02d}",
                        "state": state,
                        "district": district,
                        "market": market,
                        "min_price": mn,
                        "max_price": mx,
                        "modal_price": modal,
                    })
    logger.info(f"Created price dataset at {price_path}")
    return price_path


# ── 5. Query dataset (stub for processor) ──────────────────────────────────────

def create_query_dataset():
    """Create a farmer query dataset matching QueryProcessor expectations (questions + answers)."""
    queries = [
        ("What is the price of rice today?", "You can check current rice prices through agricultural market boards or government portals. Prices vary by region and quality."),
        ("How much does wheat cost in Delhi?", "Wheat prices in Delhi depend on the variety and season. Check the local mandi for current rates or use the AGMARKNET portal."),
        ("Current market rate for cotton", "Cotton market rates fluctuate daily. Check your nearest APMC market or the data.gov.in portal for the latest prices."),
        ("What will be the weather tomorrow?", "Weather forecasts are available through the Indian Meteorological Department (IMD) website, local news, or weather apps on your phone."),
        ("Is it going to rain this week?", "Check the weekly weather forecast from IMD or use weather apps. Rainfall predictions help plan irrigation and harvesting."),
        ("Weather forecast for Mumbai", "Mumbai's weather is tropical with monsoon season from June to September. Check IMD for daily forecasts and alerts."),
        ("My tomato leaves are turning yellow", "Yellow tomato leaves can indicate nitrogen deficiency, overwatering, or diseases like early blight. Check soil nutrition and drainage."),
        ("How to treat leaf blight on maize?", "Treat maize leaf blight by removing infected leaves, improving air circulation, applying fungicides, and using resistant varieties in the next season."),
        ("What disease has spots on cassava leaves?", "Spots on cassava leaves could indicate brown spot disease or bacterial blight. Check the pattern and color for accurate diagnosis."),
        ("Recommend a crop for my sandy soil", "Sandy soils are good for root vegetables like carrots, radishes, potatoes, as well as peanuts and watermelon. Amend with organic matter."),
        ("Best crop for clay soil with pH 7?", "Clay soil with neutral pH suits wheat, rice, cabbage, and beans. Improve drainage by adding organic matter and gypsum."),
        ("Which crop needs less water?", "Drought-tolerant crops include millets, sorghum, chickpea, and mustard. These crops need significantly less water than rice or sugarcane."),
        ("How do I grow tomatoes?", "Grow tomatoes in well-drained soil with pH 6.0-6.8, full sunlight, regular watering at the base, and support with stakes. Plant after last frost."),
        ("What is drip irrigation?", "Drip irrigation delivers water directly to plant roots through tubes and emitters. It saves up to 60% water compared to flood irrigation."),
        ("Benefits of crop rotation", "Crop rotation prevents soil nutrient depletion, breaks pest cycles, improves soil structure, and increases overall farm productivity."),
        ("How to control aphids organically?", "Control aphids with water sprays, neem oil, insecticidal soap, introducing ladybugs, or planting repellent herbs like basil nearby."),
        ("What fertilizer should I use for rice?", "Rice benefits from NPK fertilizer with higher nitrogen. Apply urea at transplanting and tillering stages. Base amounts on soil test results."),
        ("Temperature in Chennai today", "Chennai has a tropical climate. Check IMD or weather apps for today's temperature. Generally hot and humid with temperatures 25-40°C."),
        ("Onion prices in Bangalore market", "Onion prices at Yeshwanthpur market in Bangalore vary seasonally. Check AGMARKNET or local mandi boards for current rates."),
        ("Brown spots appearing on my cashew tree", "Brown spots on cashew trees often indicate anthracnose disease caused by fungi. Prune affected branches and apply copper-based fungicides."),
        ("Suggest crops for acidic soil", "Acidic soil (pH <6) is good for potatoes, sweet potatoes, blueberries, tea, and rice. Add lime to raise pH for other crops."),
        ("What is the minimum support price?", "MSP is a guaranteed price set by the government to protect farmers from market crashes. It covers major crops like rice, wheat, and pulses."),
        ("How to prevent post-harvest losses?", "Reduce losses by harvesting at proper maturity, careful handling, correct drying, improved storage, pest control, and timely market access."),
        ("What is composting?", "Composting decomposes organic waste into nutrient-rich humus. Layer green and brown materials, keep moist, turn regularly. Ready in 2-6 months."),
        ("How do I test my soil?", "Collect soil samples from multiple spots at 6-8 inch depth, mix them, and send to a testing lab. Home kits work for basic pH and nutrient checks."),
        ("What is organic farming?", "Organic farming avoids synthetic chemicals and GMOs, using natural processes, organic inputs, and biological controls for sustainable food production."),
        ("How do I store grains properly?", "Store grains in clean, dry, airtight containers below 13% moisture. Use proper silos. Protect from pests with neem leaves. Monitor regularly."),
        ("What is NPK fertilizer?", "NPK contains Nitrogen for leaves, Phosphorus for roots and flowers, and Potassium for plant health. Numbers show the percentage of each nutrient."),
        ("How do I prevent soil erosion?", "Prevent erosion with ground cover, contour farming, terracing slopes, planting windbreaks, conservation tillage, and mulching."),
        ("What is precision agriculture?", "Precision agriculture uses GPS, sensors, drones, and data analytics to optimize farming operations, reducing waste and improving yields."),
    ]
    query_path = os.path.join(DATASET_DIR, "farmer_queries.csv")
    with open(query_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["questions", "answers"])
        writer.writeheader()
        for q, a in queries:
            writer.writerow({"questions": q, "answers": a})
    logger.info(f"Created query dataset with {len(queries)} entries at {query_path}")
    return query_path


def main():
    logger.info("=" * 60)
    logger.info("Setting up training datasets for FarmGenius")
    logger.info("=" * 60)

    create_faq_dataset()
    create_crop_dataset()
    create_weather_dataset()
    create_price_dataset()
    create_query_dataset()

    logger.info("=" * 60)
    logger.info("All datasets created successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
