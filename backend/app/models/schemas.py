from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    query: str
    language: str
    farmer_id: str

class ChatResponse(BaseModel):
    response: str
    source: str
    confidence: float

class PriceResponse(BaseModel):
    commodity: str
    district: str
    min_price: float
    max_price: float
    modal_price: float
    date: str
    unit: str

class WeatherForecast(BaseModel):
    date: str
    max_temp: float
    min_temp: float
    rainfall_mm: float
    wind_kmh: float
    farming_advisory: str

class SchemeResponse(BaseModel):
    scheme_name: str
    description: str
    eligibility: str
    link: str

class DiseaseResponse(BaseModel):
    disease_name: str
    confidence: float
    disease_name_hi: str
    organic_treatment: str
    chemical_treatment: str
    dosage: str
    source_url: str
    source_name: str
