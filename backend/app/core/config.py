from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""
    # JWT Secret from Supabase Dashboard → Settings → API → JWT Secret
    SUPABASE_JWT_SECRET: str = ""
    DATABASE_URL: str = ""
    GROQ_API_KEY: str = ""
    DATA_GOV_IN_API_KEY: str = ""
    ENVIRONMENT: str = "production"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
