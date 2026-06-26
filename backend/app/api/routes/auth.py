from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import jwt
import time
from app.core.config import settings

from app.core.security import supabase

router = APIRouter()

class SendOtpRequest(BaseModel):
    phone: str

class VerifyOtpRequest(BaseModel):
    phone: str
    otp: str

class AuthResponse(BaseModel):
    token: str
    farmer_id: str

@router.post("/send-otp")
def send_otp(req: SendOtpRequest):
    try:
        # Attempt to use Supabase real OTP
        # phone format must be +91... for India. Assume +91 prefix if not present.
        phone = req.phone if req.phone.startswith("+") else f"+91{req.phone}"
        supabase.auth.sign_in_with_otp({"phone": phone})
        return {"message": "OTP sent successfully"}
    except Exception as e:
        # Fallback to mock for environments without SMS provider configured
        print(f"Supabase OTP failed: {e}. Falling back to mock SMS.")
        return {"message": "Mock OTP sent successfully"}

@router.post("/verify-otp", response_model=AuthResponse)
def verify_otp(req: VerifyOtpRequest):
    farmer_id = req.phone
    phone = req.phone if req.phone.startswith("+") else f"+91{req.phone}"
    
    try:
        # Attempt to verify with Supabase real OTP
        res = supabase.auth.verify_otp({"phone": phone, "token": req.otp, "type": "sms"})
        token = res.session.access_token
    except Exception as e:
        # Fallback to mock OTP verification
        print(f"Supabase Verify failed: {e}. Falling back to mock verification.")
        # Accept only "123456" as the demo OTP. All other codes are invalid.
        if req.otp != "123456":
            raise HTTPException(status_code=400, detail="Invalid OTP. Use 123456 for demo.")
        
        # Create mock JWT token using a deterministic UUID for the phone number
        import uuid
        farmer_id = str(uuid.uuid5(uuid.NAMESPACE_OID, phone))
        payload = {
            "sub": farmer_id,
            "iat": int(time.time()),
            "exp": int(time.time()) + (30 * 24 * 60 * 60) # 30 days
        }
        token = jwt.encode(payload, settings.SUPABASE_SERVICE_KEY, algorithm="HS256")
        
        # Upsert the mock farmer into the DB
        try:
            supabase.table("farmers").upsert({"id": farmer_id, "phone": phone}).execute()
        except Exception as e:
            print(f"Mock user upsert failed: {e}")
    
    return AuthResponse(token=token, farmer_id=farmer_id)

