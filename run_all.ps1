# Start the Backend using Uvicorn
Write-Host "Starting Backend..." -ForegroundColor Green
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "cd backend; .\venv\Scripts\activate; uvicorn main:app --host 0.0.0.0 --port 8001 --reload"

# Start the Frontend
Write-Host "Starting React Frontend..." -ForegroundColor Green
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"

Write-Host "Services started!" -ForegroundColor Cyan
Write-Host "Backend API: http://localhost:8001"
Write-Host "React Frontend: http://localhost:5173"
Write-Host "To run the Flutter app on an emulator, open another terminal and run:"
Write-Host "cd app"
Write-Host "flutter run"
