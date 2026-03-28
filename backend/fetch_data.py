import requests
import os

def download_weather_data(lat=28.61, lon=77.20): # Default: New Delhi
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # NASA POWER API Settings
    START, END = "20230101", "20251231"
    params = "T2M,ALLSKY_SFC_SW_DWN,RH2M,WS2M,PS"
    url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
           f"parameters={params}&community=AG&longitude={lon}&latitude={lat}&"
           f"start={START}&end={END}&format=CSV")

    print("🛰️ Connecting to NASA POWER...")
    response = requests.get(url)
    
    if response.status_code == 200:
        path = os.path.join('data', 'raw_weather.csv')
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"✅ Data saved to: {path}")
    else:
        print("❌ Download failed.")

if __name__ == "__main__":
    download_weather_data()