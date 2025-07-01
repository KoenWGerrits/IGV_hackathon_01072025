# Volg de onderstaande stappen om de fast API zelf te hosten via uvicorn

# Stap 1:
Installeer Python (liefst de nieuwste versie)

# Stap 2 Zorg voor een virtuele omgeving (optioneel maar aanbevolen)
*bash*

python -m venv venv

source venv/bin/activate    # Op Linux/macOS

venv\Scripts\activate       # Op Windows

# Stap 3 Installeer de benodigde modules
*bash*

pip install -r requirements.txt

# Stap 4 Host de fast API met uvicorn
*bash*

uvicorn main:app --reload

# Stap 5 Startup fastAPI
Na het runnen van je command in stap 4 zou je de volgende regels terug moeten krijgen:
*INFO←[0m:     Uvicorn running on ←[1mhttp://127.0.0.1:8000←[0m (Press CTRL+C to quit)

←[32mINFO←[0m:     Started reloader process [←[36m←[1m6788←[0m] using ←[36m←[1mWatchFiles←[0m

←[32mINFO←[0m:     Started server process [←[36m12040←[0m]

←[32mINFO←[0m:     Waiting for application startup.

←[32mINFO←[0m:     Application startup complete.*

Dit betekend dat je fast API gehost wordt op de port die genoemd wordt in de *INFO* regel (wss automatisch http://127.0.0.1:8000)

# Stap 6 Testen van zelf gehoste API
Nadat stap 5 succesvol is gedaan, kun je de API lokaal via je machine bevragen op *http://127.0.0.1:8000/natuurbrand_model/{weerstation_nummer}*

Hiervoor heb je de inloggegevens nodig die ook op deze github pagina te vinden zijn.
