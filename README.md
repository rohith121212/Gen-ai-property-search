# Property Search Assistant (Python - Streamlit) - Enhanced

This is the enhanced Python version of the Property Search Assistant for Bengaluru.

## Features
- Filters: keyword, location, price range, type, area
- Interactive Folium map showing markers for properties
- Favorites saved to `data/favorites.json`
- Average price and locality charts

## Run locally
1. Create a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows use: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Streamlit:
   ```bash
   streamlit run app.py
   ```

## Dataset
The dataset `data/Bengaluru_House_Data_with_coords.json` is included. To update, replace this file and restart the app.
