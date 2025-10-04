# Corn Prices Search Integration Guide

## 🎯 Your SerpApi URL Integration

Based on your SerpApi URL: `https://serpapi.com/search.json?q=corn+crop+prices&location=United+States&hl=en&gl=us&google_domain=google.com`

I've created a complete integration that matches your exact parameters and extends it for broader agricultural commodity searches.

## ✅ What's Been Implemented

### 1. Specialized Corn Prices Endpoint
- **URL**: `GET /api/serpapi/corn-prices`
- **Parameters**: Matches your exact SerpApi URL format
- **Demo Mode**: Works immediately without API key
- **Real Mode**: Uses your SerpApi URL when configured

### 2. Agricultural Commodity Search
- **URL**: `POST /api/serpapi/agricultural-prices`
- **Flexible**: Supports any commodity (corn, wheat, soybeans, etc.)
- **Location-based**: Supports different countries/regions
- **Price Types**: Crop prices, futures, spot prices, etc.

### 3. Frontend Interface
- **Quick Corn Prices Button**: One-click corn prices search
- **Commodity Selector**: Choose from 10+ agricultural commodities
- **Location Selector**: Search by country/region
- **Price Type Selector**: Different types of price data
- **Demo Mode**: Test functionality without API costs

## 🚀 How to Use Your SerpApi URL

### Option 1: Direct API Call (Matches Your URL)
```bash
curl -X GET "http://localhost:5001/api/serpapi/corn-prices"
```

This generates the exact URL format you provided:
```
https://serpapi.com/search.json?q=corn+crop+prices&location=United+States&hl=en&gl=us&google_domain=google.com
```

### Option 2: Custom Agricultural Search
```bash
curl -X POST "http://localhost:5001/api/serpapi/agricultural-prices" \
  -H "Content-Type: application/json" \
  -d '{
    "commodity": "corn",
    "location": "United States"
  }'
```

### Option 3: Frontend Interface
1. Start the app: `python backend/app.py`
2. Navigate to "Crop Prices" section
3. Click "Get Corn Prices" for your exact search
4. Or use the form for custom searches

## 🔧 Configuration for Real Data

### Step 1: Get SerpApi Key
1. Sign up at [serpapi.com](https://serpapi.com)
2. Get your API key from the dashboard

### Step 2: Install Dependencies
```bash
pip install google-search-results
```

### Step 3: Set Environment Variable
```bash
echo "SERPAPI_KEY=your_actual_api_key_here" >> .env
```

### Step 4: Restart Application
```bash
python backend/app.py
```

## 📊 API Response Format

Your corn prices search returns data in this format:

```json
{
  "query": "corn crop prices",
  "engine": "google",
  "timestamp": "2025-10-04T12:35:31.630346",
  "success": true,
  "data_source": "SerpApi",
  "total_results": 10,
  "search_type": "corn_prices",
  "commodity": "corn",
  "location": "United States",
  "serpapi_url": "https://serpapi.com/search.json?q=corn+crop+prices&location=United+States&hl=en&gl=us&google_domain=google.com",
  "results": [
    {
      "title": "Corn Futures Prices - CBOT Corn Price Chart",
      "link": "https://www.investing.com/commodities/corn",
      "snippet": "Get live corn futures prices and charts...",
      "position": 1
    }
  ],
  "metadata": {
    "search_info": {
      "total_results": "About 2,500,000 results",
      "time_taken_displayed": "0.32 seconds"
    }
  }
}
```

## 🌾 Supported Commodities

The system supports these agricultural commodities:
- **Corn** (your primary focus)
- **Wheat**
- **Soybeans**
- **Rice**
- **Cotton**
- **Sugar**
- **Coffee**
- **Cocoa**
- **Barley**
- **Oats**

## 🌍 Supported Locations

- **United States** (your default)
- **Canada**
- **Mexico**
- **Brazil**
- **Argentina**
- **China**
- **India**
- **European Union**
- **Australia**
- **Russia**

## 💡 Usage Examples

### Python Example
```python
import requests

# Your exact corn prices search
response = requests.get('http://localhost:5001/api/serpapi/corn-prices')
corn_data = response.json()

print(f"Found {corn_data['total_results']} corn price results")
print(f"SerpApi URL: {corn_data['serpapi_url']}")

# Custom commodity search
response = requests.post('http://localhost:5001/api/serpapi/agricultural-prices', 
    json={
        'commodity': 'wheat',
        'location': 'Canada'
    }
)
wheat_data = response.json()
```

### JavaScript Example
```javascript
// Corn prices search
const response = await fetch('/api/serpapi/corn-prices');
const cornData = await response.json();

console.log('Corn prices:', cornData.results);

// Custom search
const customResponse = await fetch('/api/serpapi/agricultural-prices', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        commodity: 'soybeans',
        location: 'Brazil'
    })
});
```

## 🎨 Frontend Features

### Quick Corn Prices
- One-click button for your exact search
- Demo mode for testing
- Real-time results display

### Advanced Search Form
- Commodity selection dropdown
- Location selection
- Price type selection
- Example buttons for common searches

### Results Display
- Clean card-based layout
- Price-specific styling (green theme)
- Source domain extraction
- External link buttons
- SerpApi URL display

## 🔄 Integration with BloomWatch

The corn prices search integrates seamlessly with your existing BloomWatch features:

1. **Plant Bloom Monitoring** + **Crop Prices** = Complete agricultural insight
2. **NASA Satellite Data** + **Market Prices** = Comprehensive crop analysis
3. **Climate Data** + **Price Trends** = Predictive agricultural modeling

## 🚀 Next Steps

1. **Test Demo Mode**: Use the interface to see how it works
2. **Get SerpApi Key**: Sign up for real data access
3. **Configure API**: Set your SERPAPI_KEY environment variable
4. **Customize**: Modify search parameters as needed
5. **Integrate**: Connect price data with your bloom monitoring features

## 📝 Files Modified

- `backend/app.py` - Added corn prices and agricultural endpoints
- `templates/index.html` - Added agricultural prices interface
- `static/js/app.js` - Added agricultural prices JavaScript
- `requirements.txt` - Added SerpApi dependency
- `README.md` - Updated documentation

## 🎉 Ready to Use!

Your corn prices search is now fully integrated and ready to use. The system works in demo mode immediately and will use your exact SerpApi URL format when you configure the API key.

Start the application and navigate to the "Crop Prices" section to see your corn prices search in action!
