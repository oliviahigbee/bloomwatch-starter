# Quick Test Guide

## 🚀 Test Your SerpApi Integration

### 1. Start the Application
```bash
cd /Users/olivehigbee/bloomwatch-starter
source venv/bin/activate
python backend/app.py
```

### 2. Test the API Endpoints Directly

**Test Corn Prices:**
```bash
curl -X GET "http://localhost:5000/api/serpapi/corn-prices"
```

**Test Status:**
```bash
curl -X GET "http://localhost:5000/api/serpapi/status"
```

**Test Demo:**
```bash
curl -X GET "http://localhost:5000/api/serpapi/demo"
```

### 3. Test in Browser

**Debug Page (Simplified):**
- Go to: `http://localhost:5000/debug`
- Should automatically test the API and show results

**Main Page:**
- Go to: `http://localhost:5000/`
- Scroll to "Web Scraping & Data Collection" section
- Click "Load Demo" button
- Should show demo search results

**Corn Prices:**
- Scroll to "Agricultural Commodity Prices" section
- Click "Demo Corn Prices" button
- Should show corn prices demo results

### 4. Check Browser Console

1. Open browser console (F12)
2. Look for console.log messages:
   - "DOM loaded, initializing SerpApi managers..."
   - "SerpApiManager: Checking status..."
   - "SerpApiManager: Status response:"
   - "SerpApiManager: Status data:"

### 5. Expected Results

**In Demo Mode (no API key):**
- Status shows: "SerpApi not configured"
- Demo searches return sample data
- All functionality works with demo data

**With Real API Key:**
- Status shows: "SerpApi service is ready"
- Real search results from SerpApi
- Your exact URL format: `https://serpapi.com/search.json?q=corn+crop+prices&location=United+States&hl=en&gl=us&google_domain=google.com`

## 🔧 Troubleshooting

### If No Results Show:
1. Check browser console for errors
2. Verify Flask app is running on port 5000
3. Test API endpoints directly with curl
4. Check if JavaScript is loading properly

### If API Calls Fail:
1. Check if Flask app is running
2. Verify port 5000 is not blocked
3. Check for CORS issues
4. Test with the debug page first

### If Demo Mode Doesn't Work:
1. Check JavaScript console for errors
2. Verify HTML elements exist
3. Test with simplified debug page
4. Check if Bootstrap CSS is loading

## 📞 Next Steps

1. **Test the debug page first**: `http://localhost:5000/debug`
2. **Check browser console** for any errors
3. **Test API endpoints** with curl commands
4. **Report what you see** - are there any error messages?

The integration should work in demo mode immediately. If you're not seeing results, there's likely a JavaScript error or the page isn't loading properly.
