# BloomWatch - NASA Earth Observation Plant Bloom Monitor

## Overview

BloomWatch is a comprehensive web application that harnesses NASA's powerful suite of Earth observation satellites to monitor and visualize plant blooming events across the globe. Built for the NASA Space Apps Challenge, this tool addresses critical vegetation monitoring, prediction, and management needs at both global and local scales.

## Features

### 🌍 Global Bloom Monitoring
- **Interactive Global Map**: Real-time visualization of bloom events worldwide using NASA satellite data
- **Multi-Scale Analysis**: Seamlessly zoom from global patterns to local regions
- **Temporal Coverage**: Leverages decades of NASA Earth observation data

### 📊 Advanced Analytics
- **Vegetation Indices**: Multiple indices including NDVI, EVI, SAVI, and GNDVI
- **Temporal Trend Analysis**: Track bloom patterns across multiple years
- **Peak Season Identification**: Identify optimal blooming periods for different regions
- **Statistical Analysis**: Comprehensive metrics and trend calculations

### 🌱 Conservation Insights
- **Actionable Recommendations**: Data-driven conservation strategies
- **Ecological Implications**: Understanding of bloom events as bio-indicators
- **Management Support**: Tools for agricultural and conservation decision-making
- **Pollinator Coordination**: Timing recommendations for pollinator-dependent activities

### 🛰️ NASA Data Integration
- **Landsat Archive**: Decades of high-resolution satellite imagery
- **MODIS Data**: Daily global vegetation monitoring
- **Multi-Sensor Fusion**: Combines data from multiple NASA missions
- **Real-time Updates**: Continuous monitoring with latest satellite passes

## Technical Architecture

### Backend (Flask)
- **RESTful API**: Clean API endpoints for data access
- **NASA API Integration**: Direct connection to NASA Earth observation services
- **Vegetation Index Calculations**: Advanced algorithms for bloom detection
- **Temporal Analysis**: Multi-year trend analysis and pattern recognition

### Frontend (Interactive Dashboard)
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Interactive Maps**: Leaflet-based global bloom visualization
- **Real-time Charts**: Dynamic time series analysis with Chart.js
- **Modern UI**: Bootstrap 5 with custom styling and animations

### Data Processing
- **Bloom Detection Algorithms**: Sophisticated vegetation index analysis
- **Trend Analysis**: Statistical modeling of temporal patterns
- **Conservation Scoring**: Automated assessment of conservation priorities
- **Scalable Architecture**: Designed to handle global-scale data processing

## Installation

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd bloomwatch-starter
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
# Create .env file
echo "NASA_API_KEY=your_nasa_api_key_here" > .env
```

5. Run the application:
```bash
python backend/app.py
```

6. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Global Bloom Map
- View real-time bloom events across the globe
- Click on markers to see detailed bloom data
- Filter by vegetation index type
- Adjust time range for historical analysis

### Temporal Analysis
- Select different time ranges (1, 3, 5, or 10 years)
- View trend analysis and peak season identification
- Compare bloom patterns across different regions
- Export data for further analysis

### Conservation Insights
- Get automated recommendations based on bloom data
- Understand ecological implications of bloom events
- Access management strategies for different scenarios
- Coordinate with local conservation efforts

## NASA Data Sources

### Primary Satellites
- **Landsat**: High-resolution multispectral imagery (30m resolution)
- **MODIS**: Daily global vegetation monitoring (250m-1km resolution)
- **VIIRS**: Visible Infrared Imaging Radiometer Suite
- **AVIRIS**: Airborne Visible/Infrared imaging Spectrometer

### Vegetation Indices
- **NDVI (Normalized Difference Vegetation Index)**: Standard vegetation health indicator
- **EVI (Enhanced Vegetation Index)**: Improved sensitivity in high biomass regions
- **SAVI (Soil Adjusted Vegetation Index)**: Accounts for soil background effects
- **GNDVI (Green NDVI)**: Enhanced sensitivity to chlorophyll content

## Applications

### Agricultural Monitoring
- **Crop Bloom Timing**: Optimize planting and harvesting schedules
- **Yield Prediction**: Early indicators of crop health and productivity
- **Disease Management**: Pre and post-bloom disease monitoring
- **Pollinator Coordination**: Timing for pollinator-dependent crops

### Conservation Biology
- **Ecosystem Health**: Monitor vegetation health across protected areas
- **Invasive Species Detection**: Identify unusual bloom patterns
- **Climate Change Impact**: Track phenological shifts over time
- **Biodiversity Assessment**: Understand bloom diversity patterns

### Public Health
- **Pollen Forecasting**: Predict pollen production and distribution
- **Allergy Management**: Early warning systems for allergy sufferers
- **Air Quality**: Vegetation impact on atmospheric conditions
- **Disease Vector Monitoring**: Track vegetation changes affecting disease vectors

## Future Enhancements

### Planned Features
- **Machine Learning Models**: Predictive bloom forecasting
- **Species Identification**: AI-powered plant species recognition
- **Mobile App**: Native mobile application for field use
- **API Access**: Public API for third-party integrations
- **Data Export**: Multiple format support for data analysis

### Advanced Analytics
- **Climate Correlation**: Link bloom patterns to climate data
- **Ecosystem Modeling**: Complex ecosystem interaction analysis
- **Predictive Modeling**: Forecast future bloom events
- **Anomaly Detection**: Identify unusual bloom patterns

## Contributing

We welcome contributions to BloomWatch! Please see our contributing guidelines for details on:
- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **NASA**: For providing open access to Earth observation data
- **Space Apps Challenge**: For inspiring this project
- **Open Source Community**: For the amazing tools and libraries used
- **Conservation Organizations**: For their valuable feedback and use cases

## Contact

For questions, suggestions, or collaboration opportunities, please contact:
- Project Lead: [Your Name]
- Email: [your.email@example.com]
- GitHub: [your-github-username]

---

*BloomWatch - Where NASA's eyes in the sky meet Earth's blooming beauty*
