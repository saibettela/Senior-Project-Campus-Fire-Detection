# Dataset Directory

## Purpose
Contains all dataset files for training the fire detection AI model.

## Structure

### raw/
Original, unprocessed data downloaded from Kaggle.

**File:** `smoke_detection_iot.csv`
- Source: https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset
- Size: ~62,630 samples
- Features: Temperature, Humidity, TVOC, eCO2, Fire Alarm (and others)

### processed/
Cleaned and preprocessed data ready for model training.

**Files:**
- `train_data.csv` - 80% of data for training
- `test_data.csv` - 20% of data for testing
- `scaler_parameters.json` - **CRITICAL:** Min/max values for normalization

## How to Download Dataset

### Method 1: Kaggle CLI (Recommended)
```bash
# Install Kaggle CLI
pip install kaggle

# Set up credentials
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:\Users\<You>\.kaggle\ (Windows)

# Download dataset
cd 02_Dataset/raw/
kaggle datasets download -d deepcontractor/smoke-detection-dataset
unzip smoke-detection-dataset.zip
```

### Method 2: Manual Download
1. Go to: https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset
2. Click "Download" (requires Kaggle account)
3. Extract `smoke_detection_iot.csv` to `raw/` folder

## Dataset Features

| Column | Description | Your Sensor | Used in Model? |
|--------|-------------|-------------|----------------|
| UTC | Timestamp | - | ❌ Not used |
| Temperature[C] | Temperature in Celsius | SHT41 | ✅ **YES** |
| Humidity[%] | Relative humidity | SHT41 | ✅ **YES** |
| TVOC[ppb] | Total Volatile Organic Compounds | SGP41 | ✅ **YES** |
| eCO2[ppm] | Equivalent CO₂ | SCD41 | ✅ **YES** |
| Raw H2 | Hydrogen (raw sensor) | - | ❌ Not available |
| Raw Ethanol | Ethanol (raw sensor) | - | ❌ Not available |
| Pressure[hPa] | Atmospheric pressure | - | ❌ Not available |
| PM1.0 | Particulate matter 1.0μm | - | ❌ Not available |
| PM2.5 | Particulate matter 2.5μm | - | ❌ Not available |
| NC0.5 | Particle count 0.5μm | - | ❌ Not available |
| NC1.0 | Particle count 1.0μm | - | ❌ Not available |
| NC2.5 | Particle count 2.5μm | - | ❌ Not available |
| CNT | Sample counter | - | ❌ Not used |
| Fire Alarm | **Target variable** (0=No, 1=Yes) | - | ✅ **TARGET** |

## Quick Stats

After preprocessing:
- **Total samples:** ~60,000
- **Fire samples:** ~15,000 (25%)
- **No-fire samples:** ~45,000 (75%)
- **Features used:** 4 (Temperature, Humidity, TVOC, eCO2)

## Important Notes

⚠️ **CRITICAL:** The `scaler_parameters.json` file contains the exact min/max values used during normalization. These MUST be hardcoded into your STM32 firmware for correct predictions.

Example `scaler_parameters.json`:
```json
{
  "features": ["temperature", "humidity", "tvoc", "eco2"],
  "min_values": [15.2, 25.3, 0.0, 400.0],
  "max_values": [65.8, 95.2, 60000.0, 5000.0],
  "range": [50.6, 69.9, 60000.0, 4600.0]
}
```

These values go directly into your C code:
```c
float temp_min = 15.2;
float temp_max = 65.8;
// ... etc
```

## Next Steps

After downloading the dataset:
1. Run `03_Model_Training/01_data_exploration.py` to explore the data
2. Run `03_Model_Training/02_data_preprocessing.py` to create processed files
3. Proceed to model training
