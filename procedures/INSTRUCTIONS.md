# STM32Cube.AI Deployment Instructions
## Converting ONNX Model to STM32 C Code

---

## Prerequisites

✅ **Software Required:**
1. STM32CubeMX (latest version)
2. X-CUBE-AI expansion pack
3. Your trained ONNX model: `fire_detection.onnx`
4. STM32CubeIDE (for development)

✅ **Hardware:**
- STM32L4Q5T6P evaluation board (or custom board)
- ST-LINK programmer

---

## Step-by-Step Instructions

### Step 1: Install STM32CubeMX and X-CUBE-AI

1. Download STM32CubeMX:
   - Go to: https://www.st.com/en/development-tools/stm32cubemx.html
   - Create ST account (free)
   - Download and install

2. Install X-CUBE-AI Pack:
   ```
   STM32CubeMX → Help → Manage embedded software packages
   → STMicroelectronics → X-CUBE-AI → Install latest version
   ```

### Step 2: Create New STM32 Project

1. Open STM32CubeMX
2. File → New Project
3. **Board Selector** tab → Search "STM32L4Q5" → Select STM32L4Q5T6P
4. Click "Start Project"
5. When asked "Initialize all peripherals with default Mode?": **NO**

### Step 3: Configure Peripherals

**Enable these peripherals for your sensors:**

#### I2C1 (for sensors: SHT41, SGP41, SCD41)
```
Pinout & Configuration → Connectivity → I2C1
Mode: I2C
I2C Speed Mode: Standard Mode (100 kHz) or Fast Mode (400 kHz)
Pins: 
  - PB8: I2C1_SCL
  - PB9: I2C1_SDA
```

#### SPI1 (for LoRa: RFM95W)
```
Pinout & Configuration → Connectivity → SPI1
Mode: Full-Duplex Master
Data Size: 8 bits
Pins:
  - PA5: SPI1_SCK
  - PA6: SPI1_MISO
  - PA7: SPI1_MOSI
  - PA4: GPIO_Output (NSS - Chip Select)
```

#### USART2 (for debugging)
```
Pinout & Configuration → Connectivity → USART2
Mode: Asynchronous
Baud Rate: 115200
```

#### RTC (for timing)
```
Pinout & Configuration → Timers → RTC
Activate Clock Source: LSE (Low Speed External)
```

### Step 4: Add AI Model to Project

1. **Pinout & Configuration** → **Software Packs** → **Select Components**

2. Check: **STMicroelectronics.X-CUBE-AI**

3. Click **OK**

4. In **Software Packs** → **STMicroelectronics.X-CUBE-AI**:
   - **Mode**: Application Template → Check ✅

5. **Configuration** → Click **X-CUBE-AI**

6. **Add network** button:
   - Click **Browse** → Select your `fire_detection.onnx`
   - Network name: `fire_detection`
   - Click **Analyze**

### Step 5: Review Model Analysis

After clicking "Analyze", you'll see:

```
Model Information:
├── Input:  [1, 4] float32      (4 features: temp, humidity, tvoc, eco2)
├── Output: [1, 1] float32      (fire probability: 0 or 1)
├── Weights: ~50-100 KB
├── Activations: ~10-20 KB
└── Total Flash: ~80-150 KB
```

**Important Metrics to Check:**
- ✅ Flash usage < 1 MB (your MCU has 1 MB)
- ✅ RAM usage < 320 KB (your MCU has 320 KB)
- ✅ Inference time < 10 ms @ 120 MHz

### Step 6: Configure Quantization (CRITICAL for Size/Speed)

```
Configuration → X-CUBE-AI → fire_detection

Compression:      Medium (good balance)
Quantization:     int8 (reduces size ~75%, minimal accuracy loss)
```

**Expected Results:**
- Before quantization: ~100 KB Flash, ~94.2% accuracy
- After int8 quantization: ~25-30 KB Flash, ~93.5% accuracy

Click **Validate on Target** (optional but recommended)

### Step 7: Configure Clock

```
Clock Configuration tab:

Input frequency: 8 MHz (HSE)
PLL Source: HSE
System Clock: 120 MHz
```

Make sure:
- HCLK = 120 MHz (max for STM32L4Q5)
- APB1 = 120 MHz
- APB2 = 120 MHz

### Step 8: Configure Project Settings

```
Project Manager tab:

Project Name: CampusFireDetection
Project Location: [Your workspace]
Toolchain/IDE: STM32CubeIDE
```

**Code Generation Settings:**
- ✅ Generate peripheral initialization as pair of '.c/.h'
- ✅ Keep user code when re-generating
- ✅ Delete previously generated files when not re-generated

### Step 9: Generate Code

1. Click **GENERATE CODE** (top right)
2. Wait for code generation (1-2 minutes)
3. Click **Open Project** when prompted

### Step 10: Verify Generated Files

Check that these files exist:

```
CubeMX_Project/
├── X-CUBE-AI/
│   ├── App/
│   │   ├── fire_detection.c          ✅ AI model C code
│   │   ├── fire_detection.h          ✅ AI header
│   │   ├── fire_detection_data.c     ✅ Model weights
│   │   ├── fire_detection_data.h
│   │   └── app_x-cube-ai.c           ✅ Integration template
│   └── Target/
│       └── (platform-specific files)
└── Core/
    ├── Inc/
    │   └── main.h
    └── Src/
        └── main.c
```

---

## Generated Code Structure

### Key Files Explained

#### 1. `fire_detection.h`
Contains AI model interface:
```c
// Model input size (4 features)
#define AI_FIRE_DETECTION_IN_1_SIZE  4

// Model output size (1 prediction)
#define AI_FIRE_DETECTION_OUT_1_SIZE 1

// Main functions
ai_bool ai_fire_detection_init(ai_handle network);
ai_i32 ai_fire_detection_run(ai_handle network, 
                              const ai_buffer* input, 
                              ai_buffer* output);
```

#### 2. `fire_detection_data.c`
Contains your trained XGBoost model weights (DO NOT EDIT!)

#### 3. `app_x-cube-ai.c`
Template for AI integration (you'll modify this)

---

## Integration Example

### Create `fire_detection_app.h`:

```c
#ifndef FIRE_DETECTION_APP_H
#define FIRE_DETECTION_APP_H

#include "fire_detection.h"

// Scaler parameters from Python training
typedef struct {
    float temp_min;
    float temp_max;
    float hum_min;
    float hum_max;
    float tvoc_min;
    float tvoc_max;
    float eco2_min;
    float eco2_max;
} ScalerParams_t;

// Initialize AI model
int8_t FireDetection_Init(void);

// Run inference
int8_t FireDetection_Predict(float temperature, float humidity, 
                              float tvoc, float eco2, 
                              float *fire_probability);

#endif
```

### Create `fire_detection_app.c`:

```c
#include "fire_detection_app.h"
#include <string.h>

// ⚠️ CRITICAL: Copy these exact values from scaler_parameters.json
static const ScalerParams_t scaler = {
    .temp_min = 15.234567f,   // ← FROM YOUR scaler_parameters.json
    .temp_max = 65.876543f,   // ← FROM YOUR scaler_parameters.json
    .hum_min  = 25.345678f,   // ← FROM YOUR scaler_parameters.json
    .hum_max  = 95.234567f,   // ← FROM YOUR scaler_parameters.json
    .tvoc_min = 0.000000f,    // ← FROM YOUR scaler_parameters.json
    .tvoc_max = 60000.000f,   // ← FROM YOUR scaler_parameters.json
    .eco2_min = 400.000000f,  // ← FROM YOUR scaler_parameters.json
    .eco2_max = 5000.000000f  // ← FROM YOUR scaler_parameters.json
};

static ai_handle fire_detection_model = AI_HANDLE_NULL;
static ai_float input_data[AI_FIRE_DETECTION_IN_1_SIZE];
static ai_float output_data[AI_FIRE_DETECTION_OUT_1_SIZE];

// Min-max normalization
static inline float normalize(float value, float min_val, float max_val) {
    float range = max_val - min_val;
    if (range == 0.0f) return 0.5f;
    float normalized = (value - min_val) / range;
    
    // Clamp to [0, 1]
    if (normalized < 0.0f) normalized = 0.0f;
    if (normalized > 1.0f) normalized = 1.0f;
    
    return normalized;
}

int8_t FireDetection_Init(void) {
    ai_error err;
    
    // Create AI model
    err = ai_fire_detection_create(&fire_detection_model, 
                                    AI_FIRE_DETECTION_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE) {
        return -1;
    }
    
    // Initialize AI model
    if (!ai_fire_detection_init(fire_detection_model)) {
        return -2;
    }
    
    return 0;
}

int8_t FireDetection_Predict(float temperature, float humidity, 
                              float tvoc, float eco2, 
                              float *fire_probability) {
    ai_i32 batch;
    ai_buffer ai_input[AI_FIRE_DETECTION_IN_NUM];
    ai_buffer ai_output[AI_FIRE_DETECTION_OUT_NUM];
    
    // Normalize inputs
    input_data[0] = normalize(temperature, scaler.temp_min, scaler.temp_max);
    input_data[1] = normalize(humidity, scaler.hum_min, scaler.hum_max);
    input_data[2] = normalize(tvoc, scaler.tvoc_min, scaler.tvoc_max);
    input_data[3] = normalize(eco2, scaler.eco2_min, scaler.eco2_max);
    
    // Setup input buffer
    ai_input[0] = AI_BUFFER_OBJ_INIT(
        AI_FLAG_NONE, AI_BUFFER_FORMAT_FLOAT,
        NULL, NULL,
        input_data, AI_FIRE_DETECTION_IN_1_SIZE
    );
    
    // Setup output buffer
    ai_output[0] = AI_BUFFER_OBJ_INIT(
        AI_FLAG_NONE, AI_BUFFER_FORMAT_FLOAT,
        NULL, NULL,
        output_data, AI_FIRE_DETECTION_OUT_1_SIZE
    );
    
    // Run inference
    batch = ai_fire_detection_run(fire_detection_model, &ai_input[0], &ai_output[0]);
    
    if (batch != 1) {
        return -3;
    }
    
    // Get result
    *fire_probability = output_data[0];
    
    return 0;
}
```

### Use in `main.c`:

```c
#include "fire_detection_app.h"

int main(void) {
    HAL_Init();
    SystemClock_Config();
    
    // Initialize peripherals
    MX_I2C1_Init();
    MX_SPI1_Init();
    MX_USART2_UART_Init();
    
    // Initialize AI model
    if (FireDetection_Init() != 0) {
        Error_Handler();
    }
    
    while (1) {
        // Read sensors
        float temp = SHT41_ReadTemperature();
        float hum = SHT41_ReadHumidity();
        float tvoc = SGP41_ReadTVOC();
        float eco2 = 0.0f;  // Read after ML triggers
        
        // Run AI inference
        float fire_prob;
        if (FireDetection_Predict(temp, hum, tvoc, eco2, &fire_prob) == 0) {
            if (fire_prob > 0.5f) {
                // FIRE DETECTED!
                TriggerAlarm();
            }
        }
        
        HAL_Delay(10000);  // 10 second interval
    }
}
```

---

## Troubleshooting

### Problem: "Network analysis failed"
**Solution:**
- Check ONNX file is valid
- Try re-exporting from Python with `opset=12`
- Update X-CUBE-AI to latest version

### Problem: "Model doesn't fit in memory"
**Solution:**
- Enable int8 quantization
- Reduce model complexity (fewer trees in XGBoost)
- Use compression: Medium or High

### Problem: "Inference time > 10ms"
**Solution:**
- Reduce model complexity
- Enable compiler optimization (-O3)
- Use FPU for float calculations

### Problem: "Predictions are wrong"
**Solution:**
- ⚠️ **CHECK SCALER PARAMETERS!**
- Verify you copied exact values from `scaler_parameters.json`
- Test with known fire/no-fire samples
- Verify sensor readings are in correct units

---

## Validation

### On-Device Testing:

1. Click **X-CUBE-AI → Validate on Target**
2. Connect ST-LINK to your board
3. Wait for validation (uploads test data to MCU)
4. Check accuracy matches Python results (~93-94%)

### Expected Performance:

| Metric | Python (training) | STM32 (int8) | Acceptable? |
|--------|-------------------|--------------|-------------|
| Accuracy | 94.2% | 93.5% | ✅ Yes (≥90% required) |
| Inference Time | N/A | ~9 ms | ✅ Yes (<10 ms target) |
| Flash Usage | N/A | ~30 KB | ✅ Yes (<<1 MB) |
| RAM Usage | N/A | ~15 KB | ✅ Yes (<<320 KB) |

---

## Next Steps

After successful deployment:

1. ✅ Test with real sensor data
2. ✅ Verify detection accuracy
3. ✅ Measure power consumption
4. ✅ Test false alarm scenarios (cooking, steam, etc.)
5. ✅ Integrate with LoRa communication
6. ✅ Field test on campus

---

## Additional Resources

- **X-CUBE-AI Documentation**: https://www.st.com/x-cube-ai
- **STM32L4 Reference Manual**: Search "RM0432" on st.com
- **Application Notes**: AN5392 (Getting started with X-CUBE-AI)
- **YouTube**: Search "STM32 X-CUBE-AI tutorial"

---

**Good luck with deployment! 🚀**
