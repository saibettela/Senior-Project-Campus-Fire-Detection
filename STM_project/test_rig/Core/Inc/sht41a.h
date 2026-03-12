#ifndef INC_SHT41A_H_
#define INC_SHT41A_H_

#include "main.h"
#include "sensirion_crc.h"
#include <stdint.h>

#define SHT41A_I2C_ADDR         (0x44 << 1)

/* measurement commands */
#define SHT41A_CMD_MEAS_HIGH    0xFD
#define SHT41A_CMD_MEAS_MED     0xF6
#define SHT41A_CMD_MEAS_LOW     0xE0

/* heater commands */
#define SHT41A_CMD_HEAT_200_1S  0x39
#define SHT41A_CMD_HEAT_200_01S 0x32
#define SHT41A_CMD_HEAT_110_1S  0x2F
#define SHT41A_CMD_HEAT_110_01S 0x24
#define SHT41A_CMD_HEAT_20_1S   0x1E
#define SHT41A_CMD_HEAT_20_01S  0x15

/* utility commands */
#define SHT41A_CMD_SERIAL       0x89
#define SHT41A_CMD_RESET        0x94

/* measurement timeouts (ms) */
#define SHT41A_DELAY_HIGH_MS    9
#define SHT41A_DELAY_MED_MS     5
#define SHT41A_DELAY_LOW_MS     2

typedef struct {
    float temperature_c;
    float humidity_pct;
} sht41a_data_t;

HAL_StatusTypeDef sht41a_measure(I2C_HandleTypeDef *hi2c, uint8_t cmd,
                                  uint32_t delay_ms, sht41a_data_t *out);
HAL_StatusTypeDef sht41a_get_serial(I2C_HandleTypeDef *hi2c, uint32_t *serial);
HAL_StatusTypeDef sht41a_reset(I2C_HandleTypeDef *hi2c);
HAL_StatusTypeDef sht41a_heater(I2C_HandleTypeDef *hi2c, uint8_t cmd,
                                 uint32_t delay_ms);

#endif /* INC_SHT41A_H_ */
