#ifndef INC_SGP41_H_
#define INC_SGP41_H_

#include "main.h"
#include "sensirion_crc.h"
#include <stdint.h>

#define SGP41_I2C_ADDR              (0x59 << 1)

/* commands */
#define SGP41_CMD_CONDITIONING      0x2612
#define SGP41_CMD_MEASURE_RAW       0x2619
#define SGP41_CMD_SELF_TEST         0x280E
#define SGP41_CMD_HEATER_OFF        0x3615
#define SGP41_CMD_SERIAL            0x3682
#define SGP41_CMD_RESET             0x0006  /* general call */

/* timing (ms) */
#define SGP41_DELAY_CONDITIONING_MS 50
#define SGP41_DELAY_MEASURE_MS      50
#define SGP41_DELAY_SELF_TEST_MS    320
#define SGP41_CONDITIONING_DURATION_MS 10000

/* default compensation values (no humidity sensor available) */
#define SGP41_DEFAULT_RH_TICKS      0x8000
#define SGP41_DEFAULT_RH_CRC        0xA2
#define SGP41_DEFAULT_T_TICKS       0x6666
#define SGP41_DEFAULT_T_CRC         0x93

typedef struct {
    uint16_t sraw_voc;
    uint16_t sraw_nox;
} sgp41_raw_t;

/* run once at startup for exactly SGP41_CONDITIONING_DURATION_MS before measuring */
HAL_StatusTypeDef sgp41_conditioning(I2C_HandleTypeDef *hi2c, uint16_t *sraw_voc);

/* call every 1s during normal operation */
HAL_StatusTypeDef sgp41_measure_raw(I2C_HandleTypeDef *hi2c,
                                     float rh_pct, float temp_c,
                                     sgp41_raw_t *out);

HAL_StatusTypeDef sgp41_self_test(I2C_HandleTypeDef *hi2c);
HAL_StatusTypeDef sgp41_get_serial(I2C_HandleTypeDef *hi2c, uint64_t *serial);
HAL_StatusTypeDef sgp41_reset(I2C_HandleTypeDef *hi2c);

#endif /* INC_SGP41_H_ */
